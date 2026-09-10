"""Routed span bank for the hybrid's guest layers (``guest_mode="routed"``).

Fixed-size per-session attention memory, the Gemma routed-span mechanism
(``docs/gemma_native_contract.md``) ported to Qwen3.5's 8 full-attention
layers. Per layer, per session, one static column space:

    [ sink S | ring W | pending 2P | summaries M | reps M x R | scratch ]

- **sink / ring**: exactly as ring mode (first S tokens forever, last W
  tokens exact, keys post-RoPE at absolute positions).
- **pending**: tokens evicted from the ring, still exact and visible, in
  eviction (= position) order. Capacity 2P; a routing pass runs once P are
  waiting (out-graph, host-driven).
- **routing pass**: pending tokens are cut into spans at sentence
  punctuation / newline tokens (``max_span`` cap, fixed ``fallback_span``
  when a run has no break). Each span is routed *atomically* by its mean
  **value** vector (post-RoPE keys cluster by position; values do not) to
  one of M slots: nearest centroid, or a free slot when the best cosine is
  below ``new_slot_sim``. A slot keeps a running mean of its tokens' K and V
  (the **summary**, one column) and exact **representatives**: whole spans
  kept under a budget of R tokens by greedy farthest-point selection with a
  near-duplicate floor, so repeated filler cannot crowd out distinct facts.
- **readout**: attention sees every valid column (sink, ring within the
  band, pending, summaries, representatives). For q=1 decode all stored
  columns are in the past, so the mask is just ``valid``.
- **scratch**: one extra column that in-graph evictions of rows with
  nothing to evict write into (never valid), so the decode step is a fixed
  set of index operations with no host branch.

A :class:`RoutedStore` holds these tensors for N rows; the bank owns one
per arena slot and the CUDA-graph runner one per resident row. The
per-(session, layer) routing bookkeeping (:class:`SlotState`) lives on the
host, keyed by session, and travels with the session between the two.

Bytes per session per layer = (S + W + 2P + M + M*R + 1) tokens of K/V;
with the defaults (16, 1024, 2x512, 64, 48) that is 5,201 columns, 163 MiB
across the 8 layers — bounded, whatever the context.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RoutedParams:
    sink: int = 16
    ring: int = 1024
    pending: int = 512  # routing pass threshold; the buffer holds 2x
    slots: int = 64
    reps: int = 48  # representative token budget per slot
    max_span: int = 48
    fallback_span: int = 32
    dup_floor: float = 0.02  # a new span within 1-floor cosine of a kept one is a duplicate
    new_slot_sim: float = 0.60  # open a free slot when the best centroid cosine is below this
    shared: bool = True  # decide once per session from all layers' features; else per layer
    retention: str = "novelty"  # "novelty": one pool, spans unlike their neighbours in time; "surprisal": one pool by
    # token surprisal; "fps": per-slot farthest-point retention (the original Gemma rule)

    @property
    def pend_cap(self) -> int:
        return 2 * self.pending

    @property
    def ring_base(self) -> int:
        return self.sink

    @property
    def pend_base(self) -> int:
        return self.sink + self.ring

    @property
    def summ_base(self) -> int:
        return self.pend_base + self.pend_cap

    @property
    def reps_base(self) -> int:
        return self.summ_base + self.slots

    @property
    def scratch(self) -> int:
        return self.reps_base + self.slots * self.reps

    @property
    def columns(self) -> int:
        """All columns including the scratch column."""
        return self.scratch + 1

    def rep_block(self, slot: int) -> tuple[int, int]:
        b = self.reps_base + slot * self.reps
        return b, b + self.reps


def split_spans(is_break: torch.Tensor, max_span: int, fallback_span: int) -> list[tuple[int, int]]:
    """Cut ``n`` pending tokens (position order) into [start, end) spans:
    after each break token, at ``max_span``, or every ``fallback_span``
    tokens when the run contains no break at all."""
    n = int(is_break.numel())
    if n == 0:
        return []
    breaks = is_break.nonzero().flatten().tolist()
    if not breaks:
        return [(s, min(s + fallback_span, n)) for s in range(0, n, fallback_span)]
    bset = set(breaks)
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(n):
        if (i in bset) or (i - start + 1 >= max_span) or i == n - 1:
            spans.append((start, i + 1))
            start = i + 1
    return spans


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _unit_np(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-6)


def farthest_point_keep(cands: list[dict], budget: int, dup_floor: float, centroid: np.ndarray) -> list[dict]:
    """Greedy farthest-point selection of spans under a token budget (host
    numpy): start from the span farthest from the slot's centroid, repeatedly
    add the candidate farthest from every kept span, skip near-duplicates
    (cosine > 1 - dup_floor to a kept span) and spans that no longer fit."""
    if not cands:
        return []
    feats = np.stack([c["feat"] for c in cands])  # [n, F]
    sims = feats @ feats.T
    # The summary column already represents the slot's mean, so the kept
    # spans should complement it: start from the span least like the
    # centroid, then keep adding the one farthest from everything kept.
    start = int((feats @ centroid).argmin()) if float(np.linalg.norm(centroid)) > 0 else 0
    kept = [start]
    used = cands[start]["len"]
    remaining = [i for i in range(len(cands)) if i != start]
    while remaining:
        d = 1.0 - sims[np.ix_(remaining, kept)].max(axis=1)
        j = int(d.argmax())
        i, best_d = remaining.pop(j), float(d[j])
        if best_d < dup_floor:
            continue
        if used + cands[i]["len"] > budget:
            continue
        kept.append(i)
        used += cands[i]["len"]
    return [cands[i] for i in kept]


class SlotState:
    """Host-side bookkeeping for one (session, layer): unit centroids and
    span features live in numpy so routing decisions never touch the GPU."""

    __slots__ = ("centroid", "count", "spans", "pool")

    def __init__(self, params: RoutedParams, feat_dim: int) -> None:
        self.centroid = np.zeros((params.slots, feat_dim), dtype=np.float32)  # unit vectors, 0 = free
        self.count = [0] * params.slots
        self.spans: list[list[dict]] = [[] for _ in range(params.slots)]  # {"feat","len","col","src","pos"}
        self.pool: list[dict] = []  # retention="surprisal": the one global pool of kept spans (+"sal")


class RoutedStore:
    """Column-space tensors for ``rows`` rows (bank slots or graph rows)."""

    def __init__(self, params: RoutedParams, n_attn: int, rows: int, kv_heads: int, head_dim: int,
                 dtype: torch.dtype, device) -> None:
        self.p = params
        self.n_attn, self.rows = n_attn, rows
        c = params.columns
        self.k = torch.zeros((n_attn, rows, kv_heads, c, head_dim), dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.valid = torch.zeros((n_attn, rows, c), dtype=torch.bool, device=device)
        self.pos = torch.full((rows, c), -1, dtype=torch.long, device=device)
        self.is_break = torch.zeros((rows, c), dtype=torch.bool, device=device)
        self.sal = torch.zeros((rows, c), dtype=torch.float32, device=device)  # token surprisal per column
        self.pend = torch.zeros((rows,), dtype=torch.long, device=device)
        self.device = device

    def reset(self, r: int) -> None:
        self.valid[:, r].zero_()
        self.pos[r].fill_(-1)
        self.is_break[r].zero_()
        self.sal[r].zero_()
        self.pend[r] = 0

    def copy_row(self, src: "RoutedStore", src_r: int, dst_r: int) -> None:
        self.k[:, dst_r].copy_(src.k[:, src_r])
        self.v[:, dst_r].copy_(src.v[:, src_r])
        self.valid[:, dst_r].copy_(src.valid[:, src_r])
        self.pos[dst_r].copy_(src.pos[src_r])
        self.is_break[dst_r].copy_(src.is_break[src_r])
        self.sal[dst_r].copy_(src.sal[src_r])
        self.pend[dst_r] = src.pend[src_r]

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.valid, self.pos, self.is_break, self.sal))


class RoutedGuest:
    """The routing logic over any :class:`RoutedStore`, with per-session
    host state. One per bank; the graph runner calls it on its own rows."""

    def __init__(self, params: RoutedParams, n_attn: int, kv_heads: int, head_dim: int, device) -> None:
        self.p = params
        self.n_attn = n_attn
        self.feat_dim = (kv_heads * head_dim) * (n_attn if params.shared else 1)
        self.device = device
        if params.retention not in ("novelty", "surprisal", "fps"):
            raise ValueError(f"routed retention must be novelty|surprisal|fps, got {params.retention!r}")
        if params.max_span > params.reps or params.fallback_span > params.reps:
            raise ValueError("routed: max_span and fallback_span must not exceed the representative budget")
        self.state: dict[tuple[int, int], SlotState] = {}  # (session id, layer) -> state
        self.stats = {"routing_passes": 0, "spans": 0, "new_slots": 0, "spills": 0, "dropped_spans": 0}
        self.trace: list[dict] | None = None  # set to a list to record per-span salience signals (diagnostics)
        self._sink_feat: np.ndarray | None = None

    # -- host state per session ------------------------------------------------------

    def slot_state(self, session: int, layer: int) -> SlotState:
        key = (session, 0 if self.p.shared else layer)
        if key not in self.state:
            self.state[key] = SlotState(self.p, self.feat_dim)
        return self.state[key]

    def spans_for(self, session: int, layer: int) -> list[list[dict]]:
        """Kept spans per slot as seen by ``layer`` (shared: the same for all)."""
        st = self.state.get((session, 0 if self.p.shared else layer))
        if st is None:
            return []
        return st.spans if self.p.retention == "fps" else [st.pool]

    def drop_session(self, session: int) -> None:
        for layer in range(self.n_attn):
            self.state.pop((session, layer), None)

    def export_session(self, session: int) -> torch.Tensor:
        """Host state as a uint8 tensor (safetensors-friendly)."""
        payload = {}
        for layer in range(self.n_attn):
            st = self.state.get((session, layer))
            if st is None:
                continue
            def pack(sp):
                return {"feat": torch.from_numpy(np.ascontiguousarray(sp["feat"])), "len": sp["len"], "col": sp["col"],
                        "pos": sp.get("pos", -1), "sal": sp.get("sal", 0.0)}

            payload[layer] = {"centroid": torch.from_numpy(st.centroid.copy()), "count": list(st.count),
                              "spans": [[pack(sp) for sp in sl] for sl in st.spans], "pool": [pack(sp) for sp in st.pool]}
        buf = io.BytesIO()
        torch.save(payload, buf)
        return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)

    def import_session(self, session: int, blob: torch.Tensor) -> None:
        self.drop_session(session)
        payload = torch.load(io.BytesIO(blob.numpy().tobytes()), weights_only=False)
        for layer, d in payload.items():
            st = self.slot_state(session, layer)
            st.centroid[:] = d["centroid"].numpy()
            st.count = list(d["count"])
            def unpack(sp):
                return {"feat": sp["feat"].numpy().astype(np.float32), "len": sp["len"], "col": sp["col"], "src": None,
                        "pos": sp.get("pos", -1), "sal": sp.get("sal", 0.0)}

            st.spans = [[unpack(sp) for sp in sl] for sl in d["spans"]]
            st.pool = [unpack(sp) for sp in d.get("pool", [])]

    # -- column arithmetic ---------------------------------------------------------------

    def ring_columns(self, positions: torch.Tensor) -> torch.Tensor:
        p = self.p
        return torch.where(positions < p.sink, positions, p.sink + (positions - p.sink) % p.ring)

    # -- eager eviction: ring -> pending ----------------------------------------------------

    def evict_before_write(self, store: RoutedStore, r: int, start: int, count: int) -> None:
        """Positions [start, start+count) are about to overwrite their ring
        columns; move the valid tokens those columns hold to pending, in
        position order."""
        p = self.p
        if count <= 0:
            return
        pos = torch.arange(start, start + count, device=store.device)
        cols = self.ring_columns(pos)
        cols = cols[pos >= p.sink]
        if cols.numel() == 0:
            return
        cols = cols[store.valid[0, r, cols]]
        if cols.numel() == 0:
            return
        cols = cols[store.pos[r, cols].argsort()]
        n = int(cols.numel())
        cur = int(store.pend[r])
        if cur + n > p.pend_cap:
            raise RuntimeError(f"pending overflow: {cur}+{n} > {p.pend_cap} (prefill sub-chunk too large)")
        dst = torch.arange(p.pend_base + cur, p.pend_base + cur + n, device=store.device)
        store.k[:, r, :, dst] = store.k[:, r, :, cols]
        store.v[:, r, :, dst] = store.v[:, r, :, cols]
        store.valid[:, r, dst] = True
        store.pos[r, dst] = store.pos[r, cols]
        store.is_break[r, dst] = store.is_break[r, cols]
        store.sal[r, dst] = store.sal[r, cols]
        store.pend[r] = cur + n

    # -- routing pass -----------------------------------------------------------------------

    def needs_routing(self, store: RoutedStore, r: int) -> bool:
        return int(store.pend[r]) >= self.p.pending

    def route(self, store: RoutedStore, r: int, session: int) -> None:
        """Route row ``r``'s pending tokens into its span bank, every layer.

        Decisions run on the host from one device->host transfer (the spans'
        unit mean values, all layers); the store is then updated with a few
        batched index operations per layer: summaries blended as one linear
        combination (equal to the sequential running mean), representatives
        gathered from their old columns / pending and scattered into the
        re-laid-out slot blocks."""
        p = self.p
        n = int(store.pend[r])
        if n == 0:
            return
        dev = store.device
        pend = slice(p.pend_base, p.pend_base + n)
        is_break = store.is_break[r, pend].cpu()
        spans = split_spans(is_break, p.max_span, p.fallback_span)
        tail = None
        if len(spans) > 1 and n < p.pend_cap and not bool(is_break[spans[-1][1] - 1]):
            # The buffer ends mid-sentence: leave that tail pending for the
            # next pass instead of cutting a span at an arbitrary boundary.
            tail = spans.pop()
            n = tail[0]
            pend = slice(p.pend_base, p.pend_base + n)
        self.stats["routing_passes"] += 1
        self.stats["spans"] += len(spans)
        n_layers, kvh, hd = self.n_attn, store.k.shape[2], store.k.shape[4]
        feat_dim = kvh * hd
        assign = np.zeros((len(spans), n), dtype=np.float32)
        for i, (a, b) in enumerate(spans):
            assign[i, a:b] = 1.0 / (b - a)
        assign_t = torch.from_numpy(assign).to(dev)
        k_p = store.k[:, r, :, pend].permute(0, 2, 1, 3).reshape(n_layers, n, feat_dim).float()
        v_p = store.v[:, r, :, pend].permute(0, 2, 1, 3).reshape(n_layers, n, feat_dim).float()
        k_mean = torch.matmul(assign_t, k_p)  # [L, S, F]
        v_mean = torch.matmul(assign_t, v_p)
        pos_pend = store.pos[r, pend]
        pos_host = pos_pend.cpu().numpy()
        sal_host = store.sal[r, pend].cpu().numpy() if p.retention == "surprisal" else None
        if tail is not None:  # snapshot the tail before pending is cleared
            ta, tb = tail
            tcols = slice(p.pend_base + ta, p.pend_base + tb)
            keep = (store.k[:, r, :, tcols].clone(), store.v[:, r, :, tcols].clone(), store.pos[r, tcols].clone(),
                    store.is_break[r, tcols].clone(), store.sal[r, tcols].clone())
        if self.trace is not None:  # diagnostics: the prompt's own prefix (sink columns) as a standing query
            sink_v = store.v[:, r, :, :p.sink].float().mean(dim=2).reshape(n_layers, -1)  # [L, F]
            self._sink_feat = _unit(_unit(sink_v).reshape(-1)).cpu().numpy()
        if p.shared:
            # One decision per span from every layer's unit mean value
            # (concatenated, re-normalised), applied to all layers: the bank
            # is one memory, every layer keeps the same spans.
            feats = _unit(_unit(v_mean).permute(1, 0, 2).reshape(len(spans), -1)).cpu().numpy()
            plan = self._decide(self.slot_state(session, 0), spans, feats, pos_host, sal_host)
            for layer in range(n_layers):
                self._apply(store, r, layer, plan, k_mean[layer], v_mean[layer], pos_pend)
        else:
            feats = _unit(v_mean).cpu().numpy()  # the pass's one device->host transfer
            for layer in range(n_layers):
                plan = self._decide(self.slot_state(session, layer), spans, feats[layer], pos_host, sal_host)
                self._apply(store, r, layer, plan, k_mean[layer], v_mean[layer], pos_pend)
        full = slice(p.pend_base, p.pend_base + int(store.pend[r]))
        store.valid[:, r, full] = False
        store.pos[r, full] = -1
        store.is_break[r, full] = False
        store.sal[r, full] = 0.0
        store.pend[r] = 0
        if tail is not None:  # the unfinished span moves to the front of pending
            k_t, v_t, pos_t, brk_t, sal_t = keep
            m = tb - ta
            head = slice(p.pend_base, p.pend_base + m)
            store.k[:, r, :, head] = k_t
            store.v[:, r, :, head] = v_t
            store.valid[:, r, head] = True
            store.pos[r, head] = pos_t
            store.is_break[r, head] = brk_t
            store.sal[r, head] = sal_t
            store.pend[r] = m

    def _decide(self, st: SlotState, spans: list[tuple[int, int]], feats: np.ndarray, pos_host: np.ndarray,
                sal_host: np.ndarray | None = None):
        """Host routing of one layer's spans, in order: slot choice, running
        centroid/count, retention. Returns the summary blend per touched
        slot and the column moves that lay the kept spans out."""
        p = self.p
        if p.retention != "fps":
            return self._decide_pool(st, spans, feats, pos_host, sal_host)
        touched: dict[int, dict] = {}  # slot -> {"c0": count before the pass, "parts": [(span, len)]}
        for si, (a, b) in enumerate(spans):
            feat, length = feats[si], b - a
            sims = st.centroid @ feat
            best = int(sims.argmax())
            free = next((i for i in range(p.slots) if st.count[i] == 0), None)
            if free is not None and (st.count[best] == 0 or float(sims[best]) < p.new_slot_sim):
                target = free
                self.stats["new_slots"] += 1
            elif (free is not None and sum(sp["len"] for sp in st.spans[best]) + length > p.reps
                  and not any(float(sp["feat"] @ feat) > 1.0 - p.dup_floor for sp in st.spans[best])):
                # The nearest slot would have to drop something to hold this
                # span and it is no duplicate there: spill to a free slot, so
                # the bank's whole representative capacity gets used before
                # retention starts choosing.
                target = free
                self.stats["spills"] += 1
            else:
                target = best
            c_old = st.count[target]
            info = touched.setdefault(target, {"c0": c_old, "parts": [], "first_pos": a})
            info["parts"].append((si, length))
            if c_old == 0:
                st.centroid[target] = feat
            else:
                w_old, w_new = c_old / (c_old + length), length / (c_old + length)
                st.centroid[target] = _unit_np(st.centroid[target] * w_old + feat * w_new)
            st.count[target] = c_old + length
            cand = list(st.spans[target]) + [{"feat": feat, "len": length, "src": (a, b), "col": None,
                                              "pos": int(pos_host[a])}]
            kept = farthest_point_keep(cand, p.reps, p.dup_floor, st.centroid[target])
            self.stats["dropped_spans"] += len(cand) - len(kept)
            st.spans[target] = kept
        src, dst, blocks = [], [], []
        for m in touched:
            b0, b1 = p.rep_block(m)
            blocks.append((b0, b1))
            cursor = b0
            for sp in st.spans[m]:
                length = sp["len"]
                src.append((sp["col"] if sp["col"] is not None else p.pend_base + sp["src"][0], length))
                dst.append((cursor, length))
                sp["col"], sp["src"] = cursor, None
                cursor += length
        return touched, src, dst, blocks

    def _decide_pool(self, st: SlotState, spans, feats: np.ndarray, pos_host: np.ndarray, sal_host: np.ndarray):
        """Summaries by value cluster as before; representatives are ONE pool
        under the whole budget (slots x reps tokens), kept by token
        surprisal (mean of a span's top-4 -log p): the least predictable
        spans — random keys, numbers, names — are what the summaries and
        the recurrent state cannot reconstruct, so they stay exact."""
        p = self.p
        touched: dict[int, dict] = {}
        new = []
        for si, (a, b) in enumerate(spans):
            feat, length = feats[si], b - a
            sims = st.centroid @ feat
            best = int(sims.argmax())
            free = next((i for i in range(p.slots) if st.count[i] == 0), None)
            if free is not None and (st.count[best] == 0 or float(sims[best]) < p.new_slot_sim):
                target = free
                self.stats["new_slots"] += 1
            else:
                target = best
            c_old = st.count[target]
            info = touched.setdefault(target, {"c0": c_old, "parts": [], "first_pos": a})
            info["parts"].append((si, length))
            if c_old == 0:
                st.centroid[target] = feat
            else:
                w_old, w_new = c_old / (c_old + length), length / (c_old + length)
                st.centroid[target] = _unit_np(st.centroid[target] * w_old + feat * w_new)
            st.count[target] = c_old + length
            new.append({"feat": feat, "len": length, "src": (a, b), "col": None, "pos": int(pos_host[a]),
                        "sal": float(np.sort(sal_host[a:b])[-4:].mean()) if sal_host is not None else 0.0})
        if p.retention == "novelty" and new:
            # Local novelty: 1 - max cosine to the other spans of this pass —
            # a span unlike its neighbours in time (a needle in prose, a new
            # entry in a ledger) is what the summaries cannot stand in for.
            f_new = np.stack([c["feat"] for c in new])
            sims_new = f_new @ f_new.T
            np.fill_diagonal(sims_new, -1.0)
            for i, c in enumerate(new):
                c["sal"] = float(1.0 - sims_new[i].max()) if len(new) > 1 else 0.5
        if self.trace is not None and new:
            f_new = np.stack([c["feat"] for c in new])
            sims_new = f_new @ f_new.T
            np.fill_diagonal(sims_new, -1.0)
            for i, c in enumerate(new):
                a, b = c["src"]
                sal_span = sal_host[a:b] if sal_host is not None else np.zeros(1, dtype=np.float32)
                self.trace.append({"pos": c["pos"], "len": c["len"], "sal": c["sal"],
                                   "sal_top4": float(np.sort(sal_span)[-4:].mean()),
                                   "sal_mean": float(sal_span.mean()), "sal_sum": float(sal_span.sum()),
                                   "distinct": float(1.0 - sims_new[i].max()) if len(new) > 1 else 1.0,
                                   "sink_sim": float(c["feat"] @ self._sink_feat) if self._sink_feat is not None else 0.0})
        cand = list(st.pool) + new  # stable sort: on ties the resident span wins
        order = sorted(range(len(cand)), key=lambda i: -cand[i]["sal"])
        feats_all = np.stack([c["feat"] for c in cand])
        sims_all = feats_all @ feats_all.T
        budget = p.slots * p.reps
        kept_idx: list[int] = []
        used = 0
        for i in order:
            if used + cand[i]["len"] > budget:
                continue
            if kept_idx and float(sims_all[i, kept_idx].max()) > 1.0 - p.dup_floor:
                continue
            kept_idx.append(i)
            used += cand[i]["len"]
        kept = [cand[i] for i in sorted(kept_idx, key=lambda i: cand[i]["pos"])]  # pool laid out in position order
        self.stats["dropped_spans"] += len(cand) - len(kept)
        st.pool = kept
        src, dst = [], []
        cursor = p.reps_base
        for sp in kept:
            length = sp["len"]
            src.append((sp["col"] if sp["col"] is not None else p.pend_base + sp["src"][0], length))
            dst.append((cursor, length))
            sp["col"], sp["src"] = cursor, None
            cursor += length
        return touched, src, dst, [(p.reps_base, p.scratch)]

    def _apply(self, store: RoutedStore, r: int, layer: int, plan, k_mean: torch.Tensor, v_mean: torch.Tensor,
               pos_pend: torch.Tensor) -> None:
        touched, src, dst, blocks = plan
        if not touched:
            return
        p, dev = self.p, store.device
        kvh, hd = store.k.shape[2], store.k.shape[4]
        slots_t = list(touched)
        # Summaries: new = w_old * old + sum_s w_s * mean_s (the running mean over tokens).
        weights = np.zeros((len(slots_t), k_mean.shape[0]), dtype=np.float32)
        w_old = np.zeros((len(slots_t), 1), dtype=np.float32)
        for i, m in enumerate(slots_t):
            info = touched[m]
            total = info["c0"] + sum(length for _, length in info["parts"])
            w_old[i, 0] = info["c0"] / total
            for si, length in info["parts"]:
                weights[i, si] += length / total
        summ_cols = torch.tensor([p.summ_base + m for m in slots_t], device=dev)
        weights_t, w_old_t = torch.from_numpy(weights).to(dev), torch.from_numpy(w_old).to(dev)
        for tensor, mean in ((store.k, k_mean), (store.v, v_mean)):
            old = tensor[layer, r, :, summ_cols].float().permute(1, 0, 2).reshape(len(slots_t), -1)  # [T, F]
            new = w_old_t * old + weights_t @ mean
            tensor[layer, r, :, summ_cols] = new.reshape(len(slots_t), kvh, hd).permute(1, 0, 2).to(tensor.dtype)
        store.valid[layer, r, summ_cols] = True
        if layer == 0:
            fresh = [(p.summ_base + m, touched[m]["first_pos"]) for m in slots_t if touched[m]["c0"] == 0]
            if fresh:
                cols = torch.tensor([c for c, _ in fresh], device=dev)
                store.pos[r, cols] = pos_pend[torch.tensor([a for _, a in fresh], device=dev)]
        # Representatives: gather every source before any block is rewritten.
        block_t = torch.from_numpy(np.concatenate([np.arange(b0, b1) for b0, b1 in blocks])).to(dev)
        if not src:
            store.valid[layer, r, block_t] = False
            if layer == 0:
                store.pos[r, block_t] = -1
            return
        src_t = torch.from_numpy(np.concatenate([np.arange(c, c + length) for c, length in src])).to(dev)
        dst_t = torch.from_numpy(np.concatenate([np.arange(c, c + length) for c, length in dst])).to(dev)
        k_src = store.k[layer, r, :, src_t]
        v_src = store.v[layer, r, :, src_t]
        pos_src = store.pos[r, src_t] if layer == 0 else None
        sal_src = store.sal[r, src_t] if layer == 0 else None
        store.valid[layer, r, block_t] = False
        store.k[layer, r, :, dst_t] = k_src
        store.v[layer, r, :, dst_t] = v_src
        store.valid[layer, r, dst_t] = True
        if layer == 0:
            store.pos[r, block_t] = -1
            store.pos[r, dst_t] = pos_src
            store.sal[r, block_t] = 0.0
            store.sal[r, dst_t] = sal_src

    # -- masks ----------------------------------------------------------------------------

    def prefill_mask(self, store: RoutedStore, rows: list[int], lens: list[int], t: int) -> torch.Tensor:
        """[B, 1, T, C+T]: every valid column (the chunk's evictions already
        sit in pending, see ``Qwen35StateBank.gather``), causal over the new
        tokens. Visibility == validity: a token never blinks out between its
        ring column and pending, which is exactly what the CUDA-graph decode
        path computes in-graph."""
        p = self.p
        if t > p.ring:
            raise ValueError(f"prefill chunk {t} exceeds the ring ({p.ring}); tokens would be overwritten unseen")
        rid = torch.tensor(rows, device=store.device)
        q = torch.arange(t, device=store.device)
        allow_past = store.valid[0].index_select(0, rid)[:, None, :].expand(-1, t, -1)
        allow_new = (q[None, :] <= q[:, None])[None].expand(len(rows), -1, -1)
        return torch.cat([allow_past, allow_new], dim=-1)[:, None]
