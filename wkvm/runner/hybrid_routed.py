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
    dup_floor: float = 0.10  # a new span within 1-floor cosine of a kept one is a duplicate
    new_slot_sim: float = 0.60  # open a free slot when the best centroid cosine is below this

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


def farthest_point_keep(cands: list[dict], budget: int, dup_floor: float, centroid: torch.Tensor) -> list[dict]:
    """Greedy farthest-point selection of spans under a token budget: start
    from the span closest to the slot's centroid, repeatedly add the
    candidate farthest from every kept span, skip near-duplicates (cosine >
    1 - dup_floor to a kept span) and spans that no longer fit."""
    if not cands:
        return []
    feats = torch.stack([c["feat"] for c in cands])  # [n, F]
    sims = feats @ feats.T
    start = int((feats @ centroid).argmax()) if float(centroid.norm()) > 0 else 0
    kept = [start]
    used = cands[start]["len"]
    remaining = [i for i in range(len(cands)) if i != start]
    while remaining:
        best_i, best_d = None, -1.0
        for i in remaining:
            d = float(1.0 - sims[i, kept].max())
            if d > best_d:
                best_i, best_d = i, d
        remaining.remove(best_i)
        if best_d < dup_floor:
            continue
        if used + cands[best_i]["len"] > budget:
            continue
        kept.append(best_i)
        used += cands[best_i]["len"]
    return [cands[i] for i in kept]


class SlotState:
    """Host-side bookkeeping for one (session, layer)."""

    __slots__ = ("centroid", "count", "spans")

    def __init__(self, params: RoutedParams, feat_dim: int, device) -> None:
        self.centroid = torch.zeros((params.slots, feat_dim), device=device)  # unit vectors, 0 = free
        self.count = [0] * params.slots
        self.spans: list[list[dict]] = [[] for _ in range(params.slots)]  # {"feat","len","col"}


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
        self.pend = torch.zeros((rows,), dtype=torch.long, device=device)
        self.device = device

    def reset(self, r: int) -> None:
        self.valid[:, r].zero_()
        self.pos[r].fill_(-1)
        self.is_break[r].zero_()
        self.pend[r] = 0

    def copy_row(self, src: "RoutedStore", src_r: int, dst_r: int) -> None:
        self.k[:, dst_r].copy_(src.k[:, src_r])
        self.v[:, dst_r].copy_(src.v[:, src_r])
        self.valid[:, dst_r].copy_(src.valid[:, src_r])
        self.pos[dst_r].copy_(src.pos[src_r])
        self.is_break[dst_r].copy_(src.is_break[src_r])
        self.pend[dst_r] = src.pend[src_r]

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.valid, self.pos, self.is_break))


class RoutedGuest:
    """The routing logic over any :class:`RoutedStore`, with per-session
    host state. One per bank; the graph runner calls it on its own rows."""

    def __init__(self, params: RoutedParams, n_attn: int, kv_heads: int, head_dim: int, device) -> None:
        self.p = params
        self.n_attn = n_attn
        self.feat_dim = kv_heads * head_dim
        self.device = device
        self.state: dict[tuple[int, int], SlotState] = {}  # (session id, layer) -> state
        self.stats = {"routing_passes": 0, "spans": 0, "new_slots": 0, "dropped_spans": 0}

    # -- host state per session ------------------------------------------------------

    def slot_state(self, session: int, layer: int) -> SlotState:
        key = (session, layer)
        if key not in self.state:
            self.state[key] = SlotState(self.p, self.feat_dim, self.device)
        return self.state[key]

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
            payload[layer] = {"centroid": st.centroid.cpu(), "count": list(st.count),
                              "spans": [[{"feat": s["feat"].cpu(), "len": s["len"], "col": s["col"]} for s in sl]
                                        for sl in st.spans]}
        buf = io.BytesIO()
        torch.save(payload, buf)
        return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)

    def import_session(self, session: int, blob: torch.Tensor) -> None:
        self.drop_session(session)
        payload = torch.load(io.BytesIO(blob.numpy().tobytes()), weights_only=False)
        for layer, d in payload.items():
            st = self.slot_state(session, layer)
            st.centroid.copy_(d["centroid"].to(self.device))
            st.count = list(d["count"])
            st.spans = [[{"feat": s["feat"].to(self.device), "len": s["len"], "col": s["col"], "src": None}
                         for s in sl] for sl in d["spans"]]

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
        store.pend[r] = cur + n

    # -- routing pass -----------------------------------------------------------------------

    def needs_routing(self, store: RoutedStore, r: int) -> bool:
        return int(store.pend[r]) >= self.p.pending

    def route(self, store: RoutedStore, r: int, session: int) -> None:
        """Route row ``r``'s pending tokens into its span bank (every layer)."""
        p = self.p
        n = int(store.pend[r])
        if n == 0:
            return
        pend_cols = torch.arange(p.pend_base, p.pend_base + n, device=store.device)
        spans = split_spans(store.is_break[r, pend_cols], p.max_span, p.fallback_span)
        self.stats["routing_passes"] += 1
        self.stats["spans"] += len(spans)
        pos_p = store.pos[r, pend_cols]
        for layer in range(self.n_attn):
            st = self.slot_state(session, layer)
            k_p = store.k[layer, r, :, pend_cols]  # [kvh, n, hd]
            v_p = store.v[layer, r, :, pend_cols]
            for a, b in spans:
                length = b - a
                v_span = v_p[:, a:b]
                feat = _unit(v_span.float().mean(dim=1).flatten())
                sims = st.centroid @ feat
                best = int(sims.argmax())
                free = next((i for i in range(p.slots) if st.count[i] == 0), None)
                if free is not None and (st.count[best] == 0 or float(sims[best]) < p.new_slot_sim):
                    target = free
                    self.stats["new_slots"] += 1
                else:
                    target = best
                c_old = st.count[target]
                summ_col = p.summ_base + target
                k_mean = k_p[:, a:b].float().mean(dim=1)
                v_mean = v_span.float().mean(dim=1)
                if c_old == 0:
                    store.k[layer, r, :, summ_col] = k_mean.to(store.k.dtype)
                    store.v[layer, r, :, summ_col] = v_mean.to(store.v.dtype)
                    st.centroid[target] = feat
                    if layer == 0:
                        store.pos[r, summ_col] = int(pos_p[a])
                else:
                    w_old, w_new = c_old / (c_old + length), length / (c_old + length)
                    store.k[layer, r, :, summ_col] = (store.k[layer, r, :, summ_col].float() * w_old + k_mean * w_new).to(store.k.dtype)
                    store.v[layer, r, :, summ_col] = (store.v[layer, r, :, summ_col].float() * w_old + v_mean * w_new).to(store.v.dtype)
                    st.centroid[target] = _unit(st.centroid[target] * w_old + feat * w_new)
                st.count[target] = c_old + length
                store.valid[layer, r, summ_col] = True
                cand = list(st.spans[target]) + [{"feat": feat, "len": length, "src": (a, b), "col": None}]
                kept = farthest_point_keep(cand, p.reps, p.dup_floor, st.centroid[target])
                self.stats["dropped_spans"] += len(cand) - len(kept)
                self._rewrite_reps(store, r, layer, target, kept, k_p, v_p, pos_p)
                st.spans[target] = kept
        store.valid[:, r, pend_cols] = False
        store.pos[r, pend_cols] = -1
        store.is_break[r, pend_cols] = False
        store.pend[r] = 0

    def _rewrite_reps(self, store: RoutedStore, r: int, layer: int, target: int, kept: list[dict],
                      k_p: torch.Tensor, v_p: torch.Tensor, pos_p: torch.Tensor) -> None:
        """Lay the kept spans out contiguously in the slot's rep block;
        resident spans move within the block, the new span comes from
        pending. Positions are one tensor shared by the layers (they are
        bookkeeping only, K is stored after RoPE); layer 0 writes them."""
        p = self.p
        b0, b1 = p.rep_block(target)
        old_k = store.k[layer, r, :, b0:b1].clone()
        old_v = store.v[layer, r, :, b0:b1].clone()
        old_pos = store.pos[r, b0:b1].clone()
        store.valid[layer, r, b0:b1] = False
        cursor = b0
        for sp in kept:
            length = sp["len"]
            if sp["col"] is not None:
                o = sp["col"] - b0
                store.k[layer, r, :, cursor:cursor + length] = old_k[:, o:o + length]
                store.v[layer, r, :, cursor:cursor + length] = old_v[:, o:o + length]
                if layer == 0:
                    store.pos[r, cursor:cursor + length] = old_pos[o:o + length]
            else:
                a, b = sp["src"]
                store.k[layer, r, :, cursor:cursor + length] = k_p[:, a:b]
                store.v[layer, r, :, cursor:cursor + length] = v_p[:, a:b]
                if layer == 0:
                    store.pos[r, cursor:cursor + length] = pos_p[a:b]
            store.valid[layer, r, cursor:cursor + length] = True
            sp["col"], sp["src"] = cursor, None  # this layer's own bookkeeping (SlotState is per layer)
            cursor += length
        if layer == 0:
            store.pos[r, cursor:b1] = -1

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
