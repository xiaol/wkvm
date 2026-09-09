"""CUDA-graph decode with resident rows for the Qwen3.5 hybrid runner (H5).

A decode step through the HF decoder layers is ~6,300 kernel launches for
~23 ms of GPU work at B=1 (``experiments/hybrid_decode_profile.py``), so the
step is launch-bound. This module captures the *forward* once per
``(batch bucket, length bucket)`` and replays it.

Between steps, a running request's state stays **resident** in a row of the
static buffers: the Gated DeltaNet state/conv are updated in place by the
captured forward, and the new attention K/V token is written in-graph at
column ``lens[row]`` of the row's K/V window. Nothing is gathered or
scattered per step; only ``ids`` and ``lens`` are copied host -> device and
the logits are reordered to the batch order.

The bank (recurrent slots + paged K/V) stays the durable owner. A row is
filled from the bank once when the request first decodes (``acquire``),
written back when anything eager needs the state (``flush``: store export,
snapshot) or when the request leaves the graph path (``evict``: prefill,
eager fallback, parking), and simply dropped on finish/abort (``release``).
Rows are kept contiguous ``[0, n)`` by moving the last row into a departing
row's place, so bucket ``b >= n`` reads rows ``[0, b)`` with rows ``[n, b)``
as harmless dummies (their outputs are discarded, their state is garbage
that is never read for a real request).

Static state, allocated once at the largest bucket and sliced as views for
smaller buckets (views keep the addresses graphs captured):

- ``ids [Bmax, 1]``, ``lens [Bmax]`` (host -> device copies before replay)
- per GDN layer: ``state [Bmax, H_v, K, V]`` fp32, ``conv [Bmax, conv_dim, k]``
- per attention layer: ``k_win / v_win [Bmax, kv_heads, Lmax+1, hd]``

Invariant: every resident row is in every graphed decode batch. A resident
request that is not in the batch (parked, mid-prefill crumb) is evicted
before the step, so the replay can never advance a row with stale inputs.
"""

from __future__ import annotations

import torch

from wkvm.models.qwen35 import GDN_CONV_FAMILY, GDN_STATE_DTYPE, GDN_STATE_FAMILY, GUEST_KV_FAMILY
from wkvm.runner.hybrid_state import Qwen35StateBank, SlotCache, _AttnEntry, _GdnEntry


def _bucket(n: int, buckets: tuple[int, ...]) -> int | None:
    for b in buckets:
        if n <= b:
            return b
    return None


class ResidentSlotCache(SlotCache):
    """SlotCache over static row views: the new K/V token is written in
    place at column ``col_t[row]`` (``lens[row]`` for the paged layout, the
    ring column for ring mode; no concatenation), and attention reads the
    whole static window under the mask computed in ``_forward_static``."""

    def __init__(self, layout, b: int, rows_idx: torch.Tensor, lens_t: torch.Tensor, layers: dict) -> None:
        super().__init__(layout, [0] * b, layers)
        self.rows_idx = rows_idx
        self.lens_t = lens_t
        self.col_t = lens_t  # ring mode replaces this per forward (a graph-pool tensor)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *_, **__):
        entry = self.layers[layer_idx]
        entry.k_new, entry.v_new = key_states, value_states
        entry.k_past[self.rows_idx, :, self.col_t] = key_states[:, :, 0]
        entry.v_past[self.rows_idx, :, self.col_t] = value_states[:, :, 0]
        return entry.k_past, entry.v_past


class HybridDecodeGraphs:
    def __init__(
        self,
        model,
        bank: Qwen35StateBank,
        batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16),
        length_buckets: tuple[int, ...] = (256, 512, 1024, 2048, 4096),
        warmup_iters: int = 2,
    ) -> None:
        self.model = model
        self.bank = bank
        self.layout = bank.layout
        self.device = bank.device
        self.batch_buckets = tuple(sorted(set(batch_buckets)))
        self.ring = bank.ring
        self.routed = getattr(bank, "routed", False)
        # Ring / routed mode: the window is the only "length bucket"; its
        # columns are the static row itself, so no +1 column and no length dispatch.
        fixed = self.ring or self.routed
        self.length_buckets = (self.layout.window_tokens,) if fixed else tuple(sorted(set(length_buckets)))
        self.warmup_iters = warmup_iters
        bmax, lmax = self.batch_buckets[-1], self.length_buckets[-1]
        lay = self.layout
        dt = lay.dtype
        self.bmax = bmax
        self.ids = torch.zeros((bmax, 1), dtype=torch.long, device=self.device)
        self.lens = torch.zeros((bmax,), dtype=torch.long, device=self.device)
        self.true_b = torch.ones(bmax, dtype=torch.bool, device=self.device)
        self.rows_idx = torch.arange(bmax, device=self.device)
        self.gdn_state = torch.zeros((lay.n_gdn, bmax, *lay.gdn_state_shape), dtype=GDN_STATE_DTYPE, device=self.device)
        self.gdn_conv = torch.zeros((lay.n_gdn, bmax, *lay.gdn_conv_shape), dtype=dt, device=self.device)
        if self.routed:
            from wkvm.runner.hybrid_routed import RoutedStore

            self.rstore = RoutedStore(bank.rg.p, lay.n_attn, bmax, lay.num_kv_heads, lay.head_dim, dt, self.device)
            self.k_win, self.v_win = self.rstore.k, self.rstore.v
        else:
            # Paged: +1 column so a request whose length equals the bucket writes at column l.
            cols = lmax if self.ring else lmax + 1
            self.k_win = torch.zeros((lay.n_attn, bmax, lay.num_kv_heads, cols, lay.head_dim), dtype=dt, device=self.device)
            self.v_win = torch.zeros_like(self.k_win)
        self._graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, ResidentSlotCache, torch.Tensor]] = {}
        self._pool = None
        # residency
        self.row_of: dict[int, int] = {}  # gdn_state slot id -> row
        self.slots_of: list[dict | None] = [None] * bmax  # row -> slots
        self.synced_len: list[int] = [0] * bmax  # guest tokens already in the pages
        self.n_resident = 0
        self.stats = {"replays": 0, "captures": 0, "eager_fallbacks": 0, "acquires": 0,
                      "flushes": 0, "evicts": 0, "releases": 0, "row_moves": 0}
        bank.resident = self

    # -- buckets --------------------------------------------------------------------

    def bucket_for(self, batch: int, lmax: int) -> tuple[int, int] | None:
        """Bucket key, or None when the step must run eagerly. Ring mode
        never falls back on length: the window absorbs any context."""
        b = _bucket(batch, self.batch_buckets)
        if b is None:
            return None
        if self.ring or self.routed:
            return (b, self.layout.window_tokens)
        l = _bucket(lmax, self.length_buckets)
        return None if l is None else (b, l)

    def _cache(self, b: int, l: int) -> ResidentSlotCache:
        layers: dict = {}
        for j, li in enumerate(self.layout.gdn_layers):
            layers[li] = _GdnEntry(conv=self.gdn_conv[j, :b], state=self.gdn_state[j, :b])
        ncol = l if (self.ring or self.routed) else l + 1
        for j, li in enumerate(self.layout.attn_layers):
            layers[li] = _AttnEntry(k_past=self.k_win[j, :b, :, :ncol], v_past=self.v_win[j, :b, :, :ncol])
        return ResidentSlotCache(self.layout, b, self.rows_idx[:b], self.lens[:b], layers)

    def _forward_static(self, b: int, l: int, cache: ResidentSlotCache) -> torch.Tensor:
        ids = self.ids[:b]
        lens = self.lens[:b]
        positions = lens[:, None]  # [b, 1], T == 1
        if self.routed:
            # In-graph: the ring column about to be overwritten is moved to
            # pending (or to the scratch column when it holds nothing), the
            # new token's column becomes valid, and attention reads every
            # valid column. Routing of pending runs out-graph after the step.
            p, st, rows = self.bank.rg.p, self.rstore, self.rows_idx[:b]
            col = self.bank.rg.ring_columns(lens)
            do_evict = (lens >= p.sink) & st.valid[0, rows, col]
            dst = torch.where(do_evict, p.pend_base + st.pend[:b], torch.full_like(lens, p.scratch))
            for j in range(self.layout.n_attn):
                st.k[j][rows, :, dst] = st.k[j][rows, :, col]
                st.v[j][rows, :, dst] = st.v[j][rows, :, col]
                st.valid[j][rows, dst] = do_evict
            st.pos[rows, dst] = st.pos[rows, col]
            st.is_break[rows, dst] = st.is_break[rows, col]
            st.pend[:b] += do_evict.long()
            st.valid[:, rows, col] = self.true_b[:b]  # device bool, no host scalar under capture
            st.pos[rows, col] = lens
            st.is_break[rows, col] = self.bank.break_lut[ids[:, 0]]
            cache.col_t = col
            mask = st.valid[0, :b][:, None, None, :]
            return self.model(ids, cache, positions, mask)
        if self.ring:
            # Window columns [0, min(len+1, window)) are the sinks + the last
            # ring tokens including the one written this step at ring_column(len).
            wt = self.layout.window_tokens
            cache.col_t = self.bank.ring_columns(lens)
            cols = torch.arange(wt, device=self.device)
            mask = (cols[None, :] < (lens[:, None] + 1).clamp(max=wt))[:, None, None, :]
        else:
            cache.col_t = lens
            cols = torch.arange(l + 1, device=self.device)
            mask = (cols[None, :] <= lens[:, None])[:, None, None, :]  # [b,1,1,l+1]; new token at column lens
        return self.model(ids, cache, positions, mask)

    def _capture(self, b: int, l: int):
        """Capture bucket (b, l). Callers copy the current ``ids``/``lens``
        in first (warm-up indexes the K/V windows at ``lens``); the rows'
        state is saved and restored around the warm-up/capture forwards."""
        cache = self._cache(b, l)
        saved = self._save_rows(b, l)
        try:
            # Capture must happen on a stream of the bank's device. Relying on
            # torch.cuda.graph's default side stream is not enough: once graphs
            # exist on another device in the same process, a capture on
            # ``cuda:1`` came out empty and its replay was a silent no-op
            # (caught by the recompute check below). An explicit stream bound
            # to ``self.device`` for warm-up and capture is what works.
            with torch.inference_mode(), torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup_iters):  # Triton autotune + allocator warm-up
                        self._forward_static(b, l, cache)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                if self._pool is None:
                    self._pool = torch.cuda.graph_pool_handle()
                with torch.cuda.graph(graph, pool=self._pool, stream=stream):
                    logits = self._forward_static(b, l, cache)
                # Sanity: a replay must actually recompute (guards the empty-graph case).
                before = logits.clone()
                vocab = self.layout.vocab_size
                self.ids[:b, 0] = (self.ids[:b, 0] + 1) % vocab
                graph.replay()
                torch.cuda.synchronize(self.device)
                changed = not torch.equal(before, logits)
                self.ids[:b, 0] = (self.ids[:b, 0] - 1) % vocab
                if not changed:
                    raise RuntimeError(
                        f"CUDA graph for bucket {(b, l)} replays without recomputing "
                        f"(captured on the wrong device/stream? bank device {self.device})"
                    )
        finally:
            self._restore_rows(b, l, saved)
        self.stats["captures"] += 1
        return graph, cache, logits

    def _save_rows(self, b: int, l: int):
        """Snapshot exactly what a forward mutates: the GDN state/conv of the
        first ``b`` rows and, per attention layer, the single K/V column each
        row writes at ``lens[row]`` (not the whole window — at B=16 x 14k
        tokens that clone alone is 3.4 GiB)."""
        rows = self.rows_idx[:b]
        if self.routed:
            # Touched per forward: the ring column, the scratch column and the
            # next few pending columns (warm-up + capture + check run ~4x).
            p, st = self.bank.rg.p, self.rstore
            col = self.bank.rg.ring_columns(self.lens[:b])
            span = torch.arange(self.warmup_iters + 3, device=self.device)
            pend_cols = (p.pend_base + st.pend[:b, None] + span[None, :]).clamp(max=p.summ_base - 1)
            cols = torch.cat([col[:, None], torch.full_like(col[:, None], p.scratch), pend_cols], dim=1)  # [b, m]
            r = rows[:, None].expand_as(cols)
            return ("routed", self.gdn_state[:, :b].clone(), self.gdn_conv[:, :b].clone(),
                    st.k[:, r, :, cols].clone(), st.v[:, r, :, cols].clone(), st.valid[:, r, cols].clone(),
                    st.pos[r, cols].clone(), st.is_break[r, cols].clone(), st.pend[:b].clone(), r, cols)
        cols = self.bank.ring_columns(self.lens[:b]) if self.ring else self.lens[:b]
        return ("plain", self.gdn_state[:, :b].clone(), self.gdn_conv[:, :b].clone(),
                self.k_win[:, rows, :, cols].clone(), self.v_win[:, rows, :, cols].clone(), rows, cols)

    def _restore_rows(self, b: int, l: int, saved) -> None:
        if saved[0] == "routed":
            _, gdn_state, gdn_conv, k, v, valid, pos, brk, pend, r, cols = saved
            st = self.rstore
            self.gdn_state[:, :b].copy_(gdn_state)
            self.gdn_conv[:, :b].copy_(gdn_conv)
            st.k[:, r, :, cols] = k
            st.v[:, r, :, cols] = v
            st.valid[:, r, cols] = valid
            st.pos[r, cols] = pos
            st.is_break[r, cols] = brk
            st.pend[:b].copy_(pend)
            return
        _, gdn_state, gdn_conv, k_cols, v_cols, rows, cols = saved
        self.gdn_state[:, :b].copy_(gdn_state)
        self.gdn_conv[:, :b].copy_(gdn_conv)
        self.k_win[:, rows, :, cols] = k_cols
        self.v_win[:, rows, :, cols] = v_cols

    # -- residency ------------------------------------------------------------------

    @staticmethod
    def _sid(slots: dict) -> int:
        return slots[GDN_STATE_FAMILY]

    def is_resident(self, slots: dict) -> bool:
        return self._sid(slots) in self.row_of

    def _acquire(self, slots: dict) -> int:
        """Fill the next free row from the bank."""
        if self.n_resident >= self.bmax:
            raise RuntimeError("no free graph row")
        row = self.n_resident
        bank = self.bank
        sid, cid = slots[GDN_STATE_FAMILY], slots[GDN_CONV_FAMILY]
        for j in range(self.layout.n_gdn):
            self.gdn_state[j, row].copy_(bank.gdn_state[j, sid])
            self.gdn_conv[j, row].copy_(bank.gdn_conv[j, cid])
        n = bank.slot_len(slots)
        if self.layout.n_attn and self.routed:
            self.rstore.copy_row(bank.rstore, slots[GUEST_KV_FAMILY], row)
        elif self.layout.n_attn and self.ring:
            g = slots[GUEST_KV_FAMILY]
            self.k_win[:, row].copy_(bank.ring_k[:, g])
            self.v_win[:, row].copy_(bank.ring_v[:, g])
        elif n and self.layout.n_attn:
            pg, off = bank._positions_index(tuple(slots[GUEST_KV_FAMILY]), 0, n)
            for j in range(self.layout.n_attn):
                self.k_win[j, row, :, :n] = bank.pool_k[j][pg, :, off].permute(1, 0, 2)
                self.v_win[j, row, :, :n] = bank.pool_v[j][pg, :, off].permute(1, 0, 2)
        self.row_of[sid] = row
        self.slots_of[row] = slots
        self.synced_len[row] = n
        self.n_resident += 1
        self.stats["acquires"] += 1
        return row

    def flush(self, slots: dict) -> None:
        """Row -> bank (recurrent state and the guest tokens not yet in the
        pages). The row stays resident and consistent with the bank."""
        row = self.row_of.get(self._sid(slots))
        if row is None:
            return
        bank = self.bank
        sid, cid = slots[GDN_STATE_FAMILY], slots[GDN_CONV_FAMILY]
        for j in range(self.layout.n_gdn):
            bank.gdn_state[j, sid].copy_(self.gdn_state[j, row])
            bank.gdn_conv[j, cid].copy_(self.gdn_conv[j, row])
        n, start = bank.slot_len(slots), self.synced_len[row]
        if self.layout.n_attn and self.routed:
            if n > start:
                bank.rstore.copy_row(self.rstore, row, slots[GUEST_KV_FAMILY])
        elif self.layout.n_attn and self.ring:
            if n > start:
                g = slots[GUEST_KV_FAMILY]
                bank.ring_k[:, g].copy_(self.k_win[:, row])
                bank.ring_v[:, g].copy_(self.v_win[:, row])
                bank.ring_pos[g].copy_(bank.ring_positions_for(n))
        elif n > start and self.layout.n_attn:
            pg, off = bank._positions_index(tuple(slots[GUEST_KV_FAMILY]), start, n - start)
            for j in range(self.layout.n_attn):
                bank.pool_k[j][pg, :, off] = self.k_win[j, row, :, start:n].permute(1, 0, 2)
                bank.pool_v[j][pg, :, off] = self.v_win[j, row, :, start:n].permute(1, 0, 2)
        self.synced_len[row] = n
        self.stats["flushes"] += 1

    def evict(self, slots: dict) -> None:
        """Flush, then free the row (the request continues eagerly)."""
        if self.is_resident(slots):
            self.flush(slots)
            self._free_row(self.row_of[self._sid(slots)])
            self.stats["evicts"] += 1

    def release(self, slots: dict) -> None:
        """Drop the row without writing back (state discarded: finish/abort,
        or re-admission zeroing the slot)."""
        row = self.row_of.get(self._sid(slots))
        if row is not None:
            self._free_row(row)
            self.stats["releases"] += 1

    def _free_row(self, row: int) -> None:
        last = self.n_resident - 1
        sid = self._sid(self.slots_of[row])
        del self.row_of[sid]
        if row != last:  # keep rows contiguous: move the last row into the hole
            moved = self.slots_of[last]
            n = self.bank.slot_len(moved)
            self.gdn_state[:, row].copy_(self.gdn_state[:, last])
            self.gdn_conv[:, row].copy_(self.gdn_conv[:, last])
            if self.layout.n_attn and self.routed:
                self.rstore.copy_row(self.rstore, last, row)
            elif self.layout.n_attn and self.ring:
                self.k_win[:, row].copy_(self.k_win[:, last])
                self.v_win[:, row].copy_(self.v_win[:, last])
            elif n and self.layout.n_attn:
                self.k_win[:, row, :, :n].copy_(self.k_win[:, last, :, :n])
                self.v_win[:, row, :, :n].copy_(self.v_win[:, last, :, :n])
            self.row_of[self._sid(moved)] = row
            self.slots_of[row] = moved
            self.synced_len[row] = self.synced_len[last]
            self.stats["row_moves"] += 1
        if self.routed:  # a vacated row must not keep valid columns (it may pad a bucket later)
            self.rstore.reset(last)
        self.slots_of[last] = None
        self.synced_len[last] = 0
        self.n_resident = last

    # -- the step -----------------------------------------------------------------

    def decode_step(self, slot_batch: list[dict], last_tokens: list[int]) -> torch.Tensor | None:
        """Replay a captured graph for this batch; None => caller runs eager
        (after evicting any resident rows of the batch, via bank.gather)."""
        bank = self.bank
        lens = [bank.slot_len(s) for s in slot_batch]
        key = self.bucket_for(len(slot_batch), max(lens))
        if key is None:
            self.stats["eager_fallbacks"] += 1
            return None
        for slots in slot_batch:
            bank.check_capacity(slots, 1)
        # Invariant: resident rows == this batch. Evict residents that are absent.
        in_batch = {self._sid(s) for s in slot_batch}
        for sid in [s for s in self.row_of if s not in in_batch]:
            self.evict(self.slots_of[self.row_of[sid]])
        for slots in slot_batch:
            if not self.is_resident(slots):
                self._acquire(slots)
        b, l = key
        n = len(slot_batch)
        order = [self.row_of[self._sid(s)] for s in slot_batch]
        host_ids = [0] * b
        host_lens = [0] * b
        for row, tok, ln in zip(order, last_tokens, lens):
            host_ids[row] = tok
            host_lens[row] = ln
        self.ids[:b, 0].copy_(torch.tensor(host_ids, dtype=torch.long), non_blocking=True)
        self.lens[:b].copy_(torch.tensor(host_lens, dtype=torch.long), non_blocking=True)
        if key not in self._graphs:
            self._graphs[key] = self._capture(b, l)
        graph, cache, logits = self._graphs[key]
        with torch.cuda.device(self.device):
            graph.replay()
        self.stats["replays"] += 1
        for slots, ln in zip(slot_batch, lens):
            bank.guest_len[self._sid(slots)] = ln + 1
        if self.routed:
            # Route rows whose pending buffer reached the threshold (out-graph).
            pend = self.rstore.pend[:n].tolist()
            for row_i, cnt in enumerate(pend):
                if cnt >= bank.rg.p.pending:
                    slots = self.slots_of[row_i]
                    bank.rg.route(self.rstore, row_i, session=self._sid(slots))
        if order == list(range(n)):
            return logits[:n]
        return logits.index_select(0, torch.tensor(order, device=self.device))
