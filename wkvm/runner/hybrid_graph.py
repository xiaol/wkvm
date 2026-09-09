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
    place at column ``lens[row]`` (no concatenation), and attention reads the
    whole ``l + 1``-column window under the mask ``col <= lens[row]``."""

    def __init__(self, layout, b: int, rows_idx: torch.Tensor, lens_t: torch.Tensor, layers: dict) -> None:
        super().__init__(layout, [0] * b, layers)
        self.rows_idx = rows_idx
        self.lens_t = lens_t

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *_, **__):
        entry = self.layers[layer_idx]
        entry.k_new, entry.v_new = key_states, value_states
        entry.k_past[self.rows_idx, :, self.lens_t] = key_states[:, :, 0]
        entry.v_past[self.rows_idx, :, self.lens_t] = value_states[:, :, 0]
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
        self.length_buckets = tuple(sorted(set(length_buckets)))
        self.warmup_iters = warmup_iters
        bmax, lmax = self.batch_buckets[-1], self.length_buckets[-1]
        lay = self.layout
        dt = lay.dtype
        self.bmax = bmax
        self.ids = torch.zeros((bmax, 1), dtype=torch.long, device=self.device)
        self.lens = torch.zeros((bmax,), dtype=torch.long, device=self.device)
        self.rows_idx = torch.arange(bmax, device=self.device)
        self.gdn_state = torch.zeros((lay.n_gdn, bmax, *lay.gdn_state_shape), dtype=GDN_STATE_DTYPE, device=self.device)
        self.gdn_conv = torch.zeros((lay.n_gdn, bmax, *lay.gdn_conv_shape), dtype=dt, device=self.device)
        # +1 column: a request whose length equals the bucket writes at column l.
        self.k_win = torch.zeros((lay.n_attn, bmax, lay.num_kv_heads, lmax + 1, lay.head_dim), dtype=dt, device=self.device)
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
        """Bucket key, or None when the step must run eagerly."""
        b = _bucket(batch, self.batch_buckets)
        l = _bucket(lmax, self.length_buckets)
        return None if b is None or l is None else (b, l)

    def _cache(self, b: int, l: int) -> ResidentSlotCache:
        layers: dict = {}
        for j, li in enumerate(self.layout.gdn_layers):
            layers[li] = _GdnEntry(conv=self.gdn_conv[j, :b], state=self.gdn_state[j, :b])
        for j, li in enumerate(self.layout.attn_layers):
            layers[li] = _AttnEntry(k_past=self.k_win[j, :b, :, : l + 1], v_past=self.v_win[j, :b, :, : l + 1])
        return ResidentSlotCache(self.layout, b, self.rows_idx[:b], self.lens[:b], layers)

    def _forward_static(self, b: int, l: int, cache: ResidentSlotCache) -> torch.Tensor:
        ids = self.ids[:b]
        lens = self.lens[:b]
        positions = lens[:, None]  # [b, 1], T == 1
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
            with torch.inference_mode():
                for _ in range(self.warmup_iters):  # Triton autotune + allocator warm-up
                    self._forward_static(b, l, cache)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                if self._pool is None:
                    self._pool = torch.cuda.graph_pool_handle()
                with torch.cuda.graph(graph, pool=self._pool):
                    logits = self._forward_static(b, l, cache)
        finally:
            self._restore_rows(b, l, saved)
        self.stats["captures"] += 1
        return graph, cache, logits

    def _save_rows(self, b: int, l: int):
        return (self.gdn_state[:, :b].clone(), self.gdn_conv[:, :b].clone(),
                self.k_win[:, :b, :, : l + 1].clone(), self.v_win[:, :b, :, : l + 1].clone())

    def _restore_rows(self, b: int, l: int, saved) -> None:
        self.gdn_state[:, :b].copy_(saved[0])
        self.gdn_conv[:, :b].copy_(saved[1])
        self.k_win[:, :b, :, : l + 1].copy_(saved[2])
        self.v_win[:, :b, :, : l + 1].copy_(saved[3])

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
        if n and self.layout.n_attn:
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
        if n > start and self.layout.n_attn:
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
            if n and self.layout.n_attn:
                self.k_win[:, row, :, :n].copy_(self.k_win[:, last, :, :n])
                self.v_win[:, row, :, :n].copy_(self.v_win[:, last, :, :n])
            self.row_of[self._sid(moved)] = row
            self.slots_of[row] = moved
            self.synced_len[row] = self.synced_len[last]
            self.stats["row_moves"] += 1
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
        graph.replay()
        self.stats["replays"] += 1
        for slots, ln in zip(slot_batch, lens):
            bank.guest_len[self._sid(slots)] = ln + 1
        if order == list(range(n)):
            return logits[:n]
        return logits.index_select(0, torch.tensor(order, device=self.device))
