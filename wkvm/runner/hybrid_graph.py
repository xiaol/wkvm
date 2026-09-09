"""CUDA-graph decode for the Qwen3.5 hybrid runner (H5).

A decode step through the HF decoder layers is ~6,300 kernel launches for
~23 ms of GPU work at B=1 (``experiments/hybrid_decode_profile.py``), so the
step is launch-bound. This module captures the *forward* once per
``(batch bucket, length bucket)`` and replays it; the paged gather before
and the scatter after stay eager (the SGLang out-graph/in-graph split).

Static state, allocated once at the largest bucket and sliced as views for
smaller buckets (views keep the addresses graphs captured):

- ``ids [Bmax, 1]``, ``lens [Bmax]`` (host -> device copies before replay)
- per GDN layer: ``state [Bmax, H_v, K, V]`` fp32, ``conv [Bmax, conv_dim, k]``
- per attention layer: ``k_past / v_past [Bmax, kv_heads, Lmax, hd]``

Inside the graph: positions = ``lens``, the attention mask is computed from
``lens`` and the length bucket (columns ``>= lens[b]`` masked; the new token
sits at column ``L_bucket``), then ``Qwen35Decoder.forward``. The Gated
DeltaNet layers update ``state``/``conv`` in place through the captured
``SlotCache`` (static addresses), and the attention layers leave their new
K/V in graph-pool tensors that the same ``SlotCache`` object keeps pointing
at, so ``bank.scatter`` works unchanged after replay.

Padded rows use the reserved ids: slot 0 and page 0. Their outputs are
discarded and their scatter lands in the dummy slot, which is exactly what
slot 0 is for (docs/ANGLE.md §2).
"""

from __future__ import annotations

import torch

from wkvm.models.qwen35 import GDN_CONV_FAMILY, GDN_STATE_DTYPE, GDN_STATE_FAMILY, GUEST_KV_FAMILY
from wkvm.runner.hybrid_state import Qwen35StateBank, SlotCache, _AttnEntry, _GdnEntry

DUMMY_SLOTS = {GDN_STATE_FAMILY: 0, GDN_CONV_FAMILY: 0, GUEST_KV_FAMILY: (0,)}


def _bucket(n: int, buckets: tuple[int, ...]) -> int | None:
    for b in buckets:
        if n <= b:
            return b
    return None


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
        self.ids = torch.zeros((bmax, 1), dtype=torch.long, device=self.device)
        self.lens = torch.zeros((bmax,), dtype=torch.long, device=self.device)
        self.gdn_state = torch.zeros((lay.n_gdn, bmax, *lay.gdn_state_shape), dtype=GDN_STATE_DTYPE, device=self.device)
        self.gdn_conv = torch.zeros((lay.n_gdn, bmax, *lay.gdn_conv_shape), dtype=dt, device=self.device)
        self.k_past = torch.zeros((lay.n_attn, bmax, lay.num_kv_heads, lmax, lay.head_dim), dtype=dt, device=self.device)
        self.v_past = torch.zeros_like(self.k_past)
        self._graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, SlotCache, torch.Tensor]] = {}
        self._pool = None
        self.stats = {"replays": 0, "captures": 0, "eager_fallbacks": 0}

    # -- buckets --------------------------------------------------------------------

    def bucket_for(self, batch: int, lmax: int) -> tuple[int, int] | None:
        """Bucket key, or None when the step must run eagerly."""
        b = _bucket(batch, self.batch_buckets)
        l = _bucket(lmax, self.length_buckets)
        return None if b is None or l is None else (b, l)

    def _views(self, b: int, l: int) -> SlotCache:
        """A SlotCache over static views for bucket (b, l). ``lens`` are
        placeholders; the real ones are copied in before each replay."""
        layers: dict = {}
        for j, li in enumerate(self.layout.gdn_layers):
            layers[li] = _GdnEntry(conv=self.gdn_conv[j, :b], state=self.gdn_state[j, :b])
        for j, li in enumerate(self.layout.attn_layers):
            layers[li] = _AttnEntry(k_past=self.k_past[j, :b, :, :l], v_past=self.v_past[j, :b, :, :l])
        return SlotCache(self.layout, [0] * b, layers)

    def _forward_static(self, b: int, l: int, cache: SlotCache) -> torch.Tensor:
        ids = self.ids[:b]
        lens = self.lens[:b]
        positions = lens[:, None]  # [b, 1], T == 1
        cols = torch.arange(l + 1, device=self.device)
        mask = ((cols[None, :] < lens[:, None]) | (cols[None, :] == l))[:, None, None, :]  # [b,1,1,l+1]
        return self.model(ids, cache, positions, mask)

    def _capture(self, b: int, l: int):
        cache = self._views(b, l)
        with torch.inference_mode():
            for _ in range(self.warmup_iters):  # Triton autotune + allocator warm-up
                self._forward_static(b, l, cache)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._pool):
                logits = self._forward_static(b, l, cache)
        self.stats["captures"] += 1
        return graph, cache, logits

    # -- the step -----------------------------------------------------------------

    def _fill_static(self, slot_batch: list[dict], lens: list[int], b: int, l: int) -> None:
        """Eager gather into the static buffers (out-graph)."""
        bank = self.bank
        sids = torch.tensor([s[GDN_STATE_FAMILY] for s in slot_batch], dtype=torch.long, device=self.device)
        cids = torch.tensor([s[GDN_CONV_FAMILY] for s in slot_batch], dtype=torch.long, device=self.device)
        n = len(slot_batch)
        for j in range(self.layout.n_gdn):
            torch.index_select(bank.gdn_state[j], 0, sids, out=self.gdn_state[j, :n])
            torch.index_select(bank.gdn_conv[j], 0, cids, out=self.gdn_conv[j, :n])
        if self.layout.n_attn:
            lmax = max(lens) if lens else 0
            if lmax:
                page_idx, off_idx = bank._past_index(slot_batch, lens, lmax)
                for j in range(self.layout.n_attn):
                    self.k_past[j, :n, :, :lmax] = bank.pool_k[j][page_idx, :, off_idx].permute(0, 2, 1, 3)
                    self.v_past[j, :n, :, :lmax] = bank.pool_v[j][page_idx, :, off_idx].permute(0, 2, 1, 3)
        self.lens[:n].copy_(torch.tensor(lens, dtype=torch.long), non_blocking=True)
        if b > n:
            self.lens[n:b].zero_()

    def decode_step(self, slot_batch: list[dict], last_tokens: list[int]) -> torch.Tensor | None:
        """Replay a captured graph for this batch; None => caller runs eager."""
        bank = self.bank
        lens = [bank.slot_len(s) for s in slot_batch]
        key = self.bucket_for(len(slot_batch), max(lens))
        if key is None:
            self.stats["eager_fallbacks"] += 1
            return None
        for slots in slot_batch:
            bank.check_capacity(slots, 1)
        b, l = key
        if key not in self._graphs:
            self._graphs[key] = self._capture(b, l)
        graph, cache, logits = self._graphs[key]
        n = len(slot_batch)
        padded = list(slot_batch) + [DUMMY_SLOTS] * (b - n)
        self._fill_static(slot_batch, lens, b, l)
        self.ids[:n, 0].copy_(torch.tensor(last_tokens, dtype=torch.long), non_blocking=True)
        if b > n:
            self.ids[n:b].zero_()
        cache.lens = lens + [0] * (b - n)
        cache.lmax = l  # column layout of the captured buffers
        graph.replay()
        self.stats["replays"] += 1
        bank.scatter(padded, cache)
        bank.guest_len[0] = 0  # dummy rows never accumulate
        return logits[:n]
