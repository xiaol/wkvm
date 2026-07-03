"""RWKV7StateBank: the GPU half of the StateArena split.

The arena (`wkvm.core.arena.StateArena`) is pure integer bookkeeping; this
class materialises the tensors it promises: one dense GPU tensor per state
family, slot-indexed, with slot 0 reserved as the padding write target for
CUDA-graph padded batch rows (M2). Request state lives *only* here — python
request objects carry slot ids, never tensors.

Layout choice: tensors are layer-major (``[n_layer, num_slots+1, ...]``)
rather than slot-major, so gathering a decode batch for layer ``i`` is one
contiguous ``index_select`` on ``tensor[i]`` and scattering back is one
``index_copy_`` — no per-layer ``.contiguous()`` repacks in the hot loop.
The slot dimension is still what the arena allocates against.
"""

from __future__ import annotations

import torch

from wkvm.models.rwkv7 import WKV_DTYPE, RWKV7StateLayout


class RWKV7StateBank:
    def __init__(
        self,
        layout: RWKV7StateLayout,
        num_slots: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.layout = layout
        self.num_slots = num_slots
        self.device = torch.device(device)
        # [L, S+1, H, head_dim, head_v_dim], fp32: the fla kernels emit and
        # consume fp32 recurrent states regardless of activation dtype.
        self.wkv = torch.zeros(
            (layout.n_layer, num_slots + 1, *layout.wkv_shape),
            dtype=WKV_DTYPE,
            device=self.device,
        )
        # [L, 2, S+1, hidden] in model dtype; [:, 0] = attn conv_state,
        # [:, 1] = ffn_state (both are single-token token-shift caches).
        self.shift = torch.zeros(
            (layout.n_layer, 2, num_slots + 1, *layout.shift_shape),
            dtype=layout.dtype,
            device=self.device,
        )

    # -- slot lifecycle -----------------------------------------------------

    def zero_slots(self, slots: dict[str, int]) -> None:
        """Reset a freshly admitted request's slots. Zero state == empty
        prefix for every RWKV-7 state component (see models/rwkv7.py)."""
        self.wkv[:, slots["wkv"]].zero_()
        self.shift[:, :, slots["shift"]].zero_()

    def _ids(self, slot_batch: list[dict[str, int]], family: str) -> torch.Tensor:
        return torch.tensor(
            [s[family] for s in slot_batch], dtype=torch.long, device=self.device
        )

    # -- gather / scatter -----------------------------------------------------

    def gather_cache(self, slot_batch: list[dict[str, int]]):
        """Copy the batch's states out of the bank into a seeded fla Cache.

        The Cache is a transient staging object for one forward call; the
        bank remains the owner of record (`scatter_cache` commits results).
        ``offset=0`` so seeding never advances the cache's token count.
        """
        from fla.models.utils import Cache

        wkv_ids = self._ids(slot_batch, "wkv")
        shift_ids = self._ids(slot_batch, "shift")
        cache = Cache()
        for i in range(self.layout.n_layer):
            cache.update(
                recurrent_state=self.wkv[i].index_select(0, wkv_ids),
                conv_state=self.shift[i, 0].index_select(0, shift_ids),
                ffn_state=self.shift[i, 1].index_select(0, shift_ids),
                layer_idx=i,
                offset=0,
            )
        return cache

    def scatter_cache(self, slot_batch: list[dict[str, int]], cache) -> None:
        """Commit post-forward cache states back into the bank rows."""
        wkv_ids = self._ids(slot_batch, "wkv")
        shift_ids = self._ids(slot_batch, "shift")
        for i in range(self.layout.n_layer):
            state = cache[i]
            self.wkv[i].index_copy_(0, wkv_ids, state["recurrent_state"].to(WKV_DTYPE))
            self.shift[i, 0].index_copy_(
                0, shift_ids, state["conv_state"].to(self.shift.dtype)
            )
            self.shift[i, 1].index_copy_(
                0, shift_ids, state["ffn_state"].to(self.shift.dtype)
            )

    def state_bytes(self) -> int:
        return self.wkv.numel() * self.wkv.element_size() + self.shift.numel() * self.shift.element_size()
