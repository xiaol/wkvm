"""Qwen35StateBank: the GPU half of the StateArena split for hybrid models.

Three families, all layer-major (``[n_family_layers, num_slots+1, ...]``)
so one layer's decode batch is one contiguous ``index_select``; slot 0 is the
reserved padding write target (docs/ANGLE.md §2):

- ``gdn_state`` fp32 ``[n_gdn, S+1, H_v, K, V]``
- ``gdn_conv``  model dtype ``[n_gdn, S+1, conv_dim, kernel]``
- ``guest_kv``  model dtype K and V ``[n_attn, S+1, kv_heads, guest_ctx, head_dim]``
  plus a host-side per-slot ``guest_len`` (tokens resident in the window).

``gather`` builds a :class:`SlotCache` — the one-forward staging object that
satisfies exactly the calls the HF Qwen3.5 layers make on their cache
(``has_previous_state``, ``layers[i].conv_states[0]``,
``layers[i].recurrent_states[0]``, ``update_conv_state``,
``update_recurrent_state`` for Gated DeltaNet; ``update`` for attention).
``scatter`` commits it back. The bank remains the owner of record; request
state never survives a step inside python objects.
"""

from __future__ import annotations

import torch

from wkvm.models.qwen35 import (
    FULL,
    GDN_CONV_FAMILY,
    GDN_STATE_DTYPE,
    GDN_STATE_FAMILY,
    GUEST_KV_FAMILY,
    LINEAR,
    Qwen35HybridLayout,
)


class GuestWindowExceeded(RuntimeError):
    """A request needs more guest-KV tokens than its slot window holds."""


class _GdnEntry:
    """Per-layer view the HF Gated DeltaNet layer reads/writes."""

    __slots__ = ("conv_states", "recurrent_states", "record_past")

    def __init__(self, conv: torch.Tensor, state: torch.Tensor) -> None:
        self.conv_states = {0: conv}
        self.recurrent_states = {0: state}
        self.record_past = False


class _AttnEntry:
    __slots__ = ("k_past", "v_past", "k_new", "v_new")

    def __init__(self, k_past: torch.Tensor, v_past: torch.Tensor) -> None:
        self.k_past = k_past
        self.v_past = v_past
        self.k_new: torch.Tensor | None = None
        self.v_new: torch.Tensor | None = None


class SlotCache:
    """Staging cache for one forward over ``B`` rows.

    Attention rows are laid out as ``[past_0 .. past_{Lmax-1} | new_0 .. new_{T-1}]``
    where ``Lmax = max(lens)``; a row with fewer resident tokens has its unused
    past columns masked. Because attention is permutation-invariant over keys
    (positions travel in RoPE, not in column index), this padded layout is
    exact — the mask is the whole contract.
    """

    def __init__(self, layout: Qwen35HybridLayout, lens: list[int], layers: dict) -> None:
        self.layout = layout
        self.lens = list(lens)
        self.lmax = max(self.lens) if self.lens else 0
        self.layers = layers  # model layer idx -> _GdnEntry | _AttnEntry

    # -- Gated DeltaNet contract --------------------------------------------------

    def has_previous_state(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        # Always seeded: zero state == fresh sequence for both components, so
        # the layer can take the cached path unconditionally (and the decode
        # path when seq_len == 1).
        return True

    def update_conv_state(
        self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None, **_
    ) -> torch.Tensor:
        entry = self.layers[layer_idx]
        window = entry.conv_states[0]
        full = torch.cat([window, conv_states], dim=-1)
        window.copy_(full[..., -window.shape[-1] :])
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **_) -> torch.Tensor:
        entry = self.layers[layer_idx]
        entry.recurrent_states[0].copy_(recurrent_states)
        return entry.recurrent_states[0]

    # -- attention contract -------------------------------------------------------

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *_, **__):
        entry = self.layers[layer_idx]
        entry.k_new, entry.v_new = key_states, value_states
        if entry.k_past.shape[-2] == 0:
            return key_states, value_states
        return (
            torch.cat([entry.k_past, key_states], dim=-2),
            torch.cat([entry.v_past, value_states], dim=-2),
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lmax

    # -- mask ---------------------------------------------------------------------

    def attention_mask(self, query_len: int, device) -> torch.Tensor | None:
        """Bool ``[B, 1, T, Lmax+T]`` (True = attend), or None when the
        mask-free SDPA paths are exact: a single fresh prefill row (pure
        causal) or a decode batch whose rows all share one length."""
        b, t, lmax = len(self.lens), query_len, self.lmax
        if lmax == 0 and b == 1:
            return None  # SDPA is_causal over exactly the new tokens
        if t == 1 and all(n == lmax for n in self.lens):
            return None  # every column valid for every row
        lens = torch.tensor(self.lens, device=device)
        cols = torch.arange(lmax + t, device=device)
        past_ok = cols[None, :] < lens[:, None]  # [B, KV]
        rows = torch.arange(t, device=device)
        new_ok = (cols[None, :] >= lmax) & (cols[None, :] <= lmax + rows[:, None])  # [T, KV]
        mask = past_ok[:, None, :] | new_ok[None, :, :]  # [B, T, KV]
        return mask[:, None]


class Qwen35StateBank:
    memory_families = (GDN_STATE_FAMILY,)

    def __init__(
        self,
        layout: Qwen35HybridLayout,
        num_slots: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.layout = layout
        self.num_slots = num_slots
        self.device = torch.device(device)
        s = num_slots + 1
        self.gdn_state = torch.zeros(
            (layout.n_gdn, s, *layout.gdn_state_shape), dtype=GDN_STATE_DTYPE, device=self.device
        )
        self.gdn_conv = torch.zeros(
            (layout.n_gdn, s, *layout.gdn_conv_shape), dtype=layout.dtype, device=self.device
        )
        self.guest_k = torch.zeros(
            (layout.n_attn, s, *layout.guest_kv_shape), dtype=layout.dtype, device=self.device
        )
        self.guest_v = torch.zeros_like(self.guest_k)
        self.guest_len: list[int] = [0] * s
        # model layer idx -> position inside the family bank
        self._gdn_pos = {li: j for j, li in enumerate(layout.gdn_layers)}
        self._attn_pos = {li: j for j, li in enumerate(layout.attn_layers)}

    # -- slot lifecycle -----------------------------------------------------------

    def zero_slots(self, slots: dict[str, int]) -> None:
        """Reset a freshly admitted request's slots (zero == fresh sequence)."""
        self.gdn_state[:, slots[GDN_STATE_FAMILY]].zero_()
        self.gdn_conv[:, slots[GDN_CONV_FAMILY]].zero_()
        g = slots[GUEST_KV_FAMILY]
        self.guest_k[:, g].zero_()
        self.guest_v[:, g].zero_()
        self.guest_len[g] = 0

    def slot_len(self, slots: dict[str, int]) -> int:
        return self.guest_len[slots[GUEST_KV_FAMILY]]

    def check_capacity(self, slots: dict[str, int], new_tokens: int) -> None:
        if self.slot_len(slots) + new_tokens > self.layout.guest_ctx:
            raise GuestWindowExceeded(
                f"guest window {self.layout.guest_ctx} tokens: slot holds "
                f"{self.slot_len(slots)}, cannot add {new_tokens}"
            )

    def _ids(self, slot_batch: list[dict[str, int]], family: str) -> torch.Tensor:
        return torch.tensor([s[family] for s in slot_batch], dtype=torch.long, device=self.device)

    # -- gather / scatter -----------------------------------------------------------

    def gather(self, slot_batch: list[dict[str, int]], new_tokens: int) -> SlotCache:
        for slots in slot_batch:
            self.check_capacity(slots, new_tokens)
        lens = [self.slot_len(s) for s in slot_batch]
        lmax = max(lens)
        layers: dict = {}
        if self.layout.n_gdn:
            sids = self._ids(slot_batch, GDN_STATE_FAMILY)
            cids = self._ids(slot_batch, GDN_CONV_FAMILY)
            for li, j in self._gdn_pos.items():
                layers[li] = _GdnEntry(
                    conv=self.gdn_conv[j].index_select(0, cids),
                    state=self.gdn_state[j].index_select(0, sids),
                )
        if self.layout.n_attn:
            gids = self._ids(slot_batch, GUEST_KV_FAMILY)
            for li, j in self._attn_pos.items():
                layers[li] = _AttnEntry(
                    k_past=self.guest_k[j, :, :, :lmax].index_select(0, gids),
                    v_past=self.guest_v[j, :, :, :lmax].index_select(0, gids),
                )
        return SlotCache(self.layout, lens, layers)

    def scatter(self, slot_batch: list[dict[str, int]], cache: SlotCache) -> None:
        if self.layout.n_gdn:
            sids = self._ids(slot_batch, GDN_STATE_FAMILY)
            cids = self._ids(slot_batch, GDN_CONV_FAMILY)
            for li, j in self._gdn_pos.items():
                entry = cache.layers[li]
                self.gdn_state[j].index_copy_(0, sids, entry.recurrent_states[0].to(GDN_STATE_DTYPE))
                self.gdn_conv[j].index_copy_(0, cids, entry.conv_states[0].to(self.gdn_conv.dtype))
        if self.layout.n_attn:
            t = None
            for li, j in self._attn_pos.items():
                entry = cache.layers[li]
                if entry.k_new is None:
                    raise RuntimeError(f"layer {li}: attention cache was never updated this forward")
                t = entry.k_new.shape[-2]
                for row, slots in enumerate(slot_batch):
                    g = slots[GUEST_KV_FAMILY]
                    n = cache.lens[row]
                    self.guest_k[j, g, :, n : n + t].copy_(entry.k_new[row].to(self.guest_k.dtype))
                    self.guest_v[j, g, :, n : n + t].copy_(entry.v_new[row].to(self.guest_v.dtype))
            for row, slots in enumerate(slot_batch):
                self.guest_len[slots[GUEST_KV_FAMILY]] = cache.lens[row] + t

    # -- durable-state protocol (wkvm/store.py) ------------------------------------

    def fingerprint_key(self) -> str:
        l = self.layout
        return (
            f"qwen35:L{l.n_layer}:types{''.join('l' if t == LINEAR else 'f' for t in l.layer_types)}"
            f":gdn{tuple(l.gdn_state_shape)}:conv{tuple(l.gdn_conv_shape)}"
            f":kv{tuple(l.guest_kv_shape)}:{l.dtype}"
        )

    def export_slot(self, slots: dict[str, int]) -> dict[str, torch.Tensor]:
        """Device -> host copies of one slot's state. The guest window is
        trimmed to its resident length so a snapshot costs O(tokens), not
        O(guest_ctx)."""
        g = slots[GUEST_KV_FAMILY]
        n = self.guest_len[g]
        out = {
            GDN_STATE_FAMILY: self.gdn_state[:, slots[GDN_STATE_FAMILY]],
            GDN_CONV_FAMILY: self.gdn_conv[:, slots[GDN_CONV_FAMILY]],
            "guest_k": self.guest_k[:, g, :, :n],
            "guest_v": self.guest_v[:, g, :, :n],
        }
        host = {k: _to_host(v) for k, v in out.items()}
        host["guest_len"] = torch.tensor([n], dtype=torch.int64)
        return host

    def import_slot(self, slots: dict[str, int], tensors: dict[str, torch.Tensor]) -> None:
        """Host -> device into already-allocated slots. Missing families stay
        zero (fresh), which is what makes a tuned-initial-state import — only
        ``gdn_state`` present — a valid slot."""
        self.zero_slots(slots)
        known = {GDN_STATE_FAMILY, GDN_CONV_FAMILY, "guest_k", "guest_v", "guest_len"}
        unknown = set(tensors) - known
        if unknown:
            raise KeyError(f"unknown state tensors for Qwen3.5 bank: {sorted(unknown)}")
        if GDN_STATE_FAMILY in tensors:
            dst = self.gdn_state[:, slots[GDN_STATE_FAMILY]]
            _check(tensors[GDN_STATE_FAMILY], dst.shape, GDN_STATE_FAMILY)
            dst.copy_(tensors[GDN_STATE_FAMILY], non_blocking=True)
        if GDN_CONV_FAMILY in tensors:
            dst = self.gdn_conv[:, slots[GDN_CONV_FAMILY]]
            _check(tensors[GDN_CONV_FAMILY], dst.shape, GDN_CONV_FAMILY)
            dst.copy_(tensors[GDN_CONV_FAMILY], non_blocking=True)
        g = slots[GUEST_KV_FAMILY]
        n = 0
        if "guest_k" in tensors:
            k, v = tensors["guest_k"], tensors["guest_v"]
            n = k.shape[-2]
            if n > self.layout.guest_ctx:
                raise GuestWindowExceeded(f"snapshot holds {n} guest tokens > window {self.layout.guest_ctx}")
            _check(k, (self.layout.n_attn, self.layout.num_kv_heads, n, self.layout.head_dim), "guest_k")
            _check(v, k.shape, "guest_v")
            self.guest_k[:, g, :, :n].copy_(k, non_blocking=True)
            self.guest_v[:, g, :, :n].copy_(v, non_blocking=True)
        if "guest_len" in tensors and int(tensors["guest_len"].reshape(-1)[0]) != n:
            raise ValueError(f"guest_len {int(tensors['guest_len'].reshape(-1)[0])} != guest tokens {n}")
        self.guest_len[g] = n

    def state_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.gdn_state, self.gdn_conv, self.guest_k, self.guest_v)
        )


def _check(t: torch.Tensor, shape, name: str) -> None:
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name}: shape {tuple(t.shape)} != expected {tuple(shape)}")


def _to_host(view: torch.Tensor) -> torch.Tensor:
    pin = view.is_cuda
    host = torch.empty(view.shape, dtype=view.dtype, pin_memory=pin)
    host.copy_(view, non_blocking=pin)
    return host
