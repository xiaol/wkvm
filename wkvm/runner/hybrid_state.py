"""Qwen35StateBank: the GPU half of the StateArena split for hybrid models.

Two slot families and one paged guest family, all layer-major so one layer's
decode batch is one gather; slot/page 0 is the reserved dummy
(docs/ANGLE.md §2):

- ``gdn_state`` fp32 ``[n_gdn, S+1, H_v, K, V]``
- ``gdn_conv``  model dtype ``[n_gdn, S+1, conv_dim, kernel]``
- ``guest_kv``  model dtype K and V pools ``[n_attn, P+1, kv_heads, page_tokens, head_dim]``
  plus a host-side per-request ``guest_len`` (tokens resident in the pages).

A request's guest tokens live in the pages the arena reserved for it
(``slots["guest_kv"]`` is a tuple of page ids); token ``t`` is at page
``pages[t // page_tokens]``, offset ``t % page_tokens``. ``gather`` builds a
:class:`SlotCache` — the one-forward staging object that satisfies exactly
the calls the HF Qwen3.5 layers make on their cache (``has_previous_state``,
``layers[i].conv_states[0]``, ``layers[i].recurrent_states[0]``,
``update_conv_state``, ``update_recurrent_state`` for Gated DeltaNet;
``update`` for attention). ``scatter`` commits it back. The bank remains the
owner of record; request state never survives a step inside python objects.
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


class GuestCapacityExceeded(RuntimeError):
    """A request needs more guest-KV tokens than its reserved pages hold."""


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
        num_pages: int = 0,
    ) -> None:
        if layout.n_gdn < 1:
            raise ValueError("Qwen35StateBank needs at least one linear_attention layer")
        if layout.n_attn and num_pages < 1:
            raise ValueError("num_pages must be >= 1 for a model with full-attention layers")
        self.layout = layout
        self.num_slots = num_slots
        self.num_pages = num_pages if layout.n_attn else 0
        self.device = torch.device(device)
        s = num_slots + 1
        self.gdn_state = torch.zeros(
            (layout.n_gdn, s, *layout.gdn_state_shape), dtype=GDN_STATE_DTYPE, device=self.device
        )
        self.gdn_conv = torch.zeros(
            (layout.n_gdn, s, *layout.gdn_conv_shape), dtype=layout.dtype, device=self.device
        )
        self.pool_k = torch.zeros(
            (layout.n_attn, self.num_pages + 1, *layout.guest_page_shape),
            dtype=layout.dtype, device=self.device,
        )
        self.pool_v = torch.zeros_like(self.pool_k)
        # Request identity is the gdn_state slot; guest length lives with it.
        self.guest_len: list[int] = [0] * s
        # model layer idx -> position inside the family bank
        self._gdn_pos = {li: j for j, li in enumerate(layout.gdn_layers)}
        self._attn_pos = {li: j for j, li in enumerate(layout.attn_layers)}
        # Set by HybridDecodeGraphs: rows of running requests may live in its
        # static buffers between decode steps (write-back cache over this bank).
        self.resident = None

    # -- slot lifecycle -----------------------------------------------------------

    @staticmethod
    def _pages(slots: dict) -> tuple[int, ...]:
        return tuple(slots.get(GUEST_KV_FAMILY, ()))

    def _sync_resident(self, slot_batch, evict: bool) -> None:
        if self.resident is None:
            return
        for slots in slot_batch:
            if evict:
                self.resident.evict(slots)
            else:
                self.resident.flush(slots)

    def release_slots(self, slots: dict) -> None:
        """The request is done with these slots (finish/abort): drop any
        resident row without writing back."""
        if self.resident is not None:
            self.resident.release(slots)

    def zero_slots(self, slots: dict) -> None:
        """Reset a freshly admitted request's slots (zero == fresh sequence).
        Pages are not zeroed: positions beyond ``guest_len`` are never read
        unmasked, and stale page contents are finite model outputs."""
        self.release_slots(slots)
        self.gdn_state[:, slots[GDN_STATE_FAMILY]].zero_()
        self.gdn_conv[:, slots[GDN_CONV_FAMILY]].zero_()
        self.guest_len[slots[GDN_STATE_FAMILY]] = 0

    def slot_len(self, slots: dict) -> int:
        return self.guest_len[slots[GDN_STATE_FAMILY]]

    def capacity(self, slots: dict) -> int:
        return len(self._pages(slots)) * self.layout.page_tokens

    def check_capacity(self, slots: dict, new_tokens: int) -> None:
        if self.slot_len(slots) + new_tokens > self.capacity(slots):
            raise GuestCapacityExceeded(
                f"reserved {len(self._pages(slots))} pages x {self.layout.page_tokens} tokens: "
                f"slot holds {self.slot_len(slots)}, cannot add {new_tokens}"
            )

    def _ids(self, slot_batch: list[dict], family: str) -> torch.Tensor:
        return torch.tensor([s[family] for s in slot_batch], dtype=torch.long, device=self.device)

    # -- page addressing ------------------------------------------------------------

    def _positions_index(self, pages: tuple[int, ...], start: int, count: int):
        """(page ids, offsets) of tokens ``start .. start+count-1``."""
        pt = self.layout.page_tokens
        pos = torch.arange(start, start + count, device=self.device)
        page_ids = torch.tensor(pages, dtype=torch.long, device=self.device)
        return page_ids[pos // pt], pos % pt

    def _past_index(self, slot_batch: list[dict], lens: list[int], lmax: int):
        """``[B, Lmax]`` page ids / offsets; positions beyond a row's length
        point at the dummy page 0 (masked out by the attention mask)."""
        pt = self.layout.page_tokens
        b = len(slot_batch)
        page_idx = torch.zeros((b, lmax), dtype=torch.long, device=self.device)
        off_idx = torch.zeros((b, lmax), dtype=torch.long, device=self.device)
        if lmax == 0:
            return page_idx, off_idx
        t = torch.arange(lmax, device=self.device)
        for row, slots in enumerate(slot_batch):
            n = lens[row]
            if n == 0:
                continue
            pages = torch.tensor(self._pages(slots), dtype=torch.long, device=self.device)
            valid = t < n
            page_idx[row] = torch.where(valid, pages[(t // pt).clamp_(max=len(pages) - 1)], 0)
            off_idx[row] = torch.where(valid, t % pt, 0)
        return page_idx, off_idx

    # -- gather / scatter -----------------------------------------------------------

    def gather(self, slot_batch: list[dict], new_tokens: int) -> SlotCache:
        # The eager path takes over these requests: resident rows are written
        # back and freed so the bank is the only copy again.
        self._sync_resident(slot_batch, evict=True)
        for slots in slot_batch:
            self.check_capacity(slots, new_tokens)
        lens = [self.slot_len(s) for s in slot_batch]
        lmax = max(lens)
        layers: dict = {}
        sids = self._ids(slot_batch, GDN_STATE_FAMILY)
        cids = self._ids(slot_batch, GDN_CONV_FAMILY)
        for li, j in self._gdn_pos.items():
            layers[li] = _GdnEntry(
                conv=self.gdn_conv[j].index_select(0, cids),
                state=self.gdn_state[j].index_select(0, sids),
            )
        if self.layout.n_attn:
            page_idx, off_idx = self._past_index(slot_batch, lens, lmax)
            for li, j in self._attn_pos.items():
                # advanced indices on dims 0 and 2 -> [B, Lmax, kv_heads, hd]
                layers[li] = _AttnEntry(
                    k_past=self.pool_k[j][page_idx, :, off_idx].permute(0, 2, 1, 3),
                    v_past=self.pool_v[j][page_idx, :, off_idx].permute(0, 2, 1, 3),
                )
        return SlotCache(self.layout, lens, layers)

    def scatter(self, slot_batch: list[dict], cache: SlotCache) -> None:
        sids = self._ids(slot_batch, GDN_STATE_FAMILY)
        cids = self._ids(slot_batch, GDN_CONV_FAMILY)
        for li, j in self._gdn_pos.items():
            entry = cache.layers[li]
            self.gdn_state[j].index_copy_(0, sids, entry.recurrent_states[0].to(GDN_STATE_DTYPE))
            self.gdn_conv[j].index_copy_(0, cids, entry.conv_states[0].to(self.gdn_conv.dtype))
        if not self.layout.n_attn:
            return
        first = cache.layers[self.layout.attn_layers[0]]
        if first.k_new is None:
            raise RuntimeError("attention cache was never updated this forward")
        t = first.k_new.shape[-2]
        if t == 1:  # batched decode: one fused write per layer
            pg = torch.cat([self._positions_index(self._pages(s), cache.lens[r], 1)[0]
                            for r, s in enumerate(slot_batch)])
            off = torch.cat([self._positions_index(self._pages(s), cache.lens[r], 1)[1]
                             for r, s in enumerate(slot_batch)])
            for li, j in self._attn_pos.items():
                entry = cache.layers[li]
                self.pool_k[j][pg, :, off] = entry.k_new[:, :, 0].to(self.pool_k.dtype)
                self.pool_v[j][pg, :, off] = entry.v_new[:, :, 0].to(self.pool_v.dtype)
        else:  # prefill chunk(s): per row
            for row, slots in enumerate(slot_batch):
                pg, off = self._positions_index(self._pages(slots), cache.lens[row], t)
                for li, j in self._attn_pos.items():
                    entry = cache.layers[li]
                    self.pool_k[j][pg, :, off] = entry.k_new[row].permute(1, 0, 2).to(self.pool_k.dtype)
                    self.pool_v[j][pg, :, off] = entry.v_new[row].permute(1, 0, 2).to(self.pool_v.dtype)
        for row, slots in enumerate(slot_batch):
            self.guest_len[slots[GDN_STATE_FAMILY]] = cache.lens[row] + t

    # -- durable-state protocol (wkvm/store.py) ------------------------------------

    def fingerprint_key(self) -> str:
        l = self.layout
        # Page size is deliberately not part of the identity: snapshots are
        # token-trimmed and page-size independent.
        return (
            f"qwen35:L{l.n_layer}:types{''.join('l' if t == LINEAR else 'f' for t in l.layer_types)}"
            f":gdn{tuple(l.gdn_state_shape)}:conv{tuple(l.gdn_conv_shape)}"
            f":kv{(l.num_kv_heads, l.head_dim)}:{l.dtype}"
        )

    def export_slot(self, slots: dict) -> dict[str, torch.Tensor]:
        """Device -> host copies of one request's state. The guest tokens are
        gathered out of their pages, trimmed to ``guest_len`` — a snapshot
        costs O(tokens) and is page-size independent."""
        self._sync_resident([slots], evict=False)  # the row stays; the bank is now current
        n = self.slot_len(slots)
        out = {
            GDN_STATE_FAMILY: self.gdn_state[:, slots[GDN_STATE_FAMILY]],
            GDN_CONV_FAMILY: self.gdn_conv[:, slots[GDN_CONV_FAMILY]],
        }
        if self.layout.n_attn:
            pg, off = self._positions_index(self._pages(slots), 0, n)
            # Advanced indices on dims 1 and 3 (separated by a slice) come
            # first: [n, n_attn, kv_heads, hd] -> [n_attn, kv_heads, n, hd].
            out["guest_k"] = self.pool_k[:, pg, :, off].permute(1, 2, 0, 3)
            out["guest_v"] = self.pool_v[:, pg, :, off].permute(1, 2, 0, 3)
        host = {k: _to_host(v) for k, v in out.items()}
        host["guest_len"] = torch.tensor([n], dtype=torch.int64)
        return host

    def import_slot(self, slots: dict, tensors: dict[str, torch.Tensor]) -> None:
        """Host -> device into already-allocated slots/pages. Missing families
        stay zero (fresh), which is what makes a tuned-initial-state import —
        only ``gdn_state`` present — a valid request."""
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
        n = 0
        if "guest_k" in tensors:
            k, v = tensors["guest_k"], tensors["guest_v"]
            n = k.shape[-2]
            if n > self.capacity(slots):
                raise GuestCapacityExceeded(
                    f"snapshot holds {n} guest tokens > reserved {self.capacity(slots)}"
                )
            _check(k, (self.layout.n_attn, self.layout.num_kv_heads, n, self.layout.head_dim), "guest_k")
            _check(v, k.shape, "guest_v")
            if n:
                pg, off = self._positions_index(self._pages(slots), 0, n)
                # Target layout is [n, n_attn, kv_heads, hd] (see export_slot).
                self.pool_k[:, pg, :, off] = k.permute(2, 0, 1, 3).to(
                    self.pool_k.device, self.pool_k.dtype, non_blocking=True
                )
                self.pool_v[:, pg, :, off] = v.permute(2, 0, 1, 3).to(
                    self.pool_v.device, self.pool_v.dtype, non_blocking=True
                )
        if "guest_len" in tensors and int(tensors["guest_len"].reshape(-1)[0]) != n:
            raise ValueError(f"guest_len {int(tensors['guest_len'].reshape(-1)[0])} != guest tokens {n}")
        self.guest_len[slots[GDN_STATE_FAMILY]] = n

    def state_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.gdn_state, self.gdn_conv, self.pool_k, self.pool_v)
        )


def _check(t: torch.Tensor, shape, name: str) -> None:
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name}: shape {tuple(t.shape)} != expected {tuple(shape)}")


def _to_host(view: torch.Tensor) -> torch.Tensor:
    pin = view.is_cuda
    host = torch.empty(view.shape, dtype=view.dtype, pin_memory=pin)
    host.copy_(view, non_blocking=pin)
    return host
