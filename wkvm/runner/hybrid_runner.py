"""Qwen35HybridRunner: prefill and batched decode against arena slots.

Same contract as ``RWKV7Runner`` (so ``wkvm.engine.Engine`` is agnostic):
``prefill(token_ids, slots) -> last logits`` and
``decode_step(slot_batch, last_tokens) -> [batch, vocab] logits``. Both
gather the batch's state into a :class:`SlotCache`, run the HF layer loop
through ``Qwen35Decoder.forward`` with wkvm-built positions and mask, then
scatter back.

Kernel dispatch inside the HF Gated DeltaNet layer follows ``seq_len``:
``seq_len == 1`` (with a seeded cache) takes the recurrent step, anything
longer the chunked scan (chunk 64) — the same prefill/decode split M1 uses,
so a prefill chunk that is a multiple of 64 re-enters the scan on its own
chunk boundaries.
"""

from __future__ import annotations

import torch

from wkvm.runner.hybrid_state import Qwen35StateBank


class Qwen35HybridRunner:
    def __init__(self, model, bank: Qwen35StateBank, prefill_chunk: int = 512) -> None:
        if prefill_chunk < 1:
            raise ValueError("prefill_chunk must be >= 1")
        self.model = model
        self.bank = bank
        self.prefill_chunk = prefill_chunk
        self.device = bank.device
        self.graphs = None  # HybridDecodeGraphs once enable_cuda_graphs() is called

    def enable_cuda_graphs(self, **kwargs) -> None:
        """Capture-and-replay decode forwards per (batch, length) bucket
        (``wkvm/runner/hybrid_graph.py``). Steps outside every bucket run
        eagerly; prefill always runs eagerly."""
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graphs need a CUDA bank")
        from wkvm.runner.hybrid_graph import HybridDecodeGraphs

        self.graphs = HybridDecodeGraphs(self.model, self.bank, **kwargs)

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], slots: dict[str, int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("empty prompt")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        logits = None
        step = self.chunk_size()
        for start in range(0, len(token_ids), step):
            chunk = ids[start : start + step].unsqueeze(0)
            cache = self.bank.gather([slots], new_tokens=chunk.shape[1], token_ids=chunk)
            logits = self._forward(chunk, cache)
            self.bank.scatter([slots], cache)
        return logits[0].float()

    def chunk_size(self) -> int:
        """Prefill sub-chunk: in routed mode bounded by the pending threshold
        (one chunk's evictions always fit the pending buffer) and by the ring
        (no token is overwritten by a later token of the same chunk)."""
        if getattr(self.bank, "routed", False):
            return min(self.prefill_chunk, self.bank.rg.p.pending, self.bank.rg.p.ring)
        return self.prefill_chunk

    # Cap on the past K/V one batched prefill forward gathers from the bank
    # (plus the concatenation with the new tokens): a routed store is 5,201
    # columns per row, 32 rows would copy 11 GiB per forward.
    prefill_gather_bytes = 4 << 30

    def _rows_per_forward(self, t: int, lens: list[int]) -> int:
        lay = self.bank.layout
        if not lay.n_attn:
            return max(1, len(lens))
        cols = (lay.window_tokens if getattr(self.bank, "ring", False) or getattr(self.bank, "routed", False)
                else max(lens)) + t
        return max(1, self.prefill_gather_bytes // (2 * cols * lay.guest_bytes_per_token))

    @torch.inference_mode()
    def prefill_batch(self, items: list[tuple[list[int], dict]]) -> list[torch.Tensor]:
        """Prefill several requests' chunks of EQUAL length as one forward.

        Equal lengths keep the batch rectangular (no padding, no ragged
        kernels): the bank gathers each row's state, the mask handles each
        row's own past, and the scatter writes each row back. Callers group
        chunks by length; the multi-turn case (every session gets the same
        turn length) and the bulk of a same-shape ladder land here, the
        remainder runs one request at a time. Returns last-position logits
        per row, in order."""
        if not items:
            return []
        t = len(items[0][0])
        if any(len(tokens) != t for tokens, _ in items):
            raise ValueError("prefill_batch needs equal-length chunks")
        if t > self.chunk_size():
            raise ValueError("prefill_batch chunk exceeds the runner's chunk size")
        cap = self._rows_per_forward(t, [self.bank.slot_len(slots) for _, slots in items])
        if len(items) > cap:  # keep the gathered past K/V under the byte budget
            out: list[torch.Tensor] = []
            for i in range(0, len(items), cap):
                out.extend(self.prefill_batch(items[i:i + cap]))
            return out
        ids = torch.tensor([tokens for tokens, _ in items], dtype=torch.long, device=self.device)
        slot_batch = [slots for _, slots in items]
        cache = self.bank.gather(slot_batch, new_tokens=t, token_ids=ids)
        logits = self._forward(ids, cache)
        self.bank.scatter(slot_batch, cache)
        return [row.float() for row in logits]

    @torch.inference_mode()
    def decode_step(self, slot_batch: list[dict[str, int]], last_tokens: list[int]) -> torch.Tensor:
        if len(slot_batch) != len(last_tokens):
            raise ValueError("slot_batch and last_tokens length mismatch")
        if self.graphs is not None:
            logits = self.graphs.decode_step(slot_batch, last_tokens)
            if logits is not None:
                return logits.float()
        ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        cache = self.bank.gather(slot_batch, new_tokens=1, token_ids=ids)
        logits = self._forward(ids, cache)
        self.bank.scatter(slot_batch, cache)
        return logits.float()

    def _forward(self, input_ids: torch.Tensor, cache) -> torch.Tensor:
        b, t = input_ids.shape
        lens = torch.tensor(cache.lens, dtype=torch.long, device=self.device)
        positions = lens[:, None] + torch.arange(t, device=self.device)[None, :]
        mask = cache.attention_mask(t, self.device)
        return self.model(input_ids, cache, positions, mask)
