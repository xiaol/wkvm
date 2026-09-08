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
        # Exact admission needs this at intake: prompt + max_new_tokens must
        # fit the guest window (the engine checks it before queueing).
        self.max_tokens_per_request = bank.layout.guest_ctx

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], slots: dict[str, int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("empty prompt")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        logits = None
        for start in range(0, len(token_ids), self.prefill_chunk):
            chunk = ids[start : start + self.prefill_chunk].unsqueeze(0)
            cache = self.bank.gather([slots], new_tokens=chunk.shape[1])
            logits = self._forward(chunk, cache)
            self.bank.scatter([slots], cache)
        return logits[0].float()

    @torch.inference_mode()
    def decode_step(self, slot_batch: list[dict[str, int]], last_tokens: list[int]) -> torch.Tensor:
        if len(slot_batch) != len(last_tokens):
            raise ValueError("slot_batch and last_tokens length mismatch")
        ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        cache = self.bank.gather(slot_batch, new_tokens=1)
        logits = self._forward(ids, cache)
        self.bank.scatter(slot_batch, cache)
        return logits.float()

    def _forward(self, input_ids: torch.Tensor, cache) -> torch.Tensor:
        b, t = input_ids.shape
        lens = torch.tensor(cache.lens, dtype=torch.long, device=self.device)
        positions = lens[:, None] + torch.arange(t, device=self.device)[None, :]
        mask = cache.attention_mask(t, self.device)
        return self.model(input_ids, cache, positions, mask)
