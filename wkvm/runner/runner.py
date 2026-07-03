"""RWKV7Runner: prefill and batched decode against arena slots.

Both entry points follow one pattern: gather states from the bank into an
fla Cache, run the reference model forward (which internally dispatches
``chunk_rwkv7`` for seq_len >= 64 and the fused recurrent kernel for short
sequences — exactly the chunked-prefill / recurrent-decode split M1 wants),
then scatter the updated states back to the slots. State never survives a
call inside python objects.

M1 keeps gather/scatter as explicit copies (a few MB per step at 0.1B); M2
replaces them with slot-indexed views + CUDA-graph-captured whole steps.
Prefill deliberately round-trips the bank between chunks so chunk-resume
exercises the same persistence path a preempted request will use.
"""

from __future__ import annotations

import torch

from wkvm.runner.state import RWKV7StateBank


class RWKV7Runner:
    def __init__(
        self,
        model,
        bank: RWKV7StateBank,
        prefill_chunk: int = 512,
    ) -> None:
        """``prefill_chunk`` should stay a multiple of ``chunk_rwkv7``'s
        internal chunk (64) so re-entry boundaries align with the kernel's
        own; misaligned chunks add avoidable bf16 accumulation-order noise
        (~1 ulp on last-position logits, measured on the 0.1B)."""
        if prefill_chunk < 1:
            raise ValueError("prefill_chunk must be >= 1")
        self.model = model
        self.bank = bank
        self.prefill_chunk = prefill_chunk
        self.device = bank.device

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], slots: dict[str, int]) -> torch.Tensor:
        """Run a prompt through the model in chunks, leaving the final
        recurrent state in ``slots``. Returns last-position logits (fp32,
        ``[vocab]``) — the distribution the first output token is sampled
        from. ``logits_to_keep=1`` keeps the lm_head cost at O(1) tokens.
        """
        if not token_ids:
            raise ValueError("empty prompt")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        logits = None
        for start in range(0, len(token_ids), self.prefill_chunk):
            chunk = ids[start:start + self.prefill_chunk].unsqueeze(0)
            cache = self.bank.gather_cache([slots])
            out = self.model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            self.bank.scatter_cache([slots], cache)
            logits = out.logits
        return logits[0, -1].float()

    @torch.inference_mode()
    def decode_step(
        self,
        slot_batch: list[dict[str, int]],
        last_tokens: list[int],
    ) -> torch.Tensor:
        """One batched decode step: advance every slot by its last sampled
        token and return next-token logits (fp32, ``[batch, vocab]``).

        seq_len == 1 puts the fla layer on the fused recurrent kernel; the
        initial state per batch row comes from the gathered slot rows, so
        batch composition can change freely between steps.
        """
        if len(slot_batch) != len(last_tokens):
            raise ValueError("slot_batch and last_tokens length mismatch")
        ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device)
        cache = self.bank.gather_cache(slot_batch)
        out = self.model(
            input_ids=ids.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.bank.scatter_cache(slot_batch, cache)
        return out.logits[:, -1].float()
