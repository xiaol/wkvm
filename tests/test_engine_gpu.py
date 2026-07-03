"""GPU tests for the M2 engine: the M0 scheduler driving the M1 runner.

Skipped when CUDA/torch is unavailable; weight-bound classes skip when their
checkpoint is missing (``WKVM_RWKV7_PATH`` / ``WKVM_RWKV7_1P5B_PATH``
override the defaults).

Determinism note: the engine config used here satisfies the invariant in
``wkvm/engine.py`` (budget >= max_running * per-request cap), so every
request's prefill chunk boundaries are min(gap, cap) regardless of what else
is in flight — outputs must then match a sequential run token-for-token.

Run: ``python -m unittest tests.test_engine_gpu -v``
"""

from __future__ import annotations

import os
import unittest

from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - core stays torch-free
    torch = None  # type: ignore[assignment]
    HAS_CUDA = False

_WEIGHTS_ROOT = "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla"
WEIGHTS = os.environ.get("WKVM_RWKV7_PATH", f"{_WEIGHTS_ROOT}/rwkv7-191M-world")
WEIGHTS_1P5B = os.environ.get(
    "WKVM_RWKV7_1P5B_PATH", f"{_WEIGHTS_ROOT}/rwkv7-1.5B-world"
)


def _has(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


# Deterministic-chunking config (see module docstring): 8 slots, per-request
# cap 512 (a chunk_rwkv7-aligned multiple of 64), budget >= 8 * 512.
def _config(num_slots: int = 8) -> SchedulerConfig:
    return SchedulerConfig(
        max_tokens_per_step=8192,
        max_running_requests=num_slots,
        max_tokens_per_request_per_step=512,
    )


_shared: dict = {}


def _engine():
    """191M engine + a >=2000-token real-text id sequence, once per module."""
    if not _shared:
        from transformers import AutoTokenizer

        from wkvm.engine import Engine

        engine = Engine.from_pretrained(
            WEIGHTS, num_slots=8, scheduler_config=_config(8), prefill_chunk=512
        )
        tokenizer = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        # Real text, not random ids: random prompts give a near-flat next-token
        # distribution where greedy flips on 1-ulp bf16 noise (see M1 tests).
        base = tokenizer(
            "Once upon a time, there was a little girl who lived in a village "
            "near the forest. Whenever she went out, the little girl wore a "
            "red riding cloak, so everyone in the village called her Little "
            "Red Riding Hood. One morning her mother asked her to take a "
            "basket of bread and butter to her grandmother, who lived on the "
            "far side of the wood and had been feeling poorly. " * 30
        )["input_ids"]
        assert len(base) >= 2000
        _shared.update(engine=engine, base=base)
    return _shared


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
@unittest.skipUnless(_has(WEIGHTS), f"no RWKV-7 weights at {WEIGHTS}")
class TestContinuousBatching(unittest.TestCase):
    N = 24

    def _make_requests(self) -> list[Request]:
        """24 requests: prompt lengths spread over [10, 2000], max_new_tokens
        cycling over [8, 128]. Same shapes for the engine and reference runs."""
        base = _engine()["base"]
        reqs = []
        for i in range(self.N):
            plen = 10 + round((2000 - 10) * i / (self.N - 1))
            reqs.append(
                Request(
                    prompt_token_ids=list(base[:plen]),
                    max_new_tokens=[8, 16, 32, 48, 64, 96, 128][i % 7],
                )
            )
        return reqs

    def test_staggered_arrivals_match_sequential(self) -> None:
        eng = _engine()["engine"]
        reqs = self._make_requests()

        # Staggered arrival against 8 slots: 4 up front, then one every two
        # steps — admission pressure the whole run (24 requests, 8 slots).
        arrived = 4
        for r in reqs[:arrived]:
            eng.add_request(r)
        finished: list[Request] = []
        steps = 0
        while eng.has_unfinished or arrived < self.N:
            if steps % 2 == 0 and arrived < self.N:
                eng.add_request(reqs[arrived])
                arrived += 1
            finished.extend(eng.step())
            steps += 1
            self.assertLess(steps, 10_000, "engine did not converge")

        # Every request finished, slots fully freed.
        self.assertEqual(len(finished), self.N)
        for req in reqs:
            self.assertTrue(req.status.is_finished, req.req_id)
            self.assertEqual(len(req.output_token_ids), req.max_new_tokens)
            self.assertEqual(req.slots, {})
        self.assertEqual(eng.arena.num_free_slots(), 8)
        self.assertFalse(eng.has_unfinished)

        # Greedy outputs must equal the same requests run sequentially
        # (one request in flight at a time, same engine config).
        for i, req in enumerate(reqs):
            ref = Request(
                prompt_token_ids=list(req.prompt_token_ids),
                max_new_tokens=req.max_new_tokens,
            )
            eng.add_request(ref)
            while eng.has_unfinished:
                eng.step()
            self.assertEqual(
                req.output_token_ids,
                ref.output_token_ids,
                f"request {i} (prompt len {req.num_prompt_tokens})",
            )
        self.assertEqual(eng.arena.num_free_slots(), 8)


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
@unittest.skipUnless(_has(WEIGHTS_1P5B), f"no 1.5B weights at {WEIGHTS_1P5B}")
class TestEngine1p5BParity(unittest.TestCase):
    """M2 parity gate at 1.5B: greedy continuations engine-vs-reference on
    the M1 fixture prompts — exact token equality, batch-1 vs batch-1.

    Like-for-like matters: the reference decodes at batch 1, so the engine
    side runs one request at a time. Decoding the same 3 prompts as one
    batch flips exactly one token (prompt 2, step 23) where the reference
    top-2 margin is 0.0312 — one bf16 ulp at |logit| ~ 40 — because lm_head/
    MLP GEMM accumulation order is batch-shape-dependent (the reason vLLM
    ships a separate batch-invariant mode). Batch-vs-sequential exactness
    is certified at 191M by TestContinuousBatching (batch sizes 1..8, 24
    requests, zero flips); at 1.5B it holds everywhere except that one
    sub-ulp tie."""

    TEXTS = [
        "The Eiffel Tower is located in the city of",
        "In a shocking finding, scientists discovered a herd of dragons living "
        "in a remote valley in Tibet. " * 2 + "The lead researcher explained that",
        "Once upon a time, there was a little girl who lived in a village near "
        "the forest. " * 14 + "One day her mother said",
    ]
    NEW_TOKENS = 32

    def test_greedy_continuations_match_reference(self) -> None:
        from fla.models.utils import Cache
        from transformers import AutoTokenizer

        from wkvm.engine import Engine

        engine = Engine.from_pretrained(
            WEIGHTS_1P5B, num_slots=4, scheduler_config=_config(4), prefill_chunk=512
        )
        tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_1P5B, trust_remote_code=True)
        prompts = [tokenizer(t)["input_ids"] for t in self.TEXTS]

        reqs = [
            Request(prompt_token_ids=list(p), max_new_tokens=self.NEW_TOKENS)
            for p in prompts
        ]
        for r in reqs:  # batch-1: one request in flight at a time
            engine.add_request(r)
            while engine.has_unfinished:
                engine.step()

        model = engine.runner.model
        for req, prompt in zip(reqs, prompts):
            ref: list[int] = []
            cache = Cache()
            ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                step = model(input_ids=ids, past_key_values=cache,
                             use_cache=True, logits_to_keep=1)
                for _ in range(self.NEW_TOKENS):
                    tok = int(step.logits[0, -1].argmax().item())
                    ref.append(tok)
                    step = model(
                        input_ids=torch.tensor([[tok]], dtype=torch.long,
                                               device="cuda"),
                        past_key_values=cache, use_cache=True, logits_to_keep=1,
                    )
            self.assertEqual(req.output_token_ids, ref, f"len={len(prompt)}")
        self.assertEqual(engine.arena.num_free_slots(), 4)


if __name__ == "__main__":
    unittest.main()
