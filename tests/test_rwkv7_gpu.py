"""GPU tests for the M1 runner: RWKV-7 decode from arena slots.

Skipped entirely when CUDA (or torch/fla) is unavailable; the weight-bound
tests additionally skip when the 0.1B checkpoint is missing. Point
``WKVM_RWKV7_PATH`` at an fla-format RWKV-7 directory to override.

Run: ``python -m unittest tests.test_rwkv7_gpu -v``
"""

from __future__ import annotations

import os
import unittest

from wkvm.core.arena import StateArena
from wkvm.core.request import Request

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - core stays torch-free
    torch = None  # type: ignore[assignment]
    HAS_CUDA = False

WEIGHTS = os.environ.get(
    "WKVM_RWKV7_PATH",
    "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla/rwkv7-191M-world",
)
HAS_WEIGHTS = os.path.isfile(os.path.join(WEIGHTS, "config.json"))

_shared: dict = {}

# Real-text prompts, not random ids: random-token prompts give the model a
# degenerate, near-flat next-token distribution where greedy argmax flips on
# 1-ulp bf16 kernel noise, making the continuation check meaningless.
# Tokenized lengths (11, 43, 271) cover: single fused-recurrent short
# prefill, single chunk_rwkv7 chunk, and a multi-chunk prefill with a
# sub-64 fused tail plus mid-prompt bank round-trips.
_TEXTS = [
    "The Eiffel Tower is located in the city of",
    "In a shocking finding, scientists discovered a herd of dragons living "
    "in a remote valley in Tibet. " * 2 + "The lead researcher explained that",
    "Once upon a time, there was a little girl who lived in a village near "
    "the forest. " * 14 + "One day her mother said",
]


def _engine():
    """Load the 0.1B model + bank + arena + tokenizer once per module."""
    if not _shared:
        from transformers import AutoTokenizer

        from wkvm.models.rwkv7 import load_rwkv7
        from wkvm.runner import RWKV7Runner, RWKV7StateBank

        model, layout = load_rwkv7(WEIGHTS, device="cuda")
        bank = RWKV7StateBank(layout, num_slots=8, device="cuda")
        arena = StateArena(layout.state_spec(), num_slots=8)
        tokenizer = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        _shared.update(
            model=model, layout=layout, bank=bank, arena=arena,
            # 128: a multiple of chunk_rwkv7's internal chunk (64), so chunk
            # boundaries align with the reference one-shot scan's internal ones.
            runner=RWKV7Runner(model, bank, prefill_chunk=128),
            prompts=[tokenizer(t)["input_ids"] for t in _TEXTS],
        )
    return _shared


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
class TestStateBankRoundTrip(unittest.TestCase):
    """Bank-only tests: no weights needed, just a layout."""

    def _bank(self):
        from wkvm.models.rwkv7 import RWKV7StateLayout
        from wkvm.runner.state import RWKV7StateBank

        layout = RWKV7StateLayout(
            n_layer=3, hidden_size=64, num_heads=2, head_dim=32,
            head_v_dim=32, dtype=torch.bfloat16,
        )
        return RWKV7StateBank(layout, num_slots=4, device="cuda")

    def test_gather_scatter_roundtrip(self) -> None:
        bank = self._bank()
        batch = [{"wkv": 2, "shift": 2}, {"wkv": 4, "shift": 4}]
        cache = bank.gather_cache(batch)
        for i in range(bank.layout.n_layer):
            cache[i]["recurrent_state"].normal_()
            cache[i]["conv_state"].normal_()
            cache[i]["ffn_state"].normal_()
        expected = [
            (
                cache[i]["recurrent_state"].clone(),
                cache[i]["conv_state"].clone(),
                cache[i]["ffn_state"].clone(),
            )
            for i in range(bank.layout.n_layer)
        ]
        bank.scatter_cache(batch, cache)
        back = bank.gather_cache(batch)
        for i, (wkv, conv, ffn) in enumerate(expected):
            torch.testing.assert_close(back[i]["recurrent_state"], wkv)
            torch.testing.assert_close(back[i]["conv_state"], conv)
            torch.testing.assert_close(back[i]["ffn_state"], ffn)

    def test_scatter_leaves_other_slots_untouched(self) -> None:
        bank = self._bank()
        cache = bank.gather_cache([{"wkv": 3, "shift": 3}])
        for i in range(bank.layout.n_layer):
            cache[i]["recurrent_state"].fill_(1.0)
            cache[i]["conv_state"].fill_(1.0)
            cache[i]["ffn_state"].fill_(1.0)
        bank.scatter_cache([{"wkv": 3, "shift": 3}], cache)
        # Slot 0 (reserved padding target) and slots 1,2,4 stay zero.
        for s in (0, 1, 2, 4):
            self.assertEqual(bank.wkv[:, s].abs().sum().item(), 0.0, f"slot {s}")
            self.assertEqual(bank.shift[:, :, s].abs().sum().item(), 0.0, f"slot {s}")
        self.assertGreater(bank.wkv[:, 3].abs().sum().item(), 0.0)


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
@unittest.skipUnless(HAS_WEIGHTS, f"no RWKV-7 weights at {WEIGHTS}")
class TestRWKV7Parity(unittest.TestCase):
    """The parity gate is split by what bf16 can certify:

    1. State *plumbing* (arena gather -> forward -> scatter) must be
       bit-preserving, so a single-chunk prefill through the arena is gated
       at 1e-2 against the reference forward — measured 0.0.
    2. *Chunked* prefill re-enters the kernels with a carried state; even
       64-aligned, that reorders bf16 accumulation by 1-2 ulp (~0.25-0.5 at
       |logit| ~ 40 here). The fla reference disagrees with its own
       ``use_cache=False`` path by the same order (measured up to 0.5), so a
       1e-2 absolute gate on raw bf16 logits is not meaningful for this leg;
       it is gated on argmax equality + a 1-ulp-scale bound, with the actual
       diff printed for the record. In fp32 the same comparison lands at
       ~1.4e-2.
    3. Greedy continuations must match the reference exactly (32 tokens x 3
       real prompts) — the end-to-end certificate that ulp noise never
       changes sampled output on real text.
    """

    PLUMBING_TOL = 1e-2  # single-chunk: bit-exact expected
    CHUNKED_TOL = 1.0    # multi-chunk: 1-2 bf16 ulp at |logit| ~ 40

    def _reference_last_logits(
        self, prompt: list[int], seed_zero_state: bool = False
    ) -> torch.Tensor:
        """Reference forward: fla's own cache path (what `generate` uses),
        one un-chunked pass, states never touching the arena.

        ``seed_zero_state`` seeds the cache with explicit zero states, which
        is semantically identical to an empty cache (zero wkv = empty prefix
        sum; zero token-shift = the ZeroPad shift-in) but selects the same
        ``USE_INITIAL_STATE`` kernel specialization the runner uses — needed
        when the gate is bit-exactness rather than ulp-closeness. The
        zero == empty equivalence itself is covered by the chunked-parity
        and greedy gates, which compare against the *empty* cache."""
        from fla.models.utils import Cache

        eng = _engine()
        model, layout = eng["model"], eng["layout"]
        cache = Cache()
        if seed_zero_state:
            zeros = lambda *shape, dtype: torch.zeros(  # noqa: E731
                1, *shape, dtype=dtype, device="cuda"
            )
            for i in range(layout.n_layer):
                cache.update(
                    recurrent_state=zeros(*layout.wkv_shape, dtype=torch.float32),
                    conv_state=zeros(*layout.shift_shape, dtype=layout.dtype),
                    ffn_state=zeros(*layout.shift_shape, dtype=layout.dtype),
                    layer_idx=i, offset=0,
                )
        with torch.inference_mode():
            return model(
                input_ids=torch.tensor([prompt], dtype=torch.long, device="cuda"),
                past_key_values=cache, use_cache=True, logits_to_keep=1,
            ).logits[0, -1].float()

    def _reference_greedy(self, model, prompt: list[int], n: int) -> list[int]:
        """Reference decode: fla's own Cache management, no arena."""
        from fla.models.utils import Cache

        cache = Cache()
        ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
        out: list[int] = []
        with torch.inference_mode():
            step = model(input_ids=ids, past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
            for _ in range(n):
                tok = int(step.logits[0, -1].argmax().item())
                out.append(tok)
                step = model(
                    input_ids=torch.tensor([[tok]], dtype=torch.long, device="cuda"),
                    past_key_values=cache, use_cache=True, logits_to_keep=1,
                )
        return out

    def test_plumbing_parity_single_chunk(self) -> None:
        """Gate 1: arena round-trip is lossless. Prefill with the whole
        prompt as one chunk takes the identical kernel path as the
        reference, so any diff would be a state-plumbing bug."""
        from wkvm.runner import RWKV7Runner

        eng = _engine()
        runner = RWKV7Runner(eng["model"], eng["bank"], prefill_chunk=4096)
        worst = 0.0
        for prompt in eng["prompts"]:
            slots = eng["arena"].allocate()
            eng["bank"].zero_slots(slots)
            ours = runner.prefill(prompt, slots)
            eng["arena"].free(slots)
            ref = self._reference_last_logits(prompt, seed_zero_state=True)
            diff = (ours - ref).abs().max().item()
            worst = max(worst, diff)
            self.assertLess(diff, self.PLUMBING_TOL, f"len={len(prompt)}")
        print(f"\n[parity/plumbing] max abs last-logit diff: {worst:.3e}")

    def test_chunked_prefill_parity(self) -> None:
        """Gate 2: chunked prefill (128/chunk, states parked in the arena
        between chunks) vs the reference one-shot pass."""
        eng = _engine()
        worst = 0.0
        for prompt in eng["prompts"]:
            slots = eng["arena"].allocate()
            eng["bank"].zero_slots(slots)
            ours = eng["runner"].prefill(prompt, slots)
            eng["arena"].free(slots)
            ref = self._reference_last_logits(prompt)
            diff = (ours - ref).abs().max().item()
            worst = max(worst, diff)
            self.assertEqual(int(ours.argmax()), int(ref.argmax()), f"len={len(prompt)}")
            self.assertLess(diff, self.CHUNKED_TOL, f"len={len(prompt)}")
        print(f"\n[parity/chunked] max abs last-logit diff: {worst:.3e} (bf16 ulp scale)")

    def test_greedy_continuations_match_reference(self) -> None:
        """Gate 3: 32 greedy tokens x 3 prompts, runner-from-arena vs the
        reference decoding with its own cache. Exact token equality."""
        from wkvm.runner import GenerationLoop, SamplingParams

        eng = _engine()
        loop = GenerationLoop(eng["runner"], eng["arena"])
        reqs = [
            Request(prompt_token_ids=list(p), max_new_tokens=32)
            for p in eng["prompts"]
        ]
        loop.generate(reqs, SamplingParams(temperature=0.0))
        for req, prompt in zip(reqs, eng["prompts"]):
            ref = self._reference_greedy(eng["model"], prompt, 32)
            self.assertEqual(req.output_token_ids, ref, f"len={len(prompt)}")

    def test_concurrent_matches_sequential(self) -> None:
        """State isolation: two requests decoded as one batch produce the
        same tokens as the same requests run alone."""
        from wkvm.runner import GenerationLoop, SamplingParams

        eng = _engine()
        loop = GenerationLoop(eng["runner"], eng["arena"])
        prompts = eng["prompts"][:2]
        greedy = SamplingParams(temperature=0.0)

        sequential = []
        for p in prompts:
            (req,) = loop.generate(
                [Request(prompt_token_ids=list(p), max_new_tokens=24)], greedy
            )
            sequential.append(req.output_token_ids)

        concurrent = loop.generate(
            [Request(prompt_token_ids=list(p), max_new_tokens=24) for p in prompts],
            greedy,
        )
        for seq, con in zip(sequential, concurrent):
            self.assertEqual(con.output_token_ids, seq)


if __name__ == "__main__":
    unittest.main()
