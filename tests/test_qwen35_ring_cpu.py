"""CPU tests for ring-mode guest memory (sink + sliding window) on the hybrid path.

Reference: exact attention recomputed from scratch at every step over the
full sequence with a *band mask* (query at position q sees key position p
iff p <= q and (p < sink or p > q - ring)). The ring engine — chunked
prefill with in-chunk eviction, ring decode, resident-row graphs on GPU —
must reproduce it token for token in fp32. Tiny model, sink 2, ring 8, so
prompts of 7/23/31 tokens wrap the window several times.

Run: ``CUDA_VISIBLE_DEVICES= python -m unittest tests.test_qwen35_ring_cpu -v``
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request

try:
    import torch

    from wkvm.runner.kernels import select_kernels

    # CPU-only run: force the torch path. Imported by the GPU suite (CUDA
    # visible): leave the selection to that suite ("auto" -> fla).
    select_kernels("torch" if not torch.cuda.is_available() else None)
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    HAS_DEPS = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_DEPS = False

LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
SINK, RING = 2, 8
_shared: dict = {}


def _fixture():
    if not _shared:
        from wkvm.models.qwen35 import Qwen35Decoder, Qwen35HybridLayout

        torch.manual_seed(0)
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=len(LAYER_TYPES),
            num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_conv_kernel_dim=4,
            linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=list(LAYER_TYPES),
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_section": [3, 3, 2], "mrope_interleaved": True},
            pad_token_id=0,
        )
        hf = Qwen3_5ForCausalLM._from_config(config, attn_implementation="sdpa").float().eval().requires_grad_(False)
        decoder = Qwen35Decoder.from_hf(hf)
        ring = Qwen35HybridLayout.from_config(config, dtype=torch.float32, guest_mode="ring",
                                              sink_tokens=SINK, ring_tokens=RING)
        exact = Qwen35HybridLayout.from_config(config, dtype=torch.float32, page_tokens=8)
        gen = torch.Generator().manual_seed(1)
        prompts = [torch.randint(1, 128, (n,), generator=gen).tolist() for n in (23, 7, 31)]
        _shared.update(decoder=decoder, ring=ring, exact=exact, prompts=prompts)
    return _shared


def _engine(chunk: int = 5, num_slots: int = 4, cuda_graphs=False, device: str = "cpu"):
    from wkvm.engine import Engine

    fx = _fixture()
    return Engine(
        fx["decoder"], fx["ring"], num_slots=num_slots, device=device,
        scheduler_config=SchedulerConfig(max_tokens_per_step=64, max_running_requests=num_slots,
                                         max_tokens_per_request_per_step=chunk),
        prefill_chunk=chunk, cuda_graphs=cuda_graphs,
    )


def _run(engine, reqs):
    while engine.has_unfinished:
        engine.step()
    return [list(r.output_token_ids) for r in reqs]


def band_reference(prompt: list[int], n_new: int, device: str = "cpu") -> list[int]:
    """Greedy continuation with sink+window attention recomputed from scratch
    every step (exact paged bank, one fresh slot per step, explicit mask)."""
    from wkvm.core.arena import StateArena

    fx = _fixture()
    layout, decoder = fx["exact"], fx["decoder"]
    tokens = list(prompt)
    out: list[int] = []
    for _ in range(n_new):
        t = len(tokens)
        bank = layout.make_bank(1, device, num_pages=layout.state_spec().pages_for(t))
        arena = StateArena(layout.state_spec(), num_slots=1, num_pages=layout.state_spec().pages_for(t))
        slots = arena.allocate(pages=arena.pages_for(t))
        bank.zero_slots(slots)
        cache = bank.gather([slots], new_tokens=t)
        pos = torch.arange(t, device=device)
        q, k = pos[:, None], pos[None, :]
        mask = ((k <= q) & ((k < SINK) | (k > q - RING)))[None, None]
        with torch.inference_mode():
            logits = decoder(torch.tensor([tokens], device=device), cache, pos[None, :], mask)
        tok = int(logits[0].argmax())
        out.append(tok)
        tokens.append(tok)
    return out


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestRingLayout(unittest.TestCase):
    def test_families_and_capacity(self) -> None:
        from wkvm.core.arena import StateArena

        fx = _fixture()
        spec = fx["ring"].state_spec()
        self.assertEqual([f.name for f in spec.families], ["gdn_state", "gdn_conv", "guest_kv"])
        self.assertFalse(spec.families[2].is_paged)
        # 1 attention layer x K,V x 2 kv-heads x (2+8) columns x 64 x fp32
        self.assertEqual(spec.families[2].bytes_per_slot, 1 * 2 * 2 * 10 * 64 * 4)
        arena = StateArena(spec, num_slots=3)
        self.assertIsNone(arena.max_tokens_per_request)  # unbounded context
        self.assertEqual(arena.pages_for(100000), 0)

    def test_ring_columns_and_positions(self) -> None:
        fx = _fixture()
        bank = fx["ring"].make_bank(1, "cpu")
        pos = torch.tensor([0, 1, 2, 9, 10, 17, 18])
        self.assertEqual(bank.ring_columns(pos).tolist(), [0, 1, 2, 9, 2, 9, 2])
        self.assertEqual(bank.ring_positions_for(0).tolist(), [-1] * 10)
        self.assertEqual(bank.ring_positions_for(3).tolist(), [0, 1, 2, -1, -1, -1, -1, -1, -1, -1])
        # after 13 tokens the ring holds positions 5..12: column 2+j holds the p ≡ j (mod 8), p in [5, 12]
        self.assertEqual(bank.ring_positions_for(13).tolist(), [0, 1, 10, 11, 12, 5, 6, 7, 8, 9])


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestRingParity(unittest.TestCase):
    def test_engine_matches_band_reference(self) -> None:
        fx = _fixture()
        for chunk in (5, 64):  # chunks shorter than the ring, and one chunk longer than the window
            engine = _engine(chunk=chunk)
            for prompt in fx["prompts"]:
                req = Request(prompt_token_ids=list(prompt), max_new_tokens=12)
                engine.add_request(req)
                self.assertEqual(_run(engine, [req])[0], band_reference(prompt, 12), f"chunk={chunk} len={len(prompt)}")

    def test_batched_equals_alone_and_slots_reused(self) -> None:
        fx = _fixture()
        alone = [band_reference(p, 10) for p in fx["prompts"]]
        engine = _engine()
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=10) for p in fx["prompts"]]
        for r in reqs:
            engine.add_request(r)
        self.assertEqual(_run(engine, reqs), alone)
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=10) for p in fx["prompts"]]
        for r in reqs:
            engine.add_request(r)
        self.assertEqual(_run(engine, reqs), alone)

    def test_context_beyond_any_window_is_admitted(self) -> None:
        fx = _fixture()
        engine = _engine()
        prompt = list(fx["prompts"][2]) * 4  # 124 tokens with a 10-column window
        req = Request(prompt_token_ids=prompt, max_new_tokens=6)
        engine.add_request(req)
        self.assertEqual(_run(engine, [req])[0], band_reference(prompt, 6))
        self.assertEqual(engine.arena.num_free_slots(), 4)


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestRingStore(unittest.TestCase):
    def test_hibernate_resume_exact(self) -> None:
        fx = _fixture()
        tmp = tempfile.mkdtemp()
        try:
            engine = _engine()
            engine.attach_store(tmp)
            prompt = fx["prompts"][0]
            reference = band_reference(prompt, 20)
            twin = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
            engine.add_request(twin)
            engine.save_on_finish(twin.req_id, "twin")
            _run(engine, [twin])
            handle = engine._finish_handles[twin.req_id]
            rest = _run(engine, [engine.submit_from_handle(handle, max_new_tokens=12)])[0]
            self.assertEqual(reference, twin.output_token_ids + rest)
            engine.store.evict(handle)
            cold = _run(engine, [engine.submit_from_handle(handle, max_new_tokens=12)])[0]
            self.assertEqual(rest, cold)
            record = engine.store._tensors(handle)
            self.assertEqual(tuple(record["ring_pos"].shape), (SINK + RING,))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
