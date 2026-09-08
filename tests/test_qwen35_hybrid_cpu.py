"""CPU tests for the M4 hybrid path: Qwen3.5 (Gated DeltaNet + full attention).

A randomly initialised tiny Qwen3.5 text model (4 layers, 3 linear + 1 full
attention) is enough to certify the *plumbing*: state gather -> HF layer
forward -> scatter must reproduce HF's own cached path exactly, in fp32 on
CPU, for chunked prefill, batched decode with mixed lengths, hibernate /
resume across store tiers, and a tuned initial state imported as a handle.

Run: ``CUDA_VISIBLE_DEVICES= python -m unittest tests.test_qwen35_hybrid_cpu -v``
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest

from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request

try:
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    HAS_DEPS = True
except ImportError:  # pragma: no cover - core stays torch-free
    torch = None  # type: ignore[assignment]
    HAS_DEPS = False

LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
GUEST_CTX = 64


def _tiny_config() -> "Qwen3_5TextConfig":
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=list(LAYER_TYPES),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        pad_token_id=0,
    )


_shared: dict = {}


def _fixture():
    """Tiny HF model + wkvm decoder/layout, built once (CPU, fp32)."""
    if not _shared:
        from wkvm.models.qwen35 import Qwen35Decoder, Qwen35HybridLayout

        torch.manual_seed(0)
        config = _tiny_config()
        hf = Qwen3_5ForCausalLM._from_config(config, attn_implementation="sdpa")
        hf = hf.float().eval().requires_grad_(False)
        decoder = Qwen35Decoder.from_hf(hf)
        layout = Qwen35HybridLayout.from_config(config, dtype=torch.float32, guest_ctx=GUEST_CTX)
        gen = torch.Generator().manual_seed(1)
        prompts = [
            torch.randint(1, 128, (n,), generator=gen).tolist() for n in (23, 7, 31)
        ]
        _shared.update(hf=hf, decoder=decoder, layout=layout, prompts=prompts)
    return _shared


def _sched(num_slots: int = 4, chunk: int = 5) -> SchedulerConfig:
    return SchedulerConfig(
        max_tokens_per_step=64,
        max_running_requests=num_slots,
        max_tokens_per_request_per_step=chunk,
    )


def _engine(num_slots: int = 4, chunk: int = 5):
    from wkvm.engine import Engine

    fx = _fixture()
    return Engine(
        fx["decoder"], fx["layout"], num_slots=num_slots, device="cpu",
        scheduler_config=_sched(num_slots, chunk), prefill_chunk=chunk,
    )


def _run(engine, req):
    while not req.status.is_finished:
        engine.step()
    return list(req.output_token_ids)


def _hf_greedy(hf, prompt: list[int], n: int) -> list[int]:
    ids = torch.tensor([prompt])
    out = hf.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    return out[0, len(prompt):].tolist()


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestQwen35Layout(unittest.TestCase):
    def test_families_and_bytes(self) -> None:
        fx = _fixture()
        layout = fx["layout"]
        spec = layout.state_spec()
        self.assertEqual([f.name for f in spec.families], ["gdn_state", "gdn_conv", "guest_kv"])
        self.assertEqual(layout.gdn_layers, (0, 1, 3))
        self.assertEqual(layout.attn_layers, (2,))
        # gdn_state: 3 layers x 4 heads x 16 x 16 x fp32
        self.assertEqual(spec.families[0].bytes_per_slot, 3 * 4 * 16 * 16 * 4)
        # gdn_conv: 3 layers x (2*2*16 + 4*16) x 4 x fp32
        self.assertEqual(spec.families[1].bytes_per_slot, 3 * 128 * 4 * 4)
        # guest_kv: 1 layer x 2 (K,V) x 2 kv heads x 64 ctx x 64 head_dim x fp32
        self.assertEqual(spec.families[2].bytes_per_slot, 1 * 2 * 2 * 64 * 64 * 4)
        self.assertEqual(spec.families[2].layer_ids, (2,))
        self.assertEqual(layout.bytes_per_slot, spec.bytes_per_request)

    def test_qwen35_9b_layout_numbers(self) -> None:
        """The real checkpoint's per-slot footprint, from config numbers only."""
        from wkvm.models.qwen35 import Qwen35HybridLayout

        types = tuple("full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32))
        layout = Qwen35HybridLayout(
            n_layer=32, layer_types=types, hidden_size=4096,
            num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel=4,
            num_kv_heads=4, head_dim=256, guest_ctx=4096, vocab_size=248320, dtype=torch.bfloat16,
        )
        spec = {f.name: f.bytes_per_slot for f in layout.state_spec().families}
        self.assertEqual(spec["gdn_state"], 24 * 32 * 128 * 128 * 4)  # 48 MiB
        self.assertEqual(spec["gdn_conv"], 24 * 8192 * 4 * 2)  # 1.5 MiB
        self.assertEqual(layout.guest_bytes_per_token, 8 * 2 * 4 * 256 * 2)  # 32 KiB/token
        self.assertEqual(spec["guest_kv"], 4096 * 32 * 1024)  # 128 MiB at a 4k window


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestQwen35RunnerParity(unittest.TestCase):
    def test_chunked_prefill_matches_reference_logits(self) -> None:
        from wkvm.core.arena import StateArena

        fx = _fixture()
        layout, hf = fx["layout"], fx["hf"]
        bank = layout.make_bank(2, "cpu")
        arena = StateArena(layout.state_spec(), num_slots=2)
        for chunk in (5, 64):  # sub-kernel-chunk re-entry and a single chunk
            runner = layout.make_runner(fx["decoder"], bank, prefill_chunk=chunk)
            for prompt in fx["prompts"]:
                slots = arena.allocate()
                bank.zero_slots(slots)
                ours = runner.prefill(prompt, slots)
                with torch.inference_mode():
                    ref = hf(torch.tensor([prompt]), use_cache=False).logits[0, -1].float()
                torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)
                self.assertEqual(bank.slot_len(slots), len(prompt))
                arena.free(slots)

    def test_engine_greedy_matches_hf_generate(self) -> None:
        fx = _fixture()
        engine = _engine()
        for prompt in fx["prompts"]:
            req = Request(prompt_token_ids=list(prompt), max_new_tokens=12)
            engine.add_request(req)
            self.assertEqual(_run(engine, req), _hf_greedy(fx["hf"], prompt, 12))

    def test_batched_decode_independent_of_batch_composition(self) -> None:
        fx = _fixture()
        alone = []
        for prompt in fx["prompts"]:
            engine = _engine()
            req = Request(prompt_token_ids=list(prompt), max_new_tokens=10)
            engine.add_request(req)
            alone.append(_run(engine, req))
        engine = _engine(num_slots=4, chunk=5)
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=10) for p in fx["prompts"]]
        for req in reqs:
            engine.add_request(req)
        while engine.has_unfinished:
            engine.step()
        self.assertEqual([r.output_token_ids for r in reqs], alone)
        # Slots are back; guest lengths of freed slots are reset on reuse only.
        self.assertEqual(engine.arena.num_free_slots(), 4)

    def test_guest_window_enforced_at_intake(self) -> None:
        engine = _engine()
        with self.assertRaises(ValueError):
            engine.add_request(Request(prompt_token_ids=[1] * 60, max_new_tokens=8))
        ok = Request(prompt_token_ids=[1] * 60, max_new_tokens=4)
        engine.add_request(ok)
        self.assertEqual(len(_run(engine, ok)), 4)


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestQwen35Store(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hibernate_resume_exact_across_tiers(self) -> None:
        fx = _fixture()
        prompt = fx["prompts"][0]
        engine = _engine()
        engine.attach_store(f"{self.tmp}/store_a")
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=30)
        engine.add_request(req)
        reference = _run(engine, req)

        twin = Request(prompt_token_ids=list(prompt), max_new_tokens=10)
        engine.add_request(twin)
        engine.save_on_finish(twin.req_id, "twin")
        _run(engine, twin)
        handle = engine._finish_handles[twin.req_id]
        record = engine.store.get(handle)
        self.assertEqual(record.num_computed_tokens, len(prompt) + 9)
        warm = _run(engine, engine.submit_from_handle(handle, max_new_tokens=20))
        self.assertEqual(reference, twin.output_token_ids + warm)

        engine.store.evict(handle)
        cold = _run(engine, engine.submit_from_handle(handle, max_new_tokens=20))
        self.assertEqual(warm, cold)

        from wkvm.store import StateStore

        rebuilt_store = StateStore(engine.bank, f"{self.tmp}/store_a")
        engine.store = rebuilt_store
        rebuilt = _run(engine, engine.submit_from_handle(handle, max_new_tokens=20))
        self.assertEqual(warm, rebuilt)

    def test_decay_targets_gdn_state_only(self) -> None:
        fx = _fixture()
        engine = _engine()
        engine.attach_store(f"{self.tmp}/store_b")
        req = Request(prompt_token_ids=list(fx["prompts"][2]), max_new_tokens=4)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, "m")
        _run(engine, req)
        parent = engine._finish_handles[req.req_id]
        child = engine.store.mutate(parent, "decay", {"alpha": 0.0})
        src, dst = engine.store._tensors(parent), engine.store._tensors(child)
        self.assertEqual(dst["gdn_state"].abs().sum().item(), 0.0)
        torch.testing.assert_close(dst["gdn_conv"], src["gdn_conv"])
        torch.testing.assert_close(dst["guest_k"], src["guest_k"])
        self.assertEqual(engine.store.get(child).parent, parent)

    def _write_adapter(self, name: str, layers: dict[int, torch.Tensor]) -> str:
        from safetensors.torch import save_file

        d = f"{self.tmp}/{name}"
        import os

        os.makedirs(d, exist_ok=True)
        save_file({f"layers.{i}.recurrent_state": t.contiguous() for i, t in layers.items()},
                  f"{d}/adapter_model.safetensors")
        with open(f"{d}/adapter_config.json", "w") as f:
            json.dump({"backend": "qwen3_5", "method": "direct_recurrent_state",
                       "base_model": "tiny", "format_version": 2}, f)
        return d

    def test_import_tuned_initial_state(self) -> None:
        """A tuned initial state is a handle at length 0; generating from it
        must equal HF's cached path seeded with the same recurrent states."""
        from transformers import DynamicCache

        fx = _fixture()
        layout, hf = fx["layout"], fx["hf"]
        gen = torch.Generator().manual_seed(7)
        tuned = {i: torch.randn(*layout.gdn_state_shape, generator=gen) * 0.5 for i in layout.gdn_layers}
        path = self._write_adapter("tuned", tuned)

        engine = _engine()
        engine.attach_store(f"{self.tmp}/store_c")
        handle = engine.import_state("tuned", path)
        record = engine.store.get(handle)
        self.assertEqual(record.rule, "import")
        self.assertEqual(record.num_computed_tokens, 0)
        self.assertEqual(record.token_ids, ())
        self.assertEqual(record.rule_params["tuned_layers"], list(layout.gdn_layers))
        self.assertIn("sha256", record.rule_params)

        prompt = fx["prompts"][1]
        ours = _run(engine, engine.submit_from_handle(handle, suffix_tokens=list(prompt), max_new_tokens=8))

        # Zero-state generation must differ from the tuned-state one.
        plain = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
        engine.add_request(plain)
        self.assertNotEqual(_run(engine, plain), ours)

        # Independent reference: HF's own LinearAttentionLayer cache, seeded.
        cache = DynamicCache(config=hf.config)
        for i in layout.gdn_layers:
            layer = cache.layers[i]
            layer.lazy_initialization(
                conv_states=torch.zeros(1, layout.conv_dim, layout.conv_kernel),
                recurrent_states=tuned[i][None],
                state_idx=0,
                conv_kernel_size=layout.conv_kernel,
            )
            layer.recurrent_states[0].copy_(tuned[i][None])
            layer.has_previous_state[0] = True
        ids = torch.tensor([prompt])
        ref = []
        with torch.inference_mode():
            logits = hf(ids, past_key_values=cache, use_cache=True).logits[0, -1]
            for _ in range(8):
                tok = int(logits.argmax())
                ref.append(tok)
                logits = hf(torch.tensor([[tok]]), past_key_values=cache, use_cache=True).logits[0, -1]
        self.assertEqual(ours, ref)

    def test_import_rejects_ple_and_wrong_shapes(self) -> None:
        from safetensors.torch import save_file

        fx = _fixture()
        layout = fx["layout"]
        engine = _engine()
        engine.attach_store(f"{self.tmp}/store_d")
        bad = f"{self.tmp}/bad_ple.safetensors"
        save_file({"layers.0.recurrent_state": torch.zeros(*layout.gdn_state_shape),
                   "ple.token_embeddings.weight": torch.zeros(4, 4)}, bad)
        with self.assertRaises(ValueError):
            engine.import_state("bad", bad)
        wrong = f"{self.tmp}/bad_layer.safetensors"
        save_file({"layers.2.recurrent_state": torch.zeros(*layout.gdn_state_shape)}, wrong)
        with self.assertRaises(ValueError):  # layer 2 is full attention
            engine.import_state("bad", wrong)
        partial = f"{self.tmp}/partial.safetensors"
        save_file({"layers.0.recurrent_state": torch.zeros(*layout.gdn_state_shape)}, partial)
        with self.assertRaises(KeyError):
            engine.import_state("partial", partial)
        handle = engine.import_state("partial", partial, strict=False)
        self.assertEqual(engine.store.get(handle).rule_params["missing_layers"], [1, 3])


if __name__ == "__main__":
    unittest.main()
