"""CPU tests for the routed span bank (``guest_mode="routed"``).

Tiny model; params sink 2, ring 8, pending 4 (buffer 8), 3 slots, 6-token
representative budget, spans cut at three "break" token ids. Prompts of 7 /
23 / 31 tokens plus decode drive several evictions and routing passes.

Checks: (1) inside the window nothing is evicted, so the routed engine
equals the band-mask reference of ring mode; (2) invariants after routing —
pending drained, no column beyond the budget, summaries valid iff a slot
has tokens, no scratch leak; (3) batched == alone (routing is per row);
(4) hibernate/resume round-trips the store and the host routing state.

Run: ``CUDA_VISIBLE_DEVICES= python -m unittest tests.test_qwen35_routed_cpu -v``
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

    select_kernels("torch" if not torch.cuda.is_available() else None)
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    HAS_DEPS = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_DEPS = False

LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
BREAKS = (5, 17, 33)
ROUTED = dict(routed_pending=4, routed_slots=3, routed_reps=6, routed_max_span=4, routed_fallback_span=3,
              break_token_ids=BREAKS)
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
        routed = Qwen35HybridLayout.from_config(config, dtype=torch.float32, guest_mode="routed",
                                                sink_tokens=2, ring_tokens=8, **ROUTED)
        ring = Qwen35HybridLayout.from_config(config, dtype=torch.float32, guest_mode="ring",
                                              sink_tokens=2, ring_tokens=8)
        exact = Qwen35HybridLayout.from_config(config, dtype=torch.float32, page_tokens=8)
        gen = torch.Generator().manual_seed(1)
        prompts = [torch.randint(1, 128, (n,), generator=gen).tolist() for n in (23, 7, 31)]
        _shared.update(hf=hf, decoder=decoder, routed=routed, ring=ring, exact=exact, prompts=prompts)
    return _shared


def _engine(layout_key: str = "routed", chunk: int = 4, num_slots: int = 4, cuda_graphs=False, device: str = "cpu"):
    from wkvm.engine import Engine

    fx = _fixture()
    return Engine(
        fx["decoder"], fx[layout_key], num_slots=num_slots, device=device,
        scheduler_config=SchedulerConfig(max_tokens_per_step=64, max_running_requests=num_slots,
                                         max_tokens_per_request_per_step=chunk),
        prefill_chunk=chunk, cuda_graphs=cuda_graphs,
    )


def _run(engine, reqs):
    while engine.has_unfinished:
        engine.step()
    return [list(r.output_token_ids) for r in reqs]


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestRoutedLayout(unittest.TestCase):
    def test_columns_and_bytes(self) -> None:
        fx = _fixture()
        p = fx["routed"].routed_params()
        # sink 2 + ring 8 + pending 8 + summaries 3 + reps 18 + scratch 1
        self.assertEqual(p.columns, 2 + 8 + 8 + 3 + 18 + 1)
        spec = fx["routed"].state_spec()
        self.assertFalse(spec.families[2].is_paged)
        self.assertEqual(spec.families[2].bytes_per_slot, p.columns * fx["routed"].guest_bytes_per_token)


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class TestRoutedEngine(unittest.TestCase):
    def test_within_window_equals_ring_reference(self) -> None:
        """A 7-token prompt + 3 new tokens never leaves a 2+8 window: no
        eviction, no routing, identical to the band-mask reference."""
        try:
            from tests import test_qwen35_ring_cpu as R
        except ImportError:  # run from the tests directory
            import test_qwen35_ring_cpu as R

        fx = _fixture()
        saved = dict(R._shared)  # borrow the ring module's reference without leaving our fixture in it
        R._shared.update(decoder=fx["decoder"], exact=fx["exact"], prompts=fx["prompts"])
        try:
            ref = R.band_reference(fx["prompts"][1], 3)
        finally:
            R._shared.clear()
            R._shared.update(saved)
        engine = _engine()
        req = Request(prompt_token_ids=list(fx["prompts"][1]), max_new_tokens=3)
        engine.add_request(req)
        self.assertEqual(_run(engine, [req])[0], ref)
        self.assertEqual(engine.bank.rg.stats["routing_passes"], 0)

    def test_routing_invariants_and_batch_independence(self) -> None:
        fx = _fixture()
        alone = []
        for p in fx["prompts"]:
            e = _engine()
            req = Request(prompt_token_ids=list(p), max_new_tokens=12)
            e.add_request(req)
            alone.append(_run(e, [req])[0])
            self.assertGreater(e.bank.rg.stats["routing_passes"], 0)
        engine = _engine()
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=12) for p in fx["prompts"]]
        for r in reqs:
            engine.add_request(r)
        # Snapshot invariants while the requests are still running (slots live).
        while engine.has_unfinished:
            engine.step()
            for r in reqs:
                if r.slots:
                    self._check_invariants(engine, r.slots)
        self.assertEqual([r.output_token_ids for r in reqs], alone)
        self.assertGreater(engine.bank.rg.stats["spans"], 0)

    def _check_invariants(self, engine, slots) -> None:
        bank = engine.bank
        p, st = bank.rg.p, bank.rstore
        g = slots["guest_kv"]
        self.assertLess(int(st.pend[g]), p.pend_cap)
        valid = st.valid[0, g]
        self.assertFalse(bool(valid[p.scratch]))
        self.assertLessEqual(int(valid[p.pend_base:p.summ_base].sum()), int(st.pend[g]))
        for s in range(p.slots):
            b0, b1 = p.rep_block(s)
            self.assertLessEqual(int(valid[b0:b1].sum()), p.reps)
            state = bank.rg.state.get((slots["gdn_state"], 0))
            if state is not None:
                self.assertEqual(bool(valid[p.summ_base + s]), state.count[s] > 0)
        # positions of valid columns are never -1
        self.assertTrue(bool((st.pos[g][valid] >= 0).all()))

    def test_hibernate_resume_round_trips_routing_state(self) -> None:
        fx = _fixture()
        tmp = tempfile.mkdtemp()
        try:
            engine = _engine()
            engine.attach_store(tmp)
            prompt = fx["prompts"][2]
            full = Request(prompt_token_ids=list(prompt), max_new_tokens=20)
            engine.add_request(full)
            reference = _run(engine, [full])[0]
            twin = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
            engine.add_request(twin)
            engine.save_on_finish(twin.req_id, "twin")
            _run(engine, [twin])
            handle = engine._finish_handles[twin.req_id]
            engine.store.persist(handle)
            engine.store.evict(handle)
            rest = _run(engine, [engine.submit_from_handle(handle, max_new_tokens=12)])[0]
            self.assertEqual(reference, twin.output_token_ids + rest)
            rec = engine.store._tensors(handle)
            self.assertIn("rt_state", rec)
            self.assertEqual(rec["rt_state"].dtype, torch.uint8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
