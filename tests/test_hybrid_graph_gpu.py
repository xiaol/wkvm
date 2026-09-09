"""GPU tests for CUDA-graph decode on the hybrid path (H5).

A tiny random Qwen3.5 (fp32) on one GPU: the graph runner must reproduce
eager decode exactly across batch padding (3 rows in a 4-row bucket),
length-bucket transitions (8 -> 16 -> 32 token buckets), page boundaries,
the eager fallback beyond the largest bucket, and hibernate/resume through
the store. Skipped without CUDA.

Run: ``python -m unittest tests.test_hybrid_graph_gpu -v``
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_CUDA = False

LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
PAGE_TOKENS = 8
_shared: dict = {}


def _fixture():
    if not _shared:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        from wkvm.models.qwen35 import Qwen35Decoder, Qwen35HybridLayout
        from wkvm.runner.kernels import select_kernels

        select_kernels()
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
        hf = Qwen3_5ForCausalLM._from_config(config, attn_implementation="sdpa").float().cuda().eval()
        hf.requires_grad_(False)
        decoder = Qwen35Decoder.from_hf(hf)
        layout = Qwen35HybridLayout.from_config(config, dtype=torch.float32, page_tokens=PAGE_TOKENS)
        gen = torch.Generator().manual_seed(1)
        prompts = [torch.randint(1, 128, (n,), generator=gen).tolist() for n in (23, 7, 31)]
        _shared.update(decoder=decoder, layout=layout, prompts=prompts)
    return _shared


def _engine(cuda_graphs, num_slots: int = 4, chunk: int = 5, num_pages: int = 48):
    from wkvm.engine import Engine

    fx = _fixture()
    return Engine(
        fx["decoder"], fx["layout"], num_slots=num_slots, device="cuda",
        scheduler_config=SchedulerConfig(max_tokens_per_step=64, max_running_requests=num_slots,
                                         max_tokens_per_request_per_step=chunk),
        prefill_chunk=chunk, num_pages=num_pages, cuda_graphs=cuda_graphs,
    )


GRAPH_CFG = {"batch_buckets": (1, 2, 4), "length_buckets": (8, 16, 32), "warmup_iters": 1}


def _run_all(engine, reqs):
    while engine.has_unfinished:
        engine.step()
    return [list(r.output_token_ids) for r in reqs]


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
class TestHybridGraphs(unittest.TestCase):
    def test_graph_matches_eager_batched_and_alone(self) -> None:
        fx = _fixture()
        eager = _engine(False)
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=12) for p in fx["prompts"]]
        for r in reqs:
            eager.add_request(r)
        ref = _run_all(eager, reqs)

        graphed = _engine(GRAPH_CFG)
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=12) for p in fx["prompts"]]
        for r in reqs:
            graphed.add_request(r)
        out = _run_all(graphed, reqs)
        self.assertEqual(out, ref)
        stats = graphed.runner.graphs.stats
        self.assertGreater(stats["replays"], 0)
        self.assertGreaterEqual(stats["captures"], 2)  # 3 rows -> bucket 4; lengths cross 32 -> 8/16/32
        self.assertEqual(graphed.bank.guest_len[0], 0)
        # alone, B=1 bucket
        for p, r in zip(fx["prompts"], ref):
            req = Request(prompt_token_ids=list(p), max_new_tokens=12)
            graphed.add_request(req)
            self.assertEqual(_run_all(graphed, [req])[0], r)

    def test_eager_fallback_beyond_buckets(self) -> None:
        fx = _fixture()
        eager = _engine(False, num_pages=64)
        req = Request(prompt_token_ids=list(fx["prompts"][2]) * 2, max_new_tokens=6)  # 62 + 6 > 32 bucket
        eager.add_request(req)
        ref = _run_all(eager, [req])[0]
        graphed = _engine(GRAPH_CFG, num_pages=64)
        req = Request(prompt_token_ids=list(fx["prompts"][2]) * 2, max_new_tokens=6)
        graphed.add_request(req)
        self.assertEqual(_run_all(graphed, [req])[0], ref)
        self.assertGreater(graphed.runner.graphs.stats["eager_fallbacks"], 0)

    def test_resume_from_store_under_graphs(self) -> None:
        fx = _fixture()
        tmp = tempfile.mkdtemp()
        try:
            eng = _engine(GRAPH_CFG)
            eng.attach_store(tmp)
            prompt = fx["prompts"][0]
            full = Request(prompt_token_ids=list(prompt), max_new_tokens=20)
            eng.add_request(full)
            reference = _run_all(eng, [full])[0]
            twin = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
            eng.add_request(twin)
            eng.save_on_finish(twin.req_id, "twin")
            _run_all(eng, [twin])
            handle = eng._finish_handles[twin.req_id]
            resumed = _run_all(eng, [eng.submit_from_handle(handle, max_new_tokens=12)])[0]
            self.assertEqual(reference, twin.output_token_ids + resumed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
