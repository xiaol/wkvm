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
        _shared.update(hf=hf, decoder=decoder, layout=layout, prompts=prompts)
    return _shared


def _decoder_on(device: str):
    """The fixture's decoder on another device, as an independent copy (the
    shared one stays on the default device for the other tests)."""
    import copy

    from wkvm.models.qwen35 import Qwen35Decoder

    fx = _fixture()
    if device == "cuda":
        return fx["decoder"]
    key = f"decoder@{device}"
    if key not in _shared:
        hf = copy.deepcopy(fx["hf"]).to(device)
        _shared[key] = Qwen35Decoder.from_hf(hf)
    return _shared[key]


def _engine(cuda_graphs, num_slots: int = 4, chunk: int = 5, num_pages: int = 48, device: str = "cuda"):
    from wkvm.engine import Engine

    fx = _fixture()
    return Engine(
        _decoder_on(device), fx["layout"], num_slots=num_slots, device=device,
        scheduler_config=SchedulerConfig(max_tokens_per_step=64, max_running_requests=num_slots,
                                         max_tokens_per_request_per_step=chunk),
        prefill_chunk=chunk, num_pages=num_pages, cuda_graphs=cuda_graphs,
    )


# Length buckets cover prompt + new tokens of the fixture (max 31 + 15), so
# rows stay resident for whole runs; the fallback test exceeds 64 on purpose.
GRAPH_CFG = {"batch_buckets": (1, 2, 4), "length_buckets": (8, 16, 64), "warmup_iters": 1}


def _run_all(engine, reqs):
    while engine.has_unfinished:
        engine.step()
    return [list(r.output_token_ids) for r in reqs]


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
class TestHybridGraphs(unittest.TestCase):
    NEW = (12, 7, 15)  # staggered departures: rows leave mid-batch, compaction moves rows

    def test_graph_matches_eager_batched_and_alone(self) -> None:
        fx = _fixture()
        eager = _engine(False)
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=n) for p, n in zip(fx["prompts"], self.NEW)]
        for r in reqs:
            eager.add_request(r)
        ref = _run_all(eager, reqs)

        graphed = _engine(GRAPH_CFG)
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=n) for p, n in zip(fx["prompts"], self.NEW)]
        for r in reqs:
            graphed.add_request(r)
        out = _run_all(graphed, reqs)
        self.assertEqual(out, ref)
        g = graphed.runner.graphs
        self.assertGreater(g.stats["replays"], 0)
        self.assertGreaterEqual(g.stats["captures"], 2)  # 3 rows -> bucket 4; lengths cross 16 -> 16/64
        self.assertEqual(g.stats["acquires"], 3)
        self.assertEqual(g.stats["releases"], 3)  # every row left through finish, none through fallback
        self.assertEqual(g.stats["eager_fallbacks"], 0)
        self.assertGreater(g.stats["row_moves"], 0)
        self.assertEqual(g.n_resident, 0)
        self.assertEqual(g.row_of, {})
        # alone, B=1 bucket, slots reused
        for p, r, n in zip(fx["prompts"], ref, self.NEW):
            req = Request(prompt_token_ids=list(p), max_new_tokens=n)
            graphed.add_request(req)
            self.assertEqual(_run_all(graphed, [req])[0], r)
        self.assertEqual(g.n_resident, 0)

    def test_staggered_arrivals_match_alone(self) -> None:
        """A request joining a running graphed batch is acquired mid-flight;
        results equal the alone runs (mixed residency + prefill eviction)."""
        fx = _fixture()
        alone = []
        for p, n in zip(fx["prompts"], self.NEW):
            e = _engine(GRAPH_CFG)
            req = Request(prompt_token_ids=list(p), max_new_tokens=n)
            e.add_request(req)
            alone.append(_run_all(e, [req])[0])
        eng = _engine(GRAPH_CFG)
        reqs = []
        for i, (p, n) in enumerate(zip(fx["prompts"], self.NEW)):
            req = Request(prompt_token_ids=list(p), max_new_tokens=n)
            eng.add_request(req)
            reqs.append(req)
            for _ in range(3):  # let the batch decode a few steps before the next arrival
                eng.step()
        while eng.has_unfinished:
            eng.step()
        self.assertEqual([r.output_token_ids for r in reqs], alone)

    def test_hibernate_mid_decode_under_graphs(self) -> None:
        """Snapshot a resident request (flush row -> bank -> store), abort it
        (release), resume from the handle: identical to the uninterrupted run."""
        fx = _fixture()
        tmp = tempfile.mkdtemp()
        try:
            eng = _engine(GRAPH_CFG)
            eng.attach_store(tmp)
            prompt = fx["prompts"][2]
            full = Request(prompt_token_ids=list(prompt), max_new_tokens=16)
            eng.add_request(full)
            reference = _run_all(eng, [full])[0]

            req = Request(prompt_token_ids=list(prompt), max_new_tokens=16)
            eng.add_request(req)
            while len(req.output_token_ids) < 6:
                eng.step()
            self.assertTrue(eng.runner.graphs.is_resident(req.slots))
            handle = eng.hibernate(req.req_id, "mid")
            self.assertEqual(eng.runner.graphs.n_resident, 0)
            done = list(req.output_token_ids)
            rest = _run_all(eng, [eng.submit_from_handle(handle, max_new_tokens=16 - len(done))])[0]
            self.assertEqual(reference, done + rest)
            self.assertGreater(eng.runner.graphs.stats["flushes"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(HAS_CUDA and torch.cuda.device_count() >= 2, "needs a second GPU")
    def test_graphs_on_non_default_device(self) -> None:
        """Capture on cuda:1 must recompute on replay (a capture recorded on
        the default device's stream yields an empty graph whose replay is a
        silent no-op; the engine now guards against exactly that)."""
        fx = _fixture()
        eager = _engine(False, device="cuda:1")
        req = Request(prompt_token_ids=list(fx["prompts"][0]), max_new_tokens=10)
        eager.add_request(req)
        ref = _run_all(eager, [req])[0]
        graphed = _engine(GRAPH_CFG, device="cuda:1")
        req = Request(prompt_token_ids=list(fx["prompts"][0]), max_new_tokens=10)
        graphed.add_request(req)
        self.assertEqual(_run_all(graphed, [req])[0], ref)
        self.assertGreater(graphed.runner.graphs.stats["replays"], 0)

    def test_ring_mode_graphs_match_band_reference(self) -> None:
        """Ring-mode guests (sink 2 + ring 8) under graphs: one graph per
        batch bucket, context beyond any bucket, results equal the
        from-scratch band-mask reference and the eager ring engine."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        import test_qwen35_ring_cpu as R

        rfx = R._fixture()
        rfx["decoder"].to("cuda")
        try:
            prompts = [list(rfx["prompts"][2]) * 3, list(rfx["prompts"][0]), list(rfx["prompts"][1])]  # 93, 23, 7 tokens
            refs = [R.band_reference(p, 10, device="cuda") for p in prompts]
            eager = R._engine(chunk=5, cuda_graphs=False, device="cuda")
            reqs = [Request(prompt_token_ids=list(p), max_new_tokens=10) for p in prompts]
            for r in reqs:
                eager.add_request(r)
            self.assertEqual(_run_all(eager, reqs), refs)
            graphed = R._engine(chunk=5, cuda_graphs={"batch_buckets": (1, 2, 4), "warmup_iters": 1}, device="cuda")
            reqs = [Request(prompt_token_ids=list(p), max_new_tokens=10) for p in prompts]
            for r in reqs:
                graphed.add_request(r)
            self.assertEqual(_run_all(graphed, reqs), refs)
            g = graphed.runner.graphs
            self.assertEqual(g.stats["eager_fallbacks"], 0)
            self.assertEqual(g.length_buckets, (10,))
            self.assertGreater(g.stats["replays"], 0)
        finally:
            rfx["decoder"].to("cpu")

    NEAR_TIE = 0.05  # eager top-2 logit margin below which an argmax flip is numeric, not semantic

    def test_routed_mode_graphs_match_eager(self) -> None:
        """Routed span bank under graphs, in lockstep with the eager routed
        engine. After every step the guest store of each request (K/V,
        validity, positions, breaks, pending count) must be bit-identical
        between the eager bank row and the graph row — that covers in-graph
        eviction, out-graph routing on graph rows, acquisition mid-flight and
        row compaction — and the tokens must agree. The GDN recurrent state
        differs by ~1e-4 between an eager batch of 3 and a padded bucket of 4
        (kernel tiling order), which on this random tiny model can flip a
        near-tie argmax; such a flip is accepted only when the eager top-2
        margin is below NEAR_TIE, and the request is then excluded (its
        continuation legitimately differs)."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        import test_qwen35_routed_cpu as RT

        fx = RT._fixture()
        fx["decoder"].to("cuda")
        try:
            prompts = [list(fx["prompts"][2]), list(fx["prompts"][0]), list(fx["prompts"][2]) * 2]
            eager = RT._engine(cuda_graphs=False, device="cuda")
            graphed = RT._engine(cuda_graphs={"batch_buckets": (1, 2, 4), "warmup_iters": 1}, device="cuda")
            re_ = [Request(prompt_token_ids=list(p), max_new_tokens=14) for p in prompts]
            rg_ = [Request(prompt_token_ids=list(p), max_new_tokens=14) for p in prompts]
            for r in re_:
                eager.add_request(r)
            for r in rg_:
                graphed.add_request(r)
            margins: dict = {}
            orig = eager.runner.decode_step

            def recording_decode(slot_batch, last_tokens):
                logits = orig(slot_batch, last_tokens)
                top2 = logits.float().topk(2, dim=-1).values
                for s, m in zip(slot_batch, (top2[:, 0] - top2[:, 1]).tolist()):
                    margins[s["gdn_state"]] = m
                return logits

            eager.runner.decode_step = recording_decode
            live, flips, checked = set(range(len(prompts))), 0, 0
            while eager.has_unfinished or graphed.has_unfinished:
                sid_of = {r.req_id: r.slots["gdn_state"] for r in re_ if r.slots}
                eager.step()
                graphed.step()
                for i in sorted(live):
                    a, b = re_[i], rg_[i]
                    if a.output_token_ids != b.output_token_ids:
                        n = min(len(a.output_token_ids), len(b.output_token_ids))
                        self.assertEqual(a.output_token_ids[:n - 1], b.output_token_ids[:n - 1])
                        margin = margins.get(sid_of.get(a.req_id), float("inf"))
                        self.assertLess(margin, self.NEAR_TIE, f"request {i} flipped with eager margin {margin:.4f}")
                        flips += 1
                        live.discard(i)
                        continue
                    if a.slots and b.slots:
                        self._assert_same_guest(eager, graphed, a.slots, b.slots, i)
                        checked += 1
            self.assertGreaterEqual(len(live), 1, "every request hit a near-tie flip; nothing ran to completion in lockstep")
            self.assertGreater(checked, 20)
            g = graphed.runner.graphs
            self.assertGreater(g.stats["replays"], 0)
            self.assertGreater(g.stats["row_moves"], 0)
            self.assertGreater(graphed.bank.rg.stats["routing_passes"], 0)
        finally:
            fx["decoder"].to("cpu")

    def _assert_same_guest(self, eager, graphed, se: dict, sg: dict, i: int) -> None:
        ge, gg = se["guest_kv"], sg["guest_kv"]
        E = eager.bank.rstore
        gr = graphed.runner.graphs
        if gr.is_resident(sg):
            row, G = gr.row_of[sg["gdn_state"]], gr.rstore
        else:
            row, G = gg, graphed.bank.rstore
        for name, a, b in (("k", E.k[:, ge], G.k[:, row]), ("v", E.v[:, ge], G.v[:, row]),
                           ("valid", E.valid[:, ge], G.valid[:, row]), ("pos", E.pos[ge], G.pos[row]),
                           ("is_break", E.is_break[ge], G.is_break[row]), ("sal", E.sal[ge], G.sal[row]),
                           ("pend", E.pend[ge], G.pend[row])):
            self.assertTrue(torch.equal(a, b), f"request {i}: routed store '{name}' differs")

    def test_gqa_decode_attention_matches_sdpa(self) -> None:
        """The GQA-native single-query attention (ring/routed default on the
        9B) must reproduce HF's sdpa path: per-step logits close, in
        lockstep on the ring engine until any near-tie token flip."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        import test_qwen35_ring_cpu as R

        rfx = R._fixture()
        dec = rfx["decoder"]
        dec.to("cuda")
        try:
            prompt = list(rfx["prompts"][2]) * 2
            runs = {}
            for mode in ("sdpa", "gqa"):
                dec.set_decode_attention(mode == "gqa")
                self.assertEqual(dec.decode_attention, mode)
                eng = R._engine(chunk=5, cuda_graphs=False, device="cuda")
                req = Request(prompt_token_ids=list(prompt), max_new_tokens=12)
                eng.add_request(req)
                logits, orig = [], eng.runner.decode_step

                def rec(slot_batch, last_tokens, _orig=orig, _logits=logits):
                    out = _orig(slot_batch, last_tokens)
                    _logits.append(out[0].clone())
                    return out

                eng.runner.decode_step = rec
                _run_all(eng, [req])
                runs[mode] = (list(req.output_token_ids), logits)
            a_tok, a_log = runs["sdpa"]
            b_tok, b_log = runs["gqa"]
            same = 0
            for i, (x, y) in enumerate(zip(a_log, b_log)):
                self.assertTrue(torch.allclose(x, y, atol=1e-4, rtol=1e-4), f"decode step {i} logits differ")
                same += 1
                if a_tok[i + 1] != b_tok[i + 1]:
                    break
            self.assertGreaterEqual(same, 3)
        finally:
            dec.set_decode_attention(False)
            dec.to("cpu")

    def test_eager_fallback_beyond_buckets(self) -> None:
        """62 prompt tokens + 6 new crosses the 64 bucket mid-decode: the
        first steps replay, the rest run eagerly after the row is evicted."""
        fx = _fixture()
        eager = _engine(False, num_pages=64)
        req = Request(prompt_token_ids=list(fx["prompts"][2]) * 2, max_new_tokens=6)
        eager.add_request(req)
        ref = _run_all(eager, [req])[0]
        graphed = _engine(GRAPH_CFG, num_pages=64)
        req = Request(prompt_token_ids=list(fx["prompts"][2]) * 2, max_new_tokens=6)
        graphed.add_request(req)
        self.assertEqual(_run_all(graphed, [req])[0], ref)
        g = graphed.runner.graphs
        self.assertGreater(g.stats["replays"], 0)
        self.assertGreater(g.stats["eager_fallbacks"], 0)
        self.assertEqual(g.stats["evicts"], 1)
        self.assertEqual(g.n_resident, 0)

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
