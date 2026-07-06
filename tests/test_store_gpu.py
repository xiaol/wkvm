"""GPU tests for the M3 StateStore: durable/forkable/mutable state handles.

The load-bearing gate is EXACTNESS: a continuation resumed from a handle must
be token-identical to the same session never having been interrupted — across
every tier (HOT->WARM, WARM->COLD->load) and across a full engine rebuild
from the on-disk index alone.

Run: ``python -m unittest tests.test_store_gpu -v``
"""

from __future__ import annotations

import os
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

_WEIGHTS_ROOT = "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla"
WEIGHTS = os.environ.get("WKVM_RWKV7_PATH", f"{_WEIGHTS_ROOT}/rwkv7-191M-world")
PROMPT = "The city of Paris is known for its cafes, and one morning a writer"


def _has(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _config(num_slots: int = 8) -> SchedulerConfig:
    return SchedulerConfig(
        max_tokens_per_step=8192,
        max_running_requests=num_slots,
        max_tokens_per_request_per_step=512,
    )


_shared: dict = {}


def _engine():
    if not _shared:
        from transformers import AutoTokenizer

        from wkvm.engine import Engine

        _shared["engine"] = Engine.from_pretrained(
            WEIGHTS, num_slots=8, scheduler_config=_config()
        )
        base = "/run/media/xiaol/B214449214445C0B/wkvm_bench/statestore"
        parent = None
        if os.path.isdir(os.path.dirname(base)):
            os.makedirs(base, exist_ok=True)
            parent = base
        _shared["store_dir"] = tempfile.mkdtemp(dir=parent)
        _shared["engine"].attach_store(_shared["store_dir"])
        tok = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        _shared["prompt_ids"] = tok(PROMPT)["input_ids"]
    return _shared["engine"], _shared["prompt_ids"]


def _run_to_finish(engine, req):
    while not req.status.is_finished:
        engine.step()
    return list(req.output_token_ids)


def _generate(engine, prompt_ids, max_new):
    req = Request(prompt_token_ids=list(prompt_ids), max_new_tokens=max_new)
    engine.add_request(req)
    return req, _run_to_finish(engine, req)


@unittest.skipUnless(HAS_CUDA, "CUDA unavailable")
@unittest.skipUnless(_has(WEIGHTS), f"weights missing at {WEIGHTS}")
class TestStateStore(unittest.TestCase):
    def test_1_save_load_exactness_across_tiers(self) -> None:
        engine, prompt = _engine()
        # Reference: one uninterrupted 48-token greedy generation.
        _, reference = _generate(engine, prompt, 48)
        # Interrupted twin: generate 16, snapshot-on-finish, then resume 32
        # from WARM, and again from COLD after evicting.
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=16)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, "twin")
        _run_to_finish(engine, req)
        handle = engine._finish_handles[req.req_id]

        resumed = engine.submit_from_handle(handle, max_new_tokens=32)
        warm_out = _run_to_finish(engine, resumed)
        self.assertEqual(reference, req.output_token_ids + warm_out)

        engine.store.evict(handle)
        self.assertNotIn(handle, engine.store._warm)
        resumed2 = engine.submit_from_handle(handle, max_new_tokens=32)
        cold_out = _run_to_finish(engine, resumed2)
        self.assertEqual(warm_out, cold_out)

    def test_2_fork_isolation(self) -> None:
        engine, prompt = _engine()
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, "parent")
        _run_to_finish(engine, req)
        parent = engine._finish_handles[req.req_id]

        before = _run_to_finish(
            engine, engine.submit_from_handle(parent, max_new_tokens=16)
        )
        child = engine.store.fork(parent, "child")
        child_out = _run_to_finish(
            engine, engine.submit_from_handle(child, max_new_tokens=16)
        )
        self.assertEqual(before, child_out)  # fork starts identical (greedy)
        after = _run_to_finish(
            engine, engine.submit_from_handle(parent, max_new_tokens=16)
        )
        self.assertEqual(before, after)  # parent unperturbed by the fork

    def test_3_mutate_provenance_and_effect(self) -> None:
        engine, prompt = _engine()
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, "mut")
        _run_to_finish(engine, req)
        parent = engine._finish_handles[req.req_id]

        base = _run_to_finish(
            engine, engine.submit_from_handle(parent, max_new_tokens=24)
        )
        mutated = engine.store.mutate(parent, "decay", {"alpha": 0.2})
        record = engine.store.get(mutated)
        self.assertEqual(record.parent, parent)
        self.assertEqual(record.rule, "decay")
        mut_out = _run_to_finish(
            engine, engine.submit_from_handle(mutated, max_new_tokens=24)
        )
        self.assertNotEqual(base, mut_out)  # a strong decay must change decoding
        again = _run_to_finish(
            engine, engine.submit_from_handle(parent, max_new_tokens=24)
        )
        self.assertEqual(base, again)  # parent version untouched

    def test_4_rebuild_from_cold_index(self) -> None:
        engine, prompt = _engine()
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, "durable")
        _run_to_finish(engine, req)
        handle = engine._finish_handles[req.req_id]
        reference = _run_to_finish(
            engine, engine.submit_from_handle(handle, max_new_tokens=24)
        )
        engine.store.persist(handle)

        from wkvm.store import StateStore

        # Fresh store over the same directory: only index + safetensors.
        engine.store = StateStore(engine.bank, _shared["store_dir"])
        engine.scheduler.on_finish = engine._snapshot_on_finish
        self.assertIn(handle, {r.handle for r in engine.store.list()})
        rebuilt = _run_to_finish(
            engine, engine.submit_from_handle(handle, max_new_tokens=24)
        )
        self.assertEqual(reference, rebuilt)

    def test_5_fingerprint_mismatch_rejected(self) -> None:
        engine, prompt = _engine()
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=8)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, "fp")
        _run_to_finish(engine, req)
        handle = engine._finish_handles[req.req_id]

        from wkvm.store import FingerprintMismatch

        object.__setattr__(engine.store.get(handle), "fingerprint", "bogus")
        with self.assertRaises(FingerprintMismatch):
            engine.submit_from_handle(handle, max_new_tokens=8)
        # The failed load must not leak arena slots.
        self.assertEqual(engine.arena.num_free_slots(), engine.arena.num_slots)

    @classmethod
    def tearDownClass(cls) -> None:
        if "store_dir" in _shared:
            shutil.rmtree(_shared["store_dir"], ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
