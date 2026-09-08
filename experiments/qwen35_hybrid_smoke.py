"""M4 smoke: Qwen3.5-9B hybrid (Gated DeltaNet + full attention) on wkvm.

What it certifies, on the real checkpoint, one A100 for the engine and one
for an independent HF reference:

  A. parity        greedy continuations vs HF ``generate`` on chat prompts;
                   prefill last-logit max |diff| vs an HF full forward.
  B. batching      N distinct scene prompts continuously batched vs each
                   alone: outputs identical (bf16 noise is reported, not
                   hidden); decode step time and tok/s at the batch size.
  C. tuned state   an RNN-StateTuning adapter imported as a state handle
                   (``num_computed_tokens = 0``): generation from the handle
                   vs the adapter's own HF runtime, plus valid-JSON / exact
                   match on scene-boundary rows for zero vs tuned state.
  D. footprint     bytes per slot, weights, peak memory, slot capacity.

Usage:
  python experiments/qwen35_hybrid_smoke.py --json experiments/results/m4_qwen35_hybrid_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import torch

SCENES = (
    "/root/x/.cache/huggingface/datasets--mikuhhn1239--novel-agent-sft-dataset/snapshots/"
    "5d3040d21f51b3ce90b9396b058e552c47f43cd5/training/v4-scene-boundary-detection/test.jsonl"
)
CHAT_PROMPTS = [
    "Explain in two sentences why recurrent state is cheaper to serve than a KV cache.",
    "Write a haiku about a hypervisor for model state.",
    "List three cities on the Silk Road and one fact about each.",
]


def render(tok, messages) -> list[int]:
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "ids"):
        ids = ids.ids
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def boundaries(text: str):
    """Same parse as RNN-StateTuning's benchmark: a JSON object with a
    ``boundaries`` list of ints, else invalid."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    b = obj.get("boundaries") if isinstance(obj, dict) else None
    if not isinstance(b, list) or not all(isinstance(x, int) for x in b):
        return None
    return sorted(set(b))


def hf_greedy(model, prompt: list[int], n: int, eos: int, pad: int, device) -> list[int]:
    ids = torch.tensor([prompt], device=device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
            do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad,
        )
    return out[0, len(prompt):].tolist()


def run_all(engine, reqs):
    while engine.has_unfinished:
        engine.step()
    return [list(r.output_token_ids) for r in reqs]


def prefix_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/x/Qwen3.5-9B")
    ap.add_argument("--adapter", default="/root/x/RNN-StateTuning/outputs/qwen35-state-novel-scene-e1")
    ap.add_argument("--rst-src", default="/root/x/RNN-StateTuning/src")
    ap.add_argument("--scenes", default=SCENES)
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--engine-device", default="cuda:0")
    ap.add_argument("--ref-device", default="cuda:1")
    ap.add_argument("--guest-pool-tokens", type=int, default=None,
                    help="total guest-KV pool shared by all requests (default 4096 x slots)")
    ap.add_argument("--page-tokens", type=int, default=256)
    ap.add_argument("--long-prompt-tokens", type=int, default=12288,
                    help="phase E: long-context parity prompt length (0 disables)")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--scene-max-new", type=int, default=48)
    ap.add_argument("--json", default="experiments/results/m4_qwen35_hybrid_smoke.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from wkvm.core.config import SchedulerConfig
    from wkvm.core.request import Request
    from wkvm.engine import Engine

    tok = AutoTokenizer.from_pretrained(args.model)
    eos, pad = tok.eos_token_id, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    result: dict = {
        "engine": "wkvm-m4-hybrid",
        "model": args.model,
        "dtype": "bfloat16",
        "page_tokens": args.page_tokens,
        "slots": args.slots,
        "prefill_chunk": args.prefill_chunk,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "kernels": "transformers pure-torch GDN fallback (no fla/triton on host), SDPA guest attention",
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
    }

    # -- engine -------------------------------------------------------------------
    t0 = time.time()
    engine = Engine.from_qwen35(
        args.model, num_slots=args.slots, guest_pool_tokens=args.guest_pool_tokens,
        page_tokens=args.page_tokens, device=args.engine_device,
        stop_token_ids=frozenset({eos}), prefill_chunk=args.prefill_chunk,
        scheduler_config=SchedulerConfig(
            max_tokens_per_step=8192, max_running_requests=args.slots,
            max_tokens_per_request_per_step=args.prefill_chunk,
        ),
    )
    load_s = time.time() - t0
    layout = engine.layout
    dev = torch.device(args.engine_device)
    weights = sum(p.numel() * p.element_size() for p in engine.runner.model._hf_model.parameters())
    spec = {f.name: f.bytes_per_slot for f in layout.state_spec().families}
    result["guest_pool_tokens"] = engine.arena.max_tokens_per_request
    result["footprint"] = {
        "load_s": load_s,
        "weights_bytes": weights,
        "bytes_per_family": spec,  # guest_kv is per PAGE
        "recurrent_bytes_per_slot": layout.bytes_per_slot,
        "bytes_per_page": layout.bytes_per_page,
        "guest_bytes_per_token": layout.guest_bytes_per_token,
        "num_pages": engine.arena.num_pages,
        "bank_bytes": engine.bank.state_bytes(),
        "layers": {"gdn": layout.n_gdn, "attn": layout.n_attn},
        "after_load_allocated_gib": torch.cuda.memory_allocated(dev) / 2**30,
    }
    total = torch.cuda.get_device_properties(dev).total_memory
    for ctx in (4096, 16384, 65536):
        per_session = layout.bytes_per_slot + ctx * layout.guest_bytes_per_token
        result["footprint"][f"sessions_at_ctx{ctx}_on_this_gpu"] = int((total - weights - 2 * 2**30) // per_session)
    print("loaded engine", json.dumps(result["footprint"], indent=1), flush=True)

    # -- reference model (HF, other GPU) ------------------------------------------
    sys.path.insert(0, args.rst_src)
    from rnn_state_tuning import load_adapter, prepare_model_for_state_tuning
    from rnn_state_tuning.modeling import load_qwen35_model

    ref = load_qwen35_model(args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    ref = ref.to(args.ref_device).eval().requires_grad_(False)
    rdev = torch.device(args.ref_device)

    # -- A. parity on chat prompts ------------------------------------------------
    parity = []
    for text in CHAT_PROMPTS:
        prompt = render(tok, [{"role": "user", "content": text}])
        with torch.inference_mode():
            ref_logits = ref(input_ids=torch.tensor([prompt], device=rdev), use_cache=False,
                             logits_to_keep=1).logits[0, -1].float().cpu()
        slots = engine.arena.allocate(pages=engine.arena.pages_for(len(prompt)))
        engine.bank.zero_slots(slots)
        ours_logits = engine.runner.prefill(prompt, slots).cpu()
        engine.arena.free(slots)
        ref_out = hf_greedy(ref, prompt, args.max_new, eos, pad, rdev)
        req = Request(prompt_token_ids=prompt, max_new_tokens=args.max_new)
        engine.add_request(req)
        ours = run_all(engine, [req])[0]
        parity.append({
            "prompt_tokens": len(prompt),
            "logit_max_abs_diff": (ours_logits - ref_logits).abs().max().item(),
            "logit_argmax_equal": int(ours_logits.argmax()) == int(ref_logits.argmax()),
            "greedy_equal": ours == ref_out,
            "greedy_prefix_match": prefix_match(ours, ref_out),
            "generated": len(ours),
            "text": tok.decode(ours, skip_special_tokens=True),
            "ref_text": tok.decode(ref_out, skip_special_tokens=True),
        })
        print("parity", json.dumps(parity[-1], ensure_ascii=False), flush=True)
    result["parity"] = parity

    # -- B. continuous batching on scene prompts ----------------------------------
    rows = [json.loads(l) for l in open(args.scenes)][: args.n_scenes]
    prompts = [render(tok, r["messages"][:-1]) for r in rows]
    targets = [boundaries(r["messages"][-1]["content"]) for r in rows]
    alone = []
    t_alone = time.time()
    for p in prompts:
        req = Request(prompt_token_ids=p, max_new_tokens=args.scene_max_new)
        engine.add_request(req)
        alone.append(run_all(engine, [req])[0])
    t_alone = time.time() - t_alone
    torch.cuda.synchronize(dev)
    reqs = [Request(prompt_token_ids=p, max_new_tokens=args.scene_max_new) for p in prompts]
    for r in reqs:
        engine.add_request(r)
    def pure_decode_step() -> bool:
        """Every running row has a 1-token gap and nothing is waiting: the
        step is one batched decode, no prefill chunk inside it."""
        sched = engine.scheduler
        return not sched.waiting and all(r.num_scheduled_gap == 1 for r in sched.running)

    step_times = []  # (seconds, rows) for pure-decode steps only
    t_batch = time.time()
    while engine.has_unfinished:
        pure = pure_decode_step()
        rows = len(engine.scheduler.running)
        t = time.time()
        engine.step()
        torch.cuda.synchronize(dev)
        if pure:
            step_times.append((time.time() - t, rows))
    t_batch = time.time() - t_batch
    batched = [list(r.output_token_ids) for r in reqs]
    result["batching"] = {
        "n": len(prompts),
        "prompt_tokens": [len(p) for p in prompts],
        "identical_to_alone": [a == b for a, b in zip(alone, batched)],
        "prefix_match": [prefix_match(a, b) for a, b in zip(alone, batched)],
        "generated": [len(b) for b in batched],
        "wall_alone_s": t_alone,
        "wall_batched_s": t_batch,
        "pure_decode_steps": len(step_times),
        "pure_decode_tok_s": (sum(b for _, b in step_times) / sum(dt for dt, _ in step_times)) if step_times else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
    }
    print("batching", json.dumps(result["batching"], indent=1), flush=True)

    # -- B2. pure-decode ladder (same ~300-token prompt replicated) ----------------
    ladder = []
    base = prompts[1]
    for b in (1, 4, 8, 16):
        if b > args.slots:
            break
        reqs_b = [Request(prompt_token_ids=list(base), max_new_tokens=24) for _ in range(b)]
        for r in reqs_b:
            engine.add_request(r)
        while not pure_decode_step():  # prefill everything first
            engine.step()
        torch.cuda.synchronize(dev)
        times = []
        for _ in range(8):
            assert pure_decode_step()
            t = time.time()
            engine.step()
            torch.cuda.synchronize(dev)
            times.append(time.time() - t)
        for r in reqs_b:
            engine.abort_request(r.req_id)
        ms = sum(times) / len(times) * 1000
        ladder.append({"B": b, "decode_step_ms": ms, "decode_tok_s": b / (ms / 1000),
                       "resident_tokens_per_row": len(base) + 24})
        print("ladder", ladder[-1], flush=True)
    result["decode_ladder"] = ladder

    # -- C. tuned initial state as a durable handle -------------------------------
    store_dir = Path("/tmp/wkvm_m4_store")
    engine.attach_store(store_dir)
    t0 = time.time()
    handle = engine.import_state("scene-ce", args.adapter)
    import_s = time.time() - t0
    record = engine.store.get(handle)
    zero_outs = alone  # zero-state generations from step B (same prompts, same max_new)
    tuned_reqs = [engine.submit_from_handle(handle, suffix_tokens=p, max_new_tokens=args.scene_max_new)
                  for p in prompts]
    tuned_outs = run_all(engine, tuned_reqs)
    # HF reference for the tuned state: the adapter's own runtime.
    prepare_model_for_state_tuning(ref)
    load_adapter(ref, args.adapter)
    ref_tuned = [hf_greedy(ref, p, args.scene_max_new, eos, pad, rdev) for p in prompts]

    def score(outs):
        preds = [boundaries(tok.decode(o, skip_special_tokens=True)) for o in outs]
        valid = [p is not None for p in preds]
        exact = [p is not None and p == t for p, t in zip(preds, targets)]
        return {"valid_json": sum(valid), "exact_match": sum(exact), "n": len(outs),
                "texts": [tok.decode(o, skip_special_tokens=True)[:120] for o in outs]}

    result["tuned_state"] = {
        "adapter": args.adapter,
        "handle": handle,
        "record": {"rule": record.rule, "num_computed_tokens": record.num_computed_tokens,
                   "token_ids": len(record.token_ids), "sha256": record.rule_params.get("sha256"),
                   "tuned_layers": record.rule_params.get("tuned_layers")},
        "import_s": import_s,
        "zero_state": score(zero_outs),
        "tuned_state": score(tuned_outs),
        "tuned_vs_adapter_runtime_equal": [a == b for a, b in zip(tuned_outs, ref_tuned)],
        "tuned_vs_adapter_runtime_prefix": [prefix_match(a, b) for a, b in zip(tuned_outs, ref_tuned)],
        "adapter_runtime": score(ref_tuned),
        "targets": targets,
    }
    # Durable: persist, evict, resume from COLD, must be identical.
    engine.store.persist(handle)
    engine.store.evict(handle)
    cold = run_all(engine, [engine.submit_from_handle(handle, suffix_tokens=prompts[0],
                                                      max_new_tokens=args.scene_max_new)])[0]
    result["tuned_state"]["cold_resume_identical"] = cold == tuned_outs[0]
    result["tuned_state"]["cold_file_bytes"] = os.path.getsize(engine.store._cold_path(handle))
    print("tuned", json.dumps({k: v for k, v in result["tuned_state"].items() if k != "targets"},
                              ensure_ascii=False, indent=1), flush=True)

    # -- E. long-context parity through the paged pool -----------------------------
    if args.long_prompt_tokens:
        # Natural text: concatenate scene paragraphs until the rendered chat
        # prompt reaches the target length (the fixed 4k window of slice 1
        # could not hold this; the paged pool reserves ceil(n / page) pages).
        all_rows = [json.loads(l) for l in open(args.scenes)]
        body, i = "", 0
        while True:
            body += all_rows[i % len(all_rows)]["messages"][1]["content"] + "\n\n"
            i += 1
            probe = render(tok, [{"role": "user", "content": body + "\n用一句话总结以上内容。"}])
            if len(probe) >= args.long_prompt_tokens:
                break
        long_prompt = probe
        need = engine.arena.pages_for(len(long_prompt) + 16)
        if need > engine.arena.num_pages:
            result["long_context"] = {"skipped": f"needs {need} pages > pool {engine.arena.num_pages}"}
        else:
            t0 = time.time()
            req = Request(prompt_token_ids=long_prompt, max_new_tokens=16)
            engine.add_request(req)
            ours = run_all(engine, [req])[0]
            t_ours = time.time() - t0
            torch.cuda.synchronize(rdev)
            t0 = time.time()
            # Reference is the *base* model again (adapter states zeroed).
            for m in ref.modules():
                if hasattr(m, "initial_state") and isinstance(getattr(m, "initial_state"), torch.nn.Parameter):
                    m.initial_state.data.zero_()
            ref_out = hf_greedy(ref, long_prompt, 16, eos, pad, rdev)
            t_ref = time.time() - t0
            result["long_context"] = {
                "prompt_tokens": len(long_prompt),
                "pages_reserved": need,
                "page_tokens": args.page_tokens,
                "greedy_equal": ours == ref_out,
                "greedy_prefix_match": prefix_match(ours, ref_out),
                "generated": len(ours),
                "text": tok.decode(ours, skip_special_tokens=True),
                "wall_engine_s": t_ours,
                "wall_reference_s": t_ref,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
            }
        print("long_context", json.dumps(result["long_context"], ensure_ascii=False, indent=1), flush=True)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("M4_SMOKE_OK", args.json)


if __name__ == "__main__":
    main()
