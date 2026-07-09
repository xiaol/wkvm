#!/usr/bin/env python
"""HTTP smoke for the native Gemma token-id endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_engine_smoke import build_prompt, chunked_scheduler_config
from native_gemma_smoke import break_mask_for, load_model, resolve_model_path

from wkvm.gemma_engine import GemmaNativeEngine
from wkvm.gemma_server import BoundedGemmaService, engine_kwargs_from_args, serve
from wkvm.models.gemma import gemma4_e4b_routed_span_config


def post_json(url: str, payload: dict, timeout: float = 120.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def get_json(url: str, timeout: float = 30.0) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


def make_engine(args):
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    model = load_model(path, args.device, args.attn)
    tok = AutoTokenizer.from_pretrained(path)
    cfg = gemma4_e4b_routed_span_config(
        num_hidden_layers=model.config.num_hidden_layers,
        num_kv_shared_layers=getattr(model.config, "num_kv_shared_layers", 0),
        layer_types=tuple(model.config.layer_types),
        num_kv_heads=getattr(model.config, "num_global_key_value_heads", None)
        or getattr(model.config, "num_key_value_heads", 2),
        head_dim=getattr(model.config, "global_head_dim", None)
        or getattr(model.config, "head_dim", 512),
        sink_tokens=args.sink,
        ring_tokens=args.window,
        routed_slots=args.m_slots,
        pending_tokens=args.route_chunk,
        sliding_window=getattr(model.config, "sliding_window", 1024),
    )
    lengths = [args.ctx - i * 5 for i in range(args.concurrency)]
    prompts = [build_prompt(tok, n, i) for i, n in enumerate(lengths)]
    sched_cfg = chunked_scheduler_config(
        prompts,
        slots=args.slots,
        token_budget=None,
        chunk=args.chunk,
    )
    engine = GemmaNativeEngine(
        model,
        cfg,
        num_slots=args.slots,
        scheduler_config=sched_cfg,
        prefill_chunk=args.chunk,
        **engine_kwargs_from_args(args),
    )
    payloads = [
        {
            "req_id": f"server-{i}",
            "prompt_ids": prompt,
            "max_tokens": args.out,
            "break_mask": break_mask_for(tok, prompt),
        }
        for i, prompt in enumerate(prompts)
    ]
    return engine, payloads


def run(args) -> None:
    import torch

    engine, payloads = make_engine(args)
    service = BoundedGemmaService(
        engine,
        max_queue=args.max_queue,
        batch_wait_s=args.batch_wait_s,
        max_completed_requests=args.max_completed_requests,
    )
    server = serve(service, port=0)
    host, port = server.server_address
    base = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code, health = get_json(f"{base}/health")
        if code != 200 or not health.get("ok"):
            raise SystemExit(f"health failed: {code} {health}")

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(post_json, f"{base}/v1/submit", payload)
                for payload in payloads
            ]
            submits = [fut.result() for fut in futures]

        submit_errors = [
            (code, body)
            for code, body in submits
            if code != 202 or body.get("req_id") is None
        ]
        if submit_errors:
            raise SystemExit(f"submit failures: {submit_errors[:2]}")

        deadline = time.perf_counter() + args.timeout_s
        by_id = {body["req_id"]: None for _, body in submits}
        while any(body is None or not body.get("finished") for body in by_id.values()):
            if time.perf_counter() > deadline:
                raise SystemExit(f"status polling timed out: {by_id}")
            for req_id, body in list(by_id.items()):
                if body is not None and body.get("finished"):
                    continue
                code, status = get_json(f"{base}/v1/status/{req_id}", timeout=30.0)
                if code != 200:
                    raise SystemExit(f"status failed for {req_id}: {code} {status}")
                by_id[req_id] = status
            time.sleep(args.poll_s)
        elapsed = time.perf_counter() - start

        errors = []
        results = [by_id[body["req_id"]] for _, body in submits]
        for body in results:
            if body is None or len(body.get("tokens", [])) != args.out:
                errors.append(body)

        cancel_code, cancel_body = post_json(
            f"{base}/v1/cancel", {"req_id": "server-cancel-noop"}
        )
        metrics_code, metrics = get_json(f"{base}/metrics")
        if metrics_code != 200:
            raise SystemExit("metrics endpoint failed")
        if cancel_code != 200 or "cancelled" not in cancel_body:
            raise SystemExit("cancel endpoint failed")

        engine_metrics = metrics["engine"]
        print(json.dumps(metrics, sort_keys=True))
        print(f"success_count={len(results) - len(errors)}")
        print(f"error_count={len(errors)}")
        print(f"elapsed_s={elapsed:.3f}")
        print(f"queue_depth={engine_metrics['queue_depth']}")
        print(f"runnable_rows={engine_metrics['runnable_rows']}")
        print(f"resident_state_slots={engine_metrics['resident_state_slots']}")
        print(f"max_decode_batch_rows={engine_metrics['max_decode_batch_rows']}")
        print(f"model_forward_backend={engine_metrics.get('model_forward_backend')}")
        print(
            "uses_hf_transformer_forward="
            f"{engine_metrics.get('uses_hf_transformer_forward')}"
        )
        print(f"first_finish_reason={results[0].get('finish_reason') if results else None}")
        if errors:
            raise SystemExit(f"server smoke failures: {errors[:2]}")
        if (
            args.use_native_gemma_forward
            and engine_metrics.get("uses_hf_transformer_forward") is not False
        ):
            raise SystemExit(
                "native-forward server smoke did not disable HF transformer forward"
            )
        expected_batch_rows = min(args.concurrency, args.slots)
        if engine_metrics["max_decode_batch_rows"] < expected_batch_rows:
            raise SystemExit(
                "server did not batch concurrent submitted requests: "
                f"max_decode_batch_rows={engine_metrics['max_decode_batch_rows']} "
                f"expected>={expected_batch_rows}"
            )
        print("SERVER_SMOKE_OK")
    finally:
        service.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--decode-microbatch-rows", type=int, default=16)
    ap.add_argument("--decode-microbatch-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-width-bucket", type=int, default=16)
    ap.add_argument("--disable-persistent-exact-decode", action="store_true")
    ap.add_argument("--disable-persistent-padded-decode", action="store_true")
    ap.add_argument("--persistent-padded-decode-steps", type=int, default=8)
    ap.add_argument("--persistent-padded-decode-cuda-graph", action="store_true")
    ap.add_argument("--persistent-padded-decode-graph-warmup-iters", type=int, default=3)
    ap.add_argument(
        "--use-native-gemma-forward",
        action="store_true",
        help=(
            "Run model calls through wkvm's NativeGemma4ForCausalLM bridge instead "
            "of transformers.Gemma4ForCausalLM.forward. Still uses loaded HF weights."
        ),
    )
    ap.add_argument(
        "--native-gemma-attention-backend",
        choices=["manual", "sdpa"],
        default="manual",
        help="Attention primitive used inside --use-native-gemma-forward.",
    )
    ap.add_argument(
        "--native-gemma-projection-backend",
        choices=["separate", "qkv_packed", "gate_up_packed", "qkv_gate_up_packed"],
        default="separate",
        help="Projection primitive used inside --use-native-gemma-forward.",
    )
    ap.add_argument(
        "--native-gemma-weight-backend",
        choices=["hf_live", "owned", "owned_cpu"],
        default="hf_live",
        help=(
            "Weight source used inside --use-native-gemma-forward. 'owned' copies "
            "decoder-layer weights into native tensors at bridge construction; "
            "'owned_cpu' keeps those snapshots on CPU and stages per operation."
        ),
    )
    ap.add_argument(
        "--native-gemma-release-hf-decoder-layers",
        action="store_true",
        help=(
            "After constructing the native owned-weight bridge, replace HF decoder "
            "layers with empty modules so serving does not keep duplicate decoder "
            "weights resident. Requires --native-gemma-weight-backend owned or "
            "owned_cpu."
        ),
    )
    ap.add_argument(
        "--decode-batch-planner",
        choices=["scheduler", "length_bucketed"],
        default="scheduler",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=256)
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--batch-wait-s", type=float, default=0.5)
    ap.add_argument("--poll-s", type=float, default=0.02)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--max-completed-requests", type=int, default=4096)
    args = ap.parse_args()
    if args.slots is None:
        args.slots = args.concurrency
    run(args)


if __name__ == "__main__":
    main()
