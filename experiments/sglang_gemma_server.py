#!/usr/bin/env python
"""Launch SGLang's OpenAI server for the text-only Gemma4 benchmark path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from incumbent_gemma_bench import sglang_language_model_override


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="gemma-4-E4B-it")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=15_232)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--cuda-graph-backend-decode", default="full")
    parser.add_argument("--cuda-graph-backend-prefill", default="disabled")
    return parser


def server_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_path": args.model_path,
        "host": args.host,
        "port": args.port,
        "served_model_name": args.served_model_name,
        "dtype": args.dtype,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "attention_backend": args.attention_backend,
        "cuda_graph_backend_decode": args.cuda_graph_backend_decode,
        "cuda_graph_backend_prefill": args.cuda_graph_backend_prefill,
        "json_model_override_args": json.dumps(
            sglang_language_model_override(args.model_path)
        ),
        "enable_cache_report": True,
        "sampling_defaults": "openai",
        "enable_multimodal": False,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    launch_server(ServerArgs(**server_kwargs(args)))


if __name__ == "__main__":
    main()
