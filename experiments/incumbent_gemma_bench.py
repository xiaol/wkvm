#!/usr/bin/env python
"""Same-shape vLLM/SGLang Gemma throughput benchmark.

This is the incumbent-engine counterpart to ``native_gemma_bench.py`` and
``hf_gemma_bench.py``. It intentionally reuses the same prompt builder, output
length, concurrency ladder, and JSON row shape where the incumbent APIs expose
the same facts.

Run this script from the target engine's environment, for example:

  /run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/vllm/bin/python \
      experiments/incumbent_gemma_bench.py --engine vllm --ctx 13824 \
      --out 128 --concurrency 1,2,4,8 \
      --json experiments/results/vllm_gemma_ctx13824_out128_ladder.json

  /run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/sglang/bin/python \
      experiments/incumbent_gemma_bench.py --engine sglang --ctx 13824 \
      --out 128 --concurrency 1,2,4,8 \
      --json experiments/results/sglang_gemma_ctx13824_out128_ladder.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_engine_smoke import build_prompt, prompt_lengths
from native_gemma_smoke import resolve_model_path
from bench_prompt_utils import (
    GENERATED_OUTPUT_FINGERPRINT_SCHEMA,
    SyntheticBenchTokenizer,
    generated_output_fingerprint,
    generated_output_fingerprint_row_fields,
    prompt_fingerprint_row_fields,
    prompt_set_fingerprint,
)
from benchmark_identity import model_checkpoint_identity, source_worktree_identity


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_concurrency(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val < 1:
            raise argparse.ArgumentTypeError("concurrency values must be >= 1")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("--concurrency must contain at least one value")
    return vals


def parse_positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def bench_prompt_lengths(ctx: int, concurrency: int, mode: str) -> list[int]:
    if mode == "staggered":
        return prompt_lengths(ctx, concurrency)
    if mode == "uniform":
        return [ctx] * concurrency
    raise ValueError(f"unknown prompt length mode: {mode}")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def round_or_none(x: float | None, ndigits: int = 3) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(x, ndigits)


RESIDENCY_TELEMETRY_SCHEMA = "wkvm.incumbent_residency_telemetry.v1"
PROVENANCE_SCHEMA = "wkvm.incumbent_gemma_bench.provenance.v1"
PROVENANCE_PACKAGES = ("wkvm", "torch", "transformers", "vllm", "sglang", "triton")


class TelemetryUnavailable(RuntimeError):
    pass


def telemetry_number(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        return None
    return float(value)


def telemetry_value(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def telemetry_items(value: Any, *container_names: str) -> list[Any]:
    for name in container_names:
        nested = telemetry_value(value, name)
        if nested is not None:
            return telemetry_items(nested)
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return [value]


def metric_name_values(snapshot: Any) -> list[tuple[str, float]]:
    if isinstance(snapshot, Mapping) and telemetry_value(snapshot, "name") is None:
        direct = []
        for name, value in snapshot.items():
            number = telemetry_number(value)
            if isinstance(name, str) and number is not None:
                direct.append((name, number))
        if direct:
            return direct
    points: list[tuple[str, float]] = []
    for metric in telemetry_items(snapshot, "metrics", "data"):
        name = telemetry_value(metric, "name")
        value = telemetry_number(telemetry_value(metric, "value"))
        if isinstance(name, str) and value is not None:
            points.append((name, value))
            continue
        for sample in telemetry_items(telemetry_value(metric, "samples")):
            sample_name = telemetry_value(sample, "name")
            sample_value = telemetry_number(telemetry_value(sample, "value"))
            if isinstance(sample_name, str) and sample_value is not None:
                points.append((sample_name, sample_value))
    return points


def normalized_metric_name(name: str) -> str:
    for suffix in ("_total", "_created"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def summed_metric(
    points: list[tuple[str, float]], aliases: set[str]
) -> float | None:
    values = [
        value
        for name, value in points
        if normalized_metric_name(name) in aliases
    ]
    return sum(values) if values else None


def vllm_capacity_telemetry(
    llm: Any,
    *,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    llm_engine = getattr(llm, "llm_engine", llm)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    candidates = [
        ("llm.llm_engine.vllm_config.cache_config", getattr(vllm_config, "cache_config", None)),
        ("llm.llm_engine.cache_config", getattr(llm_engine, "cache_config", None)),
        ("llm.cache_config", getattr(llm, "cache_config", None)),
    ]
    for source, cache_config in candidates:
        if cache_config is None:
            continue
        capacity = telemetry_number(
            telemetry_value(cache_config, "kv_cache_size_tokens")
        )
        max_concurrency = telemetry_number(
            telemetry_value(cache_config, "kv_cache_max_concurrency")
        )
        estimated = False
        if capacity is None:
            num_blocks = telemetry_number(
                telemetry_value(cache_config, "num_gpu_blocks")
            )
            block_size = telemetry_number(telemetry_value(cache_config, "block_size"))
            if num_blocks is not None and block_size is not None:
                capacity = num_blocks * block_size
                estimated = True
        model_len = max_model_len
        if model_len is None:
            model_config = getattr(vllm_config, "model_config", None)
            model_len_value = telemetry_number(
                telemetry_value(model_config, "max_model_len")
            )
            model_len = None if model_len_value is None else int(model_len_value)
        if max_concurrency is None and capacity is not None and model_len:
            max_concurrency = capacity / model_len
            estimated = True
        if capacity is not None or max_concurrency is not None:
            return {
                "kv_token_capacity": (
                    None if capacity is None else int(capacity)
                ),
                "kv_max_concurrency": max_concurrency,
                "capacity_source": source,
                "capacity_estimated": estimated,
            }
    return {
        "kv_token_capacity": None,
        "kv_max_concurrency": None,
        "capacity_source": None,
        "capacity_estimated": None,
    }


def vllm_runtime_telemetry_sample(llm: Any) -> dict[str, Any]:
    get_metrics = getattr(llm, "get_metrics", None)
    source = "llm.get_metrics"
    if not callable(get_metrics):
        llm_engine = getattr(llm, "llm_engine", None)
        get_metrics = getattr(llm_engine, "get_metrics", None)
        source = "llm.llm_engine.get_metrics"
    if not callable(get_metrics):
        raise TelemetryUnavailable("vLLM metrics snapshot API is unavailable")
    points = metric_name_values(get_metrics())
    if not points:
        raise TelemetryUnavailable("vLLM metrics snapshot contained no numeric metrics")
    return {
        "source": source,
        "running_requests": summed_metric(
            points,
            {
                "vllm:num_requests_running",
                "vllm:num_running_requests",
                "num_requests_running",
                "num_running_requests",
            },
        ),
        "waiting_requests": summed_metric(
            points,
            {
                "vllm:num_requests_waiting",
                "vllm:num_waiting_requests",
                "num_requests_waiting",
                "num_waiting_requests",
            },
        ),
        "preemptions_total": summed_metric(
            points,
            {
                "vllm:num_preemptions",
                "vllm:num_preempted_requests",
                "num_preemptions",
                "num_preempted_requests",
            },
        ),
    }


def sglang_capacity_telemetry(engine: Any) -> dict[str, Any]:
    get_server_info = getattr(engine, "get_server_info", None)
    if not callable(get_server_info):
        return {
            "effective_token_capacity": None,
            "configured_max_running_requests": None,
            "capacity_source": None,
            "capacity_error": "SGLang get_server_info API is unavailable",
        }
    try:
        info = get_server_info()
    except Exception as exc:
        return {
            "effective_token_capacity": None,
            "configured_max_running_requests": None,
            "capacity_source": None,
            "capacity_error": str(exc).splitlines()[0],
        }
    capacity = telemetry_number(
        telemetry_value(
            info,
            "max_total_num_tokens",
            "max_total_tokens",
            "token_capacity",
        )
    )
    max_running = telemetry_number(
        telemetry_value(
            info,
            "effective_max_running_requests_per_dp",
            "max_running_requests",
        )
    )
    source = "engine.get_server_info"
    internal_states = telemetry_items(info, "internal_states")
    if capacity is None:
        capacities = []
        for state in internal_states:
            memory = telemetry_value(state, "memory_usage", "memory")
            value = telemetry_number(
                telemetry_value(
                    memory,
                    "token_capacity",
                    "max_total_num_tokens",
                )
            )
            if value is not None:
                capacities.append(value)
        if capacities:
            capacity = sum(capacities)
            source = "engine.get_server_info.internal_states[].memory_usage"
    if max_running is None:
        running_limits = [
            value
            for state in internal_states
            if (
                value := telemetry_number(
                    telemetry_value(
                        state,
                        "effective_max_running_requests_per_dp",
                        "max_running_requests",
                    )
                )
            )
            is not None
        ]
        if running_limits:
            max_running = sum(running_limits)
    return {
        "effective_token_capacity": (
            None if capacity is None else int(capacity)
        ),
        "configured_max_running_requests": (
            None if max_running is None else int(max_running)
        ),
        "capacity_source": source if capacity is not None else None,
        "capacity_error": None,
    }


def sglang_runtime_telemetry_sample(engine: Any) -> dict[str, Any]:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    reader = getattr(tokenizer_manager, "load_snapshot_reader", None)
    read_all = getattr(reader, "read_all", None)
    if not callable(read_all):
        raise TelemetryUnavailable(
            "SGLang shared load-snapshot API is unavailable"
        )
    snapshots = telemetry_items(read_all(), "loads", "snapshots", "data")
    if not snapshots:
        raise RuntimeError("SGLang load-snapshot API returned no snapshots")

    def total(*names: str) -> float | None:
        values = [
            value
            for snapshot in snapshots
            if (value := telemetry_number(telemetry_value(snapshot, *names)))
            is not None
        ]
        return sum(values) if values else None

    return {
        "source": "engine.tokenizer_manager.load_snapshot_reader.read_all",
        "running_requests": total("num_running_reqs", "num_running_requests"),
        "waiting_requests": total(
            "num_waiting_reqs",
            "num_queue_reqs",
            "num_waiting_requests",
        ),
        "used_tokens": total("num_used_tokens", "used_tokens"),
    }


def sglang_output_retractions(obj: Any, expected: int) -> tuple[int | None, int]:
    if expected == 1 and isinstance(obj, Mapping):
        items = [obj]
    elif isinstance(obj, list) and len(obj) == expected:
        items = obj
    else:
        return None, 0
    total = 0
    captured = 0
    for item in items:
        meta = telemetry_value(item, "meta_info", "meta")
        value = telemetry_value(
            meta,
            "num_retractions",
            "retraction_count",
            "retraction_counts",
        )
        if value is None:
            value = telemetry_value(
                item,
                "num_retractions",
                "retraction_count",
                "retraction_counts",
            )
        values = value if isinstance(value, (list, tuple)) else [value]
        numbers = [telemetry_number(entry) for entry in values]
        if not numbers or any(number is None for number in numbers):
            return None, captured
        total += sum(int(number) for number in numbers if number is not None)
        captured += 1
    return total, captured


class ResidencyTelemetryMonitor:
    def __init__(
        self,
        *,
        engine: str,
        capacity: dict[str, Any],
        sampler: Callable[[], dict[str, Any]],
        required_fields: tuple[str, ...],
        interval_s: float = 0.05,
    ) -> None:
        self.engine = engine
        self.capacity = dict(capacity)
        self.sampler = sampler
        self.required_fields = required_fields
        self.interval_s = interval_s
        self.sample_count = 0
        self.active_sample_count = 0
        self.periodic_sample_count = 0
        self.active_periodic_sample_count = 0
        self.error_count = 0
        self.first_error: str | None = None
        self.sources: set[str] = set()
        capacity_source = capacity.get("capacity_source")
        if isinstance(capacity_source, str):
            self.sources.add(capacity_source)
        self.peak_running_requests: int | None = None
        self.peak_waiting_requests: int | None = None
        self.peak_used_tokens: int | None = None
        self.preemptions_start: int | None = None
        self.preemptions_peak: int | None = None
        self.output_retractions: int | None = None
        self.output_retraction_request_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sampling_disabled = False

    def _record_error(self, exc: Exception) -> None:
        self.error_count += 1
        if self.first_error is None:
            self.first_error = str(exc).splitlines()[0]
        if isinstance(exc, TelemetryUnavailable):
            self._sampling_disabled = True

    def _sample_once(self, *, periodic: bool = False) -> None:
        if self._sampling_disabled:
            return
        try:
            sample = self.sampler()
        except Exception as exc:
            self._record_error(exc)
            return
        source = sample.get("source")
        if isinstance(source, str):
            self.sources.add(source)
        self.sample_count += 1
        active = any(
            (telemetry_number(sample.get(key)) or 0) > 0
            for key in ("running_requests", "waiting_requests", "used_tokens")
        )
        if active:
            self.active_sample_count += 1
        if periodic:
            self.periodic_sample_count += 1
            if active:
                self.active_periodic_sample_count += 1
        for key, attribute in (
            ("running_requests", "peak_running_requests"),
            ("waiting_requests", "peak_waiting_requests"),
            ("used_tokens", "peak_used_tokens"),
        ):
            value = telemetry_number(sample.get(key))
            if value is None:
                continue
            current = getattr(self, attribute)
            setattr(
                self,
                attribute,
                int(value) if current is None else max(current, int(value)),
            )
        preemptions = telemetry_number(sample.get("preemptions_total"))
        if preemptions is not None:
            count = int(preemptions)
            if self.preemptions_start is None:
                self.preemptions_start = count
            self.preemptions_peak = (
                count
                if self.preemptions_peak is None
                else max(self.preemptions_peak, count)
            )

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once(periodic=True)
            if self._sampling_disabled:
                return

    def __enter__(self):
        self._sample_once()
        if not self._sampling_disabled:
            self._thread = threading.Thread(
                target=self._sample_loop,
                name=f"{self.engine}-residency-telemetry",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
            if self._thread.is_alive():
                self._record_error(
                    RuntimeError("residency telemetry sampler did not stop")
                )
                return
        self._sample_once()

    def record_output_retractions(
        self,
        value: int | None,
        request_count: int,
    ) -> None:
        self.output_retractions = value
        self.output_retraction_request_count = request_count
        if value is not None:
            self.sources.add("SGLang output meta_info.num_retractions")

    def result(self) -> dict[str, Any]:
        preemption_events = None
        if self.preemptions_start is not None and self.preemptions_peak is not None:
            preemption_events = max(
                0, self.preemptions_peak - self.preemptions_start
            )
        result = {
            "schema": RESIDENCY_TELEMETRY_SCHEMA,
            "engine": self.engine,
            "sample_interval_s": self.interval_s,
            "sample_count": self.sample_count,
            "active_sample_count": self.active_sample_count,
            "periodic_sample_count": self.periodic_sample_count,
            "active_periodic_sample_count": self.active_periodic_sample_count,
            "error_count": self.error_count,
            "error": self.first_error or self.capacity.get("capacity_error"),
            "sources": sorted(self.sources),
            "kv_token_capacity": self.capacity.get("kv_token_capacity"),
            "kv_max_concurrency": self.capacity.get("kv_max_concurrency"),
            "effective_token_capacity": self.capacity.get(
                "effective_token_capacity"
            ),
            "configured_max_running_requests": self.capacity.get(
                "configured_max_running_requests"
            ),
            "capacity_estimated": self.capacity.get("capacity_estimated"),
            "peak_running_requests": self.peak_running_requests,
            "peak_waiting_requests": self.peak_waiting_requests,
            "peak_used_tokens": self.peak_used_tokens,
            "preemption_events": preemption_events,
            "output_retractions": self.output_retractions,
            "output_retraction_request_count": (
                self.output_retraction_request_count
            ),
        }
        required_unavailable = [
            field for field in self.required_fields if result.get(field) is None
        ]
        unavailable = list(required_unavailable)
        runtime_fields = {
            "peak_running_requests",
            "peak_waiting_requests",
            "peak_used_tokens",
            "preemption_events",
            "output_retractions",
        }
        if (
            runtime_fields.intersection(self.required_fields)
            and not self._sampling_disabled
            and not self.active_periodic_sample_count
        ):
            unavailable.append("active_periodic_sample_coverage")
        available_count = len(self.required_fields) - len(required_unavailable)
        status = (
            "complete"
            if not unavailable
            else "partial"
            if available_count
            else "unavailable"
        )
        result.update(
            {
                "status": status,
                "available": status != "unavailable",
                "complete": status == "complete",
                "unavailable_fields": unavailable,
            }
        )
        return result


def residency_telemetry_row_fields(
    telemetry: dict[str, Any],
    *,
    tokens_per_request: int | None = None,
) -> dict[str, Any]:
    token_capacity = telemetry.get("kv_token_capacity")
    if token_capacity is None:
        token_capacity = telemetry.get("effective_token_capacity")
    full_length_context_capacity = None
    if token_capacity is not None and tokens_per_request:
        full_length_context_capacity = float(token_capacity) / float(
            tokens_per_request
        )
    return {
        "residency_telemetry_status": telemetry["status"],
        "residency_telemetry": telemetry,
        "token_capacity": token_capacity,
        "kv_max_concurrency": telemetry.get("kv_max_concurrency"),
        "configured_max_running_requests": telemetry.get(
            "configured_max_running_requests"
        ),
        "full_length_context_capacity": full_length_context_capacity,
        "max_running": telemetry.get("peak_running_requests"),
        "max_waiting": telemetry.get("peak_waiting_requests"),
        "max_used_tokens": telemetry.get("peak_used_tokens"),
        "preemption_events": telemetry.get("preemption_events"),
        "retraction_events": telemetry.get("output_retractions"),
    }


def prompt_token_source(args) -> str:
    return "synthetic" if getattr(args, "synthetic_prompts", False) else "hf_tokenizer"


def uses_hf_tokenizer(args) -> bool:
    return prompt_token_source(args) == "hf_tokenizer"


def sglang_language_model_override(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError(f"{config_path}: text_config must be an object")
    override = dict(text_config)
    if text_config.get("global_head_dim") is not None:
        override["swa_head_dim"] = text_config["head_dim"]
        override["swa_v_head_dim"] = text_config["head_dim"]
        override["head_dim"] = text_config["global_head_dim"]
        override["v_head_dim"] = text_config["global_head_dim"]
    if text_config.get("num_global_key_value_heads") is not None:
        override["swa_num_key_value_heads"] = text_config["num_key_value_heads"]
        override["num_key_value_heads"] = text_config[
            "num_global_key_value_heads"
        ]
    override["architectures"] = ["Gemma4ForCausalLM"]
    return override


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except Exception:
            versions[package] = None
    return versions


def query_nvidia_gpu(device: str) -> dict[str, Any]:
    """Return one physical GPU's identity, driver, and memory information."""

    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"nvidia-smi returned {len(rows)} GPUs for device selector {device!r}"
        )
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi output for device {device!r}")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(values[4]),
        "memory_used_mib": int(values[5]),
    }


def _single_visible_gpu_selector() -> str | None:
    """Use a sole physical GPU only when nvidia-smi makes that unambiguous."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
    except Exception:
        return None
    indexes = [line.strip() for line in output.splitlines() if line.strip()]
    return indexes[0] if len(indexes) == 1 else None


def resolve_gpu_memory_device(
    device: str | int | None = None,
) -> tuple[str | None, str]:
    """Resolve a physical nvidia-smi selector without assuming GPU 0.

    An explicit CLI value wins.  A single-value ``CUDA_VISIBLE_DEVICES`` (or
    ``WKVM_GPU_MEMORY_DEVICE``) is accepted as a physical index/UUID.  With no
    selector, an unambiguous one-GPU host is detected; multi-GPU hosts require
    the operator to pass ``--gpu-memory-device``.
    """

    if device is not None and str(device).strip():
        return str(device).strip(), "explicit"
    for env_name in ("WKVM_GPU_MEMORY_DEVICE", "WKVM_GPU_INDEX"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value, env_name
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "none", "None"}:
        values = [part.strip() for part in visible.split(",") if part.strip()]
        if len(values) == 1:
            return values[0], "CUDA_VISIBLE_DEVICES"
    discovered = _single_visible_gpu_selector()
    if discovered is not None:
        return discovered, "single_gpu_discovery"
    return None, "unresolved"


def gpu_mem_used_mib(device: str | int | None = None) -> int | None:
    selector, _source = resolve_gpu_memory_device(device)
    if selector is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
    except Exception:
        return None
    values = [line.strip() for line in out.splitlines() if line.strip()]
    if len(values) != 1:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


class VramMonitor:
    """Whole-GPU memory sampler for engines that do not expose torch peaks."""

    def __init__(
        self,
        interval_s: float = 0.1,
        device: str | int | None = None,
        *,
        device_selector: str | int | None = None,
    ) -> None:
        if device is not None and device_selector is not None:
            raise ValueError("pass only one of device or device_selector")
        self.interval_s = interval_s
        requested_device = device if device is not None else device_selector
        self.device, self.device_selector_source = resolve_gpu_memory_device(
            requested_device
        )
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self.sample_count = 0
        self.query_error_count = 0
        self.first_error: str | None = None
        self.gpu: dict[str, Any] | None = None
        self._process: subprocess.Popen[str] | None = None

    def _record_samples(self, output: str) -> None:
        for raw in output.splitlines():
            value = raw.strip()
            if not value:
                continue
            try:
                used = int(value)
            except ValueError:
                self.query_error_count += 1
                if self.first_error is None:
                    self.first_error = f"unexpected nvidia-smi sample {value!r}"
                continue
            self.sample_count += 1
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)

    def __enter__(self):
        if self.device is None:
            self.query_error_count += 1
            self.first_error = (
                "no physical GPU selector; pass --gpu-memory-device (or set "
                "CUDA_VISIBLE_DEVICES to one physical device)"
            )
            return self
        try:
            self.gpu = query_nvidia_gpu(self.device)
        except Exception as exc:
            self.query_error_count += 1
            self.first_error = str(exc).splitlines()[0]
        self.baseline_mib = gpu_mem_used_mib(self.device)
        self.peak_mib = self.baseline_mib
        interval_ms = max(1, int(round(self.interval_s * 1000)))
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--id={self.device}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self.query_error_count += 1
            self.first_error = str(exc).splitlines()[0]
        return self

    def __exit__(self, *exc) -> None:
        process = self._process
        if process is not None:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
            self._record_samples(stdout)
            if process.returncode not in (0, -15) and stderr.strip():
                self.query_error_count += 1
                if self.first_error is None:
                    self.first_error = stderr.strip().splitlines()[0]
        final = gpu_mem_used_mib(self.device)
        if final is not None:
            self.sample_count += 1
            self.peak_mib = final if self.peak_mib is None else max(self.peak_mib, final)

    @property
    def engine_peak_delta_gib(self) -> float | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0, self.peak_mib - self.baseline_mib) / 1024.0

    def result(self) -> dict[str, Any]:
        peak_delta_mib = None
        if self.baseline_mib is not None and self.peak_mib is not None:
            peak_delta_mib = max(0, self.peak_mib - self.baseline_mib)
        return {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": self.device,
            "device_selector_source": self.device_selector_source,
            "device_index": None if self.gpu is None else self.gpu.get("index"),
            "device_uuid": None if self.gpu is None else self.gpu.get("uuid"),
            "gpu_name": None if self.gpu is None else self.gpu.get("name"),
            "driver_version": (
                None if self.gpu is None else self.gpu.get("driver_version")
            ),
            "memory_total_mib": (
                None if self.gpu is None else self.gpu.get("memory_total_mib")
            ),
            "sample_interval_s": self.interval_s,
            "sample_count": self.sample_count,
            "baseline_used_mib": self.baseline_mib,
            "peak_used_mib": self.peak_mib,
            "peak_delta_mib": peak_delta_mib,
            "query_error_count": self.query_error_count,
            "error": self.first_error,
        }


def monitor_memory_row(monitor: VramMonitor) -> dict[str, Any]:
    return {
        "baseline_gpu_mib": monitor.baseline_mib,
        "peak_gpu_mib": monitor.peak_mib,
        "peak_engine_delta_gib": round_or_none(monitor.engine_peak_delta_gib),
        "gpu_memory": monitor.result(),
    }


def synchronize_cuda() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def reset_torch_peak() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def torch_peak_alloc_gib() -> float | None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**30
    return None


def torch_peak_reserved_gib() -> float | None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved() / 2**30
    return None


def cleanup_cuda() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def normalize_vllm_outputs(outputs: list[Any], expected: int) -> list[list[int]]:
    if len(outputs) != expected:
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {expected} prompts")
    rows: list[list[int]] = []
    for out in outputs:
        rows.append([int(x) for x in out.outputs[0].token_ids])
    return rows


def _finite_nonnegative_metric(value: Any) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def vllm_request_metrics_timing(
    outputs: list[Any],
    expected: int,
) -> dict[str, Any] | None:
    """Extract request TTFTs and a same-run batch decode interval.

    vLLM's ``first_token_latency`` is a per-request TTFT.  The old harness
    exposed its minimum as ``prefill_plus_first_s`` without saying so, and
    copied the batch wall time into request latency percentiles.  Keep the old
    key as a compatibility alias while emitting explicit TTFT and batch scope
    fields.  Timestamp coverage can be incomplete on some vLLM releases, so
    TTFT extraction is independent of decode-interval extraction.
    """

    if len(outputs) != expected:
        return None
    first_token_timestamps: list[float] = []
    last_token_timestamps: list[float] = []
    first_token_latencies: list[float] = []
    request_e2e_latencies: list[float] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        first_token_ts = _finite_nonnegative_metric(
            getattr(metrics, "first_token_ts", None)
        )
        last_token_ts = _finite_nonnegative_metric(
            getattr(metrics, "last_token_ts", None)
        )
        if (
            first_token_ts is not None
            and last_token_ts is not None
            and first_token_ts > 0
            and last_token_ts >= first_token_ts
        ):
            first_token_timestamps.append(first_token_ts)
            last_token_timestamps.append(last_token_ts)
            timestamp_pair_valid = True
        else:
            timestamp_pair_valid = False
        first_token_latency = _finite_nonnegative_metric(
            getattr(metrics, "first_token_latency", None)
        )
        if first_token_latency is not None and (
            first_token_latency > 0 or timestamp_pair_valid
        ):
            first_token_latencies.append(first_token_latency)
            if timestamp_pair_valid:
                request_e2e_latencies.append(
                    first_token_latency + max(0.0, last_token_ts - first_token_ts)
                )

    has_timing = bool(first_token_latencies or first_token_timestamps)
    if not has_timing:
        return None
    complete_interval = (
        len(first_token_timestamps) == expected
        and len(last_token_timestamps) == expected
    )
    decode_seconds = (
        max(last_token_timestamps) - min(first_token_timestamps)
        if complete_interval
        else None
    )
    min_ttft = (
        min(first_token_latencies) if first_token_latencies else None
    )
    max_ttft = (
        max(first_token_latencies) if first_token_latencies else None
    )
    ttft_scope = (
        "min_request_ttft" if min_ttft is not None else "unavailable_request_ttft"
    )
    cohort_scope = (
        "max_request_ttft_synchronous_cohort"
        if max_ttft is not None
        else "unavailable_request_ttft"
    )
    return {
        # Compatibility alias: this is *minimum request TTFT*, never batch wall.
        "prefill_plus_first_s": min_ttft,
        "prefill_plus_first_scope": ttft_scope,
        "prefill_plus_first_source": (
            "RequestOutput.metrics.first_token_latency"
            if min_ttft is not None
            else "RequestOutput.metrics.first_token_latency unavailable"
        ),
        "min_ttft_s": min_ttft,
        "max_ttft_s": max_ttft,
        "p50_ttft_s": percentile(first_token_latencies, 0.50),
        "p95_ttft_s": percentile(first_token_latencies, 0.95),
        "ttft_request_count": len(first_token_latencies),
        "ttft_available_count": len(first_token_latencies),
        "request_ttft_s": list(first_token_latencies),
        "request_e2e_latency_s": list(request_e2e_latencies),
        "decode_seconds": decode_seconds,
        "decode_interval_s": decode_seconds,
        "decode_interval_scope": (
            "batch_earliest_first_to_latest_last"
            if complete_interval
            else "unavailable_incomplete_request_timestamps"
        ),
        "cohort_prefill_wall_s": max_ttft,
        "cohort_prefill_scope": cohort_scope,
        "cohort_prefill_wall_scope": cohort_scope,
        "cohort_prefill_source": "RequestOutput.metrics.first_token_latency",
        "cohort_prefill_wall_source": (
            "RequestOutput.metrics.first_token_latency"
        ),
        "cohort_prefill_comparable": max_ttft is not None,
        "cohort_prefill_timing_method": "same_run_max_request_ttft",
        "cohort_prefill_note": (
            "All requests were submitted synchronously; maximum request TTFT "
            "is the cohort time until every request produced a first token."
        ),
        "batch_wall_scope": "synchronous_batch_completion",
        "decode_timing_method": (
            "same_run_request_metrics" if complete_interval else "request_metrics_partial"
        ),
        "decode_timing_source": (
            "RequestOutput.metrics.first_token_ts/last_token_ts"
        ),
        "decode_timing_comparable": complete_interval,
        "decode_timing_request_count": len(first_token_timestamps),
        "decode_timing_note": (
            "Batch interval is earliest first token to latest last token from "
            "the measured max_tokens=N run."
            if complete_interval
            else "Request timestamps were incomplete; no comparable same-run batch interval."
        ),
    }


def measure_vllm_generation(
    llm: Any,
    reqs: list[dict[str, Any]],
    sp1: Any,
    spn: Any,
    telemetry_monitor: ResidencyTelemetryMonitor | None = None,
) -> tuple[list[Any], float, dict[str, Any]]:
    """Measure vLLM, preferring exact timestamps from the full generation."""

    telemetry_context = telemetry_monitor or contextlib.nullcontext()
    with telemetry_context:
        synchronize_cuda()
        started = time.perf_counter()
        full = llm.generate(reqs, spn, use_tqdm=False)
        synchronize_cuda()
        full_wall_s = time.perf_counter() - started

    timing = vllm_request_metrics_timing(full, len(reqs))
    if timing is not None and timing.get("decode_seconds") is not None:
        timing["batch_wall_s"] = full_wall_s
        return full, full_wall_s, timing

    synchronize_cuda()
    started = time.perf_counter()
    llm.generate(reqs, sp1, use_tqdm=False)
    synchronize_cuda()
    first_wall_s = time.perf_counter() - started
    fallback_timing: dict[str, Any] = {
        "prefill_plus_first_s": first_wall_s,
        "prefill_plus_first_scope": "separate_max_tokens_1_batch_wall",
        "prefill_plus_first_source": "separate max_tokens=1 wall time",
        "separate_first_token_batch_wall_s": first_wall_s,
        "cohort_prefill_wall_s": first_wall_s,
        "cohort_prefill_scope": "separate_run_batch_wall",
        "cohort_prefill_wall_scope": "separate_run_batch_wall",
        "cohort_prefill_source": "separate max_tokens=1 batch wall time",
        "cohort_prefill_wall_source": "separate max_tokens=1 batch wall time",
        "cohort_prefill_comparable": False,
        "cohort_prefill_timing_method": "separate_run_batch_wall",
        "cohort_prefill_note": (
            "Measured in a separate max_tokens=1 run and not strictly comparable "
            "to same-run maximum request TTFT."
        ),
        "batch_wall_s": full_wall_s,
        "batch_wall_scope": "synchronous_batch_completion",
        "decode_seconds": max(full_wall_s - first_wall_s, 0.0),
        "decode_interval_s": max(full_wall_s - first_wall_s, 0.0),
        "decode_interval_scope": "separate_run_wall_time_subtraction",
        "decode_timing_method": "separate_run_subtraction",
        "decode_timing_source": (
            "max_tokens=N wall time minus a separate max_tokens=1 wall time"
        ),
        "decode_timing_comparable": False,
        "decode_timing_request_count": 0,
        "decode_timing_note": (
            "Fallback only: separate-run subtraction is not directly comparable "
            "to same-run first-token-to-finish timing."
        ),
    }
    if timing is not None:
        for key in (
            "min_ttft_s",
            "max_ttft_s",
            "p50_ttft_s",
            "p95_ttft_s",
            "ttft_request_count",
            "ttft_available_count",
            "request_ttft_s",
            "request_e2e_latency_s",
            "cohort_prefill_wall_s",
            "cohort_prefill_scope",
            "cohort_prefill_wall_scope",
            "cohort_prefill_source",
            "cohort_prefill_wall_source",
        ):
            fallback_timing[key] = timing.get(key)
        if timing.get("min_ttft_s") is not None:
            fallback_timing.update(
                {
                    "prefill_plus_first_s": timing["min_ttft_s"],
                    "prefill_plus_first_scope": "min_request_ttft",
                    "prefill_plus_first_source": (
                        "RequestOutput.metrics.first_token_latency"
                    ),
                }
            )
        if timing.get("max_ttft_s") is not None:
            fallback_timing.update(
                {
                    "cohort_prefill_wall_s": timing["max_ttft_s"],
                    "cohort_prefill_scope": (
                        "max_request_ttft_synchronous_cohort"
                    ),
                    "cohort_prefill_wall_scope": (
                        "max_request_ttft_synchronous_cohort"
                    ),
                    "cohort_prefill_source": (
                        "RequestOutput.metrics.first_token_latency"
                    ),
                    "cohort_prefill_wall_source": (
                        "RequestOutput.metrics.first_token_latency"
                    ),
                    "cohort_prefill_comparable": True,
                    "cohort_prefill_timing_method": (
                        "same_run_max_request_ttft"
                    ),
                    "cohort_prefill_note": (
                        "All requests were submitted synchronously; maximum "
                        "request TTFT is the cohort time until every request "
                        "produced a first token."
                    ),
                }
            )
        fallback_timing["decode_timing_request_count"] = timing.get(
            "decode_timing_request_count", 0
        )
    return full, full_wall_s, fallback_timing


def sglang_output_ids(obj: Any) -> list[int]:
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected SGLang output item type: {type(obj).__name__}")
    for key in ("output_ids", "output_token_ids"):
        if obj.get(key) is not None:
            return [int(x) for x in obj[key]]
    meta = obj.get("meta_info") or {}
    for key in ("output_ids", "output_token_ids"):
        if meta.get(key) is not None:
            return [int(x) for x in meta[key]]
    raise KeyError("SGLang output did not contain output token ids")


def normalize_sglang_outputs(obj: Any, expected: int) -> list[list[int]]:
    if expected == 1 and isinstance(obj, dict):
        return [sglang_output_ids(obj)]
    if not isinstance(obj, list):
        raise TypeError(f"unexpected SGLang output type: {type(obj).__name__}")
    if len(obj) != expected:
        raise RuntimeError(f"SGLang returned {len(obj)} outputs for {expected} prompts")
    return [sglang_output_ids(item) for item in obj]


def make_row(
    *,
    B: int,
    prompt_lens: list[int],
    first_wall_s: float | None,
    full_wall_s: float,
    outputs: list[list[int]],
    mem: dict[str, Any],
    error: str | None = None,
    decode_seconds: float | None = None,
    decode_timing: dict[str, Any] | None = None,
    request_ttft_s: list[float] | None = None,
    request_e2e_latency_s: list[float] | None = None,
    cohort_prefill_wall_s: float | None = None,
    cohort_prefill_scope: str | None = None,
    cohort_prefill_source: str | None = None,
) -> dict[str, Any]:
    success_count = sum(1 for ids in outputs if ids)
    timing_fields = decode_timing or {
        "prefill_plus_first_scope": "batch_max_tokens_1_wall",
        "prefill_plus_first_source": "separate max_tokens=1 wall time",
        "cohort_prefill_scope": "separate_run_batch_wall",
        "cohort_prefill_wall_scope": "separate_run_batch_wall",
        "cohort_prefill_source": "separate max_tokens=1 batch wall time",
        "cohort_prefill_wall_source": "separate max_tokens=1 batch wall time",
        "cohort_prefill_comparable": False,
        "cohort_prefill_timing_method": "separate_run_batch_wall",
        "cohort_prefill_note": (
            "Measured in a separate max_tokens=1 run and not strictly comparable "
            "to same-run maximum request TTFT."
        ),
        "batch_wall_scope": "synchronous_batch_completion",
        "decode_interval_scope": "batch_wall_minus_first_token_wall",
        "decode_timing_method": "separate_run_subtraction",
        "decode_timing_source": (
            "max_tokens=N wall time minus a separate max_tokens=1 wall time"
        ),
        "decode_timing_comparable": False,
        "decode_timing_request_count": 0,
        "decode_timing_note": (
            "Separate-run subtraction is not directly comparable to same-run "
            "first-token-to-finish timing."
        ),
    }
    if request_ttft_s is None:
        request_ttft_s = timing_fields.get("request_ttft_s")
    if request_e2e_latency_s is None:
        request_e2e_latency_s = timing_fields.get("request_e2e_latency_s")
    if cohort_prefill_wall_s is None:
        cohort_prefill_wall_s = timing_fields.get("cohort_prefill_wall_s")
    if cohort_prefill_scope is None:
        cohort_prefill_scope = timing_fields.get("cohort_prefill_scope")
    if cohort_prefill_source is None:
        cohort_prefill_source = timing_fields.get("cohort_prefill_source")
    request_ttft_s = [
        float(value)
        for value in (request_ttft_s or [])
        if _finite_nonnegative_metric(value) is not None
    ]
    request_e2e_latency_s = [
        float(value)
        for value in (request_e2e_latency_s or [])
        if _finite_nonnegative_metric(value) is not None
    ]
    output_tokens = sum(len(ids) for ids in outputs)
    decode_tokens = sum(max(0, len(ids) - 1) for ids in outputs)
    decode_s = (
        max(full_wall_s - first_wall_s, 0.0)
        if decode_seconds is None and first_wall_s is not None
        else decode_seconds
    )
    timing_min_ttft = timing_fields.get("min_ttft_s")
    timing_max_ttft = timing_fields.get("max_ttft_s")
    timing_p50_ttft = timing_fields.get("p50_ttft_s")
    timing_p95_ttft = timing_fields.get("p95_ttft_s")
    timing_fields = {
        key: value
        for key, value in timing_fields.items()
        if key
        not in {
            "prefill_plus_first_s",
            "decode_seconds",
            "decode_interval_s",
            "request_ttft_s",
            "request_e2e_latency_s",
            "batch_wall_s",
            "cohort_prefill_wall_s",
            "cohort_prefill_scope",
            "cohort_prefill_wall_scope",
            "cohort_prefill_source",
            "cohort_prefill_wall_source",
            "min_ttft_s",
            "max_ttft_s",
            "p50_ttft_s",
            "p95_ttft_s",
            "ttft_request_count",
            "ttft_available_count",
        }
    }
    min_ttft = (
        min(request_ttft_s) if request_ttft_s else timing_min_ttft
    )
    max_ttft = (
        max(request_ttft_s) if request_ttft_s else timing_max_ttft
    )
    p50_ttft = (
        percentile(request_ttft_s, 0.50)
        if request_ttft_s
        else timing_p50_ttft
    )
    p95_ttft = (
        percentile(request_ttft_s, 0.95)
        if request_ttft_s
        else timing_p95_ttft
    )
    prefill_scope = timing_fields.get(
        "prefill_plus_first_scope",
        "batch_max_tokens_1_wall",
    )
    if cohort_prefill_wall_s is None and max_ttft is not None:
        cohort_prefill_wall_s = max_ttft
    if cohort_prefill_wall_s is None:
        cohort_prefill_wall_s = first_wall_s
    cohort_scope = cohort_prefill_scope or timing_fields.get(
        "cohort_prefill_scope",
        "max_request_ttft_synchronous_cohort"
        if request_ttft_s
        else "separate_run_batch_wall",
    )
    cohort_source = cohort_prefill_source or timing_fields.get(
        "cohort_prefill_source",
        "RequestOutput.metrics.first_token_latency"
        if request_ttft_s
        else "separate max_tokens=1 batch wall time",
    )
    cohort_comparable = bool(
        timing_fields.get("cohort_prefill_comparable", bool(request_ttft_s))
    )
    cohort_input_tokens = sum(
        int(length)
        for length in prompt_lens
        if isinstance(length, int) and not isinstance(length, bool) and length >= 0
    )
    cohort_input_tok_s = (
        cohort_input_tokens / cohort_prefill_wall_s
        if cohort_prefill_wall_s is not None and cohort_prefill_wall_s > 0
        else None
    )
    row: dict[str, Any] = {
        "B": B,
        "success_count": success_count,
        "error_count": B - success_count + (1 if error else 0),
        # These are request percentiles only when the engine exposes request
        # metrics.  A synchronous batch wall is deliberately not replicated.
        "p50_latency_s": round_or_none(percentile(request_e2e_latency_s, 0.50)),
        "p95_latency_s": round_or_none(percentile(request_e2e_latency_s, 0.95)),
        "latency_metric_source": (
            "request_metrics" if request_e2e_latency_s else None
        ),
        "latency_metric_count": len(request_e2e_latency_s),
        "p50_ttft_s": round_or_none(p50_ttft),
        "p95_ttft_s": round_or_none(p95_ttft),
        "min_ttft_s": round_or_none(min_ttft),
        "max_ttft_s": round_or_none(max_ttft),
        "ttft_metric_source": "request_metrics" if request_ttft_s else None,
        "ttft_metric_count": len(request_ttft_s),
        "prefill_plus_first_s": round_or_none(first_wall_s),
        "prefill_plus_first_scope": prefill_scope,
        "cohort_prefill_wall_s": round_or_none(cohort_prefill_wall_s),
        "cohort_prefill_scope": cohort_scope,
        "cohort_prefill_wall_scope": cohort_scope,
        "cohort_prefill_source": cohort_source,
        "cohort_prefill_wall_source": cohort_source,
        "cohort_prefill_comparable": cohort_comparable,
        "cohort_input_tokens": cohort_input_tokens,
        "cohort_input_tok_s": round_or_none(cohort_input_tok_s),
        "cohort_input_tok_s_comparable": cohort_comparable,
        "cohort_input_tok_scope": "prompt_lengths_over_cohort_prefill_wall",
        "cohort_input_tok_source": "prompt_lengths",
        "decode_seconds": round_or_none(decode_s),
        "decode_interval_s": round_or_none(decode_s),
        "agg_decode_tok_s": round_or_none(
            decode_tokens / decode_s
            if decode_s is not None and decode_s > 0
            else None
        ),
        "e2e_output_tok_s": round_or_none(
            output_tokens / full_wall_s if full_wall_s > 0 else None
        ),
        "batch_wall_s": round_or_none(full_wall_s),
        "batch_wall_scope": timing_fields.get(
            "batch_wall_scope", "synchronous_batch_completion"
        ),
        "elapsed_s": round_or_none(full_wall_s),
        "prompt_lengths": prompt_lens,
        "output_token_counts": [len(ids) for ids in outputs],
        "error": error,
        **timing_fields,
        **mem,
    }
    if request_ttft_s:
        row["request_ttft_s"] = [round_or_none(value, 6) for value in request_ttft_s]
    if request_e2e_latency_s:
        row["request_e2e_latency_s"] = [
            round_or_none(value, 6) for value in request_e2e_latency_s
        ]
    if success_count == B:
        row.update(
            generated_output_fingerprint_row_fields(
                generated_output_fingerprint(
                    (f"bench-{B}-{row_index}", token_ids)
                    for row_index, token_ids in enumerate(outputs)
                )
            )
        )
    return row


def row_green(row: dict[str, Any], args) -> bool:
    if (
        row.get("success_count") != row.get("B")
        or row.get("error_count") != 0
        or row.get("error") is not None
    ):
        return False
    candidates = [
        row.get("peak_engine_delta_gib"),
        row.get("peak_reserved_gib"),
        row.get("peak_alloc_gib"),
    ]
    mem_gib = next((float(x) for x in candidates if x is not None), None)
    if mem_gib is None:
        return False
    return mem_gib <= args.mem_cap_gib - args.headroom_gib


def run_vllm(
    args,
    tok,
    prompts_by_b: dict[int, list[list[int]]],
    monitor: VramMonitor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    max_prompt_len = max(len(p) for prompts in prompts_by_b.values() for p in prompts)
    max_model_len = args.max_model_len or (max_prompt_len + args.out + 16)
    kwargs = {
        "model": args.model_path,
        "max_model_len": max_model_len,
        "max_num_seqs": max(prompts_by_b),
        "gpu_memory_utilization": args.vllm_gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "swap_space": 0,
        "disable_log_stats": False,
        "dtype": "bfloat16",
    }
    max_num_batched_tokens = getattr(args, "vllm_max_num_batched_tokens", None)
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if args.vllm_language_model_only:
        kwargs["language_model_only"] = True
    compilation_config = None
    if args.vllm_disable_inductor:
        capture_sizes = sorted({1, 2, 4, max(prompts_by_b)})
        compilation_config = {
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": max(capture_sizes),
        }
        kwargs["compilation_config"] = compilation_config
    try:
        llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "audio": 0})
        mm_note = "limit_mm_per_prompt={image:0,audio:0}"
    except TypeError:
        llm = LLM(**kwargs)
        mm_note = "default multimodal budget"

    sp1 = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True)
    spn = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.out,
        ignore_eos=True,
    )

    if args.warmup:
        with contextlib.suppress(Exception):
            llm.generate(
                [{"prompt_token_ids": next(iter(prompts_by_b.values()))[0][:128]}],
                sp1,
                use_tqdm=False,
            )

    capacity_telemetry = vllm_capacity_telemetry(
        llm,
        max_model_len=max_model_len,
    )
    rows: list[dict[str, Any]] = []
    reset_torch_peak()
    for B, prompts in prompts_by_b.items():
        fingerprint_fields = prompt_fingerprint_row_fields(
            prompt_set_fingerprint(
                prompts,
                prompt_token_source=prompt_token_source(args),
            )
        )
        row_telemetry = ResidencyTelemetryMonitor(
            engine="vllm",
            capacity=capacity_telemetry,
            sampler=lambda: vllm_runtime_telemetry_sample(llm),
            required_fields=(
                "kv_token_capacity",
                "kv_max_concurrency",
                "peak_running_requests",
                "peak_waiting_requests",
                "preemption_events",
            ),
            interval_s=getattr(args, "telemetry_sample_interval_s", 0.05),
        )
        try:
            reqs = [{"prompt_token_ids": p} for p in prompts]
            full, t_full, timing = measure_vllm_generation(
                llm,
                reqs,
                sp1,
                spn,
                telemetry_monitor=row_telemetry,
            )
            outputs = normalize_vllm_outputs(full, B)
            mem = {
                "peak_alloc_gib": round_or_none(torch_peak_alloc_gib()),
                "peak_reserved_gib": round_or_none(torch_peak_reserved_gib()),
                **monitor_memory_row(monitor),
            }
            row = make_row(
                B=B,
                prompt_lens=[len(p) for p in prompts],
                first_wall_s=timing["prefill_plus_first_s"],
                full_wall_s=t_full,
                outputs=outputs,
                mem=mem,
                decode_seconds=timing["decode_seconds"],
                decode_timing=timing,
            )
        except Exception as exc:
            row = {
                "B": B,
                "success_count": 0,
                "error_count": B,
                "error": str(exc).splitlines()[0],
                "prompt_lengths": [len(p) for p in prompts],
                "peak_alloc_gib": round_or_none(torch_peak_alloc_gib()),
                "peak_reserved_gib": round_or_none(torch_peak_reserved_gib()),
            }
        row.update(
            residency_telemetry_row_fields(
                row_telemetry.result(),
                tokens_per_request=max(len(prompt) for prompt in prompts) + args.out,
            )
        )
        row.update(fingerprint_fields)
        row["green"] = row_green(row, args)
        rows.append(row)
        print_row(args.engine, args, row)
        if row.get("error") and args.stop_on_failure:
            break

    engine_info = {
        "vllm_version": vllm.__version__,
        "gpu_memory_utilization": args.vllm_gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "max_model_len": max_model_len,
        "max_num_seqs": max(prompts_by_b),
        "max_num_batched_tokens": max_num_batched_tokens,
        "prefix_caching": False,
        "request_metrics_enabled": True,
        "decode_timing_preferred_method": "same_run_request_metrics",
        "decode_timing_fallback_comparable": False,
        "language_model_only": args.vllm_language_model_only,
        "compilation_config": compilation_config,
        "mm_note": mm_note,
        "residency_telemetry_capacity": capacity_telemetry,
    }
    with contextlib.suppress(Exception):
        del llm
    cleanup_cuda()
    return rows, engine_info


def measure_sglang_generation(
    engine: Any,
    *,
    prompts: list[list[int]],
    full_sampling_params: dict[str, Any],
    one_token_sampling_params: dict[str, Any],
    telemetry: Any,
) -> tuple[Any, float, Any, float]:
    with telemetry:
        synchronize_cuda()
        started = time.perf_counter()
        full = engine.generate(
            input_ids=prompts,
            sampling_params=full_sampling_params,
        )
        synchronize_cuda()
        full_wall_s = time.perf_counter() - started

    synchronize_cuda()
    started = time.perf_counter()
    first = engine.generate(
        input_ids=prompts,
        sampling_params=one_token_sampling_params,
    )
    synchronize_cuda()
    first_wall_s = time.perf_counter() - started
    return full, full_wall_s, first, first_wall_s


def run_sglang(
    args,
    tok,
    prompts_by_b: dict[int, list[list[int]]],
    monitor: VramMonitor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import sglang as sgl

    kwargs = {
        "model_path": args.model_path,
        "mem_fraction_static": args.sglang_mem_fraction,
        "context_length": args.sglang_context_length,
        "max_total_tokens": args.sglang_max_total_tokens,
        "disable_radix_cache": True,
        "enable_multimodal": False,
        "log_level": args.sglang_log_level,
        "max_running_requests": args.sglang_max_running_requests,
        "cuda_graph_backend_decode": args.sglang_decode_graph,
        "cuda_graph_backend_prefill": args.sglang_prefill_graph,
    }
    if args.sglang_attention_backend:
        kwargs["attention_backend"] = args.sglang_attention_backend
    chunked_prefill_size = getattr(args, "sglang_chunked_prefill_size", None)
    if chunked_prefill_size is not None:
        kwargs["chunked_prefill_size"] = chunked_prefill_size
    model_override = None
    if args.sglang_language_model_only:
        model_override = sglang_language_model_override(args.model_path)
        kwargs["json_model_override_args"] = json.dumps(model_override)

    engine = sgl.Engine(**kwargs)
    capacity_telemetry = sglang_capacity_telemetry(engine)
    sp1 = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    spn = {
        "temperature": 0.0,
        "max_new_tokens": args.out,
        "ignore_eos": True,
    }

    if args.warmup:
        with contextlib.suppress(Exception):
            engine.generate(
                input_ids=[next(iter(prompts_by_b.values()))[0][:128]],
                sampling_params=sp1,
            )

    rows: list[dict[str, Any]] = []
    for B, prompts in prompts_by_b.items():
        fingerprint_fields = prompt_fingerprint_row_fields(
            prompt_set_fingerprint(
                prompts,
                prompt_token_source=prompt_token_source(args),
            )
        )
        row_telemetry = ResidencyTelemetryMonitor(
            engine="sglang",
            capacity=capacity_telemetry,
            sampler=lambda: sglang_runtime_telemetry_sample(engine),
            required_fields=(
                "effective_token_capacity",
                "peak_running_requests",
                "peak_waiting_requests",
                "peak_used_tokens",
                "output_retractions",
            ),
            interval_s=getattr(args, "telemetry_sample_interval_s", 0.05),
        )
        try:
            full, t_full, first, t_first = measure_sglang_generation(
                engine,
                prompts=prompts,
                full_sampling_params=spn,
                one_token_sampling_params=sp1,
                telemetry=row_telemetry,
            )
            normalize_sglang_outputs(first, B)
            outputs = normalize_sglang_outputs(full, B)
            output_retractions, captured_requests = sglang_output_retractions(
                full,
                B,
            )
            row_telemetry.record_output_retractions(
                output_retractions,
                captured_requests,
            )
            row = make_row(
                B=B,
                prompt_lens=[len(p) for p in prompts],
                first_wall_s=t_first,
                full_wall_s=t_full,
                outputs=outputs,
                mem=monitor_memory_row(monitor),
            )
            row["separate_timing_probe_order"] = "full_then_max_tokens_1"
        except Exception as exc:
            row = {
                "B": B,
                "success_count": 0,
                "error_count": B,
                "error": str(exc).splitlines()[0],
                "prompt_lengths": [len(p) for p in prompts],
            }
        row.update(
            residency_telemetry_row_fields(
                row_telemetry.result(),
                tokens_per_request=max(len(prompt) for prompt in prompts) + args.out,
            )
        )
        row.update(fingerprint_fields)
        row["green"] = row_green(row, args)
        rows.append(row)
        print_row(args.engine, args, row)
        if row.get("error") and args.stop_on_failure:
            break

    engine_info = {
        "sglang_version": getattr(sgl, "__version__", "unknown"),
        "mem_fraction_static": args.sglang_mem_fraction,
        "context_length": args.sglang_context_length,
        "max_total_tokens": args.sglang_max_total_tokens,
        "chunked_prefill_size": chunked_prefill_size,
        "separate_timing_probe_order": "full_then_max_tokens_1",
        "attention_backend": args.sglang_attention_backend or "default",
        "language_model_only": args.sglang_language_model_only,
        "model_override_architectures": (
            None if model_override is None else model_override["architectures"]
        ),
        "model_override_model_type": (
            None if model_override is None else model_override.get("model_type")
        ),
        "disable_radix_cache": True,
        "max_running_requests": args.sglang_max_running_requests,
        "cuda_graph_backend_decode": args.sglang_decode_graph,
        "cuda_graph_backend_prefill": args.sglang_prefill_graph,
        "residency_telemetry_capacity": capacity_telemetry,
    }
    with contextlib.suppress(Exception):
        engine.shutdown()
    cleanup_cuda()
    return rows, engine_info


def print_row(engine: str, args, row: dict[str, Any]) -> None:
    print(
        f"[{engine} ctx={args.ctx} out={args.out} B={row['B']}] "
        f"success={row.get('success_count')}/{row['B']} "
        f"p50={row.get('p50_latency_s')}s p95={row.get('p95_latency_s')}s "
        f"agg={row.get('agg_decode_tok_s')}tok/s "
        f"reserved={row.get('peak_reserved_gib')}GiB "
        f"engine_delta={row.get('peak_engine_delta_gib')}GiB "
        f"green={row.get('green')}"
    )


def generated_output_fingerprint_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_rows = [
        row
        for row in rows
        if row.get("success_count") == row.get("B")
        and row.get("error_count") == 0
    ]
    by_batch: dict[str, dict[str, object]] = {}
    for row in successful_rows:
        fingerprint = row.get("generated_output_fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        if fingerprint.get("schema") != GENERATED_OUTPUT_FINGERPRINT_SCHEMA:
            continue
        digest = fingerprint.get("request_output_token_ids_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            continue
        by_batch[str(row["B"])] = dict(fingerprint)
    return {
        "successful_rows": len(successful_rows),
        "fingerprinted_successful_rows": len(by_batch),
        "complete": len(by_batch) == len(successful_rows),
        "by_batch": by_batch,
    }


def git_worktree_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return bool(output.strip())


def build_provenance(
    args,
    engine_info: dict[str, Any],
    monitor: VramMonitor,
    *,
    source_identity_before: dict[str, Any] | None = None,
    source_identity_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packages = installed_package_versions()
    engine_version = engine_info.get(f"{args.engine}_version")
    source_identity_after = source_identity_after or source_worktree_identity(ROOT)
    source_identity_before = source_identity_before or source_identity_after
    source_identity_unchanged = (
        source_identity_before.get("error") is None
        and source_identity_after.get("error") is None
        and source_identity_before.get("identity_sha256")
        == source_identity_after.get("identity_sha256")
    )
    gpu = None if monitor.gpu is None else dict(monitor.gpu)
    if gpu is not None:
        gpu.pop("memory_used_mib", None)
        gpu["source"] = "nvidia-smi"
    return {
        "schema": PROVENANCE_SCHEMA,
        "benchmark": {
            "git_commit": git_commit(),
            "git_worktree_dirty": source_identity_after.get(
                "git_worktree_dirty",
                git_worktree_dirty(),
            ),
            "source_identity": source_identity_after,
            "pre_run_source_identity_sha256": source_identity_before.get(
                "identity_sha256"
            ),
            "source_identity_unchanged_during_run": (
                source_identity_unchanged
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": packages,
        },
        "engine": {
            "label": args.engine,
            "version": engine_version,
            "version_source": "imported_package",
        },
        "gpu": gpu,
        "gpu_memory_monitor": {
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": monitor.device,
            "device_selector_source": monitor.device_selector_source,
            "sample_interval_s": monitor.interval_s,
            "caveat": (
                "includes every process on the selected physical GPU; baseline "
                "and peak are not process-attributed"
            ),
        },
    }


def build_payload(
    args,
    rows: list[dict[str, Any]],
    engine_info: dict[str, Any],
    monitor: VramMonitor,
    *,
    source_identity_before: dict[str, Any] | None = None,
    model_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_identity_after = source_worktree_identity(ROOT)
    gpu_memory = monitor.result()
    for row in rows:
        row["peak_engine_delta_gib"] = round_or_none(monitor.engine_peak_delta_gib)
        row["baseline_gpu_mib"] = monitor.baseline_mib
        row["peak_gpu_mib"] = monitor.peak_mib
        row["gpu_memory"] = dict(gpu_memory)
        row["green"] = row_green(row, args)
    output_fingerprints = generated_output_fingerprint_summary(rows)
    return {
        "schema": "wkvm.incumbent_gemma_bench.v1",
        "engine": args.engine,
        "context_tokens_per_session": args.ctx,
        "prompt_lengths_mode": args.prompt_lengths,
        "decode_tokens_per_session": args.out,
        "mem_cap_gib": args.mem_cap_gib,
        "headroom_gib": args.headroom_gib,
        "max_baseline_gpu_used_gib": getattr(
            args,
            "max_baseline_gpu_used_gib",
            1.0,
        ),
        "model_path": args.model_path,
        "model_identity": model_identity,
        "prompt_token_source": prompt_token_source(args),
        "uses_hf_tokenizer": uses_hf_tokenizer(args),
        "dtype": "bfloat16",
        "git_commit": git_commit(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "provenance": build_provenance(
            args,
            engine_info,
            monitor,
            source_identity_before=source_identity_before,
            source_identity_after=source_identity_after,
        ),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "generated_output_fingerprint_schema": (
            GENERATED_OUTPUT_FINGERPRINT_SCHEMA
        ),
        "generated_output_fingerprint_coverage": {
            key: output_fingerprints[key]
            for key in (
                "successful_rows",
                "fingerprinted_successful_rows",
                "complete",
            )
        },
        "generated_output_fingerprints_by_batch": output_fingerprints[
            "by_batch"
        ],
        "engine_config": engine_info,
        "memory": {
            "baseline_gpu_mib": monitor.baseline_mib,
            "peak_gpu_mib": monitor.peak_mib,
            "peak_engine_delta_gib": round_or_none(monitor.engine_peak_delta_gib),
            "gpu_memory": gpu_memory,
            "green_note": (
                "green uses peak_engine_delta_gib when available; this is "
                "whole-GPU peak minus whole-GPU baseline sampled before engine load"
            ),
        },
        "summary": {
            "bmax_green": max((r["B"] for r in rows if r.get("green")), default=0),
            "max_success_B": max(
                (r["B"] for r in rows if r.get("success_count") == r["B"]),
                default=0,
            ),
            "best_green_agg_decode_tok_s": max(
                (r.get("agg_decode_tok_s") or 0.0 for r in rows if r.get("green")),
                default=0.0,
            ),
        },
        "rows": rows,
    }


def run(args) -> dict[str, Any]:
    args.model_path = resolve_model_path(args.model_path)
    source_identity_before = source_worktree_identity(ROOT)
    model_identity = model_checkpoint_identity(args.model_path)
    if getattr(args, "synthetic_prompts", False):
        tok = SyntheticBenchTokenizer(vocab_size=args.synthetic_vocab_size)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts_by_b = {
        B: [
            build_prompt(tok, n, row)
            for row, n in enumerate(bench_prompt_lengths(args.ctx, B, args.prompt_lengths))
        ]
        for B in args.concurrency
    }

    runners = {
        "vllm": run_vllm,
        "sglang": run_sglang,
    }
    with VramMonitor(
        interval_s=args.mem_sample_interval_s,
        device=getattr(args, "gpu_memory_device", None),
    ) as monitor:
        try:
            rows, engine_info = runners[args.engine](args, tok, prompts_by_b, monitor)
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            rows = []
            for B, prompts in prompts_by_b.items():
                rows.append(
                    {
                        "B": B,
                        "success_count": 0,
                        "error_count": B,
                        "error": setup_error,
                        "prompt_lengths": [len(prompt) for prompt in prompts],
                        "green": False,
                        **prompt_fingerprint_row_fields(
                            prompt_set_fingerprint(
                                prompts,
                                prompt_token_source=prompt_token_source(args),
                            )
                        ),
                    }
                )
            engine_info = {"setup_error": setup_error}
    payload = build_payload(
        args,
        rows,
        engine_info,
        monitor,
        source_identity_before=source_identity_before,
        model_identity=model_identity,
    )
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency("1,2,4,8"))
    ap.add_argument("--prompt-lengths", choices=["staggered", "uniform"], default="staggered")
    ap.add_argument(
        "--synthetic-prompts",
        action="store_true",
        help=(
            "Generate deterministic prompt token IDs locally instead of loading "
            "a Hugging Face tokenizer. The prompt IDs are passed directly to "
            "the incumbent engine."
        ),
    )
    ap.add_argument(
        "--synthetic-vocab-size",
        type=int,
        default=262_144,
        help="Vocabulary size used by --synthetic-prompts.",
    )
    ap.add_argument("--mem-cap-gib", type=float, default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)))
    ap.add_argument("--headroom-gib", type=float, default=1.0)
    ap.add_argument(
        "--max-baseline-gpu-used-gib",
        type=float,
        default=1.0,
        help="Reject report evidence when pre-load whole-GPU use exceeds this idle ceiling.",
    )
    ap.add_argument("--mem-sample-interval-s", type=float, default=0.1)
    ap.add_argument(
        "--gpu-memory-device",
        "--monitor-gpu",
        "--gpu-index",
        default=None,
        help=(
            "Physical GPU index or UUID used by nvidia-smi memory sampling. "
            "Required on ambiguous multi-GPU hosts; a single CUDA_VISIBLE_DEVICES "
            "value is used when present."
        ),
    )
    ap.add_argument("--telemetry-sample-interval-s", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stop-on-failure", action="store_true")

    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.82)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument(
        "--vllm-max-num-batched-tokens",
        type=parse_positive_int,
        default=None,
        help=(
            "Explicit vLLM scheduler token budget. Set this high enough for the "
            "intended long-prefill cohort; the exact value is recorded."
        ),
    )
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--vllm-language-model-only", action="store_true")
    ap.add_argument(
        "--vllm-disable-inductor",
        action="store_true",
        help="Disable Inductor compilation while retaining full CUDA graphs.",
    )

    ap.add_argument("--sglang-mem-fraction", type=float, default=0.88)
    ap.add_argument("--sglang-context-length", type=int, default=None)
    ap.add_argument("--sglang-max-total-tokens", type=int, default=None)
    ap.add_argument(
        "--sglang-chunked-prefill-size",
        type=parse_positive_int,
        default=None,
        help=(
            "Explicit SGLang chunked_prefill_size passed to Engine and recorded."
        ),
    )
    ap.add_argument("--sglang-attention-backend", default="triton")
    ap.add_argument("--sglang-language-model-only", action="store_true")
    ap.add_argument("--sglang-max-running-requests", type=int, default=64)
    ap.add_argument("--sglang-decode-graph", default="full")
    ap.add_argument("--sglang-prefill-graph", default="disabled")
    ap.add_argument("--sglang-log-level", default="warning")
    return ap


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
