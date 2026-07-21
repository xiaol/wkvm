from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "scripts" / "run_a800_incumbent_profile_sweep.sh"
BENCHMARK = ROOT / "experiments" / "gemma_multiturn_http_bench.py"
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))


def write_shared_trace(
    path: Path,
    *,
    sessions: int = 1,
    turns: int = 1,
    initial_context_tokens: int = 16,
    turn_input_tokens: int = 4,
    output_tokens_per_turn: int = 2,
    source: dict[str, object] | None = None,
):
    from gemma_multiturn_bench import (
        build_shared_history_trace,
        build_workload,
        shared_history_trace_payload,
    )

    workload = build_workload(
        sessions=sessions,
        turns=turns,
        initial_context_tokens=initial_context_tokens,
        turn_input_tokens=turn_input_tokens,
        vocab_size=262_144,
    )
    outputs = [
        [
            [4 + session_index + token_index for token_index in range(output_tokens_per_turn)]
            for session_index in range(sessions)
        ]
        for _ in range(turns)
    ]
    trace = build_shared_history_trace(
        workload,
        outputs,
        sessions=sessions,
        turns=turns,
        output_tokens_per_turn=output_tokens_per_turn,
        vocab_size=262_144,
        source_path=str(path),
        source=source,
    )
    path.write_text(json.dumps(shared_history_trace_payload(trace)) + "\n")
    return trace


def metadata(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split() if "=" in field)


def option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def parser_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }


class TestA800IncumbentProfileSweep(unittest.TestCase):
    maxDiff = None

    def dry_run(
        self,
        *,
        base: Path,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        model = base / "model"
        model.mkdir()
        env = os.environ.copy()
        env.update(
            {
                "CAMPAIGN_ID": "test-a800-incumbent-sweep",
                "DRY_RUN": "1",
                "GPU_DEVICES": "4,5",
                "MODEL_PATH": str(model),
                "OUT_DIR": str(base / "results"),
                "PORTS": "8310,8311",
                "SGLANG_CHUNKED_PREFILL_SIZE_LIST": "8192",
                "SGLANG_MEM_FRACTION_STATIC_LIST": "0.92",
                "SGLANG_PY": sys.executable,
                "SGLANG_VERSION": "0.5.15.post1",
                "TRACE_JSON": str(base / "shared.trace.json"),
                "VLLM_GPU_MEMORY_UTILIZATION_LIST": "0.92",
                "VLLM_MAX_NUM_BATCHED_TOKENS_LIST": "16384",
                "VLLM_PY": sys.executable,
                "VLLM_VERSION": "0.25.1",
                "WKVM_PY": sys.executable,
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(RUNNER)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def fake_nvidia_smi(self, directory: Path, *, occupied: bool) -> Path:
        fake = directory / "nvidia-smi"
        compute_result = (
            'printf "%s, unrelated-python\\n" "${OCCUPIED_PID:?}"\n  exit 0\n'
            if occupied
            else "exit 0\n"
        )
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$*\" >>\"${NVIDIA_SMI_LOG:?}\"\n"
            "if [[ \"$*\" == *\"--query-compute-apps=pid,process_name\"* ]]; then\n"
            f"  {compute_result}"
            "fi\n"
            "if [[ \"$*\" == *\"--query-gpu=name,uuid,driver_version,memory.total,memory.used\"* ]]; then\n"
            "  printf '%s\\n' 'NVIDIA A800-SXM4-80GB, GPU-test-a800, 570.1, 81920, 0'\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"$*\" == *\"--query-gpu=memory.used\"* ]]; then\n"
            "  printf '%s\\n' 0\n"
            "  exit 0\n"
            "fi\n"
            "exit 1\n"
        )
        fake.chmod(0o755)
        return fake

    def fake_engine_python(
        self,
        directory: Path,
        *,
        name: str,
        module: str,
        version: str,
    ) -> Path:
        fake = directory / name
        probe = f"import {module}; print({module}.__version__)"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"if [[ \"${{1:-}}\" == -c && \"${{2:-}}\" == {shlex.quote(probe)} ]]; then\n"
            f"  printf '%s\\n' {shlex.quote(version)}\n"
            "  exit 0\n"
            "fi\n"
            f"exec {shlex.quote(sys.executable)} \"$@\"\n"
        )
        fake.chmod(0o755)
        return fake

    def preflight(
        self,
        *,
        base: Path,
        fake_bin: Path,
        occupied_pid: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        model = base / "model"
        model.mkdir()
        trace = base / "shared.trace.json"
        write_shared_trace(trace)
        vllm_python = self.fake_engine_python(
            fake_bin,
            name="vllm-python",
            module="vllm",
            version="0.25.1",
        )
        sglang_python = self.fake_engine_python(
            fake_bin,
            name="sglang-python",
            module="sglang",
            version="0.5.15.post1",
        )
        env = os.environ.copy()
        env.update(
            {
                "CAMPAIGN_ID": "test-a800-incumbent-preflight",
                "DRY_RUN": "0",
                "GPU_DEVICES": "4",
                "INITIAL_CONTEXT_TOKENS": "16",
                "MODEL_PATH": str(model),
                "NVIDIA_SMI_LOG": str(base / "nvidia-smi.log"),
                "OUT_DIR": str(base / "results"),
                "PATH": f"{fake_bin}:{env['PATH']}",
                "PORTS": "8310",
                "PREFLIGHT_ONLY": "1",
                "PROFILE_BASES": "vllm-inductor-full-and-piecewise",
                "SESSIONS": "1",
                "SGLANG_MAX_RUNNING_REQUESTS": "1",
                "SGLANG_PY": str(sglang_python),
                "SGLANG_VERSION": "0.5.15.post1",
                "TRACE_JSON": str(trace),
                "TURNS": "1",
                "TURN_INPUT_TOKENS": "4",
                "OUTPUT_TOKENS_PER_TURN": "2",
                "VLLM_PY": str(vllm_python),
                "VLLM_VERSION": "0.25.1",
                "WKVM_PY": sys.executable,
            }
        )
        if occupied_pid is not None:
            env["OCCUPIED_PID"] = str(occupied_pid)
        return subprocess.run(
            ["bash", str(RUNNER)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_default_dry_run_preserves_incumbent_optimizations_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result = self.dry_run(base=base)

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        starts = [
            index for index, line in enumerate(lines) if line.startswith("profile start ")
        ]
        server_starts = [
            index for index, line in enumerate(lines) if line.startswith("server start ")
        ]
        client_starts = [
            index for index, line in enumerate(lines) if line.startswith("run profile=")
        ]
        self.assertEqual(len(starts), 9)
        self.assertEqual(len(server_starts), 9)
        self.assertEqual(len(client_starts), 9)
        self.assertEqual(
            [metadata(lines[index])["gpu"] for index in starts],
            ["4", "5"] * 4 + ["4"],
        )
        self.assertEqual(
            len(
                [
                    line
                    for line in lines
                    if line.endswith("mode=parallel-different-gpus")
                ]
            ),
            5,
        )

        server_commands = {
            metadata(lines[index])["profile"]: shlex.split(lines[index + 1])
            for index in server_starts
        }
        client_commands = {
            metadata(lines[index])["profile"]: shlex.split(lines[index + 1])
            for index in client_starts
        }
        self.assertEqual(set(server_commands), set(client_commands))
        self.assertEqual(len(server_commands), 9)

        supported_options = parser_options(BENCHMARK)
        artifacts: set[str] = set()
        for profile, command in client_commands.items():
            benchmark_index = command.index(str(BENCHMARK))
            arguments = command[benchmark_index + 1 :]
            used_options = {
                value for value in arguments if value.startswith("--")
            }
            self.assertEqual(used_options - supported_options, set())
            self.assertEqual(option_value(arguments, "--sessions"), "32")
            self.assertEqual(option_value(arguments, "--turns"), "4")
            self.assertEqual(
                option_value(arguments, "--initial-context-tokens"), "98304"
            )
            self.assertEqual(option_value(arguments, "--turn-input-tokens"), "32")
            self.assertEqual(
                option_value(arguments, "--output-tokens-per-turn"), "32"
            )
            self.assertEqual(option_value(arguments, "--request-timeout-s"), "3600")
            self.assertEqual(option_value(arguments, "--semantics"), "full_kv")
            if profile.startswith("vllm-v2-native-"):
                self.assertNotIn("--shared-history-trace-json", arguments)
                self.assertIn("--write-shared-history-trace-json", arguments)
                self.assertEqual(
                    option_value(arguments, "--teacher-forcing-field"), "none"
                )
            else:
                self.assertEqual(
                    option_value(arguments, "--shared-history-trace-json"),
                    str(base / "results" / "shared_history_trace.json"),
                )
                self.assertNotIn("--write-shared-history-trace-json", arguments)
            artifact = option_value(arguments, "--json")
            self.assertNotIn(artifact, artifacts)
            artifacts.add(artifact)

            config = json.loads(
                option_value(arguments, "--target-server-config-json")
            )
            self.assertEqual(config["profile_id"], profile)
            self.assertEqual(config["dtype"], "bfloat16")
            if profile.startswith("vllm-"):
                self.assertEqual(option_value(arguments, "--engine"), "vllm")
                self.assertNotIn("--teacher-forcing-processor", arguments)
                self.assertTrue(config["enable_chunked_prefill"])
                self.assertTrue(config["enable_prefix_caching"])
                self.assertTrue(config["language_model_only"])
                self.assertEqual(config["max_num_seqs"], 32)
                self.assertEqual(config["max_num_batched_tokens"], 16384)
                self.assertEqual(config["gpu_memory_utilization"], 0.92)
                if profile.startswith("vllm-v2-native-"):
                    self.assertFalse(config["kv_sharing_fast_prefill"])
                    self.assertTrue(config["use_v2_model_runner"])
                    self.assertEqual(config["model_runner_generation"], "v2")
                else:
                    self.assertTrue(config["kv_sharing_fast_prefill"])
                    self.assertFalse(config["use_v2_model_runner"])
                    self.assertEqual(config["model_runner_generation"], "v1")
            else:
                self.assertEqual(option_value(arguments, "--engine"), "sglang")
                self.assertIn("--teacher-forcing-processor", arguments)
                self.assertNotIn("--sglang-native-generate", arguments)
                self.assertFalse(config["disable_radix_cache"])
                self.assertFalse(config["disable_chunked_prefix_cache"])
                self.assertFalse(config["disable_overlap_schedule"])
                self.assertFalse(config["disable_cuda_graph"])
                self.assertFalse(config["disable_decode_cuda_graph"])
                self.assertFalse(config["disable_prefill_cuda_graph"])
                self.assertTrue(config["enable_custom_logit_processor"])
                self.assertEqual(config["cuda_graph_backend_decode"], "full")
                self.assertEqual(config["chunked_prefill_size"], 8192)
                self.assertEqual(config["mem_fraction_static"], 0.92)
                self.assertFalse(config["enable_torch_compile"])
                self.assertFalse(config["enable_two_batch_overlap"])
                self.assertFalse(config["enable_single_batch_overlap"])
                self.assertEqual(
                    set(config["untested_supported_optimizations"]),
                    {
                        "enable_torch_compile",
                        "enable_two_batch_overlap",
                        "enable_single_batch_overlap",
                    },
                )

        candidate = next(
            profile
            for profile in server_commands
            if profile.startswith("vllm-inductor-full-and-piecewise-")
        )
        control = next(
            profile
            for profile in server_commands
            if profile.startswith("vllm-mode0-control-")
        )
        native_v2 = next(
            profile
            for profile in server_commands
            if profile.startswith("vllm-v2-native-")
        )
        for profile, expected_mode, expected_graph in (
            (candidate, 3, "FULL_AND_PIECEWISE"),
            (control, 0, "FULL_DECODE_ONLY"),
        ):
            command = server_commands[profile]
            for flag in (
                "--enable-chunked-prefill",
                "--enable-prefix-caching",
                "--kv-sharing-fast-prefill",
                "--language-model-only",
            ):
                self.assertIn(flag, command)
            self.assertEqual(option_value(command, "--max-num-seqs"), "32")
            self.assertEqual(
                option_value(command, "--max-num-batched-tokens"), "16384"
            )
            self.assertEqual(
                option_value(command, "--gpu-memory-utilization"), "0.92"
            )
            compilation = json.loads(option_value(command, "--compilation-config"))
            self.assertEqual(compilation["mode"], expected_mode)
            self.assertEqual(compilation["cudagraph_mode"], expected_graph)
            self.assertEqual(compilation["cudagraph_capture_sizes"], [1, 2, 4, 8, 16, 32])

        native_command = server_commands[native_v2]
        for flag in (
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--language-model-only",
        ):
            self.assertIn(flag, native_command)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", native_command)
        self.assertNotIn("--kv-sharing-fast-prefill", native_command)
        self.assertNotIn("--logits-processors", native_command)
        native_compilation = json.loads(
            option_value(native_command, "--compilation-config")
        )
        self.assertEqual(native_compilation["mode"], 3)
        self.assertEqual(
            native_compilation["cudagraph_mode"], "FULL_AND_PIECEWISE"
        )

        sglang_programs = [
            option_value(command, "-c")
            for profile, command in server_commands.items()
            if profile.startswith("sglang-")
        ]
        self.assertTrue(sglang_programs)
        for program in sglang_programs:
            self.assertIn("disable_radix_cache=False", program)
            self.assertIn("disable_chunked_prefix_cache=False", program)
            self.assertIn("disable_overlap_schedule=False", program)
            self.assertIn("enable_custom_logit_processor=True", program)
            self.assertIn('backend == "auto"', program)
            self.assertIn('"attention_backend":backend', program)

        sglang_configs = [
            json.loads(
                option_value(
                    client_commands[profile], "--target-server-config-json"
                )
            )
            for profile in client_commands
            if profile.startswith("sglang-")
        ]
        self.assertEqual(
            {config["attention_backend_requested"] for config in sglang_configs},
            {"auto", "triton"},
        )
        self.assertEqual(
            {config["cuda_graph_backend_prefill"] for config in sglang_configs},
            {"breakable", "tc_piecewise", "disabled"},
        )

    def test_token_and_memory_lists_expand_a_bounded_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.dry_run(
                base=Path(temporary),
                extra_env={
                    "GPU_DEVICES": "4",
                    "MAX_PROFILE_RUNS": "8",
                    "PORTS": "8310",
                    "PROFILE_BASES": (
                        "vllm-inductor-full-and-piecewise,vllm-mode0-control"
                    ),
                    "VLLM_GPU_MEMORY_UTILIZATION_LIST": "0.90,0.92",
                    "VLLM_MAX_NUM_BATCHED_TOKENS_LIST": "8192,16384",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        starts = [
            metadata(line)
            for line in lines
            if line.startswith("profile start ")
        ]
        self.assertEqual(len(starts), 8)
        self.assertEqual(
            {(row["token_chunk"], row["memory_fraction"]) for row in starts},
            {
                ("8192", "0.90"),
                ("8192", "0.92"),
                ("16384", "0.90"),
                ("16384", "0.92"),
            },
        )
        self.assertEqual(len({row["profile"] for row in starts}), 8)

    def test_unknown_profile_and_matrix_overflow_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            unknown = self.dry_run(
                base=Path(temporary),
                extra_env={"PROFILE_BASES": "vllm-imaginary-fast-mode"},
            )
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("Unknown profile base", unknown.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            overflow = self.dry_run(
                base=Path(temporary),
                extra_env={
                    "MAX_PROFILE_RUNS": "1",
                    "PROFILE_BASES": (
                        "vllm-inductor-full-and-piecewise,vllm-mode0-control"
                    ),
                },
            )
        self.assertNotEqual(overflow.returncode, 0)
        self.assertIn("exceeds MAX_PROFILE_RUNS=1", overflow.stderr)

    def test_cpu_only_preflight_records_homogeneous_idle_a800(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fake_bin = base / "bin"
            fake_bin.mkdir()
            self.fake_nvidia_smi(fake_bin, occupied=False)
            result = self.preflight(base=base, fake_bin=fake_bin)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("preflight=passed profiles=1 gpus=1", result.stdout)
            matrix = (base / "results" / "profile_matrix.tsv").read_text()
            pool = (base / "results" / "gpu_pool.tsv").read_text()
            self.assertIn("vllm-inductor-full-and-piecewise", matrix)
            self.assertIn("NVIDIA A800-SXM4-80GB", pool)
            self.assertEqual(
                list((base / "results" / "artifacts").glob("*.json")), []
            )

    def test_occupied_gpu_is_refused_without_terminating_unrelated_pid(self) -> None:
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                fake_bin = base / "bin"
                fake_bin.mkdir()
                self.fake_nvidia_smi(fake_bin, occupied=True)
                result = self.preflight(
                    base=base,
                    fake_bin=fake_bin,
                    occupied_pid=sleeper.pid,
                )

                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("refusing without terminating them", result.stderr)
                self.assertIn(str(sleeper.pid), result.stderr)
                self.assertIsNone(sleeper.poll())
                self.assertFalse((base / "results").exists())
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)


class TestA800IncumbentProfileValidation(unittest.TestCase):
    def vllm_requested(self, profile: str, *, v2: bool) -> dict[str, object]:
        return {
            "VLLM_USE_V2_MODEL_RUNNER": v2,
            "compilation_config": {
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "max_cudagraph_capture_size": 32,
                "mode": 3,
            },
            "custom_logits_processors_enabled": not v2,
            "dtype": "bfloat16",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.92,
            "kv_sharing_fast_prefill": not v2,
            "language_model_only": True,
            "max_model_len": 64,
            "max_num_batched_tokens": 16,
            "max_num_seqs": 1,
            "model_runner_generation": "v2" if v2 else "v1",
            "profile_id": profile,
            "use_v2_model_runner": v2,
        }

    def vllm_server_info(
        self,
        requested: dict[str, object],
        *,
        model_path: Path,
        served_model_name: str,
    ) -> dict[str, object]:
        v2 = bool(requested["use_v2_model_runner"])
        return {
            "vllm_config": {
                "model_config": {
                    "model": str(model_path),
                    "served_model_name": [served_model_name],
                    "dtype": "torch.bfloat16",
                    "max_model_len": requested["max_model_len"],
                    "multimodal_config": {"language_model_only": True},
                    "logits_processors": (
                        []
                        if v2
                        else [
                            "experiments.vllm_shared_history_logits:"
                            "SharedHistoryLogitsProcessor"
                        ]
                    ),
                },
                "cache_config": {
                    "enable_prefix_caching": True,
                    "kv_sharing_fast_prefill": not v2,
                    "gpu_memory_utilization": 0.92,
                },
                "scheduler_config": {
                    "enable_chunked_prefill": True,
                    "max_num_batched_tokens": 16,
                    "max_num_seqs": 1,
                },
                "compilation_config": {
                    "mode": 3,
                    "cudagraph_mode": [2, 1],
                    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
                },
            },
            "vllm_env": {"VLLM_USE_V2_MODEL_RUNNER": v2},
            "system_env": {},
        }

    def test_vllm_server_info_proves_explicit_v2_runner(self) -> None:
        from scripts.a800_incumbent_profile_validation import (
            ValidationError,
            validate_server_info_payload,
        )

        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            requested = self.vllm_requested("vllm-v2-native-test", v2=True)
            payload = self.vllm_server_info(
                requested,
                model_path=model,
                served_model_name="gemma-test",
            )
            validate_server_info_payload(
                payload,
                engine="vllm",
                requested=requested,
                version="0.25.1",
                model_path=str(model),
                served_model_name="gemma-test",
            )
            payload["vllm_env"]["VLLM_USE_V2_MODEL_RUNNER"] = False
            with self.assertRaisesRegex(ValidationError, "VLLM_USE_V2_MODEL_RUNNER"):
                validate_server_info_payload(
                    payload,
                    engine="vllm",
                    requested=requested,
                    version="0.25.1",
                    model_path=str(model),
                    served_model_name="gemma-test",
                )

    def test_sglang_auto_backend_accepts_and_records_actual_resolution(self) -> None:
        from scripts.a800_incumbent_profile_validation import (
            ValidationError,
            validate_server_info_payload,
        )

        requested = {
            "attention_backend_requested": "auto",
            "chunked_prefill_size": 8,
            "context_length": 64,
            "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": "breakable",
            "disable_chunked_prefix_cache": False,
            "disable_cuda_graph": False,
            "disable_decode_cuda_graph": False,
            "disable_overlap_schedule": False,
            "disable_prefill_cuda_graph": False,
            "disable_radix_cache": False,
            "enable_cache_report": True,
            "enable_custom_logit_processor": True,
            "enable_multimodal": False,
            "enable_single_batch_overlap": False,
            "enable_torch_compile": False,
            "enable_two_batch_overlap": False,
            "max_running_requests": 1,
            "max_total_tokens": 128,
            "mem_fraction_static": 0.92,
            "sampling_defaults": "openai",
            "skip_tokenizer_init": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            payload = {
                "version": "0.5.15.post1",
                "model_path": str(model),
                "served_model_name": "gemma-test",
                "dtype": "bfloat16",
                **{
                    field: requested[field]
                    for field in requested
                    if field
                    not in {
                        "attention_backend_requested",
                        "cuda_graph_backend_decode",
                        "cuda_graph_backend_prefill",
                    }
                },
                "attention_backend": "flashinfer",
                "cuda_graph_config": {
                    "decode": {"backend": "full"},
                    "prefill": {"backend": "breakable"},
                },
            }
            validate_server_info_payload(
                payload,
                engine="sglang",
                requested=requested,
                version="0.5.15.post1",
                model_path=str(model),
                served_model_name="gemma-test",
            )
            payload["attention_backend"] = "auto"
            with self.assertRaisesRegex(ValidationError, "must record"):
                validate_server_info_payload(
                    payload,
                    engine="sglang",
                    requested=requested,
                    version="0.5.15.post1",
                    model_path=str(model),
                    served_model_name="gemma-test",
                )

    def test_autonomous_artifact_requires_complete_exact_outputs(self) -> None:
        from scripts.a800_incumbent_profile_validation import (
            ValidationError,
            validate_artifact,
        )
        from gemma_multiturn_bench import shared_history_trace_metadata

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            artifact_path = base / "artifact.json"
            trace_path = base / "trace.json"
            profile = "vllm-v2-native-test"
            run_id = "11111111-1111-4111-8111-111111111111"
            requested = self.vllm_requested(profile, v2=True)
            server_info = self.vllm_server_info(
                requested,
                model_path=model,
                served_model_name="gemma-test",
            )
            trace = write_shared_trace(
                trace_path,
                source={
                    "engine": "vllm",
                    "engine_version": "0.25.1",
                    "model": "gemma-test",
                    "benchmark_artifact": str(artifact_path),
                    "campaign_id": "campaign",
                    "repeat_id": profile,
                    "run_id": run_id,
                },
            )
            artifact = {
                "schema": "wkvm.gemma_multiturn_http_bench.v1",
                "engine": "vllm",
                "engine_version": "0.25.1",
                "semantic_mode": "full_kv",
                "model": "gemma-test",
                "fatal_error": None,
                "benchmark_identity": {
                    "campaign_id": "campaign",
                    "repeat_id": profile,
                    "run_id": run_id,
                    "artifact_role": "http_trace_source",
                },
                "history_trace": {
                    "mode": "engine_generated",
                    "shared": False,
                    "teacher_forced": False,
                },
                "emitted_history_trace": shared_history_trace_metadata(trace),
                "workload": {
                    "sessions": 1,
                    "turns": 1,
                    "initial_context_tokens": 16,
                    "turn_input_tokens": 4,
                    "output_tokens_per_turn": 2,
                    "synchronized_turn_barriers": True,
                },
                "teacher_forcing_hook": {
                    "enabled": False,
                    "processor_present": False,
                },
                "provenance": {
                    "engine": {
                        "label": "vllm",
                        "version": "0.25.1",
                        "version_source": "runtime_import",
                    },
                    "client_environment": {"cuda_visible_devices": "5"},
                    "target_server": {
                        "config": requested,
                        "launch_command": "vllm serve",
                        "launch_profile": "vllm serve MODEL",
                    },
                },
                "gpu_memory": {
                    "enabled": True,
                    "within_memory_ceiling": True,
                    "sample_count": 1,
                },
                "summary": {
                    "all_turns_recorded": True,
                    "completed_turn_rows": 1,
                    "requested_turns": 1,
                    "turn_rows": 1,
                    "request_count": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "output_tokens": 2,
                },
                "turns": [
                    {
                        "turn_index": 0,
                        "request_count": 1,
                        "success_count": 1,
                        "error_count": 0,
                        "output_tokens": 2,
                        "response_output_fingerprint_complete": True,
                        "response_token_ids_observed_count": 1,
                        "teacher_forcing": {
                            "requested": False,
                            "trace_sha256": None,
                            "exact_response_verification_count": 0,
                            "hook_contract_verification_count": 0,
                        },
                        "requests": [
                            {
                                "output_token_ids_observed": True,
                                "output_token_ids_source": "response_token_ids",
                            }
                        ],
                    }
                ],
                "server_metrics_error": None,
                "server_metrics_after_run": server_info,
            }
            artifact_path.write_text(json.dumps(artifact) + "\n")
            kwargs = {
                "engine": "vllm",
                "profile": profile,
                "campaign_id": "campaign",
                "run_id": run_id,
                "trace_mode": "autonomous_source",
                "trace_sha256": trace.trace_sha256,
                "trace_path": str(trace_path),
                "version": "0.25.1",
                "model_path": str(model),
                "served_model_name": "gemma-test",
                "gpu_selector": "5",
                "sessions": 1,
                "turns": 1,
                "initial_context_tokens": 16,
                "turn_input_tokens": 4,
                "output_tokens_per_turn": 2,
                "requested": requested,
            }
            validate_artifact(artifact_path, **kwargs)
            artifact["turns"][0]["response_token_ids_observed_count"] = 0
            artifact_path.write_text(json.dumps(artifact) + "\n")
            with self.assertRaisesRegex(
                ValidationError, "response_token_ids_observed_count"
            ):
                validate_artifact(artifact_path, **kwargs)

    def test_bound_port_and_listener_ownership_checks(self) -> None:
        import socket

        from scripts.a800_incumbent_profile_validation import (
            ValidationError,
            assert_port_unbound,
            prove_listener_owned,
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(ValidationError, "already bound"):
                assert_port_unbound("127.0.0.1", port)
            self.assertIn(os.getpid(), prove_listener_owned(port, os.getpgrp()))


if __name__ == "__main__":
    unittest.main()
