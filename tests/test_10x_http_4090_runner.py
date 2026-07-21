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
RUNNER = ROOT / "scripts" / "run_wkvm_10x_http_4090.sh"
BENCHMARK = ROOT / "experiments" / "gemma_multiturn_http_bench.py"
REPORT = ROOT / "experiments" / "multiturn_http_10x_report.py"


def option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def metadata(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split() if "=" in field)


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


class Test10xHttp4090Runner(unittest.TestCase):
    maxDiff = None

    def dry_run(
        self,
        *,
        out_dir: Path,
        model_path: Path,
        repeats: int = 1,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "ALLOW_FAIL": "0",
                "CAMPAIGN_ID": "test-http-campaign",
                "DRY_RUN": "1",
                "GPU_DEVICE": "7",
                "MODEL_PATH": str(model_path),
                "OUT_DIR": str(out_dir),
                "OUTPUT_TOKENS_PER_TURN": "64",
                "REPEATS": str(repeats),
                "REPORT_CLAIM_SCOPE": "continuation",
                "SGLANG_CHUNKED_PREFILL_SIZE": "2048",
                "SGLANG_CUDA_GRAPH_BACKEND_PREFILL": "disabled",
                "SGLANG_MAX_RUNNING_REQUESTS": "16",
                "SGLANG_PY": sys.executable,
                "STRICT_PUBLICATION": "0",
                "VLLM_COMPILE_MODE": "0",
                "VLLM_CUDAGRAPH_MODE": "",
                "VLLM_KV_SHARING_FAST_PREFILL": "1",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.82",
                "VLLM_PY": sys.executable,
                "TURNS": "8",
                "TURN_INPUT_TOKENS": "32",
                "INITIAL_CONTEXT_TOKENS": "36864",
                "WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL": "1",
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

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_dry_run_freezes_server_order_profiles_and_trace_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                repeats=2,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        server_indexes = [
            index
            for index, line in enumerate(lines)
            if line.startswith("server start engine=")
        ]
        run_indexes = [
            index for index, line in enumerate(lines) if line.startswith("run engine=")
        ]
        expected_order = ["sglang", "wkvm", "vllm"] * 2
        self.assertEqual(
            [metadata(lines[index])["engine"] for index in server_indexes],
            expected_order,
        )
        self.assertEqual(
            [metadata(lines[index])["engine"] for index in run_indexes],
            expected_order,
        )
        self.assertEqual(
            len([line for line in lines if line.startswith("baseline engine=")]),
            6,
        )
        self.assertEqual(
            len([line for line in lines if line.startswith("server-info engine=")]),
            12,
        )

        server_commands: dict[tuple[str, int], list[str]] = {}
        for index in server_indexes:
            item = metadata(lines[index])
            command = shlex.split(lines[index + 1])
            self.assertEqual(command[:2], ["setsid", "env"])
            self.assertIn("CUDA_VISIBLE_DEVICES=7", command)
            server_commands[(item["engine"], int(item["repeat"]))] = command

        sglang_server = server_commands[("sglang", 1)]
        sglang_program = option_value(sglang_server, "-c")
        self.assertIn("ServerArgs", sglang_program)
        self.assertIn("launch_server", sglang_program)
        self.assertIn("enable_multimodal=False", sglang_program)
        self.assertIn("skip_tokenizer_init=True", sglang_program)
        self.assertIn("mem_fraction_static=0.94", sglang_program)
        self.assertIn("chunked_prefill_size=int(sys.argv[5])", sglang_program)
        self.assertIn("max_running_requests=int(sys.argv[6])", sglang_program)
        self.assertIn("cuda_graph_backend_prefill=sys.argv[7]", sglang_program)
        self.assertEqual(
            sglang_server[-5:],
            ["2048", "16", "disabled", "37616", "608000"],
        )

        wkvm_server = server_commands[("wkvm", 1)]
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", wkvm_server)
        for environment_flag in (
            "WKVM_ENABLE_TOKEN_POOL_TRITON=1",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON=1",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON=1",
            "WKVM_TOKEN_POOL_TRITON_STRICT=1",
            "WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY=1",
            "WKVM_TOKEN_POOL_ROUTE_BOUNDARY_BATCH=1",
        ):
            self.assertIn(environment_flag, wkvm_server)
        self.assertEqual(option_value(wkvm_server, "--max-queue"), "64")
        self.assertEqual(option_value(wkvm_server, "--slots"), "16")
        self.assertEqual(
            option_value(wkvm_server, "--max-completed-requests"),
            "144",
        )
        self.assertEqual(option_value(wkvm_server, "--batch-wait-s"), "0.01")
        self.assertEqual(option_value(wkvm_server, "--stream-flush-tokens"), "1")
        self.assertEqual(
            option_value(wkvm_server, "--continuation-prefill-microbatch-rows"),
            "8",
        )
        self.assertEqual(
            option_value(wkvm_server, "--native-gemma-attention-backend"),
            "triton_dense_gqa",
        )
        self.assertIn("--enable-token-session-teacher-forcing", wkvm_server)
        self.assertIn("--native-gemma-kv-sharing-fast-prefill", wkvm_server)
        self.assertNotIn("--persistent-padded-decode-cuda-graph", wkvm_server)

        vllm_server = server_commands[("vllm", 1)]
        self.assertIn("VLLM_SERVER_DEV_MODE=1", vllm_server)
        self.assertEqual(option_value(vllm_server, "--gpu-memory-utilization"), "0.82")
        self.assertEqual(option_value(vllm_server, "--max-model-len"), "37616")
        self.assertEqual(
            option_value(vllm_server, "--max-num-batched-tokens"),
            "4096",
        )
        self.assertIn("--enable-prefix-caching", vllm_server)
        self.assertIn("--kv-sharing-fast-prefill", vllm_server)
        self.assertNotIn("--no-kv-sharing-fast-prefill", vllm_server)
        self.assertEqual(
            json.loads(option_value(vllm_server, "--compilation-config")),
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                "max_cudagraph_capture_size": 16,
            },
        )
        self.assertEqual(
            option_value(vllm_server, "--logits-processors"),
            "experiments.vllm_shared_history_logits:SharedHistoryLogitsProcessor",
        )

        supported_options = parser_options(BENCHMARK)
        client_commands: dict[tuple[str, int], list[str]] = {}
        recorded_configs: dict[tuple[str, int], dict[str, object]] = {}
        run_ids: set[str] = set()
        for index in run_indexes:
            item = metadata(lines[index])
            command = shlex.split(lines[index + 1])
            benchmark_index = command.index(str(BENCHMARK))
            arguments = command[benchmark_index + 1 :]
            used_options = {
                argument for argument in arguments if argument.startswith("--")
            }
            self.assertEqual(used_options - supported_options, set())
            self.assertEqual(option_value(arguments, "--sessions"), "16")
            self.assertEqual(option_value(arguments, "--turns"), "8")
            self.assertEqual(
                option_value(arguments, "--initial-context-tokens"),
                "36864",
            )
            self.assertEqual(option_value(arguments, "--turn-input-tokens"), "32")
            self.assertEqual(
                option_value(arguments, "--output-tokens-per-turn"),
                "64",
            )
            self.assertEqual(
                option_value(arguments, "--gpu-memory-baseline-used-mib"),
                "<prelaunch-nvidia-smi>",
            )
            self.assertEqual(
                option_value(arguments, "--memory-ceiling-mib"),
                "24200",
            )
            self.assertEqual(
                option_value(arguments, "--gpu-memory-sample-interval-s"),
                "0.1",
            )
            self.assertEqual(
                option_value(arguments, "--campaign-id"),
                "test-http-campaign",
            )
            self.assertEqual(
                option_value(arguments, "--repeat-id"),
                f"r{item['repeat']}",
            )
            self.assertIn("setsid env", option_value(arguments, "--target-server-launch-command"))
            run_ids.add(option_value(arguments, "--run-id"))
            client_commands[(item["engine"], int(item["repeat"]))] = arguments
            recorded_configs[(item["engine"], int(item["repeat"]))] = json.loads(
                option_value(arguments, "--target-server-config-json")
            )
        self.assertEqual(len(run_ids), 6)

        self.assertEqual(
            recorded_configs[("sglang", 1)]["cuda_graph_backend_prefill"],
            "disabled",
        )
        self.assertEqual(recorded_configs[("sglang", 1)]["chunked_prefill_size"], 2048)
        self.assertEqual(recorded_configs[("sglang", 1)]["max_running_requests"], 16)
        self.assertEqual(recorded_configs[("sglang", 1)]["context_length"], 37616)
        self.assertEqual(recorded_configs[("sglang", 1)]["max_total_tokens"], 608000)
        self.assertEqual(
            recorded_configs[("wkvm", 1)]["token_pool_max_context_len"],
            37632,
        )
        self.assertIs(
            recorded_configs[("wkvm", 1)][
                "native_gemma_kv_sharing_fast_prefill"
            ],
            True,
        )
        self.assertIs(
            recorded_configs[("vllm", 1)]["kv_sharing_fast_prefill"],
            True,
        )
        self.assertEqual(
            recorded_configs[("vllm", 1)]["compilation_config"]["mode"],
            0,
        )
        self.assertEqual(
            recorded_configs[("vllm", 1)]["compilation_config"],
            json.loads(option_value(vllm_server, "--compilation-config")),
        )
        self.assertEqual(recorded_configs[("vllm", 1)]["max_model_len"], 37616)

        for repeat in (1, 2):
            sglang = client_commands[("sglang", repeat)]
            wkvm = client_commands[("wkvm", repeat)]
            vllm = client_commands[("vllm", repeat)]
            self.assertIn("--sglang-native-generate", sglang)
            self.assertEqual(option_value(sglang, "--endpoint"), "/generate")
            self.assertEqual(
                option_value(sglang, "--teacher-forcing-field"),
                "none",
            )
            trace = option_value(sglang, "--write-shared-history-trace-json")
            self.assertTrue(
                trace.endswith(f"/traces/b16_ctx36864_t8_o64-r{repeat}.trace.json"),
                trace,
            )
            self.assertEqual(option_value(wkvm, "--shared-history-trace-json"), trace)
            self.assertEqual(option_value(vllm, "--shared-history-trace-json"), trace)
            self.assertEqual(option_value(wkvm, "--endpoint"), "/v1/stream")
            self.assertEqual(option_value(vllm, "--endpoint"), "/v1/completions")
            self.assertNotIn("--shared-history-trace-json", sglang)

    def test_dry_run_applies_and_records_incumbent_scout_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                extra_env={
                    "SGLANG_CHUNKED_PREFILL_SIZE": "4096",
                    "SGLANG_CUDA_GRAPH_BACKEND_PREFILL": "tc_piecewise",
                    "SGLANG_MAX_RUNNING_REQUESTS": "8",
                    "VLLM_COMPILE_MODE": "VLLM_COMPILE",
                    "VLLM_CUDAGRAPH_MODE": "piecewise",
                    "VLLM_KV_SHARING_FAST_PREFILL": "0",
                    "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
                    "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
                    "WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL": "0",
                },
            )
            disabled_default = self.dry_run(
                out_dir=base / "results-disabled-default",
                model_path=model,
                extra_env={"VLLM_KV_SHARING_FAST_PREFILL": "0"},
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(disabled_default.returncode, 0, disabled_default.stderr)

        lines = result.stdout.splitlines()
        server_commands: dict[str, list[str]] = {}
        for index, line in enumerate(lines):
            if line.startswith("server start engine="):
                server_commands[metadata(line)["engine"]] = shlex.split(lines[index + 1])

        sglang_server = server_commands["sglang"]
        self.assertEqual(
            sglang_server[-5:],
            ["4096", "8", "tc_piecewise", "37616", "608000"],
        )

        wkvm_server = server_commands["wkvm"]
        self.assertNotIn("--native-gemma-kv-sharing-fast-prefill", wkvm_server)

        vllm_server = server_commands["vllm"]
        self.assertIn("--no-kv-sharing-fast-prefill", vllm_server)
        self.assertNotIn("--kv-sharing-fast-prefill", vllm_server)
        self.assertEqual(option_value(vllm_server, "--max-num-batched-tokens"), "8192")
        self.assertEqual(option_value(vllm_server, "--gpu-memory-utilization"), "0.90")
        self.assertEqual(
            json.loads(option_value(vllm_server, "--compilation-config")),
            {
                "mode": 3,
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                "max_cudagraph_capture_size": 16,
            },
        )

        recorded_configs: dict[str, dict[str, object]] = {}
        for index, line in enumerate(lines):
            if not line.startswith("run engine="):
                continue
            engine = metadata(line)["engine"]
            arguments = shlex.split(lines[index + 1])
            recorded_configs[engine] = json.loads(
                option_value(arguments, "--target-server-config-json")
            )
        self.assertEqual(recorded_configs["sglang"]["chunked_prefill_size"], 4096)
        self.assertEqual(recorded_configs["sglang"]["max_running_requests"], 8)
        self.assertEqual(
            recorded_configs["sglang"]["cuda_graph_backend_prefill"],
            "tc_piecewise",
        )
        self.assertIs(
            recorded_configs["wkvm"]["native_gemma_kv_sharing_fast_prefill"],
            False,
        )
        self.assertIs(recorded_configs["vllm"]["kv_sharing_fast_prefill"], False)
        self.assertEqual(recorded_configs["vllm"]["max_num_batched_tokens"], 8192)
        self.assertEqual(recorded_configs["vllm"]["gpu_memory_utilization"], 0.90)

        disabled_lines = disabled_default.stdout.splitlines()
        disabled_vllm = next(
            shlex.split(disabled_lines[index + 1])
            for index, line in enumerate(disabled_lines)
            if line.startswith("server start engine=vllm ")
        )
        self.assertEqual(
            json.loads(option_value(disabled_vllm, "--compilation-config")),
            {
                "mode": 0,
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                "max_cudagraph_capture_size": 16,
            },
        )

    def test_report_uses_http_artifacts_and_frozen_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                repeats=2,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        report_index = next(
            index for index, line in enumerate(lines) if line.startswith("report artifacts=")
        )
        self.assertIn("artifacts=6", lines[report_index])
        report_command = shlex.split(lines[report_index + 1])
        self.assertEqual(report_command[1], str(REPORT))
        artifacts = [
            argument
            for argument in report_command
            if "/artifacts/" in argument and argument.endswith(".json")
        ]
        self.assertEqual(len(artifacts), 6)
        self.assertFalse(any(argument.endswith(".trace.json") for argument in report_command))
        self.assertEqual(option_value(report_command, "--min-repeats"), "2")
        self.assertEqual(
            option_value(report_command, "--whole-device-memory-ceiling-mib"),
            "24200",
        )
        self.assertEqual(
            option_value(report_command, "--claim-scope"),
            "continuation",
        )
        self.assertNotIn("--allow-fail", report_command)
        self.assertNotIn("--strict-publication", report_command)
        self.assertIn("kind\trepeat\tpath", result.stdout)

    def test_dry_run_parameterizes_long_lived_workload_and_server_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                repeats=3,
                extra_env={
                    "TURNS": "48",
                    "INITIAL_CONTEXT_TOKENS": "36864",
                    "TURN_INPUT_TOKENS": "32",
                    "OUTPUT_TOKENS_PER_TURN": "64",
                    "REPORT_CLAIM_SCOPE": "full-session",
                    "STRICT_PUBLICATION": "1",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        server_commands: dict[str, list[str]] = {}
        client_commands: dict[str, list[str]] = {}
        recorded_configs: dict[str, dict[str, object]] = {}
        for index, line in enumerate(lines):
            if line.startswith("server start engine="):
                server_commands[metadata(line)["engine"]] = shlex.split(
                    lines[index + 1]
                )
            elif line.startswith("run engine="):
                engine = metadata(line)["engine"]
                command = shlex.split(lines[index + 1])
                arguments = command[command.index(str(BENCHMARK)) + 1 :]
                client_commands[engine] = arguments
                recorded_configs[engine] = json.loads(
                    option_value(arguments, "--target-server-config-json")
                )

        report_index = next(
            index for index, line in enumerate(lines) if line.startswith("report artifacts=")
        )
        report_command = shlex.split(lines[report_index + 1])
        self.assertIn("artifacts=9", lines[report_index])
        self.assertEqual(option_value(report_command, "--min-repeats"), "3")
        self.assertEqual(option_value(report_command, "--claim-scope"), "full-session")
        self.assertIn("--strict-publication", report_command)
        self.assertNotIn("--allow-fail", report_command)

        for arguments in client_commands.values():
            self.assertEqual(option_value(arguments, "--turns"), "48")
            self.assertEqual(
                option_value(arguments, "--initial-context-tokens"), "36864"
            )
            self.assertEqual(option_value(arguments, "--turn-input-tokens"), "32")
            self.assertEqual(
                option_value(arguments, "--output-tokens-per-turn"), "64"
            )

        tag = "b16_ctx36864_d32_t48_o64"
        trace = option_value(
            client_commands["sglang"],
            "--write-shared-history-trace-json",
        )
        self.assertTrue(trace.endswith(f"/traces/{tag}-r3.trace.json"), trace)
        self.assertEqual(
            option_value(client_commands["wkvm"], "--shared-history-trace-json"),
            trace,
        )
        self.assertEqual(
            option_value(client_commands["vllm"], "--shared-history-trace-json"),
            trace,
        )
        for engine, role in (
            ("sglang", "sglang-source"),
            ("wkvm", "wkvm-replay"),
            ("vllm", "vllm-replay"),
        ):
            artifact = option_value(client_commands[engine], "--json")
            self.assertTrue(
                artifact.endswith(f"/artifacts/{role}-{tag}-r3.json"),
                artifact,
            )

        # 36,864 + 48*64 + 47*32 = 41,440; server limits retain the
        # frozen 16/32-token headroom and SGLang's per-session reserve.
        self.assertEqual(server_commands["sglang"][-2:], ["41456", "669440"])
        self.assertEqual(recorded_configs["sglang"]["context_length"], 41456)
        self.assertEqual(recorded_configs["sglang"]["max_total_tokens"], 669440)
        self.assertEqual(
            option_value(
                server_commands["wkvm"], "--token-pool-max-context-len"
            ),
            "41472",
        )
        self.assertEqual(
            recorded_configs["wkvm"]["token_pool_max_context_len"],
            41472,
        )
        self.assertEqual(
            option_value(server_commands["vllm"], "--max-model-len"),
            "41456",
        )
        self.assertEqual(recorded_configs["vllm"]["max_model_len"], 41456)

        self.assertIn("# turns=48", result.stdout)
        self.assertIn("# required_model_len=41440", result.stdout)
        self.assertIn("# report_claim_scope=full-session", result.stdout)
        self.assertIn(f"# workload_tag={tag}", result.stdout)
        self.assertIn(
            f"trace\t1\t{base / 'results' / 'traces' / f'{tag}-r1.trace.json'}",
            result.stdout,
        )
        self.assertIn(
            "sglang-source\t1\t"
            f"{base / 'results' / 'artifacts' / f'sglang-source-{tag}-r1.json'}",
            result.stdout,
        )
        report_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("report artifacts=")
        )
        report_command = shlex.split(lines[report_index + 1])
        self.assertEqual(
            option_value(report_command, "--claim-scope"),
            "full-session",
        )

    def test_runner_contains_lock_parallel_guard_and_process_group_cleanup(self) -> None:
        runner = RUNNER.read_text()

        self.assertIn("flock -n 9", runner)
        self.assertIn("--query-compute-apps=pid,process_name", runner)
        self.assertIn("--query-gpu=memory.used", runner)
        self.assertIn("MAX_IDLE_BASELINE_MIB", runner)
        self.assertIn("trap on_exit EXIT", runner)
        self.assertIn('kill -TERM -- "-$pid"', runner)
        self.assertIn('kill -KILL -- "-$pid"', runner)
        self.assertIn("wait_for_gpu_clear", runner)

    def test_invalid_controls_and_duplicate_ports_fail_without_gpu_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            invalid_repeats = self.dry_run(
                out_dir=base / "results-a",
                model_path=model,
                repeats=0,
            )
            duplicate_ports = self.dry_run(
                out_dir=base / "results-b",
                model_path=model,
                extra_env={"SGLANG_PORT": "8000", "WKVM_PORT": "8000"},
            )
            invalid_turns = self.dry_run(
                out_dir=base / "results-c",
                model_path=model,
                extra_env={"TURNS": "0"},
            )
            invalid_context = self.dry_run(
                out_dir=base / "results-d",
                model_path=model,
                extra_env={"INITIAL_CONTEXT_TOKENS": "nope"},
            )
            invalid_delta = self.dry_run(
                out_dir=base / "results-e",
                model_path=model,
                extra_env={"TURN_INPUT_TOKENS": "0"},
            )
            invalid_output = self.dry_run(
                out_dir=base / "results-f",
                model_path=model,
                extra_env={"OUTPUT_TOKENS_PER_TURN": "-1"},
            )
            invalid_scope = self.dry_run(
                out_dir=base / "results-g",
                model_path=model,
                extra_env={"REPORT_CLAIM_SCOPE": "all"},
            )
            invalid_strict = self.dry_run(
                out_dir=base / "results-h",
                model_path=model,
                extra_env={"STRICT_PUBLICATION": "yes"},
            )
            invalid_vllm_utilization = self.dry_run(
                out_dir=base / "results-i",
                model_path=model,
                extra_env={"VLLM_GPU_MEMORY_UTILIZATION": "0"},
            )

        self.assertNotEqual(invalid_repeats.returncode, 0)
        self.assertIn("REPEATS must be an integer >= 1", invalid_repeats.stderr)
        self.assertNotEqual(duplicate_ports.returncode, 0)
        self.assertIn("must be distinct", duplicate_ports.stderr)
        self.assertNotEqual(invalid_turns.returncode, 0)
        self.assertIn("TURNS must be an integer >= 1", invalid_turns.stderr)
        self.assertNotEqual(invalid_context.returncode, 0)
        self.assertIn(
            "INITIAL_CONTEXT_TOKENS must be an integer >= 1",
            invalid_context.stderr,
        )
        self.assertNotEqual(invalid_delta.returncode, 0)
        self.assertIn(
            "TURN_INPUT_TOKENS must be an integer >= 1",
            invalid_delta.stderr,
        )
        self.assertNotEqual(invalid_output.returncode, 0)
        self.assertIn(
            "OUTPUT_TOKENS_PER_TURN must be an integer >= 1",
            invalid_output.stderr,
        )
        self.assertNotEqual(invalid_scope.returncode, 0)
        self.assertIn(
            "REPORT_CLAIM_SCOPE must be continuation or full-session",
            invalid_scope.stderr,
        )
        self.assertNotEqual(invalid_strict.returncode, 0)
        self.assertIn("STRICT_PUBLICATION must be 0 or 1", invalid_strict.stderr)
        self.assertNotEqual(invalid_vllm_utilization.returncode, 0)
        self.assertIn(
            "VLLM_GPU_MEMORY_UTILIZATION must be a number greater than 0 and at most 1",
            invalid_vllm_utilization.stderr,
        )


if __name__ == "__main__":
    unittest.main()
