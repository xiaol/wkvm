from __future__ import annotations

import json
import os
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_wkvm_10x_http_a800.sh"
BENCHMARK = ROOT / "experiments" / "gemma_multiturn_http_bench.py"
REPORT = ROOT / "experiments" / "multiturn_http_10x_report.py"


def option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def metadata(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split() if "=" in field)


class Test10xHttpA800Runner(unittest.TestCase):
    maxDiff = None

    def dry_run(
        self,
        *,
        out_dir: Path,
        model_path: Path,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for name in (
            "REPEATS",
            "REPEAT_GPU_DEVICES",
            "REPEAT_PORTS",
            "SESSIONS",
            "TURNS",
            "INITIAL_CONTEXT_TOKENS",
            "TURN_INPUT_TOKENS",
            "OUTPUT_TOKENS_PER_TURN",
            "SGLANG_CUDA_GRAPH_BACKEND_PREFILL",
            "TRACE_SOURCE_ENGINE",
            "VLLM_COMPILE_MODE",
            "VLLM_CUDAGRAPH_MODE",
            "WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS",
        ):
            env.pop(name, None)
        env.update(
            {
                "ALLOW_FAIL": "0",
                "CAMPAIGN_ID": "test-a800-http-campaign",
                "DRY_RUN": "1",
                "MODEL_PATH": str(model_path),
                "OUT_DIR": str(out_dir),
                "SGLANG_PY": sys.executable,
                "SGLANG_VERSION": "test-sglang-dynamic",
                "VLLM_PY": sys.executable,
                "VLLM_VERSION": "test-vllm-dynamic",
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

    @staticmethod
    def _write_executable(path: Path, source: str) -> None:
        path.write_text(textwrap.dedent(source).lstrip())
        path.chmod(0o755)

    @staticmethod
    def _free_ports(count: int = 3) -> list[int]:
        sockets: list[socket.socket] = []
        try:
            for _ in range(count):
                item = socket.socket()
                item.bind(("127.0.0.1", 0))
                sockets.append(item)
            return [int(item.getsockname()[1]) for item in sockets]
        finally:
            for item in sockets:
                item.close()

    def stubbed_run(
        self,
        *,
        base: Path,
        extra_env: dict[str, str] | None = None,
        timeout: float = 15,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        bin_dir = base / "bin"
        model = base / "model"
        pid_dir = base / "pids"
        lock_dir = base / "locks"
        for directory in (bin_dir, model, pid_dir, lock_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (model / "stub-model.bin").write_text("model-v1\n")
        events = base / "events.log"
        events.touch()

        stub_python = bin_dir / "stub-python"
        self._write_executable(
            stub_python,
            r"""
            #!/usr/bin/env bash
            set -euo pipefail

            record() {
              printf '%s\n' "$*" >>"$STUB_EVENTS"
            }

            option_value() {
              local target="$1"
              shift
              local -a values=("$@")
              local index
              for index in "${!values[@]}"; do
                if [[ "${values[index]}" == "$target" ]]; then
                  printf '%s' "${values[index + 1]}"
                  return 0
                fi
              done
              return 1
            }

            run_server() {
              local kind="$1"
              local port="$2"
              printf '%s\n' "$$" >"$STUB_PID_DIR/server-$kind-$port-$$.pid"
              record "server-start kind=$kind port=$port pid=$$"
              trap 'record "server-term kind=$kind port=$port pid=$$"; exit 0' TERM INT
              while true; do
                sleep 1
              done
            }

            if [[ "${1:-}" == "-c" ]]; then
              code="${2:-}"
              if [[ "$code" == *"import vllm;"* ]]; then
                printf '%s\n' "${STUB_VLLM_VERSION:-0.25.1}"
                exit 0
              fi
              if [[ "$code" == *"import sglang;"* ]]; then
                printf '%s\n' "${STUB_SGLANG_VERSION:-0.5.15}"
                exit 0
              fi
              if [[ "$code" == *"launch_server"* ]]; then
                run_server sglang "$6"
              fi
              exec "$REAL_PYTHON" "$@"
            fi

            if [[ "${1:-}" == "-m" ]]; then
              kind="$2"
              shift 2
              port="$(option_value --port "$@")"
              run_server "$kind" "$port"
            fi

            script="${1:-}"
            shift || true
            case "$script" in
              *gemma_multiturn_http_bench.py)
                engine="$(option_value --engine "$@")"
                repeat="$(option_value --repeat-id "$@")"
                artifact="$(option_value --json "$@")"
                printf '{}\n' >"$artifact"
                trace="$(option_value --write-shared-history-trace-json "$@" || true)"
                if [[ -n "$trace" ]]; then
                  printf '{}\n' >"$trace"
                fi
                printf '%s\n' "$$" >"$STUB_PID_DIR/client-$repeat-$engine-$$.pid"
                record "client-start repeat=$repeat engine=$engine pid=$$"
                if [[ "${STUB_MUTATE_MODEL_REPEAT:-}" == "$repeat" && "$engine" == "sglang" ]]; then
                  if mkdir "$STUB_PID_DIR/model-mutation.lock" 2>/dev/null; then
                    printf 'mutated\n' >>"$MODEL_PATH/stub-model.bin"
                    record "model-mutated repeat=$repeat"
                  fi
                fi
                if [[ "${STUB_FAIL_REPEAT:-}" == "$repeat" ]]; then
                  sleep "${STUB_FAIL_DELAY_S:-0}"
                  record "client-fail repeat=$repeat engine=$engine pid=$$"
                  exit 42
                fi
                if [[ "${STUB_HANG_REPEAT:-}" == "$repeat" ]]; then
                  trap 'record "client-term repeat=$repeat engine=$engine pid=$$"; exit 143' TERM INT
                  while true; do
                    sleep 1
                  done
                fi
                record "client-complete repeat=$repeat engine=$engine pid=$$"
                ;;
              *multiturn_http_10x_report.py)
                markdown="$(option_value --markdown "$@")"
                summary="$(option_value --summary-json "$@")"
                printf '# stub report\n' >"$markdown"
                printf '{}\n' >"$summary"
                record "report-reached pid=$$"
                ;;
              *)
                printf 'Unexpected stub-python invocation: %s %s\n' "$script" "$*" >&2
                exit 64
                ;;
            esac
            """,
        )
        self._write_executable(
            bin_dir / "git",
            """
            #!/usr/bin/env bash
            case " $* " in
              *" rev-parse HEAD "*) printf '%040d\\n' 0 ;;
              *" status "*) exit 0 ;;
              *) exit 64 ;;
            esac
            """,
        )
        self._write_executable(
            bin_dir / "nvidia-smi",
            r"""
            #!/usr/bin/env bash
            printf 'nvidia-smi %s\n' "$*" >>"$STUB_EVENTS"
            gpu=0
            while (($#)); do
              if [[ "$1" == "-i" ]]; then
                gpu="$2"
                break
              fi
              shift
            done
            case " $* " in
              *" --query-compute-apps="*) exit 0 ;;
              *" --query-gpu=memory.used "*) printf '0\n' ;;
              *" --query-gpu=name,uuid,driver_version,memory.total,memory.used "*)
                printf 'NVIDIA A800 80GB PCIe, GPU-stub-%s, 555.42, 81920, 0\n' "$gpu"
                ;;
              *) exit 64 ;;
            esac
            """,
        )
        self._write_executable(
            bin_dir / "curl",
            r"""
            #!/usr/bin/env bash
            output=""
            while (($#)); do
              if [[ "$1" == "-o" ]]; then
                output="$2"
                break
              fi
              shift
            done
            sleep 0.05
            if [[ -n "$output" ]]; then
              printf '{}\n' >"$output"
            fi
            """,
        )

        ports = self._free_ports()
        env = os.environ.copy()
        env.update(
            {
                "ALLOW_FAIL": "0",
                "CAMPAIGN_ID": "stub-a800-campaign",
                "DRY_RUN": "0",
                "GPU_CLEAR_TIMEOUT_S": "2",
                "GPU_LOCK_DIR": str(lock_dir),
                "HEALTH_POLL_INTERVAL_S": "1",
                "MODEL_PATH": str(model),
                "OUT_DIR": str(base / "results"),
                "PATH": f"{bin_dir}:{env['PATH']}",
                "REAL_PYTHON": sys.executable,
                "REPEATS": "3",
                "REPEAT_GPU_DEVICES": "4,5,6",
                "REPEAT_PORTS": ",".join(str(port) for port in ports),
                "SERVER_READY_TIMEOUT_S": "3",
                "SERVER_STOP_TIMEOUT_S": "2",
                "SGLANG_PY": str(stub_python),
                "SGLANG_VERSION": "0.5.15",
                "STUB_EVENTS": str(events),
                "STUB_PID_DIR": str(pid_dir),
                "STUB_SGLANG_VERSION": "0.5.15",
                "STUB_VLLM_VERSION": "0.25.1",
                "VLLM_PY": str(stub_python),
                "VLLM_VERSION": "0.25.1",
                "WKVM_PY": str(stub_python),
                "WORKER_KILL_TIMEOUT_S": "2",
                "WORKER_TERM_TIMEOUT_S": "4",
            }
        )
        if extra_env:
            env.update(extra_env)
        started = time.monotonic()
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return result, time.monotonic() - started

    @staticmethod
    def commands(
        lines: list[str], prefix: str
    ) -> list[tuple[dict[str, str], list[str]]]:
        return [
            (metadata(line), shlex.split(lines[index + 1]))
            for index, line in enumerate(lines)
            if line.startswith(prefix)
        ]

    def test_shell_syntax_and_executable_mode(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        self.assertTrue(RUNNER.stat().st_mode & stat.S_IXUSR)

    def test_default_dry_run_freezes_parallel_cohorts_and_fair_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(out_dir=base / "results", model_path=model)

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        workers = [
            metadata(line)
            for line in lines
            if line.startswith("worker launch repeat=")
        ]
        self.assertEqual(
            [(item["repeat"], item["gpu"], item["port"]) for item in workers],
            [("r1", "4", "8210"), ("r2", "5", "8211"), ("r3", "6", "8212")],
        )

        servers = self.commands(lines, "server start engine=")
        clients = self.commands(lines, "run engine=")
        expected_order = ["sglang", "wkvm", "vllm"] * 3
        self.assertEqual([item[0]["engine"] for item in servers], expected_order)
        self.assertEqual([item[0]["engine"] for item in clients], expected_order)

        server_commands = {
            (item["engine"], item["repeat"]): command for item, command in servers
        }
        client_arguments: dict[tuple[str, str], list[str]] = {}
        configs: dict[tuple[str, str], dict[str, object]] = {}
        profiles: dict[tuple[str, str], str] = {}
        versions: dict[tuple[str, str], str] = {}
        for item, command in clients:
            arguments = command[command.index(str(BENCHMARK)) + 1 :]
            key = (item["engine"], item["repeat"])
            client_arguments[key] = arguments
            configs[key] = json.loads(
                option_value(arguments, "--target-server-config-json")
            )
            profiles[key] = option_value(arguments, "--target-server-launch-profile")
            versions[key] = option_value(arguments, "--engine-version")
            self.assertEqual(option_value(arguments, "--sessions"), "32")
            self.assertEqual(option_value(arguments, "--turns"), "24")
            self.assertEqual(option_value(arguments, "--initial-context-tokens"), "98304")
            self.assertEqual(option_value(arguments, "--turn-input-tokens"), "32")
            self.assertEqual(option_value(arguments, "--output-tokens-per-turn"), "32")

        for repeat in ("r1", "r2", "r3"):
            source = client_arguments[("sglang", repeat)]
            trace = option_value(source, "--write-shared-history-trace-json")
            self.assertEqual(
                option_value(client_arguments[("wkvm", repeat)], "--shared-history-trace-json"),
                trace,
            )
            self.assertEqual(
                option_value(client_arguments[("vllm", repeat)], "--shared-history-trace-json"),
                trace,
            )

        sglang = server_commands[("sglang", "r1")]
        self.assertEqual(sglang[-5], "breakable")
        self.assertEqual(sglang[-7], "8192")
        self.assertEqual(sglang[-6], "32")
        wkvm = server_commands[("wkvm", "r1")]
        self.assertEqual(option_value(wkvm, "--continuation-prefill-microbatch-rows"), "32")
        vllm = server_commands[("vllm", "r1")]
        self.assertEqual(option_value(vllm, "--dtype"), "bfloat16")
        self.assertIn("--enable-chunked-prefill", vllm)
        self.assertIn("--enable-prefix-caching", vllm)
        self.assertIn("--kv-sharing-fast-prefill", vllm)
        self.assertEqual(option_value(vllm, "--max-num-batched-tokens"), "16384")
        self.assertEqual(
            json.loads(option_value(vllm, "--compilation-config")),
            {
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "max_cudagraph_capture_size": 32,
                "mode": 3,
            },
        )

        manifest_hashes = {
            config["model_identity"]["manifest_sha256"]
            for config in configs.values()
        }
        self.assertEqual(manifest_hashes, {"0" * 64})
        self.assertEqual(
            configs[("sglang", "r1")]["cuda_graph_backend_prefill"],
            "breakable",
        )
        self.assertEqual(
            configs[("wkvm", "r1")]["continuation_prefill_microbatch_rows"],
            32,
        )
        self.assertEqual(
            configs[("vllm", "r1")]["compilation_config"],
            json.loads(option_value(vllm, "--compilation-config")),
        )
        self.assertIs(configs[("vllm", "r1")]["enable_chunked_prefill"], True)
        self.assertIs(configs[("vllm", "r1")]["enable_prefix_caching"], True)
        self.assertIs(configs[("vllm", "r1")]["kv_sharing_fast_prefill"], True)

        for (engine, _repeat), profile in profiles.items():
            self.assertIn("CUDA_VISIBLE_DEVICES=GPU_DEVICE", profile)
            self.assertIn("PORT", profile)
            self.assertNotRegex(profile, r"CUDA_VISIBLE_DEVICES=[456](?:\s|$)")
            self.assertNotRegex(profile, r"(?:^|\s)821[012](?:\s|$)")
            if engine == "vllm":
                self.assertIn("FULL_AND_PIECEWISE", profile)
        self.assertEqual(
            {value for (engine, _), value in versions.items() if engine == "vllm"},
            {"test-vllm-dynamic"},
        )
        self.assertEqual(
            {value for (engine, _), value in versions.items() if engine == "sglang"},
            {"test-sglang-dynamic"},
        )
        self.assertTrue(
            all(
                value.startswith("git:")
                for (engine, _), value in versions.items()
                if engine == "wkvm"
            )
        )

        report_index = next(
            index for index, line in enumerate(lines) if line.startswith("report artifacts=")
        )
        self.assertIn("artifacts=9", lines[report_index])
        report_command = shlex.split(lines[report_index + 1])
        self.assertEqual(report_command[1], str(REPORT))
        self.assertEqual(
            len([item for item in report_command if "/artifacts/" in item]),
            9,
        )
        self.assertIn("--strict", report_command)
        self.assertEqual(option_value(report_command, "--gpu-policy"), "homogeneous-pool")
        self.assertEqual(option_value(report_command, "--min-repeats"), "3")
        self.assertIn("workload=b32_ctx98304_d32_t24_o32", result.stdout)

    def test_profile_overrides_are_applied_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                extra_env={
                    "SGLANG_CUDA_GRAPH_BACKEND_PREFILL": "tc_piecewise",
                    "VLLM_COMPILE_MODE": "none",
                    "VLLM_CUDAGRAPH_MODE": "full_decode_only",
                    "WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS": "16",
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        servers = {
            item["engine"]: command
            for item, command in self.commands(lines, "server start engine=")
            if item["repeat"] == "r1"
        }
        self.assertEqual(servers["sglang"][-5], "tc_piecewise")
        self.assertEqual(
            option_value(servers["wkvm"], "--continuation-prefill-microbatch-rows"),
            "16",
        )
        self.assertEqual(
            json.loads(option_value(servers["vllm"], "--compilation-config"))["mode"],
            0,
        )
        self.assertEqual(
            json.loads(option_value(servers["vllm"], "--compilation-config"))[
                "cudagraph_mode"
            ],
            "FULL_DECODE_ONLY",
        )

        configs = {}
        for item, command in self.commands(lines, "run engine="):
            if item["repeat"] != "r1":
                continue
            arguments = command[command.index(str(BENCHMARK)) + 1 :]
            configs[item["engine"]] = json.loads(
                option_value(arguments, "--target-server-config-json")
            )
        self.assertEqual(configs["sglang"]["cuda_graph_backend_prefill"], "tc_piecewise")
        self.assertEqual(configs["wkvm"]["continuation_prefill_microbatch_rows"], 16)
        self.assertEqual(configs["vllm"]["compilation_config"]["mode"], 0)
        self.assertEqual(
            configs["vllm"]["compilation_config"]["cudagraph_mode"],
            "FULL_DECODE_ONLY",
        )

    def test_vllm_trace_source_selects_v2_and_sglang_openai_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                extra_env={"TRACE_SOURCE_ENGINE": "vllm"},
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        servers = self.commands(lines, "server start engine=")
        clients = self.commands(lines, "run engine=")
        self.assertEqual(
            [item[0]["engine"] for item in servers],
            ["vllm", "wkvm", "sglang"] * 3,
        )
        self.assertEqual(
            [item[0]["engine"] for item in clients],
            ["vllm", "wkvm", "sglang"] * 3,
        )
        vllm_server = next(command for item, command in servers if item["repeat"] == "r1")
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", vllm_server)
        self.assertIn("--no-kv-sharing-fast-prefill", vllm_server)
        self.assertNotIn("--kv-sharing-fast-prefill", vllm_server)
        self.assertNotIn("--logits-processors", vllm_server)
        self.assertIn("--return-tokens-as-token-ids", vllm_server)

        client_by_key = {}
        configs = {}
        for item, command in clients:
            key = (item["engine"], item["repeat"])
            arguments = command[command.index(str(BENCHMARK)) + 1 :]
            client_by_key[key] = arguments
            configs[key] = json.loads(
                option_value(arguments, "--target-server-config-json")
            )
        vllm_source = client_by_key[("vllm", "r1")]
        sglang_replay = client_by_key[("sglang", "r1")]
        trace = option_value(vllm_source, "--write-shared-history-trace-json")
        self.assertNotIn("--shared-history-trace-json", vllm_source)
        self.assertEqual(
            option_value(client_by_key[("wkvm", "r1")], "--shared-history-trace-json"),
            trace,
        )
        self.assertEqual(
            option_value(sglang_replay, "--shared-history-trace-json"), trace
        )
        self.assertEqual(option_value(sglang_replay, "--endpoint"), "/v1/completions")
        self.assertEqual(
            option_value(sglang_replay, "--teacher-forcing-processor"),
            "@" + str(base / "results" / "sglang_teacher_forcing_processor.txt"),
        )
        self.assertEqual(configs[("vllm", "r1")]["trace_source_engine"], "vllm")
        self.assertEqual(configs[("vllm", "r1")]["model_runner_generation"], "v2")
        self.assertIs(configs[("vllm", "r1")]["use_v2_model_runner"], True)
        self.assertIs(configs[("vllm", "r1")]["kv_sharing_fast_prefill"], False)
        self.assertIs(configs[("vllm", "r1")]["logits_processor_enabled"], False)
        self.assertEqual(configs[("sglang", "r1")]["trace_role"], "replay")
        self.assertIs(configs[("sglang", "r1")]["enable_custom_logit_processor"], True)
        self.assertIn("vllm-source-b32_ctx98304_d32_t24_o32-r1.json", result.stdout)
        self.assertIn("sglang-replay-b32_ctx98304_d32_t24_o32-r1.json", result.stdout)

    def test_invalid_profiles_fail_before_gpu_access(self) -> None:
        cases = (
            (
                {"SGLANG_CUDA_GRAPH_BACKEND_PREFILL": "bogus"},
                "SGLANG_CUDA_GRAPH_BACKEND_PREFILL must be",
            ),
            ({"VLLM_COMPILE_MODE": "9"}, "VLLM_COMPILE_MODE must be"),
            ({"VLLM_CUDAGRAPH_MODE": "bogus"}, "VLLM_CUDAGRAPH_MODE is not"),
            (
                {"WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS": "0"},
                "WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS must be",
            ),
            ({"TRACE_SOURCE_ENGINE": "bogus"}, "TRACE_SOURCE_ENGINE must be"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            for index, (environment, message) in enumerate(cases):
                with self.subTest(environment=environment):
                    result = self.dry_run(
                        out_dir=base / f"results-{index}",
                        model_path=model,
                        extra_env=environment,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("nvidia-smi", result.stdout + result.stderr)

    def test_launch_profile_replaces_only_placement_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model-8210"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results-8210",
                model_path=model,
                extra_env={"SERVED_MODEL_NAME": "gemma-build-8210"},
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        for item, command in self.commands(result.stdout.splitlines(), "run engine="):
            arguments = command[command.index(str(BENCHMARK)) + 1 :]
            profile = option_value(arguments, "--target-server-launch-profile")
            self.assertIn("gemma-build-8210", profile, item)
            self.assertIn("model-8210", profile, item)
            self.assertNotIn("gemma-build-PORT", profile, item)
            self.assertIn("PORT", profile, item)

    def test_stubbed_worker_pool_reaches_report_and_reaps_every_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result, elapsed = self.stubbed_run(base=base)
            events = (base / "events.log").read_text().splitlines()
            pre_manifest = (base / "results" / "model_files.sha256").read_text()
            post_manifest = (
                base / "results" / "model_files.post.sha256"
            ).read_text()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(elapsed, 12)
        self.assertTrue(any(line.startswith("report-reached ") for line in events))
        self.assertIn("model manifest verified sha256=", result.stdout)
        self.assertIn("versions wkvm=git:", result.stdout)
        self.assertIn("vllm=0.25.1 sglang=0.5.15", result.stdout)
        self.assertEqual(pre_manifest, post_manifest)
        reaped = [
            metadata(line)
            for line in result.stdout.splitlines()
            if line.startswith("worker reaped ")
        ]
        self.assertEqual(len(reaped), 3)
        self.assertEqual(reaped[-1]["active"], "0")
        self.assertFalse(
            any(line.startswith("worker cleanup signal=") for line in result.stdout.splitlines())
        )

    def test_stubbed_peer_failure_cleanup_is_bounded_and_skips_reaped_pid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result, elapsed = self.stubbed_run(
                base=base,
                extra_env={
                    "STUB_FAIL_DELAY_S": "3",
                    "STUB_FAIL_REPEAT": "r2",
                    "STUB_HANG_REPEAT": "r3",
                },
                timeout=15,
            )
            events = (base / "events.log").read_text().splitlines()
            recorded_pids = [
                int(path.read_text()) for path in (base / "pids").glob("*.pid")
            ]

        self.assertNotEqual(result.returncode, 0)
        self.assertLess(elapsed, 10)
        self.assertIn("A repeat worker failed", result.stderr)
        self.assertFalse(any(line.startswith("report-reached ") for line in events))
        self.assertTrue(
            any(line.startswith("client-start repeat=r3 ") for line in events),
            events,
        )
        self.assertTrue(
            any("server-term kind=sglang" in line for line in events),
            events,
        )
        active = {
            item["repeat"]: item["pid"]
            for item in (
                metadata(line)
                for line in result.stdout.splitlines()
                if line.startswith("worker active ")
            )
        }
        reaped_zero = {
            item["pid"]
            for item in (
                metadata(line)
                for line in result.stdout.splitlines()
                if line.startswith("worker reaped ")
            )
            if item["status"] == "0"
        }
        cleanup_pids = {
            item["pid"]
            for item in (
                metadata(line)
                for line in result.stdout.splitlines()
                if line.startswith("worker cleanup signal=")
            )
        }
        self.assertIn(active["r1"], reaped_zero)
        self.assertNotIn(active["r1"], cleanup_pids)
        self.assertIn(active["r3"], cleanup_pids)
        for pid in recorded_pids:
            self.assertFalse(Path(f"/proc/{pid}").exists(), f"stub PID still exists: {pid}")

    def test_real_mode_rejects_version_override_before_gpu_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result, _ = self.stubbed_run(
                base=base,
                extra_env={"VLLM_VERSION": "wrong-version"},
            )
            events = (base / "events.log").read_text()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VLLM_VERSION override does not match imported version", result.stderr)
        self.assertNotIn("nvidia-smi ", events)

    def test_real_mode_rejects_bound_port_before_gpu_access(self) -> None:
        bound = socket.socket()
        bound.bind(("127.0.0.1", 0))
        bound_port = int(bound.getsockname()[1])
        other_ports = self._free_ports(2)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                result, _ = self.stubbed_run(
                    base=base,
                    extra_env={
                        "REPEAT_PORTS": ",".join(
                            str(port) for port in (bound_port, *other_ports)
                        )
                    },
                )
                events = (base / "events.log").read_text()
        finally:
            bound.close()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Port is already bound or unavailable", result.stderr)
        self.assertNotIn("nvidia-smi ", events)

    def test_model_manifest_change_blocks_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result, _ = self.stubbed_run(
                base=base,
                extra_env={"STUB_MUTATE_MODEL_REPEAT": "r1"},
            )
            events = (base / "events.log").read_text().splitlines()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Model files changed during campaign", result.stderr)
        self.assertTrue(any(line.startswith("model-mutated ") for line in events))
        self.assertFalse(any(line.startswith("report-reached ") for line in events))

    def test_runner_contains_parallel_locks_and_process_group_cleanup(self) -> None:
        runner = RUNNER.read_text()
        self.assertIn('flock -n "$worker_lock_fd"', runner)
        self.assertIn("--query-compute-apps=pid,process_name", runner)
        self.assertIn("--query-gpu=memory.used", runner)
        self.assertIn("WORKER_PIDS", runner)
        self.assertIn("setsid bash -c", runner)
        self.assertIn("wait -n -p reaped_pid", runner)
        self.assertIn('kill "-$signal" -- "-$worker_pid"', runner)
        self.assertIn('kill -KILL -- "-$pid"', runner)
        self.assertIn("wait_for_gpu_clear", runner)
        self.assertIn("TRACE_SOURCE_ENGINE", runner)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=", runner)
        self.assertIn("--no-kv-sharing-fast-prefill", runner)
        self.assertIn("--teacher-forcing-processor", runner)


if __name__ == "__main__":
    unittest.main()
