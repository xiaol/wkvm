from __future__ import annotations

import ast
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_wkvm_10x_e2e_4090.sh"
BENCHMARK = ROOT / "experiments" / "gemma_multiturn_bench.py"


def option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


class Test10xE2E4090Runner(unittest.TestCase):
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
                "DRY_RUN": "1",
                "GPU_DEVICE": "7",
                "MODEL_PATH": str(model_path),
                "OUT_DIR": str(out_dir),
                "REPEATS": str(repeats),
                "WKVM_PY": sys.executable,
                "VLLM_PY": sys.executable,
                "SGLANG_PY": sys.executable,
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

    def test_dry_run_freezes_exact_eight_turn_shared_trace_cell(self) -> None:
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
        run_indexes = [
            index for index, line in enumerate(lines) if line.startswith("run engine=")
        ]
        metadata = [
            dict(field.split("=", 1) for field in lines[index].split()[1:])
            for index in run_indexes
        ]
        self.assertEqual(
            [item["engine"] for item in metadata],
            ["sglang", "vllm", "wkvm", "sglang", "vllm", "wkvm"],
        )

        benchmark_tree = ast.parse(BENCHMARK.read_text())
        supported_options: set[str] = set()
        for node in ast.walk(benchmark_tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            options = [
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ]
            supported_options.update(options)
            boolean_optional = any(
                keyword.arg == "action"
                and isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "BooleanOptionalAction"
                for keyword in node.keywords
            )
            if boolean_optional:
                supported_options.update(
                    f"--no-{option[2:]}" for option in options
                )
        commands: dict[tuple[str, int], list[str]] = {}
        run_ids: set[str] = set()
        campaign_ids: set[str] = set()
        for index, item in zip(run_indexes, metadata, strict=True):
            command = shlex.split(lines[index + 1])
            self.assertEqual(command[0], "env")
            self.assertIn("CUDA_VISIBLE_DEVICES=7", command)
            benchmark_index = command.index(str(BENCHMARK))
            arguments = command[benchmark_index + 1 :]
            runner_options = {value for value in arguments if value.startswith("--")}
            self.assertEqual(runner_options - supported_options, set())
            self.assertEqual(option_value(arguments, "--sessions"), "16")
            self.assertEqual(option_value(arguments, "--turns"), "8")
            self.assertEqual(
                option_value(arguments, "--initial-context-tokens"), "36864"
            )
            self.assertEqual(option_value(arguments, "--turn-input-tokens"), "32")
            self.assertEqual(
                option_value(arguments, "--output-tokens-per-turn"), "64"
            )
            campaign_ids.add(option_value(arguments, "--campaign-id"))
            self.assertEqual(
                option_value(arguments, "--repeat-id"),
                f"r{item['repeat']}",
            )
            self.assertEqual(
                option_value(arguments, "--memory-ceiling-mib"), "24200"
            )
            run_ids.add(option_value(arguments, "--run-id"))
            commands[(item["engine"], int(item["repeat"]))] = command

        self.assertEqual(len(campaign_ids), 1)
        self.assertEqual(len(run_ids), 6)

        for repeat in (1, 2):
            sglang = commands[("sglang", repeat)]
            vllm = commands[("vllm", repeat)]
            wkvm = commands[("wkvm", repeat)]
            self.assertEqual(option_value(sglang, "--sglang-mem-fraction"), "0.94")
            self.assertEqual(
                option_value(sglang, "--sglang-chunked-prefill-size"), "2048"
            )
            self.assertEqual(option_value(vllm, "--vllm-gpu-mem-util"), "0.82")
            self.assertEqual(
                option_value(vllm, "--vllm-max-num-batched-tokens"), "4096"
            )
            self.assertEqual(
                option_value(wkvm, "--continuation-prefill-microbatch-rows"),
                "8",
            )
            self.assertEqual(
                option_value(wkvm, "--native-gemma-attention-backend"),
                "triton_dense_gqa",
            )
            self.assertIn("--no-persistent-padded-decode-cuda-graph", wkvm)
            self.assertNotIn("--persistent-padded-decode-cuda-graph", wkvm)
            self.assertNotIn("--wkvm-empty-cache-before-decode", wkvm)
            self.assertIn(
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", wkvm
            )
            trace = option_value(sglang, "--write-shared-history-trace-json")
            self.assertEqual(option_value(vllm, "--shared-history-trace-json"), trace)
            self.assertEqual(option_value(wkvm, "--shared-history-trace-json"), trace)
            self.assertNotIn("--shared-history-trace-json", sglang)

    def test_report_uses_only_engine_artifacts_and_frozen_ceiling(self) -> None:
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
        self.assertIn("ceiling_mib=24200", lines[report_index])
        report_command = shlex.split(lines[report_index + 1])
        json_paths = [value for value in report_command if value.endswith(".json")]
        artifacts = [value for value in json_paths if "/artifacts/" in value]
        self.assertEqual(len(artifacts), 6)
        self.assertFalse(any(".trace.json" in value for value in json_paths))
        self.assertEqual(
            option_value(report_command, "--whole-device-memory-ceiling-mib"),
            "24200",
        )
        self.assertEqual(option_value(report_command, "--min-repeats"), "2")
        self.assertIn("--allow-fail", report_command)
        self.assertIn("kind\trepeat\tpath", result.stdout)

    def test_runner_contains_both_parallel_gpu_guards(self) -> None:
        runner = RUNNER.read_text()

        self.assertIn("flock -n 9", runner)
        self.assertIn("--query-compute-apps=pid,process_name", runner)
        self.assertIn("GPU_PROCESS_ALLOWLIST_REGEX", runner)
        self.assertIn("name !~ allow", runner)

    def test_repeats_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                repeats=0,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REPEATS must be an integer >= 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
