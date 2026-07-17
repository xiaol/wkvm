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
RUNNER = ROOT / "scripts" / "run_wkvm_phase3_4090.sh"
BENCHMARK = ROOT / "experiments" / "native_gemma_bench.py"


def option_value(arguments: list[str], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


class TestPhase3Runner(unittest.TestCase):
    maxDiff = None

    def dry_run(
        self,
        *,
        out_dir: Path,
        model_path: Path,
        repeats: int = 3,
        extra_env: dict[str, str] | None = None,
    ):
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "GPU_DEVICE": "7",
                "MODEL_PATH": str(model_path),
                "OUT_DIR": str(out_dir),
                "PYTHON": sys.executable,
                "REPEATS": str(repeats),
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

    def test_dry_run_emits_isolated_interleaved_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(out_dir=base / "results", model_path=model)

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        profile_lines = [line for line in lines if line.startswith("profile=")]
        expected_cycle = [
            "prefill-baseline",
            "prefill-packed",
            "prefill-routed-packets",
            "prefill-native-gqa",
            "prefill-combined",
            "schedule-baseline",
            "schedule-lane8",
        ]
        observed_profiles = [line.split()[0].split("=", 1)[1] for line in profile_lines]
        self.assertEqual(
            {profile for profile in observed_profiles}, set(expected_cycle)
        )
        self.assertEqual(len(observed_profiles), 21)
        self.assertNotEqual(
            observed_profiles[:7], observed_profiles[7:14]
        )
        self.assertNotEqual(
            observed_profiles[7:14], observed_profiles[14:21]
        )
        for offset in (0, 7, 14):
            self.assertEqual(
                set(observed_profiles[offset : offset + 7]),
                set(expected_cycle),
            )
        self.assertIn("report artifacts=21", result.stdout)

    def test_dry_run_profile_options_are_feature_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(out_dir=base / "results", model_path=model)

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        invocations: dict[tuple[str, int], list[str]] = {}
        for index, line in enumerate(lines):
            if not line.startswith("profile="):
                continue
            fields = dict(field.split("=", 1) for field in line.split())
            command = shlex.split(lines[index + 1])
            self.assertEqual(command[0], "CUDA_VISIBLE_DEVICES=7")
            invocations[(fields["profile"], int(fields["repeat"]))] = command[1:]

        benchmark_tree = ast.parse(BENCHMARK.read_text())
        supported_options = {
            argument.value
            for node in ast.walk(benchmark_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        }

        for profile in (
            "prefill-baseline",
            "prefill-packed",
            "prefill-routed-packets",
            "prefill-native-gqa",
            "prefill-combined",
            "schedule-baseline",
            "schedule-lane8",
        ):
            arguments = invocations[(profile, 1)]
            runner_options = {value for value in arguments if value.startswith("--")}
            self.assertEqual(runner_options - supported_options, set())
            self.assertEqual(option_value(arguments, "--ctx"), "16384")
            self.assertEqual(option_value(arguments, "--prompt-lengths"), "uniform")
            self.assertIn("--synthetic-prompts", arguments)
            self.assertIn("--require-native-no-hf", arguments)
            self.assertEqual(option_value(arguments, "--max-baseline-gpu-used-gib"), "1")
            self.assertEqual(option_value(arguments, "--mem-cap-gib"), "24")
            self.assertEqual(option_value(arguments, "--headroom-gib"), "4")
            self.assertEqual(option_value(arguments, "--token-pool-capacity"), "65536")

        expected_prefill = {
            "prefill-baseline": ("sdpa_single_gqa", "separate", False),
            "prefill-packed": ("sdpa_single_gqa", "qkv_gate_up_packed", False),
            "prefill-routed-packets": ("sdpa_single_gqa", "separate", True),
            "prefill-native-gqa": ("triton_dense_gqa", "separate", False),
            "prefill-combined": ("triton_dense_gqa", "qkv_gate_up_packed", True),
        }
        for profile, (attention, projection, routed_packets) in expected_prefill.items():
            arguments = invocations[(profile, 1)]
            self.assertEqual(option_value(arguments, "--concurrency"), "8")
            self.assertEqual(option_value(arguments, "--out"), "1")
            self.assertEqual(option_value(arguments, "--slots"), "8")
            self.assertEqual(option_value(arguments, "--native-gemma-attention-backend"), attention)
            self.assertEqual(option_value(arguments, "--native-gemma-projection-backend"), projection)
            self.assertEqual("--batched-routed-packets" in arguments, routed_packets)
            self.assertNotIn("--completion-prefill-lane-size", arguments)

        for profile in ("schedule-baseline", "schedule-lane8"):
            arguments = invocations[(profile, 1)]
            self.assertEqual(option_value(arguments, "--concurrency"), "16")
            self.assertEqual(option_value(arguments, "--out"), "32")
            self.assertEqual(option_value(arguments, "--slots"), "16")
            self.assertEqual(
                option_value(arguments, "--native-gemma-attention-backend"),
                "sdpa_single_gqa",
            )
            self.assertEqual(
                option_value(arguments, "--native-gemma-projection-backend"),
                "separate",
            )
            self.assertNotIn("--batched-routed-packets", arguments)
        self.assertNotIn(
            "--completion-prefill-lane-size",
            invocations[("schedule-baseline", 1)],
        )
        self.assertEqual(
            option_value(
                invocations[("schedule-lane8", 1)],
                "--completion-prefill-lane-size",
            ),
            "8",
        )

    def test_dry_run_invokes_report_with_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            out_dir = base / "results"
            result = self.dry_run(out_dir=out_dir, model_path=model)

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        report_index = next(
            index for index, line in enumerate(lines) if line.startswith("report artifacts=")
        )
        report_arguments = shlex.split(lines[report_index + 1])
        json_inputs = [value for value in report_arguments if value.endswith(".json")]
        self.assertEqual(len(json_inputs), 22)
        self.assertEqual(len([value for value in json_inputs if value.endswith("summary.json")]), 1)
        artifact_inputs = [value for value in json_inputs if not value.endswith("summary.json")]
        self.assertEqual(len(artifact_inputs), 21)
        self.assertEqual(len(set(artifact_inputs)), 21)
        self.assertIn("--markdown", report_arguments)
        self.assertIn("--summary-json", report_arguments)

    def test_out_dir_must_be_external(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=ROOT / ".phase3-runner-test-output",
                model_path=model,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OUT_DIR must be outside", result.stderr)

    def test_repeats_cannot_be_reduced_below_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                repeats=2,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REPEATS must be an integer >= 3", result.stderr)

    def test_memory_policy_override_fails_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                extra_env={"MEM_CAP_GIB": "20"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires MEM_CAP_GIB=24", result.stderr)

    def test_packet_workspace_override_fails_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model"
            model.mkdir()
            result = self.dry_run(
                out_dir=base / "results",
                model_path=model,
                extra_env={"ROUTED_PACKET_WORKSPACE_BYTES": "1024"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires a 67108864-byte", result.stderr)


if __name__ == "__main__":
    unittest.main()
