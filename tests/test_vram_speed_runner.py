import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestVramSpeedRunner(unittest.TestCase):
    def test_probe_flags_are_supported_by_native_benchmark(self) -> None:
        runner = (ROOT / "scripts" / "run_wkvm_vram_speed_4090.sh").read_text()
        args_start = runner.index("  local -a args=(")
        args_end = runner.index("\n  )", args_start)
        runner_options = set(
            re.findall(r"(?m)^\s+(--[a-z0-9-]+)(?:\s|$)", runner[args_start:args_end])
        )

        benchmark_tree = ast.parse(
            (ROOT / "experiments" / "native_gemma_bench.py").read_text()
        )
        benchmark_options = {
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

        self.assertEqual(runner_options - benchmark_options, set())

    def test_4090_memory_policy_leaves_four_gib_headroom(self) -> None:
        runner = (ROOT / "scripts" / "run_wkvm_vram_speed_4090.sh").read_text()

        self.assertIn('MEM_CAP_GIB="${MEM_CAP_GIB:-24}"', runner)
        self.assertIn('HEADROOM_GIB="${HEADROOM_GIB:-4}"', runner)


if __name__ == "__main__":
    unittest.main()
