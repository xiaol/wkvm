import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "pyproject.toml").open("rb") as file:
            cls.project = tomllib.load(file)["project"]

    def test_core_remains_dependency_free(self) -> None:
        self.assertEqual(self.project.get("dependencies", []), [])

    def test_gemma_server_extra_declares_direct_runtime_dependencies(self) -> None:
        self.assertEqual(
            set(self.project["optional-dependencies"]["gemma-server"]),
            {
                "torch>=2.6",
                "transformers>=5.7,<6",
                "safetensors>=0.4.3",
                "accelerate>=1.1",
            },
        )

    def test_gemma_server_console_entrypoint(self) -> None:
        self.assertEqual(
            self.project["scripts"]["wkvm-gemma-server"],
            "wkvm.gemma_server:main",
        )

    def test_module_help_does_not_require_serving_extra(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "wkvm.gemma_server", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--native-gemma-checkpoint-loader", result.stdout)
        self.assertIn("--enable-token-pool-attention", result.stdout)


if __name__ == "__main__":
    unittest.main()
