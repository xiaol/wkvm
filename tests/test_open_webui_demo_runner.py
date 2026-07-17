from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "open_webui_demo.sh"


class TestOpenWebUIDemoRunner(unittest.TestCase):
    maxDiff = None

    def run_demo(
        self,
        command: str,
        *,
        base: Path,
        model_exists: bool = True,
        dry_run: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        model_dir = base / "model"
        if model_exists:
            model_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1" if dry_run else "0",
                "WKVM_DEMO_HOME": str(base / "demo home"),
                "WKVM_MODEL_DIR": str(model_dir),
                "WKVM_PYTHON": str(base / "wkvm python"),
                "OPEN_WEBUI_BIN": str(base / "open webui"),
                "WKVM_PORT": "18000",
                "OPEN_WEBUI_PORT": "13000",
                "SERVED_MODEL_NAME": "local-wkvm-gemma",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(RUNNER), command],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_install_dry_run_uses_isolated_python_312_environments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result = self.run_demo("install", base=base)

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertIn(["uv", "python", "install", "3.12"], commands)
        self.assertIn(
            [
                "uv",
                "venv",
                "--python",
                "3.12",
                str(base / "demo home" / "wkvm-venv"),
            ],
            commands,
        )
        editable = next(command for command in commands if command[:3] == ["uv", "pip", "install"])
        self.assertEqual(editable[3:5], ["--python", str(base / "wkvm python")])
        self.assertIn("--editable", editable)
        self.assertIn(f"{ROOT}[gemma-server]", editable)
        tool_install = next(
            command for command in commands if command[:3] == ["uv", "tool", "install"]
        )
        self.assertIn("--python", tool_install)
        self.assertIn("3.12", tool_install)
        self.assertIn("--torch-backend", tool_install)
        self.assertIn("cpu", tool_install)
        self.assertIn("--with-executables-from", tool_install)
        self.assertIn("huggingface_hub", tool_install)
        self.assertIn("open-webui==0.10.2", tool_install)
        self.assertFalse((base / "demo home").exists())

    def test_start_dry_run_uses_safe_chat_profile_and_open_webui_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            result = self.run_demo("start", base=base)

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        wkvm_launch = shlex.split(lines[lines.index(next(line for line in lines if line.startswith("launch service=wkvm"))) + 1])
        self.assertEqual(wkvm_launch[0], "env")
        self.assertIn(str(base / "wkvm python"), wkvm_launch)
        self.assertIn("-m", wkvm_launch)
        self.assertIn("wkvm.gemma_server", wkvm_launch)
        expected_flags = {
            "--enable-openai-chat",
            "--native-gemma-production-profile",
            "--slots",
            "--max-chat-sessions",
            "--max-queue",
            "--request-timeout-s",
            "--chat-session-ttl-s",
        }
        self.assertTrue(expected_flags.issubset(wkvm_launch))
        self.assertEqual(wkvm_launch[wkvm_launch.index("--slots") + 1], "4")
        self.assertEqual(wkvm_launch[wkvm_launch.index("--max-chat-sessions") + 1], "4")
        self.assertEqual(wkvm_launch[wkvm_launch.index("--max-queue") + 1], "16")
        self.assertEqual(wkvm_launch[wkvm_launch.index("--request-timeout-s") + 1], "600")
        self.assertEqual(wkvm_launch[wkvm_launch.index("--chat-session-ttl-s") + 1], "1800")
        self.assertNotIn("--ignore-eos", wkvm_launch)

        webui_marker = next(line for line in lines if line.startswith("launch service=open-webui"))
        webui_launch = shlex.split(lines[lines.index(webui_marker) + 1])
        expected_environment = {
            f"DATA_DIR={base / 'demo home' / 'open-webui-data'}",
            "WEBUI_AUTH=true",
            "ENABLE_OLLAMA_API=false",
            "ENABLE_OPENAI_API=true",
            "OPENAI_API_BASE_URLS=http://127.0.0.1:18000/v1",
            "OPENAI_API_KEYS=wkvm-local",
            "ENABLE_FORWARD_USER_INFO_HEADERS=true",
            "ENABLE_WEBSOCKET_SUPPORT=true",
            "ENABLE_PERSISTENT_CONFIG=false",
            "DEFAULT_MODELS=local-wkvm-gemma",
            'DEFAULT_MODEL_PARAMS={"temperature":0,"top_p":1,"function_calling":"legacy"}',
            'DEFAULT_MODEL_METADATA={"capabilities":{"builtin_tools":false,"vision":false,"file_upload":false,"file_context":false,"web_search":false,"image_generation":false,"code_interpreter":false,"terminal":false,"memory":false}}',
            "ENABLE_TITLE_GENERATION=false",
            "ENABLE_TAGS_GENERATION=false",
            "ENABLE_FOLLOW_UP_GENERATION=false",
            "ENABLE_CONTEXT_COMPACTION=false",
        }
        self.assertTrue(expected_environment.issubset(webui_launch))
        secret_argument = next(value for value in webui_launch if value.startswith("WEBUI_SECRET_KEY="))
        self.assertIn("open-webui-secret", secret_argument)
        self.assertNotIn("dry-run-secret", secret_argument)
        self.assertIn(str(base / "open webui"), webui_launch)
        self.assertEqual(webui_launch[-5:], ["serve", "--host", "127.0.0.1", "--port", "13000"])
        self.assertFalse((base / "demo home").exists())

    def test_smoke_dry_run_calls_all_required_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_demo("smoke", base=Path(temporary))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("http://127.0.0.1:18000/health", result.stdout)
        self.assertIn("http://127.0.0.1:18000/v1/models", result.stdout)
        self.assertIn("http://127.0.0.1:18000/v1/chat/completions", result.stdout)
        self.assertIn("http://127.0.0.1:13000/health", result.stdout)
        self.assertIn("temperature", result.stdout)
        self.assertIn("X-OpenWebUI-User-Id", result.stdout)
        self.assertIn("X-OpenWebUI-Chat-Id", result.stdout)

    def test_all_lifecycle_commands_are_dry_run_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for command in ("start", "stop", "status", "logs", "smoke", "doctor"):
                with self.subTest(command=command):
                    result = self.run_demo(command, base=base)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((base / "demo home").exists())

    def test_start_reports_missing_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_demo(
                "start", base=Path(temporary), model_exists=False
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model directory not found", result.stderr)

    def test_start_reports_missing_installed_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_demo(
                "start", base=Path(temporary), dry_run=False
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("executable not found", result.stderr)
        self.assertIn("run", result.stderr)
        self.assertIn("install", result.stderr)

    def test_invalid_command_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_demo("launch", base=Path(temporary))

        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command: launch", result.stderr)
        self.assertIn("install", result.stderr)
        self.assertIn("start", result.stderr)
        self.assertIn("smoke", result.stderr)


if __name__ == "__main__":
    unittest.main()
