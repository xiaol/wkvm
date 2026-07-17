from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from experiments import open_webui_demo_render as renderer


class TestOpenWebUIDemoRender(unittest.TestCase):
    maxDiff = None

    def make_fixture(self, base: Path) -> tuple[Path, Path, Path, Path]:
        video_names = ["long.webm", *(f"classic-{index}.webm" for index in range(4))]
        for name in video_names:
            (base / name).write_bytes(b"fixture-webm")

        sessions = []
        records = [
            {
                "prompt_id": "long-document",
                "label": "Long document needle recall",
                "prompt_summary": "Find the hidden launch code in a very long document.",
                "phase": "long_prompt",
                "ttft_s": 3.25,
                "e2e_s": 28.75,
                "success": True,
            }
        ]
        classic_labels = (
            "Python bug repair",
            "SQL query design",
            "Distributed systems explanation",
            "Creative micro-story",
        )
        for index, label in enumerate(classic_labels):
            submitted = 2.0 + index * 0.1
            first_token = submitted + 1.0 + index * 0.2
            completed = submitted + 6.0 + index
            sessions.append(
                {
                    "video_path": f"classic-{index}.webm",
                    "prompt_id": f"classic-{index}",
                    "label": label,
                    "first_turn": {
                        "timing": {
                            "submit_offset_s": submitted,
                            "first_token_offset_s": first_token,
                            "completion_offset_s": completed,
                            "ttft_s": first_token - submitted,
                            "e2e_s": completed - submitted,
                        },
                        "response_text": f"answer {index}",
                        "error": None,
                    },
                    "follow_up": {
                        "prompt_id": f"follow-up-{index}",
                        "label": "One-sentence follow-up",
                        "timing": {
                            "submit_offset_s": completed + 1.0,
                            "first_token_offset_s": completed + 1.5,
                            "completion_offset_s": completed + 3.0,
                        },
                        "error": None,
                    },
                }
            )
            records.append(
                {
                    "prompt_id": f"classic-{index}",
                    "label": label,
                    "prompt_summary": f"Classic prompt {index + 1}: {label}.",
                    "phase": "concurrency_first_turn",
                    "ttft_s": first_token - submitted,
                    "e2e_s": completed - submitted,
                    "success": True,
                }
            )

        capture = {
            "kind": "wkvm.open_webui.demo_capture",
            "artifact": {
                "provenance": {
                    "served_model": "gemma-4-E4B-it",
                    "browser": {"engine": "chromium", "version": "140.0"},
                    "open_webui": {"version": "0.10.2"},
                },
                "acts": {
                    "long_prompt": {
                        "video_path": "long.webm",
                        "prompt_id": "long-document",
                        "label": "Long document needle recall",
                        "rendered_token_count": 12000,
                        "timing": {
                            "submit_offset_s": 4.0,
                            "first_token_offset_s": 7.25,
                            "completion_offset_s": 32.75,
                            "ttft_s": 3.25,
                            "e2e_s": 28.75,
                        },
                        "response_text": "The launch code is ORBIT-47.",
                        "error": None,
                    },
                    "concurrency": {
                        "sessions": sessions,
                        "gpu": {
                            "sample_count": 20,
                            "devices": [
                                {
                                    "index": 0,
                                    "name": "NVIDIA GeForce RTX 4090",
                                    "total_mib": 24564,
                                    "baseline_used_mib": 17682.0,
                                    "peak_used_mib": 18420.5,
                                    "last_used_mib": 17900.0,
                                }
                            ],
                        },
                        "provider": {
                            "before": {
                                "metrics": {
                                    "values": {
                                        "server": {"total_errors": 0},
                                        "engine": {
                                            "max_running": 0,
                                            "max_runnable_rows": 0,
                                        },
                                    }
                                }
                            },
                            "after_first_turn": {
                                "metrics": {
                                    "values": {
                                        "server": {"total_errors": 0},
                                        "engine": {
                                            "max_running": 4,
                                            "max_runnable_rows": 4,
                                        },
                                    }
                                }
                            },
                            "after": {
                                "metrics": {
                                    "values": {
                                        "server": {"total_errors": 0},
                                        "engine": {
                                            "max_running": 4,
                                            "max_runnable_rows": 4,
                                            "session_reuse_hits": 4,
                                        },
                                    }
                                }
                            },
                            "follow_up_session_reuse_delta": {
                                "session_reuse_hits": 4,
                                "session_reuse_misses": 0,
                            },
                            "delta": {"server": {"total_errors": 0}},
                        },
                    },
                },
            },
        }
        report = {
            "kind": "wkvm.open_webui.demo_report",
            "status": "passed",
            "summary": {
                "overall_passed": True,
                "offered_concurrency": 4,
                "request_count": 9,
                "success_count": 9,
                "all_error_count": 0,
                "error_count": 0,
                "ttft_p50_s": 1.4,
                "ttft_p95_s": 3.1,
                "e2e_p50_s": 7.5,
                "e2e_p95_s": 28.0,
            },
            "acts": {
                "concurrency_first_turn": {
                    "offered_concurrency": 4,
                    "request_count": 4,
                    "success_count": 4,
                    "error_count": 0,
                    "ttft_p50_s": 1.3,
                    "ttft_p95_s": 1.57,
                    "e2e_p50_s": 7.5,
                    "e2e_p95_s": 8.85,
                    "output_tokens": 2_080,
                    "output_tokens_min": 520,
                },
                "concurrency_follow_up": {
                    "offered_concurrency": 4,
                    "request_count": 4,
                    "success_count": 4,
                    "error_count": 0,
                    "output_tokens": 2_120,
                    "output_tokens_min": 530,
                },
                "all_requests": {
                    "request_count": 9,
                    "success_count": 9,
                    "error_count": 0,
                },
            },
            "records": records,
            "caveats": [
                "Serving semantics are routed_span_approximate.",
                "This functional recording is not a comparison and not a 10x measurement.",
            ],
            "provenance": {
                "measurement": "browser monotonic clock",
                "scoped_48_turn_evidence": "separate scoped 48-turn evidence",
                "tokenizer": {"report": {"class": "GemmaTokenizer"}},
            },
        }
        capture_path = base / "capture.json"
        report_path = base / "report.json"
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return capture_path, report_path, base / "demo.mp4", base / "demo.gif"

    def test_loads_nested_capture_and_report_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture, report, _, _ = self.make_fixture(Path(temporary))
            evidence = renderer.load_evidence(capture, report)

        self.assertEqual(evidence.long_act.prompt.label, "Long document needle recall")
        self.assertEqual(evidence.long_act.ttft_s, 3.25)
        self.assertEqual(len(evidence.classic_acts), 4)
        self.assertEqual(evidence.classic_acts[3].e2e_s, 9.0)
        self.assertEqual(evidence.classic_acts[0].follow_up.e2e_s, 2.0)
        self.assertEqual(evidence.classic_acts[3].recording_duration_s, 12.0)
        self.assertEqual(evidence.metrics.offered_concurrency, 4)
        self.assertEqual(evidence.metrics.success_count, 4)
        self.assertEqual(evidence.metrics.follow_up_success_count, 4)
        self.assertEqual(evidence.metrics.error_count, 0)
        self.assertEqual(evidence.metrics.ttft_p50_s, 1.3)
        self.assertEqual(evidence.metrics.ttft_p95_s, 1.57)
        self.assertEqual(evidence.metrics.e2e_p50_s, 7.5)
        self.assertEqual(evidence.metrics.e2e_p95_s, 8.85)
        self.assertEqual(evidence.metrics.baseline_vram_mib, 17682.0)
        self.assertEqual(evidence.metrics.peak_vram_mib, 18420.5)
        self.assertEqual(evidence.metrics.max_running, 4)
        self.assertEqual(evidence.metrics.max_runnable_rows, 4)
        self.assertEqual(evidence.metrics.first_turn_min_output_tokens, 520)
        self.assertEqual(evidence.metrics.follow_up_min_output_tokens, 530)
        self.assertEqual(evidence.metrics.act_2_total_output_tokens, 4_200)
        self.assertEqual(evidence.metrics.exact_reuse_hits, 4)
        self.assertEqual(evidence.metrics.provider_error_count, 0)
        self.assertIn("exact reuse hits 4/4", evidence.metrics.provider_summary or "")
        self.assertEqual(evidence.model, "gemma-4-E4B-it")
        self.assertEqual(evidence.gpu, "NVIDIA GeForce RTX 4090")
        self.assertEqual(evidence.provenance, "chromium 140.0 · Open WebUI 0.10.2")

    def test_cards_show_actual_metrics_and_claim_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture, report, mp4, _ = self.make_fixture(Path(temporary))
            evidence = renderer.load_evidence(capture, report)
            timeline = renderer.build_timeline(evidence)
            specs = renderer.build_visual_specs(evidence, timeline, mp4.name)

        all_text = "\n".join(spec.all_text() for spec in specs.values())
        self.assertIn("Long document needle recall", all_text)
        self.assertIn("offered concurrency 4", all_text)
        self.assertIn("success 4/4", all_text)
        self.assertIn("4/4 follow-ups", all_text)
        self.assertIn("TTFT p50/p95 1.3 s / 1.57 s", all_text)
        self.assertIn("E2E p50/p95 7.5 s / 8.85 s", all_text)
        self.assertIn("18,420.5 MiB", all_text)
        self.assertIn("provider observed · max_running 4 · max_runnable_rows 4", all_text)
        self.assertIn("4/4 exact reuse hits", all_text)
        self.assertIn("report errors 0", all_text)
        self.assertIn("provider errors 0", all_text)
        self.assertIn("first E2E 6 s · follow-up E2E 2 s", all_text)
        self.assertIn("12,000 rendered tokens", all_text)
        self.assertIn("routed_span_approximate", all_text)
        self.assertIn("not a 10x measurement", all_text)
        self.assertIn("separate scoped 48-turn evidence", all_text)
        self.assertIn("Functional UI evidence ≠ cross-engine throughput comparison", all_text)

    def test_ffmpeg_commands_trim_offsets_and_encode_compact_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, mp4, gif = self.make_fixture(base)
            evidence = renderer.load_evidence(capture, report)
            timeline = renderer.build_timeline(evidence)
            assets = renderer._asset_paths(base / "cards")
            mp4_command = renderer.build_mp4_command(evidence, timeline, assets, mp4)
            gif_command = renderer.build_gif_command(timeline, mp4, gif)

        filter_graph = mp4_command[mp4_command.index("-filter_complex") + 1]
        self.assertIn("trim=start=4:end=32.75", filter_graph)
        self.assertIn("trim=start=2:end=11", filter_graph)
        self.assertNotIn("trim=start=2:end=8,", filter_graph)
        self.assertIn("xstack=inputs=4", filter_graph)
        self.assertIn("concat=n=5:v=1:a=0,fps=30", filter_graph)
        self.assertIn("libx264", mp4_command)
        self.assertIn("yuv420p", mp4_command)
        self.assertIn("+faststart", mp4_command)
        self.assertIn("-b:v", mp4_command)

        gif_filter = gif_command[gif_command.index("-filter_complex") + 1]
        self.assertIn("fps=10", gif_filter)
        self.assertIn("scale=960:540", gif_filter)
        self.assertIn("palettegen=max_colors=96", gif_filter)
        self.assertIn("paletteuse", gif_filter)

    def test_dry_run_emits_commands_and_link_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture, report, mp4, gif = self.make_fixture(Path(temporary))
            output = io.StringIO()
            with mock.patch.object(renderer.subprocess, "run") as run, redirect_stdout(output):
                renderer.main(
                    [
                        "render",
                        "--capture",
                        str(capture),
                        "--report",
                        str(report),
                        "--mp4",
                        str(mp4),
                        "--gif",
                        str(gif),
                        "--dry-run",
                    ]
                )

        run.assert_not_called()
        rendered = output.getvalue()
        self.assertIn("MP4_COMMAND ffmpeg", rendered)
        self.assertIn("GIF_COMMAND ffmpeg", rendered)
        self.assertIn("MARKDOWN [![WKVM Open WebUI demo](demo.gif)](demo.mp4)", rendered)
        self.assertFalse(mp4.exists())
        self.assertFalse(gif.exists())

    def test_render_orchestration_mocks_cards_and_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, mp4, gif = self.make_fixture(base)
            args = argparse.Namespace(
                capture=str(capture),
                report=str(report),
                mp4=str(mp4),
                gif=str(gif),
                work_dir=str(base / "cards"),
                dry_run=False,
            )

            def fake_run(command: list[str]) -> None:
                output = Path(command[-1])
                output.write_bytes(b"small-rendered-artifact")

            with (
                mock.patch.object(renderer.shutil, "which", return_value="/usr/bin/ffmpeg"),
                mock.patch.object(renderer, "_render_card") as render_card,
                mock.patch.object(renderer, "_run", side_effect=fake_run) as run,
                redirect_stdout(io.StringIO()),
            ):
                renderer.cmd_render(args)

        self.assertEqual(render_card.call_count, 5)
        self.assertEqual(run.call_count, 2)
        mp4_command = run.call_args_list[0].args[0]
        gif_command = run.call_args_list[1].args[0]
        self.assertIn("libx264", mp4_command)
        self.assertIn("paletteuse", gif_command[gif_command.index("-filter_complex") + 1])

    def test_schema_error_explains_invalid_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, _, _ = self.make_fixture(base)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["artifact"]["acts"]["long_prompt"]["timing"][
                "first_token_offset_s"
            ] = 2.0
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                renderer.SchemaError,
                "0 <= submit <= first token <= completion",
            ):
                renderer.load_evidence(capture, report)

    def test_rejects_report_without_passed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, _, _ = self.make_fixture(base)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["status"] = "failed"
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(renderer.SchemaError, "status must be 'passed'"):
                renderer.load_evidence(capture, report)

    def test_rejects_report_without_overall_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, _, _ = self.make_fixture(base)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["summary"]["overall_passed"] = False
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(renderer.SchemaError, "overall_passed must be true"):
                renderer.load_evidence(capture, report)

    def test_rejects_report_from_a_different_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, _, _ = self.make_fixture(base)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["provenance"]["capture"] = {"file_sha256": "0" * 64}
            report.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                renderer.SchemaError,
                "generated from a different capture",
            ):
                renderer.load_evidence(capture, report)

    def test_uses_tokenizer_family_when_exact_model_is_not_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            capture, report, _, _ = self.make_fixture(base)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            del payload["artifact"]["provenance"]["served_model"]
            capture.write_text(json.dumps(payload), encoding="utf-8")

            evidence = renderer.load_evidence(capture, report)

        self.assertEqual(evidence.model, "Gemma family (GemmaTokenizer)")


if __name__ == "__main__":
    unittest.main()
