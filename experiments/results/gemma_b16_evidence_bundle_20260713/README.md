# Gemma B16 Evidence and Provenance Bundle

This directory is the durable evidence bundle for the 2026-07-13 Gemma-4-E4B
WKVM, vLLM, and SGLang comparison. It contains current-source WKVM timing and
quality artifacts, a controlled-baseline 16K vLLM artifact, archived 32K
incumbent artifacts, exact commands, source/environment/model identities, and
historical failure evidence.

The correct conclusion is deliberately narrow:

- **16K E2E: tie.** WKVM's three-run mean is 64.769 tok/s; the controlled vLLM
  point is 64.713 tok/s and lies inside WKVM's observed range.
- **16K decode and memory: WKVM advantage.** 256.467 versus 67.091 comparable
  decode tok/s; 16.871 versus 18.405 GiB whole-GPU engine delta. Only WKVM is
  green under the 18 GiB gate.
- **32K E2E: current single-run WKVM win.** 34.830 versus archived vLLM 24.543
  and SGLang 26.130 tok/s, with baseline/repeat caveats.
- **Quality: final-source B2 full grid passes.** The guarded 16K B16 path has
  scorer-visible semantic parity with B2, not full token-exact or strict full-
  row-capacity parity.

See [comparison_report.md](comparison_report.md) for the evidence audit.

## Bundle layout

- `artifacts/final/`: promoted current WKVM rows and controlled 16K vLLM.
- `artifacts/`: archived incumbent and pre-fix timing evidence retained for
  provenance; promoted conclusions are listed in the audit.
- `quality/`: final-source quality artifacts, status, and physical-capacity
  root-cause note.
- `quality/history/`: smoke, OOM, unsafe-graph, and earlier-source diagnostics.
- `provenance/commands.tsv`: exact commands embedded in every JSON artifact.
- `provenance/artifact_inventory.tsv`: path, SHA-256, bytes, and preserved
  source filesystem mtime in UTC.
- `provenance/source_identity.json`: final report-time HEAD/tree, dirty status,
  and hashes for status, tracked binary diff, and all tracked/untracked files.
- `provenance/tracked_worktree.patch`: binary-capable tracked diff against HEAD.
- `provenance/worktree_files.tsv`: SHA-256, size, and repo-relative path for all
  tracked or untracked/non-ignored regular files; the bundle is excluded to
  avoid recursion.
- `provenance/untracked_source/`: copies of untracked Python source and tests.
- `provenance/model_identity.json`, `model_files.sha256`, and
  `model_files.tsv`: complete identity for all seven checkpoint files,
  including the full 15.99 GB safetensors file; no weights are copied.
- `provenance/gpu.csv`, `cuda_toolkit.txt`, `os_kernel.txt`, `glibc.txt`,
  `python_environments.jsonl`, and `tool_binaries.sha256`: hardware, driver,
  CUDA, Python, PyTorch, package, and executable identities.
- `SHA256SUMS`: digest for every bundle file except itself.
- `verify_bundle.sh`: internal verification by default; optional external model
  and current-worktree comparison.

## Captured platform

| Item | Identity |
|---|---|
| Working directory | `/home/xiaol/X/wkvm` |
| Git commit | `59f736b9b80a2c54c397a481775cb008dc2ce2ef` |
| GPU | NVIDIA GeForce RTX 4090, UUID `GPU-0c31f901-f0f2-8a23-b344-f85d0f07b57d`, 24,564 MiB, compute 8.9 |
| NVIDIA driver | `595.71.05` |
| CUDA toolkit | 13.1, `V13.1.115` |
| PyTorch | 2.11.0+cu130, compiled CUDA 13.0, cuDNN 91900 |
| Triton | 3.6.0 |
| Engines | vLLM 0.24.0; SGLang 0.5.14 |
| Model files | 7 files, 16,024,791,983 bytes, hashed in full |
| `model.safetensors` SHA-256 | `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503` |
| Model-manifest SHA-256 | `3aabdcb4cb38bf8ae3502d0549397d60549aec9881ee90d2423450c8cc6ea4d0` |

Exact capture timestamps and final source hashes are in `provenance/`.
Relevant package versions are recorded without serializing arbitrary direct-
URL metadata or credentials.

## Source boundary

The final quality artifacts embed pre/post hashes for 17 runtime source files
and report an unchanged combined identity
`3be6a1b031fdcec8630c30a6f9bc6f03f16efbb723bf3bf3b49eae4fa7b38d9c`.
Performance payloads record only committed HEAD, so the bundle adds a post-run
full worktree snapshot. This strongly identifies the final local state but does
not retroactively make every historical dirty-worktree invocation reconstruct-
able. Historical artifacts remain explicitly separated from promoted results.

## Quality boundary

The final B2 105-case/45-cell grid passes all declared recall gates. The final
16K B16 run matches B2 on all 35 scored texts, scores, and scorer-visible token
prefixes; only 29/35 full fixed-length token sequences match, with the six
differences occurring after the scoring stop point.

The physical model now accounts for the approximately 10,831-token routed
materialization bound. The 36,864-slot performance pool fully covers uniform
timing rows but cannot strictly cover every diverse B16 quality row. Partial
coverage is graph-ineligible and executes through guarded eager fallback. See
[quality/README.md](quality/README.md).

## Verification

Verify every internal bundle file:

```bash
cd /home/xiaol/X/wkvm/experiments/results/gemma_b16_evidence_bundle_20260713
bash verify_bundle.sh
```

Also rehash the external 15 GB checkpoint and compare the current worktree to
the recorded post-run snapshot:

```bash
cd /home/xiaol/X/wkvm/experiments/results/gemma_b16_evidence_bundle_20260713
bash verify_bundle.sh --external
```

The external source check is expected to fail after any later source edit. The
internal check is the durable bundle-integrity test.
