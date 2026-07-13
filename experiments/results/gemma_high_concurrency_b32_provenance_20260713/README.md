# Gemma B32 High-Concurrency Provenance

This focused provenance directory seals the source, report, and promoted raw
artifacts for the 2026-07-13 Gemma 16K/B32 comparison. The benchmark processes
ran from a dirty worktree whose committed base was `2b4e642`; the binary
tracked patch and promoted-file manifest identify the report-time source that
adds the CUDA graph generation guard and incumbent residency telemetry.

The performance conclusion is deliberately bounded: WKVM proves 32 resident
sessions but misses the 18 GiB memory gate; vLLM wins B32 E2E goodput while
queuing most requests; SGLang overlaps WKVM on E2E and also queues/retracts.

## Contents

- `provenance/git_head.txt`: committed base identity.
- `provenance/git_status.txt`: full report-time status, excluding this directory.
- `provenance/tracked_worktree.patch`: binary-capable tracked diff against HEAD.
- `provenance/promoted_files.tsv`: SHA-256, byte size, and repo-relative path
  for every promoted source, report, and raw artifact.
- `provenance/source_identity.json`: hashes and capture metadata.
- `SHA256SUMS`: internal hashes for this directory.
- `verify_bundle.sh`: internal verification by default; `--external` also
  verifies every promoted repo file.

## Verification

```bash
cd experiments/results/gemma_high_concurrency_b32_provenance_20260713
bash verify_bundle.sh
bash verify_bundle.sh --external
```

The tracked patch is expected to contain whitespace diagnostics when treated
as source text because it preserves literal diff payloads. Verify the actual
source diff with that generated snapshot excluded.
