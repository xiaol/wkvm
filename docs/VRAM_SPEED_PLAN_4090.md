# WKVM VRAM-for-Speed Plan (RTX 4090)

This runbook is for measuring the bounded-state WKVM path on one RTX 4090. It
does not change the model semantics or silently enable an unvalidated route
policy. The first pass measures existing opt-in knobs; code changes follow only
after an exact-output and memory gate passes.

## Goal

Use WKVM's fixed per-request state to spend a small, fixed amount of spare VRAM
on faster execution machinery. Keep the state arena bounded and compare every
profile against the same prompt, model, source tree, and GPU.

The current A800 reference is recorded in
`experiments/results/gemma_a800_reliable_20260716/report.md`:

| B=64, ctx=16K, out=32 | Prefill tok/s | E2E tok/s | Peak GPU |
|---|---:|---:|---:|
| WKVM routed-span | 12,368 | 21.886 | 28.44 GiB |
| vLLM full-KV | 14,247 | 27.608 | 74.74 GiB |

These are different semantics. Do not describe a routed-span speed result as
full-KV parity.

## Phase 0: Freeze the Environment

Run these commands from the repository root. Replace paths if the 4090 machine
uses a different checkout or model directory.

```bash
cd /home/aiuser/X/wkvm
git status --short --branch
git rev-parse HEAD
nvidia-smi -i "${GPU_DEVICE:-0}" \\
  --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu \\
  --format=csv
/home/aiuser/X/.venv-wkvm/bin/python -V
```

Use an otherwise idle GPU. The runner treats more than 1 GiB of pre-existing
device usage as non-comparable. Keep benchmark JSON outside the repository so
artifact creation cannot change source identity.

For a 24 GiB 4090, use `--mem-cap-gib 20 --headroom-gib 4`. A B16 run is the
first green high-concurrency target; B24/B32 are capacity probes and may exceed
the headroom gate.

## Phase 1: Four Controlled Probes

The helper `scripts/run_wkvm_vram_speed_4090.sh` runs three cold repeats of each
profile, sequentially and in interleaved order:

| Profile | Projection | Route chunk | Purpose |
|---|---|---:|---|
| baseline | `separate` | 512 | Current fixed-state baseline |
| packed | `qkv_gate_up_packed` | 512 | Existing packed-GEMM opt-in |
| larger-fold | `separate` | 2048 | Fewer routed folds, semantic-risk probe |
| combined | `qkv_gate_up_packed` | 2048 | Interaction probe |

Run it as follows:

```bash
cd /home/aiuser/X/wkvm
export GPU_DEVICE=0
export MODEL_PATH=/home/aiuser/X/models/gemma-4-E4B-it
export OUT_DIR=/home/aiuser/X/results/4090/wkvm_vram_speed_$(date +%Y%m%d_%H%M%S)
REPEATS=3 bash scripts/run_wkvm_vram_speed_4090.sh
```

The probe shape is B8, 16K input, one output token, BF16, synthetic uniform
prompts, `chunk=2048`, and `prefill_microbatch_rows=8`. This isolates prefill
and keeps the 4090 memory requirement modest. The script records the launch
configuration, source identity, model identity, output fingerprint, TTFT,
cohort input throughput, elapsed time, and whole-device peak memory.

For a command-line dry run without touching CUDA:

```bash
DRY_RUN=1 bash scripts/run_wkvm_vram_speed_4090.sh
```

## Phase 1 Acceptance Gates

For every successful artifact:

1. `git_worktree_dirty=false` and all repeats have the same source identity.
2. The baseline and packed `route_chunk=512` fingerprints match exactly.
3. A `route_chunk=2048` fingerprint mismatch is a semantic change, not a speed
   win; report its throughput separately and do not merge it as an equivalent
   optimization.
4. Peak whole-device use stays below 20 GiB for a green profile.
5. Use the median of three repeats. A single faster run is exploratory only.

The first candidate for a speed-profile change is a repeatable prefill gain of
at least 5% with no fingerprint, error, or memory-gate regression. The 5% value
is an engineering triage threshold, not a published claim.

Summarize the artifacts with:

```bash
for f in "$OUT_DIR"/*.json; do
  jq -r --arg f "$f" '(.rows[0]) as $r |
    [$f, .native_gemma_projection_backend, .config.route_chunk,
     $r.cohort_input_tok_s, $r.prefill_time_p50_s, $r.p50_ttft_s,
     $r.peak_reserved_gib, $r.request_output_token_ids_sha256] | @tsv' "$f"
done
```

## Phase 2: 4090 High-Concurrency Check

Only run this phase after selecting a Phase 1 candidate. Repeat the baseline
and candidate at B16, ctx=16K, out=32, with `slots=16`, token-pool capacity
`65536`, and the same `mem-cap-gib=20/headroom-gib=4` gate. Use three repeats
for a throughput claim.

Then run one exploratory capacity ladder at B24 and B32, increasing slots and
token-pool capacity to `B*4096`. Mark rows that exceed the 20 GiB gate as
capacity observations, not green production results.

Record both metrics that matter:

- **Prefill:** cohort input tok/s, p50/p95 TTFT, and prefill wall.
- **E2E:** batch wall and output tok/s for the requested output length.

WKVM can decode faster than vLLM while still losing E2E: its current scheduler
advances many requests in lockstep, so first tokens arrive near the end of the
cohort. A completion-biased B8 prefill lane with decode priority is a separate
Phase 3 change; do not infer its benefit from the projection probe.

## Phase 3: Code Changes, One at a Time

Implement and benchmark these in order, keeping each change opt-in until its
gate passes:

1. **Persistent packed projections.** Cache gate/up packed weights instead of
   rebuilding the concatenation on every layer/call. The duplicate BF16 gate/up
   weights cost about 4.10 GiB for this model; direct checkpoint packing can
   later remove most of that duplicate. Validate exact output and startup peak.
2. **Batched routed packets.** Collect route features for the eight rows and
   several folds into one pinned host packet, then run the existing CPU planner
   in canonical order. A double-buffered B8 K/V staging area is about 512 MiB.
3. **Completion-biased prefill scheduling.** Keep compact states resident, but
   finish one B8 lane before moving to the next and prioritize decode work. This
   targets E2E/TTFT rather than raw GEMM throughput; evaluate fairness and tail
   latency as well as batch wall.
4. **Native GQA prefill kernel.** Replace K/V head expansion with a fused GQA
   kernel. The generic PyTorch GQA path is explicitly not an acceptable proxy.

Do not spend VRAM first on `prefill_microbatch_rows=16`, `chunk=4096`, or a
larger token pool. Existing A/Bs added memory and became slower.

## Commit and Push Discipline

Keep benchmark JSON outside the checkout. Commit only source, tests, and this
runbook/helper. Before each commit:

```bash
git status --short
git diff --check
python -m pytest tests/test_gemma_native_forward.py tests/test_gemma_prefill_batch.py -q
git log -1 --oneline
```

Use a focused commit message, for example:

```bash
git add docs/VRAM_SPEED_PLAN_4090.md scripts/run_wkvm_vram_speed_4090.sh
git commit -m "Add RTX 4090 VRAM speed benchmark plan"
git push origin main
```

Do not commit raw 4090 result JSONs until the run has passed the provenance and
semantic gates; publish a report that names the exact commit used for the
measurements.
