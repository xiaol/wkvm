#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-$ROOT/../models/gemma-4-E4B-it}"
PYTHON="${PYTHON:-$ROOT/../.venv-wkvm/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/../results/4090/wkvm_vram_speed_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
MEM_CAP_GIB="${MEM_CAP_GIB:-20}"
HEADROOM_GIB="${HEADROOM_GIB:-4}"
BASELINE_MAX_MIB="${BASELINE_MAX_MIB:-1024}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' 'nvidia-smi is required' >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  printf '%s\n' 'jq is required' >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python executable not found: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  printf '%s\n' 'worktree must be clean before benchmark runs' >&2
  exit 1
fi
if [[ "$REPEATS" -lt 1 ]]; then
  printf '%s\n' 'REPEATS must be >= 1' >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

gpu_used_mib() {
  nvidia-smi -i "$GPU_DEVICE" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ' | head -n 1
}

check_gpu() {
  if [[ "$ALLOW_BUSY_GPU" == "1" ]]; then
    return
  fi
  local used
  used="$(gpu_used_mib)"
  if [[ -z "$used" || "$used" -gt "$BASELINE_MAX_MIB" ]]; then
    printf 'GPU %s is not idle enough: %s MiB used (limit %s MiB)\n' \
      "$GPU_DEVICE" "${used:-unknown}" "$BASELINE_MAX_MIB" >&2
    exit 2
  fi
}

run_probe() {
  local label="$1"
  local projection="$2"
  local route_chunk="$3"
  local repeat="$4"
  local output="$OUT_DIR/${label}-r${repeat}.json"
  local -a args=(
    "$ROOT/experiments/native_gemma_bench.py"
    --model-path "$MODEL_PATH"
    --ctx 16384
    --out 1
    --concurrency 8
    --slots 8
    --prompt-lengths uniform
    --synthetic-prompts
    --native-gemma-checkpoint-loader
    --native-gemma-attention-backend sdpa_single_gqa
    --native-gemma-projection-backend "$projection"
    --enable-token-pool-attention
    --token-pool-max-context-len 16640
    --token-pool-capacity 65536
    --token-pool-paged-block-size 16
    --enable-token-pool-triton
    --enable-token-pool-paged-triton
    --enable-token-pool-paged-split-triton
    --token-pool-triton-strict
    --token-pool-sliding-paged-metadata-only
    --persistent-padded-sliding-metadata-padding
    --persistent-padded-decode-steps 1
    --persistent-padded-decode-cuda-graph
    --persistent-padded-decode-graph-warmup-iters 0
    --sink 16
    --window 1024
    --m-slots 32
    --route-chunk "$route_chunk"
    --chunk 2048
    --prefill-microbatch-rows 8
    --decode-microbatch-rows 8
    --mem-cap-gib "$MEM_CAP_GIB"
    --headroom-gib "$HEADROOM_GIB"
    --max-baseline-gpu-used-gib 1
    --gpu-memory-device "$GPU_DEVICE"
    --gpu-memory-sample-interval-s 0.1
    --require-native-no-hf
    --no-warmup
    --stop-on-failure
    --json "$output"
  )
  printf 'profile=%s repeat=%s output=%s\n' "$label" "$repeat" "$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$GPU_DEVICE"
    printf '%q ' "$PYTHON" "${args[@]}"
    printf '\n'
    return
  fi
  check_gpu
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PYTHON" "${args[@]}"
}

profiles=(
  'separate-r512 separate 512'
  'packed-r512 qkv_gate_up_packed 512'
  'separate-r2048 separate 2048'
  'packed-r2048 qkv_gate_up_packed 2048'
)

for repeat in $(seq 1 "$REPEATS"); do
  for profile in "${profiles[@]}"; do
    read -r label projection route_chunk <<<"$profile"
    run_probe "$label" "$projection" "$route_chunk" "$repeat"
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

summary="$OUT_DIR/summary.tsv"
{
  printf 'file\tprojection\troute_chunk\tinput_tok_s\tprefill_p50_s\tttft_p50_s\tpeak_reserved_gib\toutput_hash\n'
  for file in "$OUT_DIR"/*.json; do
    jq -r --arg file "$file" '(.rows[0]) as $r |
      [$file, .native_gemma_projection_backend, .config.route_chunk,
       $r.cohort_input_tok_s, $r.prefill_time_p50_s, $r.p50_ttft_s,
       $r.peak_reserved_gib, $r.request_output_token_ids_sha256] | @tsv' "$file"
  done
} > "$summary"
cat "$summary"
