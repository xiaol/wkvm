#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-$ROOT/../models/gemma-4-E4B-it}"
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$ROOT/../.venv-wkvm/bin/python" ]]; then
  PYTHON="$ROOT/../.venv-wkvm/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
else
  PYTHON="$(command -v python3 || command -v python || true)"
fi
OUT_DIR="${OUT_DIR:-$ROOT/../results/4090/wkvm_phase3_$(date +%Y%m%d_%H%M%S)}"
PHASE3_REPORT="${PHASE3_REPORT:-$ROOT/experiments/phase3_gemma_report.py}"
REPEATS="${REPEATS:-3}"
MEM_CAP_GIB="${MEM_CAP_GIB:-24}"
HEADROOM_GIB="${HEADROOM_GIB:-4}"
ROUTED_PACKET_WORKSPACE_BYTES="${ROUTED_PACKET_WORKSPACE_BYTES:-67108864}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$MEM_CAP_GIB" != "24" || "$HEADROOM_GIB" != "4" ]]; then
  printf '%s\n' 'Phase 3 evidence requires MEM_CAP_GIB=24 and HEADROOM_GIB=4' >&2
  exit 1
fi
if [[ "$ROUTED_PACKET_WORKSPACE_BYTES" != "67108864" ]]; then
  printf '%s\n' 'Phase 3 evidence requires a 67108864-byte routed packet workspace' >&2
  exit 1
fi
if [[ ! "$REPEATS" =~ ^[0-9]+$ || "$REPEATS" -lt 3 ]]; then
  printf '%s\n' 'REPEATS must be an integer >= 3' >&2
  exit 1
fi
if [[ ! "$ROUTED_PACKET_WORKSPACE_BYTES" =~ ^[0-9]+$ || "$ROUTED_PACKET_WORKSPACE_BYTES" -lt 1 ]]; then
  printf '%s\n' 'ROUTED_PACKET_WORKSPACE_BYTES must be a positive integer' >&2
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
if ! command -v realpath >/dev/null 2>&1; then
  printf '%s\n' 'realpath is required' >&2
  exit 1
fi

root_real="$(realpath -e -- "$ROOT")"
out_dir_real="$(realpath -m -- "$OUT_DIR")"
if [[ "$out_dir_real" == "$root_real" || "$out_dir_real" == "$root_real/"* ]]; then
  printf 'OUT_DIR must be outside the source checkout: %s\n' "$OUT_DIR" >&2
  exit 1
fi
OUT_DIR="$out_dir_real"

if [[ "$DRY_RUN" != "1" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    printf '%s\n' 'git is required' >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'nvidia-smi is required' >&2
    exit 1
  fi
  if [[ ! -f "$PHASE3_REPORT" ]]; then
    printf 'Phase 3 report tool not found: %s\n' "$PHASE3_REPORT" >&2
    exit 1
  fi
  if [[ -e "$OUT_DIR" && ! -d "$OUT_DIR" ]]; then
    printf 'OUT_DIR exists and is not a directory: %s\n' "$OUT_DIR" >&2
    exit 1
  fi
  if [[ -d "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
    printf 'OUT_DIR must be empty before a benchmark run: %s\n' "$OUT_DIR" >&2
    exit 1
  fi
  mkdir -p "$OUT_DIR"
fi

check_clean_worktree() {
  if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    printf '%s\n' 'worktree must be clean before and throughout benchmark runs' >&2
    exit 1
  fi
}

check_idle_gpu() {
  local used_mib
  used_mib="$(
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ' | head -n 1
  )"
  if [[ ! "$used_mib" =~ ^[0-9]+$ ]]; then
    printf 'Could not read GPU %s memory use: %s\n' \
      "$GPU_DEVICE" "${used_mib:-unknown}" >&2
    exit 2
  fi
  if (( used_mib > 1024 )); then
    printf 'GPU %s is not idle enough: %s MiB used (limit 1024 MiB)\n' \
      "$GPU_DEVICE" "$used_mib" >&2
    exit 2
  fi
}

print_native_command() {
  local -a command=("$@")
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_DEVICE"
  printf '%q ' "${command[@]}"
  printf '\n'
}

declare -a expected_artifacts=()

run_profile() {
  local label="$1"
  local repeat="$2"
  local concurrency="$3"
  local output_tokens="$4"
  local slots="$5"
  local prefill_rows="$6"
  local decode_rows="$7"
  local padded_decode_steps="$8"
  local attention_backend="$9"
  local projection_backend="${10}"
  shift 10
  local output="$OUT_DIR/${label}-r${repeat}.json"
  local -a extra_args=("$@")
  local -a args=(
    "$PYTHON"
    "$ROOT/experiments/native_gemma_bench.py"
    --model-path "$MODEL_PATH"
    --ctx 16384
    --out "$output_tokens"
    --concurrency "$concurrency"
    --slots "$slots"
    --prompt-lengths uniform
    --synthetic-prompts
    --native-gemma-checkpoint-loader
    --native-gemma-attention-backend "$attention_backend"
    --native-gemma-projection-backend "$projection_backend"
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
    --persistent-padded-decode-steps "$padded_decode_steps"
    --persistent-padded-decode-cuda-graph
    --persistent-padded-decode-graph-warmup-iters 0
    --sink 16
    --window 1024
    --m-slots 32
    --route-chunk 512
    --chunk 2048
    --prefill-microbatch-rows "$prefill_rows"
    --decode-microbatch-rows "$decode_rows"
    --mem-cap-gib "$MEM_CAP_GIB"
    --headroom-gib "$HEADROOM_GIB"
    --max-baseline-gpu-used-gib 1
    --gpu-memory-device "$GPU_DEVICE"
    --gpu-memory-sample-interval-s 0.1
    --require-native-no-hf
    --stop-on-failure
    "${extra_args[@]}"
    --json "$output"
  )

  expected_artifacts+=("$output")
  printf 'profile=%s repeat=%s output=%s\n' "$label" "$repeat" "$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_native_command "${args[@]}"
    return
  fi

  check_clean_worktree
  check_idle_gpu
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${args[@]}"
  check_clean_worktree
}

run_named_profile() {
  local label="$1"
  local repeat="$2"
  case "$label" in
    prefill-baseline)
      run_profile "$label" "$repeat" 8 1 8 8 8 1 sdpa_single_gqa separate
      ;;
    prefill-packed)
      run_profile "$label" "$repeat" 8 1 8 8 8 1 sdpa_single_gqa qkv_gate_up_packed
      ;;
    prefill-routed-packets)
      run_profile "$label" "$repeat" 8 1 8 8 8 1 sdpa_single_gqa separate \
        --batched-routed-packets \
        --routed-packet-workspace-bytes "$ROUTED_PACKET_WORKSPACE_BYTES"
      ;;
    prefill-native-gqa)
      run_profile "$label" "$repeat" 8 1 8 8 8 1 triton_dense_gqa separate
      ;;
    prefill-combined)
      run_profile "$label" "$repeat" 8 1 8 8 8 1 triton_dense_gqa qkv_gate_up_packed \
        --batched-routed-packets \
        --routed-packet-workspace-bytes "$ROUTED_PACKET_WORKSPACE_BYTES"
      ;;
    schedule-baseline)
      run_profile "$label" "$repeat" 16 32 16 8 16 32 sdpa_single_gqa separate
      ;;
    schedule-lane8)
      run_profile "$label" "$repeat" 16 32 16 8 16 32 sdpa_single_gqa separate \
        --completion-prefill-lane-size 8
      ;;
    *)
      printf 'Unknown Phase 3 profile: %s\n' "$label" >&2
      exit 1
      ;;
  esac
}

profile_order=(
  prefill-baseline
  prefill-packed
  prefill-routed-packets
  prefill-native-gqa
  prefill-combined
  schedule-baseline
  schedule-lane8
)

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  offset=$(( ((repeat - 1) * 2) % ${#profile_order[@]} ))
  for ((index = 0; index < ${#profile_order[@]}; index++)); do
    profile_index=$(( (index + offset) % ${#profile_order[@]} ))
    run_named_profile "${profile_order[$profile_index]}" "$repeat"
  done
done

markdown="$OUT_DIR/report.md"
summary_json="$OUT_DIR/summary.json"
report_args=(
  "$PYTHON"
  "$PHASE3_REPORT"
  "${expected_artifacts[@]}"
  --markdown "$markdown"
  --summary-json "$summary_json"
)
printf 'report artifacts=%s markdown=%s summary=%s\n' \
  "${#expected_artifacts[@]}" "$markdown" "$summary_json"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${report_args[@]}"
  printf '\n'
  exit 0
fi

for artifact in "${expected_artifacts[@]}"; do
  if [[ ! -s "$artifact" ]]; then
    printf 'Missing benchmark artifact: %s\n' "$artifact" >&2
    exit 1
  fi
done
check_clean_worktree
"${report_args[@]}"
