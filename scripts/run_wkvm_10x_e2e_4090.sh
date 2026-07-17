#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it}"
WKVM_PY="${WKVM_PY:-$ROOT/../HRM-Text/.venv/bin/python}"
VLLM_PY="${VLLM_PY:-/run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/vllm/bin/python}"
SGLANG_PY="${SGLANG_PY:-/run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/sglang/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/../results/4090/wkvm_10x_e2e_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"
DRY_RUN="${DRY_RUN:-0}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"

# Frozen above the highest whole-device peak in the measured eight-turn cohort.
# Override explicitly only when starting a separately documented cohort.
MEMORY_CEILING_MIB="${MEMORY_CEILING_MIB:-24200}"

BENCHMARK="${BENCHMARK:-$ROOT/experiments/gemma_multiturn_bench.py}"
REPORT="${REPORT:-$ROOT/experiments/multiturn_10x_report.py}"
lock_key="${GPU_DEVICE//[^[:alnum:]_.-]/_}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-${TMPDIR:-/tmp}/wkvm-10x-e2e-gpu-${lock_key}.lock}"
GPU_PROCESS_ALLOWLIST_REGEX="${GPU_PROCESS_ALLOWLIST_REGEX:-gnome-remote-desktop-daemon|ptyxis|nautilus|gnome-text-editor|chrome|/papers$|/baobab$}"

generate_uuid() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    tr -d '\n' < /proc/sys/kernel/random/uuid
    return
  fi
  "$WKVM_PY" -c 'import uuid; print(uuid.uuid4())'
}

if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="wkvm-4090-$(generate_uuid)"
fi

if [[ ! "$REPEATS" =~ ^[0-9]+$ || "$REPEATS" -lt 1 ]]; then
  printf '%s\n' 'REPEATS must be an integer >= 1' >&2
  exit 1
fi
if [[ ! "$MEMORY_CEILING_MIB" =~ ^[0-9]+$ || "$MEMORY_CEILING_MIB" -lt 1 ]]; then
  printf '%s\n' 'MEMORY_CEILING_MIB must be a positive integer' >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  printf '%s\n' 'DRY_RUN must be 0 or 1' >&2
  exit 1
fi
if ! command -v realpath >/dev/null 2>&1; then
  printf '%s\n' 'realpath is required' >&2
  exit 1
fi
for python_path in "$WKVM_PY" "$VLLM_PY" "$SGLANG_PY"; do
  if [[ ! -x "$python_path" ]]; then
    printf 'Python executable not found: %s\n' "$python_path" >&2
    exit 1
  fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$BENCHMARK" ]]; then
  printf 'Benchmark not found: %s\n' "$BENCHMARK" >&2
  exit 1
fi
if [[ ! -f "$REPORT" ]]; then
  printf 'Report tool not found: %s\n' "$REPORT" >&2
  exit 1
fi

root_real="$(realpath -e -- "$ROOT")"
out_dir_real="$(realpath -m -- "$OUT_DIR")"
if [[ "$out_dir_real" == "$root_real" || "$out_dir_real" == "$root_real/"* ]]; then
  printf 'OUT_DIR must be outside the source checkout: %s\n' "$OUT_DIR" >&2
  exit 1
fi
OUT_DIR="$out_dir_real"
TRACE_DIR="$OUT_DIR/traces"
ARTIFACT_DIR="$OUT_DIR/artifacts"
MARKDOWN="$OUT_DIR/exploratory_10x_report.md"
SUMMARY_JSON="$OUT_DIR/exploratory_10x_summary.json"
PATH_MANIFEST="$OUT_DIR/artifact_paths.tsv"

check_gpu_identity() {
  local gpu_name
  local total_mib
  if ! gpu_name="$(
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=name \
      --format=csv,noheader 2>/dev/null | head -n 1
  )"; then
    printf 'Could not query GPU %s\n' "$GPU_DEVICE" >&2
    exit 2
  fi
  if [[ "$gpu_name" != *"RTX 4090"* ]]; then
    printf 'GPU %s is not an RTX 4090: %s\n' "$GPU_DEVICE" "$gpu_name" >&2
    exit 2
  fi
  total_mib="$(
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=memory.total \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n 1
  )"
  if [[ ! "$total_mib" =~ ^[0-9]+$ || "$MEMORY_CEILING_MIB" -gt "$total_mib" ]]; then
    printf 'Invalid %s MiB ceiling for GPU %s with %s MiB total\n' \
      "$MEMORY_CEILING_MIB" "$GPU_DEVICE" "${total_mib:-unknown}" >&2
    exit 2
  fi
}

refuse_parallel_gpu_run() {
  local query_output
  local blocking_processes
  if ! query_output="$(
    nvidia-smi -i "$GPU_DEVICE" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>&1
  )"; then
    printf 'Could not inspect compute processes on GPU %s: %s\n' \
      "$GPU_DEVICE" "$query_output" >&2
    exit 2
  fi
  blocking_processes="$(
    printf '%s\n' "$query_output" | awk -F',' \
      -v allow="$GPU_PROCESS_ALLOWLIST_REGEX" '
        {
          pid = $1
          name = $2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
          if (pid ~ /^[0-9]+$/ && name !~ allow) {
            print pid ":" name
          }
        }
      '
  )"
  if [[ -n "$blocking_processes" ]]; then
    printf 'GPU %s has non-allowlisted process(es); refusing a parallel run: %s\n' \
      "$GPU_DEVICE" "$(printf '%s' "$blocking_processes" | tr '\n' ',')" >&2
    exit 2
  fi
}

print_command() {
  local -a command=("$@")
  printf '%q ' "${command[@]}"
  printf '\n'
}

run_benchmark() {
  local engine="$1"
  local repeat="$2"
  local output="$3"
  shift 3
  local -a environment=("CUDA_VISIBLE_DEVICES=$GPU_DEVICE")
  local -a command=("$@")
  if [[ "$engine" == "wkvm" ]]; then
    environment+=("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
  fi

  printf 'run engine=%s repeat=%s output=%s\n' "$engine" "$repeat" "$output"
  print_command env "${environment[@]}" "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  refuse_parallel_gpu_run
  env "${environment[@]}" "${command[@]}"
  if [[ ! -s "$output" ]]; then
    printf 'Missing benchmark artifact: %s\n' "$output" >&2
    exit 1
  fi
}

declare -a report_artifacts=()
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  report_artifacts+=(
    "$ARTIFACT_DIR/sglang-source-r${repeat}.json"
    "$ARTIFACT_DIR/vllm-replay-r${repeat}.json"
    "$ARTIFACT_DIR/wkvm-replay-r${repeat}.json"
  )
done

print_path_manifest() {
  local repeat
  printf '# campaign_id=%s\n' "$CAMPAIGN_ID"
  printf '# memory_ceiling_mib=%s\n' "$MEMORY_CEILING_MIB"
  printf 'kind\trepeat\tpath\n'
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    printf 'trace\t%s\t%s\n' "$repeat" "$TRACE_DIR/b16_ctx36864_t8_o64-r${repeat}.trace.json"
    printf 'sglang-source\t%s\t%s\n' "$repeat" "$ARTIFACT_DIR/sglang-source-r${repeat}.json"
    printf 'vllm-replay\t%s\t%s\n' "$repeat" "$ARTIFACT_DIR/vllm-replay-r${repeat}.json"
    printf 'wkvm-replay\t%s\t%s\n' "$repeat" "$ARTIFACT_DIR/wkvm-replay-r${repeat}.json"
  done
  printf 'report\t-\t%s\n' "$MARKDOWN"
  printf 'summary\t-\t%s\n' "$SUMMARY_JSON"
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'output_dir=%s\n' "$OUT_DIR"
  printf 'path_manifest=%s\n' "$PATH_MANIFEST"
  print_path_manifest
else
  if ! command -v flock >/dev/null 2>&1; then
    printf '%s\n' 'flock is required' >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'nvidia-smi is required' >&2
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
  mkdir -p "$TRACE_DIR" "$ARTIFACT_DIR"
  exec 9>"$GPU_LOCK_FILE"
  if ! flock -n 9; then
    printf 'Another WKVM benchmark holds the GPU lock: %s\n' "$GPU_LOCK_FILE" >&2
    exit 2
  fi
  check_gpu_identity
  refuse_parallel_gpu_run
  print_path_manifest > "$PATH_MANIFEST"
  printf 'path_manifest=%s\n' "$PATH_MANIFEST"
fi

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  trace="$TRACE_DIR/b16_ctx36864_t8_o64-r${repeat}.trace.json"
  sglang_output="$ARTIFACT_DIR/sglang-source-r${repeat}.json"
  vllm_output="$ARTIFACT_DIR/vllm-replay-r${repeat}.json"
  wkvm_output="$ARTIFACT_DIR/wkvm-replay-r${repeat}.json"
  repeat_id="r${repeat}"
  sglang_run_id="$(generate_uuid)"
  vllm_run_id="$(generate_uuid)"
  wkvm_run_id="$(generate_uuid)"
  common_args=(
    --model-path "$MODEL_PATH"
    --sessions 16
    --turns 8
    --initial-context-tokens 36864
    --turn-input-tokens 32
    --output-tokens-per-turn 64
    --request-order-policy alternating
    --request-order-seed 0
    --gpu-memory-device "$GPU_DEVICE"
    --gpu-memory-sample-interval-s 0.1
    --campaign-id "$CAMPAIGN_ID"
    --repeat-id "$repeat_id"
    --memory-ceiling-mib "$MEMORY_CEILING_MIB"
  )

  run_benchmark sglang "$repeat" "$sglang_output" \
    "$SGLANG_PY" "$BENCHMARK" --engine sglang \
    "${common_args[@]}" \
    --run-id "$sglang_run_id" \
    --sglang-context-length 37616 \
    --sglang-max-total-tokens 608000 \
    --sglang-mem-fraction 0.94 \
    --sglang-chunked-prefill-size 2048 \
    --sglang-max-running-requests 16 \
    --sglang-attention-backend triton \
    --sglang-language-model-only \
    --sglang-decode-graph full \
    --sglang-prefill-graph disabled \
    --write-shared-history-trace-json "$trace" \
    --json "$sglang_output"
  if [[ "$DRY_RUN" != "1" && ! -s "$trace" ]]; then
    printf 'Missing shared-history trace: %s\n' "$trace" >&2
    exit 1
  fi

  run_benchmark vllm "$repeat" "$vllm_output" \
    "$VLLM_PY" "$BENCHMARK" --engine vllm \
    "${common_args[@]}" \
    --run-id "$vllm_run_id" \
    --max-model-len 37616 \
    --vllm-gpu-mem-util 0.82 \
    --vllm-max-num-batched-tokens 4096 \
    --vllm-language-model-only \
    --vllm-disable-inductor \
    --shared-history-trace-json "$trace" \
    --json "$vllm_output"

  run_benchmark wkvm "$repeat" "$wkvm_output" \
    "$WKVM_PY" "$BENCHMARK" --engine wkvm \
    "${common_args[@]}" \
    --run-id "$wkvm_run_id" \
    --slots 16 \
    --m-slots 32 \
    --route-chunk 2048 \
    --chunk 2048 \
    --prefill-microbatch-rows 2 \
    --continuation-prefill-microbatch-rows 8 \
    --decode-microbatch-rows 16 \
    --persistent-padded-decode-steps 64 \
    --no-persistent-padded-decode-cuda-graph \
    --persistent-padded-sliding-metadata-padding \
    --token-pool-capacity 114688 \
    --token-pool-max-context-len 37632 \
    --native-gemma-checkpoint-loader \
    --native-gemma-attention-backend triton_dense_gqa \
    --native-gemma-projection-backend separate \
    --enable-token-pool-attention \
    --enable-token-pool-triton \
    --enable-token-pool-paged-triton \
    --enable-token-pool-paged-split-triton \
    --token-pool-triton-strict \
    --token-pool-sliding-paged-metadata-only \
    --token-pool-route-boundary-batch \
    --shared-history-trace-json "$trace" \
    --json "$wkvm_output"
done

report_command=(
  "$WKVM_PY"
  "$REPORT"
  "${report_artifacts[@]}"
  --min-repeats "$REPEATS"
  --whole-device-memory-ceiling-mib "$MEMORY_CEILING_MIB"
  --allow-fail
  --markdown "$MARKDOWN"
  --summary-json "$SUMMARY_JSON"
)
printf 'report artifacts=%s markdown=%s summary=%s ceiling_mib=%s\n' \
  "${#report_artifacts[@]}" "$MARKDOWN" "$SUMMARY_JSON" "$MEMORY_CEILING_MIB"
print_command "${report_command[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${report_command[@]}"
if [[ ! -s "$MARKDOWN" || ! -s "$SUMMARY_JSON" ]]; then
  printf '%s\n' 'Exploratory report outputs are missing' >&2
  exit 1
fi
printf 'report=%s\nsummary=%s\n' "$MARKDOWN" "$SUMMARY_JSON"
