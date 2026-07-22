#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-E4B-it}"
WKVM_PY="${WKVM_PY:-$ROOT/../HRM-Text/.venv/bin/python}"
VLLM_PY="${VLLM_PY:-/run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/vllm/bin/python}"
SGLANG_PY="${SGLANG_PY:-/run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/sglang/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/../results/4090/wkvm_10x_http_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
TURNS="${TURNS:-8}"
INITIAL_CONTEXT_TOKENS="${INITIAL_CONTEXT_TOKENS:-36864}"
TURN_INPUT_TOKENS="${TURN_INPUT_TOKENS:-32}"
OUTPUT_TOKENS_PER_TURN="${OUTPUT_TOKENS_PER_TURN:-64}"
REPORT_CLAIM_SCOPE="${REPORT_CLAIM_SCOPE:-continuation}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_FAIL="${ALLOW_FAIL:-0}"
STRICT_PUBLICATION="${STRICT_PUBLICATION:-0}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
MEMORY_CEILING_MIB="${MEMORY_CEILING_MIB:-24200}"
MAX_IDLE_BASELINE_MIB="${MAX_IDLE_BASELINE_MIB:-1024}"
GPU_MEMORY_SAMPLE_INTERVAL_S="${GPU_MEMORY_SAMPLE_INTERVAL_S:-0.1}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
SERVER_STOP_TIMEOUT_S="${SERVER_STOP_TIMEOUT_S:-45}"
GPU_CLEAR_TIMEOUT_S="${GPU_CLEAR_TIMEOUT_S:-60}"
HEALTH_POLL_INTERVAL_S="${HEALTH_POLL_INTERVAL_S:-2}"
HOST="${HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-8002}"
WKVM_PORT="${WKVM_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8001}"
SGLANG_CUDA_GRAPH_BACKEND_PREFILL="${SGLANG_CUDA_GRAPH_BACKEND_PREFILL:-disabled}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL="${WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL:-1}"
VLLM_KV_SHARING_FAST_PREFILL="${VLLM_KV_SHARING_FAST_PREFILL:-1}"
VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-}"
VLLM_COMPILE_MODE="${VLLM_COMPILE_MODE:-0}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS="${WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS:-8}"
WKVM_DECODE_MICROBATCH_ROWS="${WKVM_DECODE_MICROBATCH_ROWS:-16}"
WKVM_PERSISTENT_PADDED_DECODE_STEPS="${WKVM_PERSISTENT_PADDED_DECODE_STEPS:-64}"
BENCHMARK="${BENCHMARK:-$ROOT/experiments/gemma_multiturn_http_bench.py}"
REPORT="${REPORT:-$ROOT/experiments/multiturn_http_10x_report.py}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-${TMPDIR:-/tmp}/wkvm-10x-http-gpu-${GPU_DEVICE//[^[:alnum:]_.-]/_}.lock}"
GPU_PROCESS_ALLOWLIST_REGEX="${GPU_PROCESS_ALLOWLIST_REGEX:-gnome-remote-desktop-daemon|ptyxis|nautilus|gnome-text-editor|chrome|/papers$|/baobab$}"
SESSIONS=16

TRACE_DIR="$OUT_DIR/traces"
ARTIFACT_DIR="$OUT_DIR/artifacts"
LOG_DIR="$OUT_DIR/logs"
SERVER_INFO_DIR="$OUT_DIR/server-info"
MARKDOWN="$OUT_DIR/provider_http_10x_report.md"
SUMMARY_JSON="$OUT_DIR/provider_http_10x_summary.json"
PATH_MANIFEST="$OUT_DIR/artifact_paths.tsv"

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_ENGINE=""
ACTIVE_SERVER_REPEAT=""
ACTIVE_SERVER_LOG=""

SGLANG_PROGRAM='import json,sys; from incumbent_gemma_bench import sglang_language_model_override; from sglang.srt.entrypoints.http_server import launch_server; from sglang.srt.server_args import ServerArgs; model=sys.argv[1]; launch_server(ServerArgs(model_path=model, served_model_name=sys.argv[2], host=sys.argv[3], port=int(sys.argv[4]), dtype="bfloat16", context_length=int(sys.argv[8]), max_total_tokens=int(sys.argv[9]), mem_fraction_static=0.94, chunked_prefill_size=int(sys.argv[5]), max_running_requests=int(sys.argv[6]), attention_backend="triton", json_model_override_args=json.dumps(sglang_language_model_override(model), separators=(",", ":"), sort_keys=True), cuda_graph_backend_decode="full", cuda_graph_backend_prefill=sys.argv[7], enable_cache_report=True, enable_multimodal=False, skip_tokenizer_init=True, log_level="warning"))'

generate_uuid() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    tr -d '\n' < /proc/sys/kernel/random/uuid
    return
  fi
  "$WKVM_PY" -c 'import uuid; print(uuid.uuid4())'
}

print_command() {
  local -a command=("$@")
  printf '%q ' "${command[@]}"
  printf '\n'
}

quoted_command() {
  local -a command=("$@")
  printf '%q ' "${command[@]}"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    printf '%s must be an integer >= 1: %s\n' "$name" "$value" >&2
    exit 1
  fi
}

validate_boolean() {
  local name="$1"
  local value="$2"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    printf '%s must be 0 or 1: %s\n' "$name" "$value" >&2
    exit 1
  fi
}

validate_fraction() {
  local name="$1"
  local value="$2"
  if ! awk -v value="$value" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
    printf '%s must be a number greater than 0 and at most 1: %s\n' \
      "$name" "$value" >&2
    exit 1
  fi
}

validate_port() {
  local name="$1"
  local value="$2"
  validate_positive_integer "$name" "$value"
  if ((value > 65535)); then
    printf '%s must be <= 65535: %s\n' "$name" "$value" >&2
    exit 1
  fi
}

blocking_gpu_processes() {
  local query_output
  if ! query_output="$(
    nvidia-smi -i "$GPU_DEVICE" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>&1
  )"; then
    printf 'Could not inspect compute processes on GPU %s: %s\n' \
      "$GPU_DEVICE" "$query_output" >&2
    return 2
  fi
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
}

refuse_parallel_gpu_run() {
  local blocking
  if ! blocking="$(blocking_gpu_processes)"; then
    return 2
  fi
  if [[ -n "$blocking" ]]; then
    printf 'GPU %s has non-allowlisted process(es): %s\n' \
      "$GPU_DEVICE" "$(printf '%s' "$blocking" | tr '\n' ',')" >&2
    return 2
  fi
}

wait_for_gpu_clear() {
  local deadline=$((SECONDS + GPU_CLEAR_TIMEOUT_S))
  local blocking
  while ((SECONDS < deadline)); do
    if ! blocking="$(blocking_gpu_processes)"; then
      return 2
    fi
    if [[ -z "$blocking" ]]; then
      return 0
    fi
    sleep 1
  done
  printf 'GPU %s did not clear within %s seconds: %s\n' \
    "$GPU_DEVICE" "$GPU_CLEAR_TIMEOUT_S" \
    "$(printf '%s' "$blocking" | tr '\n' ',')" >&2
  return 2
}

capture_prelaunch_memory() {
  local used_mib
  refuse_parallel_gpu_run
  if ! used_mib="$(
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n 1
  )"; then
    printf 'Could not capture prelaunch memory for GPU %s\n' "$GPU_DEVICE" >&2
    return 2
  fi
  if [[ ! "$used_mib" =~ ^[0-9]+$ ]]; then
    printf 'Invalid prelaunch memory sample for GPU %s: %s\n' \
      "$GPU_DEVICE" "$used_mib" >&2
    return 2
  fi
  if ((used_mib > MAX_IDLE_BASELINE_MIB)); then
    printf 'GPU %s prelaunch memory %s MiB exceeds idle limit %s MiB\n' \
      "$GPU_DEVICE" "$used_mib" "$MAX_IDLE_BASELINE_MIB" >&2
    return 2
  fi
  printf '%s' "$used_mib"
}

check_gpu_identity() {
  local gpu_name
  local total_mib
  gpu_name="$(
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=name \
      --format=csv,noheader 2>/dev/null | head -n 1
  )"
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

start_server() {
  local engine="$1"
  local repeat="$2"
  local port="$3"
  local log_path="$4"
  shift 4
  local -a command=("$@")
  printf 'server start engine=%s repeat=%s port=%s log=%s\n' \
    "$engine" "$repeat" "$port" "$log_path"
  print_command setsid "${command[@]}"
  ACTIVE_SERVER_ENGINE="$engine"
  ACTIVE_SERVER_REPEAT="$repeat"
  ACTIVE_SERVER_LOG="$log_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    ACTIVE_SERVER_PID="dry-run"
    return 0
  fi
  setsid "${command[@]}" >"$log_path" 2>&1 &
  ACTIVE_SERVER_PID=$!
}

wait_for_health() {
  local url="$1"
  printf 'health engine=%s repeat=%s url=%s\n' \
    "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" "$url"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl -fsS --max-time 5 "$url"
    return 0
  fi
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
      printf 'Server %s r%s exited before health check; log tail follows\n' \
        "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" >&2
      tail -n 80 "$ACTIVE_SERVER_LOG" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$HEALTH_POLL_INTERVAL_S"
  done
  printf 'Server %s r%s did not become healthy within %s seconds\n' \
    "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" \
    "$SERVER_READY_TIMEOUT_S" >&2
  tail -n 80 "$ACTIVE_SERVER_LOG" >&2 || true
  return 1
}

capture_server_info() {
  local engine="$1"
  local repeat="$2"
  local phase="$3"
  local url="$4"
  local output="$5"
  printf 'server-info engine=%s repeat=%s phase=%s url=%s output=%s\n' \
    "$engine" "$repeat" "$phase" "$url" "$output"
  print_command curl -fsS --max-time 15 "$url" -o "$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  curl -fsS --max-time 15 "$url" -o "$output"
}

stop_active_server() {
  if [[ -z "$ACTIVE_SERVER_PID" ]]; then
    return 0
  fi
  printf 'server stop engine=%s repeat=%s pid=%s\n' \
    "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" "$ACTIVE_SERVER_PID"
  if [[ "$DRY_RUN" == "1" ]]; then
    ACTIVE_SERVER_PID=""
    ACTIVE_SERVER_ENGINE=""
    ACTIVE_SERVER_REPEAT=""
    ACTIVE_SERVER_LOG=""
    return 0
  fi
  local pid="$ACTIVE_SERVER_PID"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + SERVER_STOP_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if ! kill -0 -- "-$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    printf 'Force-stopping server process group %s\n' "$pid" >&2
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_ENGINE=""
  ACTIVE_SERVER_REPEAT=""
  ACTIVE_SERVER_LOG=""
  wait_for_gpu_clear
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  stop_active_server || true
  exit "$status"
}

run_client() {
  local engine="$1"
  local repeat="$2"
  local baseline_mib="$3"
  local artifact="$4"
  local client_log="$5"
  shift 5
  local -a command=("$@")
  printf 'run engine=%s repeat=%s baseline_mib=%s artifact=%s client_log=%s\n' \
    "$engine" "$repeat" "$baseline_mib" "$artifact" "$client_log"
  print_command "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  set +e
  "${command[@]}" 2>&1 | tee "$client_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ! -s "$artifact" ]]; then
    printf 'Missing benchmark artifact: %s\n' "$artifact" >&2
    return 1
  fi
  return "$status"
}

record_baseline() {
  local engine="$1"
  local repeat="$2"
  local baseline_mib="$3"
  local output="$SERVER_INFO_DIR/${engine}-r${repeat}.prelaunch-memory-mib.txt"
  printf 'baseline engine=%s repeat=%s value_mib=%s output=%s\n' \
    "$engine" "$repeat" "$baseline_mib" "$output"
  if [[ "$DRY_RUN" != "1" ]]; then
    printf '%s\n' "$baseline_mib" >"$output"
  fi
}

print_path_manifest() {
  local repeat
  printf '# campaign_id=%s\n' "$CAMPAIGN_ID"
  printf '# memory_ceiling_mib=%s\n' "$MEMORY_CEILING_MIB"
  printf '# sessions=%s\n' "$SESSIONS"
  printf '# turns=%s\n' "$TURNS"
  printf '# initial_context_tokens=%s\n' "$INITIAL_CONTEXT_TOKENS"
  printf '# turn_input_tokens=%s\n' "$TURN_INPUT_TOKENS"
  printf '# output_tokens_per_turn=%s\n' "$OUTPUT_TOKENS_PER_TURN"
  printf '# report_claim_scope=%s\n' "$REPORT_CLAIM_SCOPE"
  printf '# strict_publication=%s\n' "$STRICT_PUBLICATION"
  printf '# required_model_len=%s\n' "$REQUIRED_MODEL_LEN"
  printf '# incumbent_context_length=%s\n' "$INCUMBENT_CONTEXT_LENGTH"
  printf '# wkvm_token_pool_max_context_len=%s\n' \
    "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN"
  printf '# wkvm_continuation_prefill_microbatch_rows=%s\n' \
    "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS"
  printf '# wkvm_decode_microbatch_rows=%s\n' \
    "$WKVM_DECODE_MICROBATCH_ROWS"
  printf '# wkvm_persistent_padded_decode_steps=%s\n' \
    "$WKVM_PERSISTENT_PADDED_DECODE_STEPS"
  printf '# sglang_max_total_tokens=%s\n' "$SGLANG_MAX_TOTAL_TOKENS"
  printf '# workload_tag=%s\n' "$WORKLOAD_TAG"
  printf 'kind\trepeat\tpath\n'
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    printf 'trace\t%s\t%s\n' "$repeat" \
      "$TRACE_DIR/${WORKLOAD_TAG}-r${repeat}.trace.json"
    printf 'sglang-source\t%s\t%s\n' "$repeat" \
      "$ARTIFACT_DIR/sglang-source${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
    printf 'wkvm-replay\t%s\t%s\n' "$repeat" \
      "$ARTIFACT_DIR/wkvm-replay${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
    printf 'vllm-replay\t%s\t%s\n' "$repeat" \
      "$ARTIFACT_DIR/vllm-replay${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
    printf 'sglang-log\t%s\t%s\n' "$repeat" "$LOG_DIR/sglang-r${repeat}.server.log"
    printf 'wkvm-log\t%s\t%s\n' "$repeat" "$LOG_DIR/wkvm-r${repeat}.server.log"
    printf 'vllm-log\t%s\t%s\n' "$repeat" "$LOG_DIR/vllm-r${repeat}.server.log"
  done
  printf 'report\t-\t%s\n' "$MARKDOWN"
  printf 'summary\t-\t%s\n' "$SUMMARY_JSON"
}

validate_positive_integer REPEATS "$REPEATS"
validate_positive_integer TURNS "$TURNS"
validate_positive_integer INITIAL_CONTEXT_TOKENS "$INITIAL_CONTEXT_TOKENS"
validate_positive_integer TURN_INPUT_TOKENS "$TURN_INPUT_TOKENS"
validate_positive_integer OUTPUT_TOKENS_PER_TURN "$OUTPUT_TOKENS_PER_TURN"
validate_positive_integer MEMORY_CEILING_MIB "$MEMORY_CEILING_MIB"
validate_positive_integer MAX_IDLE_BASELINE_MIB "$MAX_IDLE_BASELINE_MIB"
validate_positive_integer SERVER_READY_TIMEOUT_S "$SERVER_READY_TIMEOUT_S"
validate_positive_integer SERVER_STOP_TIMEOUT_S "$SERVER_STOP_TIMEOUT_S"
validate_positive_integer GPU_CLEAR_TIMEOUT_S "$GPU_CLEAR_TIMEOUT_S"
validate_positive_integer SGLANG_MAX_RUNNING_REQUESTS "$SGLANG_MAX_RUNNING_REQUESTS"
validate_positive_integer VLLM_MAX_NUM_BATCHED_TOKENS "$VLLM_MAX_NUM_BATCHED_TOKENS"
validate_positive_integer WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS \
  "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS"
validate_positive_integer WKVM_DECODE_MICROBATCH_ROWS \
  "$WKVM_DECODE_MICROBATCH_ROWS"
validate_positive_integer WKVM_PERSISTENT_PADDED_DECODE_STEPS \
  "$WKVM_PERSISTENT_PADDED_DECODE_STEPS"
validate_fraction VLLM_GPU_MEMORY_UTILIZATION "$VLLM_GPU_MEMORY_UTILIZATION"
validate_boolean WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL \
  "$WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL"
validate_boolean VLLM_KV_SHARING_FAST_PREFILL \
  "$VLLM_KV_SHARING_FAST_PREFILL"
if [[ "$SGLANG_CHUNKED_PREFILL_SIZE" != "-1" ]] && \
   [[ ! "$SGLANG_CHUNKED_PREFILL_SIZE" =~ ^[0-9]+$ || "$SGLANG_CHUNKED_PREFILL_SIZE" -lt 1 ]]; then
  printf '%s must be -1 or an integer >= 1: %s\n' \
    SGLANG_CHUNKED_PREFILL_SIZE "$SGLANG_CHUNKED_PREFILL_SIZE" >&2
  exit 1
fi
SGLANG_CUDA_GRAPH_BACKEND_PREFILL="${SGLANG_CUDA_GRAPH_BACKEND_PREFILL,,}"
case "$SGLANG_CUDA_GRAPH_BACKEND_PREFILL" in
  breakable|disabled|tc_piecewise) ;;
  *)
    printf '%s must be breakable, disabled, or tc_piecewise: %s\n' \
      SGLANG_CUDA_GRAPH_BACKEND_PREFILL \
      "$SGLANG_CUDA_GRAPH_BACKEND_PREFILL" >&2
    exit 1
    ;;
esac
case "${VLLM_COMPILE_MODE^^}" in
  0|NONE) VLLM_COMPILE_MODE=0 ;;
  1|STOCK_TORCH_COMPILE) VLLM_COMPILE_MODE=1 ;;
  2|DYNAMO_TRACE_ONCE) VLLM_COMPILE_MODE=2 ;;
  3|VLLM_COMPILE) VLLM_COMPILE_MODE=3 ;;
  *)
    printf '%s must be 0-3 or a vLLM compilation mode name: %s\n' \
      VLLM_COMPILE_MODE "$VLLM_COMPILE_MODE" >&2
    exit 1
    ;;
esac
if [[ -z "$VLLM_CUDAGRAPH_MODE" ]]; then
  if [[ "$VLLM_KV_SHARING_FAST_PREFILL" == "1" ]]; then
    if [[ "$VLLM_COMPILE_MODE" == "3" ]]; then
      VLLM_CUDAGRAPH_MODE=FULL_AND_PIECEWISE
    else
      VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY
    fi
  else
    VLLM_CUDAGRAPH_MODE=FULL
  fi
else
  VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE^^}"
fi
case "$VLLM_CUDAGRAPH_MODE" in
  NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
  *)
    printf '%s is not a supported vLLM 0.24.0 cudagraph mode: %s\n' \
      VLLM_CUDAGRAPH_MODE "$VLLM_CUDAGRAPH_MODE" >&2
    exit 1
    ;;
esac
validate_port SGLANG_PORT "$SGLANG_PORT"
validate_port WKVM_PORT "$WKVM_PORT"
validate_port VLLM_PORT "$VLLM_PORT"
if [[ "$SGLANG_PORT" == "$WKVM_PORT" || "$SGLANG_PORT" == "$VLLM_PORT" || "$WKVM_PORT" == "$VLLM_PORT" ]]; then
  printf '%s\n' 'SGLANG_PORT, WKVM_PORT, and VLLM_PORT must be distinct' >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  printf '%s\n' 'DRY_RUN must be 0 or 1' >&2
  exit 1
fi
if [[ "$ALLOW_FAIL" != "0" && "$ALLOW_FAIL" != "1" ]]; then
  printf '%s\n' 'ALLOW_FAIL must be 0 or 1' >&2
  exit 1
fi
validate_boolean STRICT_PUBLICATION "$STRICT_PUBLICATION"
case "$REPORT_CLAIM_SCOPE" in
  continuation|full-session) ;;
  *)
    printf '%s must be continuation or full-session: %s\n' \
      REPORT_CLAIM_SCOPE "$REPORT_CLAIM_SCOPE" >&2
    exit 1
    ;;
esac

REQUIRED_MODEL_LEN=$((
  INITIAL_CONTEXT_TOKENS
  + TURNS * OUTPUT_TOKENS_PER_TURN
  + (TURNS - 1) * TURN_INPUT_TOKENS
))
ALIGNED_REQUIRED_MODEL_LEN=$((((REQUIRED_MODEL_LEN + 15) / 16) * 16))
INCUMBENT_CONTEXT_LENGTH=$((ALIGNED_REQUIRED_MODEL_LEN + 16))
WKVM_TOKEN_POOL_MAX_CONTEXT_LEN=$((ALIGNED_REQUIRED_MODEL_LEN + 32))
SGLANG_MAX_TOTAL_TOKENS=$((
  SESSIONS * (ALIGNED_REQUIRED_MODEL_LEN + 400)
))
WKVM_MAX_COMPLETED_REQUESTS=$((SESSIONS * TURNS + SESSIONS))
if [[ "$TURNS" == "8" && "$INITIAL_CONTEXT_TOKENS" == "36864" && \
      "$TURN_INPUT_TOKENS" == "32" && "$OUTPUT_TOKENS_PER_TURN" == "64" ]]; then
  WORKLOAD_TAG=b16_ctx36864_t8_o64
  ARTIFACT_WORKLOAD_SUFFIX=""
else
  WORKLOAD_TAG="b16_ctx${INITIAL_CONTEXT_TOKENS}_d${TURN_INPUT_TOKENS}_t${TURNS}_o${OUTPUT_TOKENS_PER_TURN}"
  ARTIFACT_WORKLOAD_SUFFIX="-${WORKLOAD_TAG}"
fi

wkvm_fast_prefill_json=false
declare -a wkvm_fast_prefill_args=()
if [[ "$WKVM_NATIVE_GEMMA_KV_SHARING_FAST_PREFILL" == "1" ]]; then
  wkvm_fast_prefill_json=true
  wkvm_fast_prefill_args+=(--native-gemma-kv-sharing-fast-prefill)
fi
if [[ "$VLLM_KV_SHARING_FAST_PREFILL" == "1" ]]; then
  vllm_fast_prefill_json=true
  vllm_fast_prefill_arg=--kv-sharing-fast-prefill
else
  vllm_fast_prefill_json=false
  vllm_fast_prefill_arg=--no-kv-sharing-fast-prefill
fi
printf -v SGLANG_CONFIG \
  '{"attention_backend":"triton","chunked_prefill_size":%s,"context_length":%s,"cuda_graph_backend_decode":"full","cuda_graph_backend_prefill":"%s","disable_radix_cache":false,"enable_cache_report":true,"enable_multimodal":false,"max_running_requests":%s,"max_total_tokens":%s,"mem_fraction_static":0.94,"skip_tokenizer_init":true}' \
  "$SGLANG_CHUNKED_PREFILL_SIZE" "$INCUMBENT_CONTEXT_LENGTH" \
  "$SGLANG_CUDA_GRAPH_BACKEND_PREFILL" "$SGLANG_MAX_RUNNING_REQUESTS" \
  "$SGLANG_MAX_TOTAL_TOKENS"
printf -v WKVM_CONFIG \
  '{"backlog_min":64,"batch_wait_s":0.01,"continuation_prefill_microbatch_rows":%s,"decode_microbatch_rows":%s,"enable_token_pool_attention":true,"max_queue":64,"m_slots":32,"native_gemma_attention_backend":"triton_dense_gqa","native_gemma_kv_sharing_fast_prefill":%s,"native_gemma_projection_backend":"separate","persistent_padded_decode_cuda_graph":false,"persistent_padded_decode_steps":%s,"prefill_chunk":2048,"prefill_microbatch_rows":2,"route_chunk":2048,"slots":16,"stream_flush_tokens":1,"token_pool_capacity":114688,"token_pool_max_context_len":%s,"token_pool_paged_block_size":16}' \
  "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS" \
  "$WKVM_DECODE_MICROBATCH_ROWS" "$wkvm_fast_prefill_json" \
  "$WKVM_PERSISTENT_PADDED_DECODE_STEPS" \
  "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN"
printf -v VLLM_COMPILATION_CONFIG \
  '{"mode":%s,"cudagraph_mode":"%s","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  "$VLLM_COMPILE_MODE" "$VLLM_CUDAGRAPH_MODE"
printf -v VLLM_CONFIG \
  '{"compilation_config":%s,"enable_prefix_caching":true,"gpu_memory_utilization":%s,"kv_sharing_fast_prefill":%s,"language_model_only":true,"max_model_len":%s,"max_num_batched_tokens":%s,"max_num_seqs":16,"prefix_caching":true}' \
  "$VLLM_COMPILATION_CONFIG" "$VLLM_GPU_MEMORY_UTILIZATION" \
  "$vllm_fast_prefill_json" \
  "$INCUMBENT_CONTEXT_LENGTH" "$VLLM_MAX_NUM_BATCHED_TOKENS"

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
for required_file in "$BENCHMARK" "$REPORT"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Required file not found: %s\n' "$required_file" >&2
    exit 1
  fi
done
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
TRACE_DIR="$OUT_DIR/traces"
ARTIFACT_DIR="$OUT_DIR/artifacts"
LOG_DIR="$OUT_DIR/logs"
SERVER_INFO_DIR="$OUT_DIR/server-info"
MARKDOWN="$OUT_DIR/provider_http_10x_report.md"
SUMMARY_JSON="$OUT_DIR/provider_http_10x_summary.json"
PATH_MANIFEST="$OUT_DIR/artifact_paths.tsv"

if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="wkvm-http-4090-$(generate_uuid)"
fi

declare -a report_artifacts=()
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  report_artifacts+=(
    "$ARTIFACT_DIR/sglang-source${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
    "$ARTIFACT_DIR/wkvm-replay${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
    "$ARTIFACT_DIR/vllm-replay${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
  )
done

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'output_dir=%s\npath_manifest=%s\n' "$OUT_DIR" "$PATH_MANIFEST"
  print_path_manifest
else
  for command_name in flock nvidia-smi curl setsid tee; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      printf 'Required command not found: %s\n' "$command_name" >&2
      exit 1
    fi
  done
  if [[ -e "$OUT_DIR" && ! -d "$OUT_DIR" ]]; then
    printf 'OUT_DIR exists and is not a directory: %s\n' "$OUT_DIR" >&2
    exit 1
  fi
  if [[ -d "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
    printf 'OUT_DIR must be empty before a benchmark run: %s\n' "$OUT_DIR" >&2
    exit 1
  fi
  mkdir -p "$TRACE_DIR" "$ARTIFACT_DIR" "$LOG_DIR" "$SERVER_INFO_DIR"
  exec 9>"$GPU_LOCK_FILE"
  if ! flock -n 9; then
    printf 'Another WKVM HTTP campaign holds the GPU lock: %s\n' \
      "$GPU_LOCK_FILE" >&2
    exit 2
  fi
  check_gpu_identity
  refuse_parallel_gpu_run
  print_path_manifest >"$PATH_MANIFEST"
  printf 'path_manifest=%s\n' "$PATH_MANIFEST"
fi

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  repeat_id="r${repeat}"
  trace="$TRACE_DIR/${WORKLOAD_TAG}-r${repeat}.trace.json"
  sglang_artifact="$ARTIFACT_DIR/sglang-source${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
  wkvm_artifact="$ARTIFACT_DIR/wkvm-replay${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
  vllm_artifact="$ARTIFACT_DIR/vllm-replay${ARTIFACT_WORKLOAD_SUFFIX}-r${repeat}.json"
  sglang_run_id="$(generate_uuid)"
  wkvm_run_id="$(generate_uuid)"
  vllm_run_id="$(generate_uuid)"

  common_http_args=(
    --model "$SERVED_MODEL_NAME"
    --sessions "$SESSIONS"
    --turns "$TURNS"
    --initial-context-tokens "$INITIAL_CONTEXT_TOKENS"
    --turn-input-tokens "$TURN_INPUT_TOKENS"
    --output-tokens-per-turn "$OUTPUT_TOKENS_PER_TURN"
    --request-order-policy alternating
    --request-order-seed 0
    --request-timeout-s 600
    --gpu-memory-device "$GPU_DEVICE"
    --gpu-memory-sample-interval-s "$GPU_MEMORY_SAMPLE_INTERVAL_S"
    --memory-ceiling-mib "$MEMORY_CEILING_MIB"
    --campaign-id "$CAMPAIGN_ID"
    --repeat-id "$repeat_id"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    sglang_baseline="<prelaunch-nvidia-smi>"
  else
    sglang_baseline="$(capture_prelaunch_memory)"
  fi
  record_baseline sglang "$repeat" "$sglang_baseline"
  sglang_log="$LOG_DIR/sglang-r${repeat}.server.log"
  sglang_server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU_DEVICE"
    "HF_HUB_OFFLINE=1"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONPATH=$ROOT/experiments:$ROOT"
    "$SGLANG_PY"
    -c "$SGLANG_PROGRAM"
    "$MODEL_PATH" "$SERVED_MODEL_NAME" "$HOST" "$SGLANG_PORT"
    "$SGLANG_CHUNKED_PREFILL_SIZE"
    "$SGLANG_MAX_RUNNING_REQUESTS"
    "$SGLANG_CUDA_GRAPH_BACKEND_PREFILL"
    "$INCUMBENT_CONTEXT_LENGTH"
    "$SGLANG_MAX_TOTAL_TOKENS"
  )
  sglang_launch_text="$(quoted_command setsid "${sglang_server_command[@]}")"
  start_server sglang "$repeat" "$SGLANG_PORT" "$sglang_log" \
    "${sglang_server_command[@]}"
  wait_for_health "http://$HOST:$SGLANG_PORT/health"
  capture_server_info sglang "$repeat" pre \
    "http://$HOST:$SGLANG_PORT/server_info" \
    "$SERVER_INFO_DIR/sglang-r${repeat}.pre.json"
  sglang_status=0
  run_client sglang "$repeat" "$sglang_baseline" "$sglang_artifact" \
    "$LOG_DIR/sglang-r${repeat}.client.log" \
    env "CUDA_VISIBLE_DEVICES=$GPU_DEVICE" "PYTHONPATH=$ROOT" \
    "$WKVM_PY" "$BENCHMARK" --engine sglang \
    "${common_http_args[@]}" \
    --base-url "http://$HOST:$SGLANG_PORT" \
    --endpoint /generate \
    --sglang-native-generate \
    --teacher-forcing-field none \
    --gpu-memory-baseline-used-mib "$sglang_baseline" \
    --semantics full_kv \
    --engine-version 0.5.14 \
    --engine-version-source frozen_campaign \
    --target-server-launch-command "$sglang_launch_text" \
    --target-server-config-json "$SGLANG_CONFIG" \
    --server-metrics-url "http://$HOST:$SGLANG_PORT/server_info" \
    --run-id "$sglang_run_id" \
    --write-shared-history-trace-json "$trace" \
    --json "$sglang_artifact" || sglang_status=$?
  sglang_info_status=0
  capture_server_info sglang "$repeat" post \
    "http://$HOST:$SGLANG_PORT/server_info" \
    "$SERVER_INFO_DIR/sglang-r${repeat}.post.json" || sglang_info_status=$?
  stop_active_server
  if ((sglang_status != 0 || sglang_info_status != 0)); then
    exit 1
  fi
  if [[ "$DRY_RUN" != "1" && ! -s "$trace" ]]; then
    printf 'Missing shared-history trace: %s\n' "$trace" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    wkvm_baseline="<prelaunch-nvidia-smi>"
  else
    wkvm_baseline="$(capture_prelaunch_memory)"
  fi
  record_baseline wkvm "$repeat" "$wkvm_baseline"
  wkvm_log="$LOG_DIR/wkvm-r${repeat}.server.log"
  wkvm_server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU_DEVICE"
    "HF_HUB_OFFLINE=1"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONPATH=$ROOT"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "WKVM_ENABLE_TOKEN_POOL_TRITON=1"
    "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON=1"
    "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON=1"
    "WKVM_TOKEN_POOL_TRITON_STRICT=1"
    "WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY=1"
    "WKVM_TOKEN_POOL_ROUTE_BOUNDARY_BATCH=1"
    "$WKVM_PY" -m wkvm.gemma_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --port "$WKVM_PORT"
    --slots 16
    --max-queue 64
    --request-timeout-s 600
    --max-request-body-bytes 67108864
    --request-read-timeout-s 600
    --stream-flush-tokens 1
    --max-completed-requests "$WKVM_MAX_COMPLETED_REQUESTS"
    --ignore-eos
    --enable-token-session-teacher-forcing
    --batch-wait-s 0.01
    --prefill-chunk 2048
    --prefill-microbatch-rows 2
    --continuation-prefill-microbatch-rows \
    "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS"
    --decode-microbatch-rows "$WKVM_DECODE_MICROBATCH_ROWS"
    --persistent-padded-decode-steps "$WKVM_PERSISTENT_PADDED_DECODE_STEPS"
    --persistent-padded-sliding-metadata-padding
    --enable-token-pool-attention
    --token-pool-max-context-len "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN"
    --token-pool-capacity 114688
    --token-pool-paged-block-size 16
    --m-slots 32
    --route-chunk 2048
    --native-gemma-checkpoint-loader
    "${wkvm_fast_prefill_args[@]}"
    --native-gemma-attention-backend triton_dense_gqa
    --native-gemma-projection-backend separate
  )
  wkvm_launch_text="$(quoted_command setsid "${wkvm_server_command[@]}")"
  start_server wkvm "$repeat" "$WKVM_PORT" "$wkvm_log" \
    "${wkvm_server_command[@]}"
  wait_for_health "http://$HOST:$WKVM_PORT/health"
  capture_server_info wkvm "$repeat" pre \
    "http://$HOST:$WKVM_PORT/metrics" \
    "$SERVER_INFO_DIR/wkvm-r${repeat}.pre.json"
  wkvm_status=0
  run_client wkvm "$repeat" "$wkvm_baseline" "$wkvm_artifact" \
    "$LOG_DIR/wkvm-r${repeat}.client.log" \
    env "CUDA_VISIBLE_DEVICES=$GPU_DEVICE" "PYTHONPATH=$ROOT" \
    "$WKVM_PY" "$BENCHMARK" --engine wkvm \
    "${common_http_args[@]}" \
    --base-url "http://$HOST:$WKVM_PORT" \
    --endpoint /v1/stream \
    --gpu-memory-baseline-used-mib "$wkvm_baseline" \
    --semantics routed_span_approximate \
    --engine-version "git:$(git -C "$ROOT" rev-parse HEAD)" \
    --engine-version-source frozen_campaign \
    --target-server-launch-command "$wkvm_launch_text" \
    --target-server-config-json "$WKVM_CONFIG" \
    --server-metrics-url "http://$HOST:$WKVM_PORT/metrics" \
    --run-id "$wkvm_run_id" \
    --shared-history-trace-json "$trace" \
    --json "$wkvm_artifact" || wkvm_status=$?
  wkvm_info_status=0
  capture_server_info wkvm "$repeat" post \
    "http://$HOST:$WKVM_PORT/metrics" \
    "$SERVER_INFO_DIR/wkvm-r${repeat}.post.json" || wkvm_info_status=$?
  stop_active_server
  if ((wkvm_status != 0 || wkvm_info_status != 0)); then
    exit 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    vllm_baseline="<prelaunch-nvidia-smi>"
  else
    vllm_baseline="$(capture_prelaunch_memory)"
  fi
  record_baseline vllm "$repeat" "$vllm_baseline"
  vllm_log="$LOG_DIR/vllm-r${repeat}.server.log"
  vllm_server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU_DEVICE"
    "HF_HUB_OFFLINE=1"
    "TOKENIZERS_PARALLELISM=false"
    "VLLM_SERVER_DEV_MODE=1"
    "PYTHONPATH=$ROOT"
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$VLLM_PORT"
    --dtype bfloat16
    --max-model-len "$INCUMBENT_CONTEXT_LENGTH"
    --max-num-seqs 16
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --enable-prefix-caching
    "$vllm_fast_prefill_arg"
    --language-model-only
    --limit-mm-per-prompt '{"image":0,"audio":0}'
    --compilation-config "$VLLM_COMPILATION_CONFIG"
    --logits-processors experiments.vllm_shared_history_logits:SharedHistoryLogitsProcessor
    --enable-prompt-tokens-details
  )
  vllm_launch_text="$(quoted_command setsid "${vllm_server_command[@]}")"
  start_server vllm "$repeat" "$VLLM_PORT" "$vllm_log" \
    "${vllm_server_command[@]}"
  wait_for_health "http://$HOST:$VLLM_PORT/health"
  capture_server_info vllm "$repeat" pre \
    "http://$HOST:$VLLM_PORT/server_info?config_format=json" \
    "$SERVER_INFO_DIR/vllm-r${repeat}.pre.json"
  vllm_status=0
  run_client vllm "$repeat" "$vllm_baseline" "$vllm_artifact" \
    "$LOG_DIR/vllm-r${repeat}.client.log" \
    env "CUDA_VISIBLE_DEVICES=$GPU_DEVICE" "PYTHONPATH=$ROOT" \
    "$WKVM_PY" "$BENCHMARK" --engine vllm \
    "${common_http_args[@]}" \
    --base-url "http://$HOST:$VLLM_PORT" \
    --endpoint /v1/completions \
    --gpu-memory-baseline-used-mib "$vllm_baseline" \
    --semantics full_kv \
    --engine-version 0.24.0 \
    --engine-version-source frozen_campaign \
    --target-server-launch-command "$vllm_launch_text" \
    --target-server-config-json "$VLLM_CONFIG" \
    --server-metrics-url "http://$HOST:$VLLM_PORT/server_info?config_format=json" \
    --run-id "$vllm_run_id" \
    --shared-history-trace-json "$trace" \
    --json "$vllm_artifact" || vllm_status=$?
  vllm_info_status=0
  capture_server_info vllm "$repeat" post \
    "http://$HOST:$VLLM_PORT/server_info?config_format=json" \
    "$SERVER_INFO_DIR/vllm-r${repeat}.post.json" || vllm_info_status=$?
  stop_active_server
  if ((vllm_status != 0 || vllm_info_status != 0)); then
    exit 1
  fi
done

report_command=(
  "$WKVM_PY" "$REPORT"
  "${report_artifacts[@]}"
  --min-repeats "$REPEATS"
  --whole-device-memory-ceiling-mib "$MEMORY_CEILING_MIB"
  --claim-scope "$REPORT_CLAIM_SCOPE"
  --markdown "$MARKDOWN"
  --summary-json "$SUMMARY_JSON"
)
if [[ "$STRICT_PUBLICATION" == "1" ]]; then
  report_command+=(--strict-publication)
fi
if [[ "$ALLOW_FAIL" == "1" ]]; then
  report_command+=(--allow-fail)
fi
printf 'report artifacts=%s markdown=%s summary=%s ceiling_mib=%s\n' \
  "${#report_artifacts[@]}" "$MARKDOWN" "$SUMMARY_JSON" "$MEMORY_CEILING_MIB"
print_command "${report_command[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
set +e
"${report_command[@]}" 2>&1 | tee "$LOG_DIR/report.log"
report_status=${PIPESTATUS[0]}
set -e
if [[ ! -s "$MARKDOWN" || ! -s "$SUMMARY_JSON" ]]; then
  printf '%s\n' 'Provider-HTTP report outputs are missing' >&2
  exit 1
fi
printf 'report=%s\nsummary=%s\n' "$MARKDOWN" "$SUMMARY_JSON"
exit "$report_status"
