#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/aiuser/X/models/gemma-4-E4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-E4B-it}"
WKVM_PY="${WKVM_PY:-/home/aiuser/X/.venv-wkvm/bin/python}"
VLLM_PY="${VLLM_PY:-/home/aiuser/X/.venv-vllm/bin/python}"
SGLANG_PY="${SGLANG_PY:-/home/aiuser/X/.venv-sglang/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/../results/a800/wkvm_10x_http_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
REPEAT_GPU_DEVICES="${REPEAT_GPU_DEVICES:-4,5,6}"
REPEAT_PORTS="${REPEAT_PORTS:-8210,8211,8212}"
SESSIONS="${SESSIONS:-32}"
TURNS="${TURNS:-24}"
INITIAL_CONTEXT_TOKENS="${INITIAL_CONTEXT_TOKENS:-98304}"
TURN_INPUT_TOKENS="${TURN_INPUT_TOKENS:-32}"
OUTPUT_TOKENS_PER_TURN="${OUTPUT_TOKENS_PER_TURN:-32}"
REPORT_CLAIM_SCOPE="${REPORT_CLAIM_SCOPE:-full-session}"
MEMORY_CEILING_MIB="${MEMORY_CEILING_MIB:-77824}"
MAX_IDLE_BASELINE_MIB="${MAX_IDLE_BASELINE_MIB:-1024}"
GPU_MEMORY_SAMPLE_INTERVAL_S="${GPU_MEMORY_SAMPLE_INTERVAL_S:-0.1}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-3600}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
SERVER_STOP_TIMEOUT_S="${SERVER_STOP_TIMEOUT_S:-60}"
GPU_CLEAR_TIMEOUT_S="${GPU_CLEAR_TIMEOUT_S:-90}"
WORKER_TERM_TIMEOUT_S="${WORKER_TERM_TIMEOUT_S:-75}"
WORKER_KILL_TIMEOUT_S="${WORKER_KILL_TIMEOUT_S:-10}"
HEALTH_POLL_INTERVAL_S="${HEALTH_POLL_INTERVAL_S:-2}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
SGLANG_CUDA_GRAPH_BACKEND_PREFILL="${SGLANG_CUDA_GRAPH_BACKEND_PREFILL:-breakable}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.92}"
TRACE_SOURCE_ENGINE="${TRACE_SOURCE_ENGINE:-sglang}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
VLLM_COMPILE_MODE="${VLLM_COMPILE_MODE:-3}"
VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS="${WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS:-32}"
WKVM_TOKEN_POOL_CAPACITY="${WKVM_TOKEN_POOL_CAPACITY:-229376}"
WKVM_TOKEN_POOL_MAX_CONTEXT_LEN="${WKVM_TOKEN_POOL_MAX_CONTEXT_LEN:-100000}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_FAIL="${ALLOW_FAIL:-0}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
VLLM_VERSION="${VLLM_VERSION:-}"
SGLANG_VERSION="${SGLANG_VERSION:-}"
BENCHMARK="${BENCHMARK:-$ROOT/experiments/gemma_multiturn_http_bench.py}"
REPORT="${REPORT:-$ROOT/experiments/multiturn_http_10x_report.py}"
SGLANG_PROCESSOR_SOURCE="${SGLANG_PROCESSOR_SOURCE:-$ROOT/experiments/sglang_shared_history_logits.py}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-${TMPDIR:-/tmp}}"
GPU_PROCESS_ALLOWLIST_REGEX="${GPU_PROCESS_ALLOWLIST_REGEX:-gnome-remote-desktop-daemon|ptyxis|nautilus|gnome-text-editor|chrome|/papers$|/baobab$}"

TRACE_DIR="$OUT_DIR/traces"
ARTIFACT_DIR="$OUT_DIR/artifacts"
LOG_DIR="$OUT_DIR/logs"
SERVER_INFO_DIR="$OUT_DIR/server-info"
MARKDOWN="$OUT_DIR/provider_http_10x_report.md"
SUMMARY_JSON="$OUT_DIR/provider_http_10x_summary.json"
PATH_MANIFEST="$OUT_DIR/artifact_paths.tsv"
MODEL_FILE_MANIFEST="$OUT_DIR/model_files.sha256"
MODEL_FILE_MANIFEST_POST="$OUT_DIR/model_files.post.sha256"
GPU_POOL_MANIFEST="$OUT_DIR/gpu_pool.tsv"
SGLANG_PROCESSOR_FILE="$OUT_DIR/sglang_teacher_forcing_processor.txt"

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_ENGINE=""
ACTIVE_SERVER_REPEAT=""
ACTIVE_SERVER_GPU=""
ACTIVE_SERVER_LOG=""
declare -a WORKER_PIDS=()

SGLANG_PROGRAM='import json,sys; from incumbent_gemma_bench import sglang_language_model_override; from sglang.srt.entrypoints.http_server import launch_server; from sglang.srt.server_args import ServerArgs; model=sys.argv[1]; launch_server(ServerArgs(model_path=model, served_model_name=sys.argv[2], host=sys.argv[3], port=int(sys.argv[4]), dtype="bfloat16", context_length=int(sys.argv[8]), max_total_tokens=int(sys.argv[9]), mem_fraction_static=float(sys.argv[10]), chunked_prefill_size=int(sys.argv[5]), max_running_requests=int(sys.argv[6]), attention_backend="triton", json_model_override_args=json.dumps(sglang_language_model_override(model), separators=(",", ":"), sort_keys=True), cuda_graph_backend_decode="full", cuda_graph_backend_prefill=sys.argv[7], enable_cache_report=True, enable_custom_logit_processor=sys.argv[11] == "true", enable_multimodal=False, skip_tokenizer_init=True, sampling_defaults="openai", log_level="warning"))'

generate_uuid() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    tr -d '\n' < /proc/sys/kernel/random/uuid
    return
  fi
  "$WKVM_PY" -c 'import uuid; print(uuid.uuid4())'
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

quoted_command() {
  printf '%q ' "$@"
}

launch_profile() {
  local gpu="$1"
  local port="$2"
  shift 2
  local index
  local -a command=(setsid "$@")
  for index in "${!command[@]}"; do
    case "${command[index]}" in
      "CUDA_VISIBLE_DEVICES=$gpu")
        command[index]=CUDA_VISIBLE_DEVICES=GPU_DEVICE
        ;;
      "$port")
        command[index]=PORT
        ;;
      "--port=$port")
        command[index]=--port=PORT
        ;;
    esac
  done
  quoted_command "${command[@]}"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    printf '%s must be an integer >= 1: %s\n' "$name" "$value" >&2
    exit 1
  fi
}

validate_fraction() {
  local name="$1"
  local value="$2"
  if ! "$WKVM_PY" -c \
    'import math,sys; value=float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and 0 < value < 1))' \
    "$value"; then
    printf '%s must be finite and between 0 and 1: %s\n' "$name" "$value" >&2
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

assert_port_unbound() {
  local host="$1"
  local port="$2"
  printf 'port-check host=%s port=%s expected=unbound\n' "$host" "$port"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if ! "$WKVM_PY" -c \
    'import socket,sys; sock=socket.socket(); sock.bind((sys.argv[1],int(sys.argv[2]))); sock.close()' \
    "$host" "$port"; then
    printf 'Port is already bound or unavailable: %s:%s\n' "$host" "$port" >&2
    return 2
  fi
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

blocking_gpu_processes() {
  local gpu="$1"
  local query_output
  if ! query_output="$(
    nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>&1
  )"; then
    printf 'Could not inspect compute processes on GPU %s: %s\n' \
      "$gpu" "$query_output" >&2
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
  local gpu="$1"
  local blocking
  if ! blocking="$(blocking_gpu_processes "$gpu")"; then
    return 2
  fi
  if [[ -n "$blocking" ]]; then
    printf 'GPU %s has non-allowlisted process(es): %s\n' \
      "$gpu" "$(printf '%s' "$blocking" | tr '\n' ',')" >&2
    return 2
  fi
}

wait_for_gpu_clear() {
  local gpu="$1"
  local deadline=$((SECONDS + GPU_CLEAR_TIMEOUT_S))
  local blocking
  while ((SECONDS < deadline)); do
    if ! blocking="$(blocking_gpu_processes "$gpu")"; then
      return 2
    fi
    if [[ -z "$blocking" ]]; then
      return 0
    fi
    sleep 1
  done
  printf 'GPU %s did not clear within %s seconds: %s\n' \
    "$gpu" "$GPU_CLEAR_TIMEOUT_S" \
    "$(printf '%s' "$blocking" | tr '\n' ',')" >&2
  return 2
}

capture_prelaunch_memory() {
  local gpu="$1"
  local used_mib
  refuse_parallel_gpu_run "$gpu"
  used_mib="$(
    nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n 1
  )"
  if [[ ! "$used_mib" =~ ^[0-9]+$ ]]; then
    printf 'Invalid prelaunch memory sample for GPU %s: %s\n' \
      "$gpu" "$used_mib" >&2
    return 2
  fi
  if ((used_mib > MAX_IDLE_BASELINE_MIB)); then
    printf 'GPU %s prelaunch memory %s MiB exceeds idle limit %s MiB\n' \
      "$gpu" "$used_mib" "$MAX_IDLE_BASELINE_MIB" >&2
    return 2
  fi
  printf '%s' "$used_mib"
}

check_gpu_pool() {
  local reference_name=""
  local reference_total=""
  local reference_driver=""
  local repeat gpu row name uuid driver total used
  declare -A seen_uuids=()
  printf 'repeat\tselector\tuuid\tname\tmemory_total_mib\tdriver_version\n' \
    >"$GPU_POOL_MANIFEST"
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    gpu="${GPU_DEVICES[repeat - 1]}"
    refuse_parallel_gpu_run "$gpu"
    row="$(
      nvidia-smi -i "$gpu" \
        --query-gpu=name,uuid,driver_version,memory.total,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1
    )"
    IFS=',' read -r name uuid driver total used <<<"$row"
    name="$(trim "$name")"
    uuid="$(trim "$uuid")"
    driver="$(trim "$driver")"
    total="$(trim "$total")"
    used="$(trim "$used")"
    if [[ -z "$name" || -z "$uuid" || -z "$driver" || ! "$total" =~ ^[0-9]+$ ]]; then
      printf 'Invalid GPU identity for selector %s: %s\n' "$gpu" "$row" >&2
      return 2
    fi
    if [[ -n "${seen_uuids[$uuid]:-}" ]]; then
      printf 'Repeat GPU selectors resolve to the same UUID: %s\n' "$uuid" >&2
      return 2
    fi
    seen_uuids[$uuid]=1
    if [[ -z "$reference_name" ]]; then
      reference_name="$name"
      reference_total="$total"
      reference_driver="$driver"
    elif [[ "$name" != "$reference_name" || "$total" != "$reference_total" || "$driver" != "$reference_driver" ]]; then
      printf 'GPU pool is not homogeneous at selector %s: %s\n' "$gpu" "$row" >&2
      return 2
    fi
    if ((MEMORY_CEILING_MIB > total)); then
      printf 'Memory ceiling %s MiB exceeds GPU %s total %s MiB\n' \
        "$MEMORY_CEILING_MIB" "$gpu" "$total" >&2
      return 2
    fi
    if [[ ! "$used" =~ ^[0-9]+$ || "$used" -gt "$MAX_IDLE_BASELINE_MIB" ]]; then
      printf 'GPU %s is not idle: %s MiB\n' "$gpu" "${used:-unknown}" >&2
      return 2
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$repeat" "$gpu" "$uuid" "$name" "$total" "$driver" \
      >>"$GPU_POOL_MANIFEST"
  done
}

start_server() {
  local engine="$1"
  local repeat="$2"
  local gpu="$3"
  local port="$4"
  local log_path="$5"
  shift 5
  local -a command=("$@")
  assert_port_unbound 127.0.0.1 "$port"
  printf 'server start engine=%s repeat=%s gpu=%s port=%s log=%s\n' \
    "$engine" "$repeat" "$gpu" "$port" "$log_path"
  print_command setsid "${command[@]}"
  ACTIVE_SERVER_ENGINE="$engine"
  ACTIVE_SERVER_REPEAT="$repeat"
  ACTIVE_SERVER_GPU="$gpu"
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
      printf 'Server %s %s exited before health check; log tail follows\n' \
        "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" >&2
      tail -n 100 "$ACTIVE_SERVER_LOG" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$HEALTH_POLL_INTERVAL_S"
  done
  printf 'Server %s %s did not become healthy within %s seconds\n' \
    "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" \
    "$SERVER_READY_TIMEOUT_S" >&2
  tail -n 100 "$ACTIVE_SERVER_LOG" >&2 || true
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
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl -fsS --max-time 60 "$url" -o "$output"
    return 0
  fi
  curl -fsS --max-time 60 "$url" -o "$output"
}

stop_active_server() {
  if [[ -z "$ACTIVE_SERVER_PID" ]]; then
    return 0
  fi
  printf 'server stop engine=%s repeat=%s gpu=%s pid=%s\n' \
    "$ACTIVE_SERVER_ENGINE" "$ACTIVE_SERVER_REPEAT" \
    "$ACTIVE_SERVER_GPU" "$ACTIVE_SERVER_PID"
  if [[ "$DRY_RUN" == "1" ]]; then
    ACTIVE_SERVER_PID=""
    ACTIVE_SERVER_ENGINE=""
    ACTIVE_SERVER_REPEAT=""
    ACTIVE_SERVER_GPU=""
    ACTIVE_SERVER_LOG=""
    return 0
  fi
  local pid="$ACTIVE_SERVER_PID"
  local gpu="$ACTIVE_SERVER_GPU"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + SERVER_STOP_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf 'Force-stopping server process group %s\n' "$pid" >&2
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_ENGINE=""
  ACTIVE_SERVER_REPEAT=""
  ACTIVE_SERVER_GPU=""
  ACTIVE_SERVER_LOG=""
  wait_for_gpu_clear "$gpu"
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

worker_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  stop_active_server || true
  exit "$status"
}

run_sglang() {
  local role="$1"
  local repeat="$2"
  local repeat_id="$3"
  local gpu="$4"
  local port="$5"
  local trace="$6"
  local artifact="$7"
  local baseline='<prelaunch-nvidia-smi>'
  local custom_processor=false
  local status=0
  local info_status=0
  local run_id
  local server_log="$LOG_DIR/sglang-r${repeat}.server.log"
  local launch_text launch_template
  local -a client_trace_args
  local -a server_command

  if [[ "$role" == source ]]; then
    client_trace_args=(
      --endpoint /generate
      --sglang-native-generate
      --teacher-forcing-field none
      --write-shared-history-trace-json "$trace"
    )
  else
    custom_processor=true
    client_trace_args=(
      --endpoint /v1/completions
      --teacher-forcing-processor "@$SGLANG_PROCESSOR_FILE"
      --shared-history-trace-json "$trace"
    )
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    baseline="$(capture_prelaunch_memory "$gpu")"
  fi
  server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu"
    HF_HUB_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    "PYTHONPATH=$ROOT/experiments:$ROOT"
    "$SGLANG_PY" -c "$SGLANG_PROGRAM"
    "$MODEL_PATH" "$SERVED_MODEL_NAME" 127.0.0.1 "$port"
    "$SGLANG_CHUNKED_PREFILL_SIZE" "$SGLANG_MAX_RUNNING_REQUESTS"
    "$SGLANG_CUDA_GRAPH_BACKEND_PREFILL"
    "$INCUMBENT_CONTEXT_LENGTH" "$SGLANG_MAX_TOTAL_TOKENS"
    "$SGLANG_MEM_FRACTION_STATIC" "$custom_processor"
  )
  launch_text="$(quoted_command setsid "${server_command[@]}")"
  launch_template="$(launch_profile "$gpu" "$port" "${server_command[@]}")"
  start_server sglang "$repeat_id" "$gpu" "$port" "$server_log" \
    "${server_command[@]}"
  wait_for_health "http://127.0.0.1:$port/health"
  capture_server_info sglang "$repeat_id" pre \
    "http://127.0.0.1:$port/server_info" \
    "$SERVER_INFO_DIR/sglang-r${repeat}.pre.json"
  run_id="$(generate_uuid)"
  run_client sglang "$repeat_id" "$baseline" "$artifact" \
    "$LOG_DIR/sglang-r${repeat}.client.log" \
    env "CUDA_VISIBLE_DEVICES=$gpu" "PYTHONPATH=$ROOT" \
    "$WKVM_PY" "$BENCHMARK" --engine sglang \
    "${common_http_args[@]}" \
    --base-url "http://127.0.0.1:$port" \
    "${client_trace_args[@]}" \
    --gpu-memory-baseline-used-mib "$baseline" \
    --semantics full_kv \
    --engine-version "$SGLANG_VERSION" \
    --engine-version-source frozen_campaign \
    --target-server-launch-command "$launch_text" \
    --target-server-launch-profile "$launch_template" \
    --target-server-config-json "$SGLANG_CONFIG" \
    --server-metrics-url "http://127.0.0.1:$port/server_info" \
    --run-id "$run_id" \
    --json "$artifact" || status=$?
  capture_server_info sglang "$repeat_id" post \
    "http://127.0.0.1:$port/server_info" \
    "$SERVER_INFO_DIR/sglang-r${repeat}.post.json" || info_status=$?
  stop_active_server
  if ((status != 0 || info_status != 0)); then
    return 1
  fi
}

run_wkvm() {
  local repeat="$1"
  local repeat_id="$2"
  local gpu="$3"
  local port="$4"
  local trace="$5"
  local artifact="$6"
  local baseline='<prelaunch-nvidia-smi>'
  local status=0
  local info_status=0
  local run_id
  local server_log="$LOG_DIR/wkvm-r${repeat}.server.log"
  local launch_text launch_template
  local -a server_command

  if [[ "$DRY_RUN" != "1" ]]; then
    baseline="$(capture_prelaunch_memory "$gpu")"
  fi
  server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu"
    HF_HUB_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    "PYTHONPATH=$ROOT"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    WKVM_ENABLE_TOKEN_POOL_TRITON=1
    WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON=1
    WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON=1
    WKVM_TOKEN_POOL_TRITON_STRICT=1
    WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY=1
    WKVM_TOKEN_POOL_ROUTE_BOUNDARY_BATCH=1
    "$WKVM_PY" -m wkvm.gemma_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 127.0.0.1
    --port "$port"
    --slots "$SESSIONS"
    --max-queue "$MAX_QUEUE"
    --request-timeout-s "$REQUEST_TIMEOUT_S"
    --max-request-body-bytes 67108864
    --request-read-timeout-s "$REQUEST_TIMEOUT_S"
    --stream-flush-tokens 1
    --max-completed-requests "$WKVM_MAX_COMPLETED_REQUESTS"
    --ignore-eos
    --enable-token-session-teacher-forcing
    --batch-wait-s 0.01
    --prefill-chunk 2048
    --prefill-microbatch-rows 2
    --continuation-prefill-microbatch-rows "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS"
    --decode-microbatch-rows "$SESSIONS"
    --persistent-padded-decode-steps "$OUTPUT_TOKENS_PER_TURN"
    --persistent-padded-sliding-metadata-padding
    --enable-token-pool-attention
    --token-pool-max-context-len "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN"
    --token-pool-capacity "$WKVM_TOKEN_POOL_CAPACITY"
    --token-pool-paged-block-size 16
    --m-slots 32
    --route-chunk 2048
    --native-gemma-checkpoint-loader
    --native-gemma-kv-sharing-fast-prefill
    --native-gemma-attention-backend triton_dense_gqa
    --native-gemma-projection-backend separate
  )
  launch_text="$(quoted_command setsid "${server_command[@]}")"
  launch_template="$(launch_profile "$gpu" "$port" "${server_command[@]}")"
  start_server wkvm "$repeat_id" "$gpu" "$port" "$server_log" \
    "${server_command[@]}"
  wait_for_health "http://127.0.0.1:$port/health"
  capture_server_info wkvm "$repeat_id" pre \
    "http://127.0.0.1:$port/metrics" \
    "$SERVER_INFO_DIR/wkvm-r${repeat}.pre.json"
  run_id="$(generate_uuid)"
  run_client wkvm "$repeat_id" "$baseline" "$artifact" \
    "$LOG_DIR/wkvm-r${repeat}.client.log" \
    env "CUDA_VISIBLE_DEVICES=$gpu" "PYTHONPATH=$ROOT" \
    "$WKVM_PY" "$BENCHMARK" --engine wkvm \
    "${common_http_args[@]}" \
    --base-url "http://127.0.0.1:$port" \
    --endpoint /v1/stream \
    --gpu-memory-baseline-used-mib "$baseline" \
    --semantics routed_span_approximate \
    --engine-version "$WKVM_VERSION" \
    --engine-version-source frozen_campaign \
    --target-server-launch-command "$launch_text" \
    --target-server-launch-profile "$launch_template" \
    --target-server-config-json "$WKVM_CONFIG" \
    --server-metrics-url "http://127.0.0.1:$port/metrics" \
    --run-id "$run_id" \
    --shared-history-trace-json "$trace" \
    --json "$artifact" || status=$?
  capture_server_info wkvm "$repeat_id" post \
    "http://127.0.0.1:$port/metrics" \
    "$SERVER_INFO_DIR/wkvm-r${repeat}.post.json" || info_status=$?
  stop_active_server
  if ((status != 0 || info_status != 0)); then
    return 1
  fi
}

run_vllm() {
  local role="$1"
  local repeat="$2"
  local repeat_id="$3"
  local gpu="$4"
  local port="$5"
  local trace="$6"
  local artifact="$7"
  local baseline='<prelaunch-nvidia-smi>'
  local v2_enabled=0
  local status=0
  local info_status=0
  local run_id
  local server_log="$LOG_DIR/vllm-r${repeat}.server.log"
  local launch_text launch_template
  local -a fast_prefill_args logits_processor_args return_token_args trace_args
  local -a server_command

  if [[ "$role" == source ]]; then
    v2_enabled=1
    fast_prefill_args=(--no-kv-sharing-fast-prefill)
    logits_processor_args=()
    return_token_args=(--return-tokens-as-token-ids)
    trace_args=(
      --teacher-forcing-field none
      --write-shared-history-trace-json "$trace"
    )
  else
    fast_prefill_args=(--kv-sharing-fast-prefill)
    logits_processor_args=(
      --logits-processors
      experiments.vllm_shared_history_logits:SharedHistoryLogitsProcessor
    )
    return_token_args=()
    trace_args=(--shared-history-trace-json "$trace")
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    baseline="$(capture_prelaunch_memory "$gpu")"
  fi
  server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu"
    "PATH=$(dirname "$VLLM_PY"):$PATH"
    HF_HUB_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    VLLM_SERVER_DEV_MODE=1
    "VLLM_USE_V2_MODEL_RUNNER=$v2_enabled"
    "PYTHONPATH=$ROOT"
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 127.0.0.1
    --port "$port"
    --dtype bfloat16
    --max-model-len "$INCUMBENT_CONTEXT_LENGTH"
    --max-num-seqs "$SESSIONS"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --enable-chunked-prefill
    --enable-prefix-caching
    "${fast_prefill_args[@]}"
    --language-model-only
    --limit-mm-per-prompt '{"image":0,"audio":0}'
    --compilation-config "$VLLM_COMPILATION_CONFIG"
    "${logits_processor_args[@]}"
    --enable-prompt-tokens-details
    "${return_token_args[@]}"
  )
  launch_text="$(quoted_command setsid "${server_command[@]}")"
  launch_template="$(launch_profile "$gpu" "$port" "${server_command[@]}")"
  start_server vllm "$repeat_id" "$gpu" "$port" "$server_log" \
    "${server_command[@]}"
  wait_for_health "http://127.0.0.1:$port/health"
  capture_server_info vllm "$repeat_id" pre \
    "http://127.0.0.1:$port/server_info?config_format=json" \
    "$SERVER_INFO_DIR/vllm-r${repeat}.pre.json"
  run_id="$(generate_uuid)"
  run_client vllm "$repeat_id" "$baseline" "$artifact" \
    "$LOG_DIR/vllm-r${repeat}.client.log" \
    env "CUDA_VISIBLE_DEVICES=$gpu" "PYTHONPATH=$ROOT" \
    "$WKVM_PY" "$BENCHMARK" --engine vllm \
    "${common_http_args[@]}" \
    --base-url "http://127.0.0.1:$port" \
    --endpoint /v1/completions \
    --gpu-memory-baseline-used-mib "$baseline" \
    --semantics full_kv \
    --engine-version "$VLLM_VERSION" \
    --engine-version-source frozen_campaign \
    --target-server-launch-command "$launch_text" \
    --target-server-launch-profile "$launch_template" \
    --target-server-config-json "$VLLM_CONFIG" \
    --server-metrics-url "http://127.0.0.1:$port/server_info?config_format=json" \
    --run-id "$run_id" \
    "${trace_args[@]}" \
    --json "$artifact" || status=$?
  capture_server_info vllm "$repeat_id" post \
    "http://127.0.0.1:$port/server_info?config_format=json" \
    "$SERVER_INFO_DIR/vllm-r${repeat}.post.json" || info_status=$?
  stop_active_server
  if ((status != 0 || info_status != 0)); then
    return 1
  fi
}

run_repeat() {
  local repeat="$1"
  local gpu="$2"
  local port="$3"
  local repeat_id="r${repeat}"
  local trace="$TRACE_DIR/$WORKLOAD_TAG-r${repeat}.trace.json"
  local wkvm_artifact="$ARTIFACT_DIR/wkvm-replay-$WORKLOAD_TAG-r${repeat}.json"
  local sglang_artifact
  local vllm_artifact
  local source_artifact
  local lock_name="${gpu//[^[:alnum:]_.-]/_}"
  local worker_lock_fd
  if [[ "$TRACE_SOURCE_ENGINE" == sglang ]]; then
    sglang_artifact="$ARTIFACT_DIR/sglang-source-$WORKLOAD_TAG-r${repeat}.json"
    vllm_artifact="$ARTIFACT_DIR/vllm-replay-$WORKLOAD_TAG-r${repeat}.json"
    source_artifact="$sglang_artifact"
  else
    vllm_artifact="$ARTIFACT_DIR/vllm-source-$WORKLOAD_TAG-r${repeat}.json"
    sglang_artifact="$ARTIFACT_DIR/sglang-replay-$WORKLOAD_TAG-r${repeat}.json"
    source_artifact="$vllm_artifact"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    trap worker_on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    exec {worker_lock_fd}>"$GPU_LOCK_DIR/wkvm-10x-http-a800-gpu-$lock_name.lock"
    if ! flock -n "$worker_lock_fd"; then
      printf 'GPU lock is already held for repeat %s GPU %s\n' \
        "$repeat_id" "$gpu" >&2
      return 2
    fi
  fi
  printf 'worker start repeat=%s gpu=%s port=%s trace_source=%s\n' \
    "$repeat_id" "$gpu" "$port" "$TRACE_SOURCE_ENGINE"

  local -a common_http_args=(
    --model "$SERVED_MODEL_NAME"
    --sessions "$SESSIONS"
    --turns "$TURNS"
    --initial-context-tokens "$INITIAL_CONTEXT_TOKENS"
    --turn-input-tokens "$TURN_INPUT_TOKENS"
    --output-tokens-per-turn "$OUTPUT_TOKENS_PER_TURN"
    --request-order-policy alternating
    --request-order-seed 0
    --request-timeout-s "$REQUEST_TIMEOUT_S"
    --gpu-memory-device "$gpu"
    --gpu-memory-sample-interval-s "$GPU_MEMORY_SAMPLE_INTERVAL_S"
    --memory-ceiling-mib "$MEMORY_CEILING_MIB"
    --campaign-id "$CAMPAIGN_ID"
    --repeat-id "$repeat_id"
  )

  if [[ "$TRACE_SOURCE_ENGINE" == sglang ]]; then
    run_sglang source "$repeat" "$repeat_id" "$gpu" "$port" "$trace" \
      "$sglang_artifact"
  else
    run_vllm source "$repeat" "$repeat_id" "$gpu" "$port" "$trace" \
      "$vllm_artifact"
  fi
  if [[ "$DRY_RUN" != "1" && (! -s "$trace" || ! -s "$source_artifact") ]]; then
    printf '%s source did not produce the paired trace and artifact: %s\n' \
      "$TRACE_SOURCE_ENGINE" "$repeat_id" >&2
    return 1
  fi
  printf 'trace ready repeat=%s source=%s path=%s\n' \
    "$repeat_id" "$TRACE_SOURCE_ENGINE" "$trace"

  run_wkvm "$repeat" "$repeat_id" "$gpu" "$port" "$trace" "$wkvm_artifact"
  if [[ "$TRACE_SOURCE_ENGINE" == sglang ]]; then
    run_vllm replay "$repeat" "$repeat_id" "$gpu" "$port" "$trace" \
      "$vllm_artifact"
  else
    run_sglang replay "$repeat" "$repeat_id" "$gpu" "$port" "$trace" \
      "$sglang_artifact"
  fi

  printf 'worker complete repeat=%s gpu=%s port=%s\n' "$repeat_id" "$gpu" "$port"
  if [[ "$DRY_RUN" != "1" ]]; then
    trap - EXIT INT TERM
  fi
}

assert_clean_tree() {
  local status
  status="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$status" ]]; then
    printf 'Publication campaign requires a clean worktree:\n%s\n' "$status" >&2
    return 2
  fi
}

write_model_manifest() {
  local output="$1"
  find "$MODEL_PATH" -type f -print0 | sort -z | xargs -0 sha256sum >"$output"
  if [[ ! -s "$output" ]]; then
    printf 'Model manifest is empty: %s\n' "$output" >&2
    return 2
  fi
}

verify_model_manifest_unchanged() {
  write_model_manifest "$MODEL_FILE_MANIFEST_POST"
  local post_sha256
  post_sha256="$(sha256sum "$MODEL_FILE_MANIFEST_POST" | awk '{print $1}')"
  if [[ "$post_sha256" != "$MODEL_MANIFEST_SHA256" ]] || \
     ! cmp -s "$MODEL_FILE_MANIFEST" "$MODEL_FILE_MANIFEST_POST"; then
    printf 'Model files changed during campaign: before=%s after=%s\n' \
      "$MODEL_MANIFEST_SHA256" "$post_sha256" >&2
    return 2
  fi
  printf 'model manifest verified sha256=%s\n' "$post_sha256"
}

remove_worker_pid() {
  local removed_pid="$1"
  local pid
  local -a remaining_pids=()
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ "$pid" != "$removed_pid" ]]; then
      remaining_pids+=("$pid")
    fi
  done
  WORKER_PIDS=("${remaining_pids[@]}")
}

worker_owned_processes_alive() {
  local worker_pid="$1"
  ps -eo pgid=,stat= | awk -v target="$worker_pid" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit !found }
  ' && return 0
  local state
  state="$(ps -o stat= -p "$worker_pid" 2>/dev/null | tr -d '[:space:]')"
  [[ -n "$state" && "$state" != Z* ]]
}

signal_worker_group() {
  local signal="$1"
  local worker_pid="$2"
  if kill -0 -- "-$worker_pid" 2>/dev/null; then
    printf 'worker cleanup signal=%s pid=%s pgid=%s\n' \
      "$signal" "$worker_pid" "$worker_pid"
    kill "-$signal" -- "-$worker_pid" 2>/dev/null || true
  elif worker_owned_processes_alive "$worker_pid"; then
    # This fallback covers only the short interval before setsid establishes
    # the worker's process group. The PID is still our unreaped direct child.
    printf 'worker cleanup signal=%s pid=%s target=direct-child\n' \
      "$signal" "$worker_pid"
    kill "-$signal" "$worker_pid" 2>/dev/null || true
  fi
}

reap_inactive_workers() {
  local pid
  local -a snapshot=("${WORKER_PIDS[@]}")
  for pid in "${snapshot[@]}"; do
    if ! worker_owned_processes_alive "$pid"; then
      wait "$pid" 2>/dev/null || true
      remove_worker_pid "$pid"
    fi
  done
}

stop_worker_groups() {
  local pid
  local deadline
  local -a snapshot=("${WORKER_PIDS[@]}")
  for pid in "${snapshot[@]}"; do
    signal_worker_group TERM "$pid"
  done
  deadline=$((SECONDS + WORKER_TERM_TIMEOUT_S))
  while ((${#WORKER_PIDS[@]} > 0 && SECONDS < deadline)); do
    reap_inactive_workers
    if ((${#WORKER_PIDS[@]} > 0)); then
      sleep 1
    fi
  done

  snapshot=("${WORKER_PIDS[@]}")
  for pid in "${snapshot[@]}"; do
    signal_worker_group KILL "$pid"
  done
  deadline=$((SECONDS + WORKER_KILL_TIMEOUT_S))
  while ((${#WORKER_PIDS[@]} > 0 && SECONDS < deadline)); do
    reap_inactive_workers
    if ((${#WORKER_PIDS[@]} > 0)); then
      sleep 1
    fi
  done
  reap_inactive_workers
  if ((${#WORKER_PIDS[@]} > 0)); then
    printf 'Worker process groups did not exit after bounded TERM/KILL cleanup: %s\n' \
      "${WORKER_PIDS[*]}" >&2
    WORKER_PIDS=()
    return 1
  fi
}

worker_entrypoint() {
  local pgid
  pgid="$(ps -o pgid= -p "$$" | tr -d '[:space:]')"
  if [[ "$pgid" != "$$" ]]; then
    printf 'Worker is not its own process-group leader: pid=%s pgid=%s\n' \
      "$$" "${pgid:-unknown}" >&2
    return 125
  fi
  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_ENGINE=""
  ACTIVE_SERVER_REPEAT=""
  ACTIVE_SERVER_GPU=""
  ACTIVE_SERVER_LOG=""
  run_repeat "$1" "$2" "$3"
}

export_worker_context() {
  export ROOT MODEL_PATH SERVED_MODEL_NAME WKVM_PY VLLM_PY SGLANG_PY
  export BENCHMARK REPORT
  export TRACE_DIR ARTIFACT_DIR LOG_DIR SERVER_INFO_DIR SGLANG_PROGRAM
  export SGLANG_PROCESSOR_FILE TRACE_SOURCE_ENGINE
  export SESSIONS TURNS INITIAL_CONTEXT_TOKENS TURN_INPUT_TOKENS
  export OUTPUT_TOKENS_PER_TURN REQUEST_TIMEOUT_S SERVER_READY_TIMEOUT_S
  export SERVER_STOP_TIMEOUT_S GPU_CLEAR_TIMEOUT_S HEALTH_POLL_INTERVAL_S
  export GPU_MEMORY_SAMPLE_INTERVAL_S MEMORY_CEILING_MIB MAX_IDLE_BASELINE_MIB
  export GPU_LOCK_DIR GPU_PROCESS_ALLOWLIST_REGEX CAMPAIGN_ID WORKLOAD_TAG
  export SGLANG_CHUNKED_PREFILL_SIZE SGLANG_CUDA_GRAPH_BACKEND_PREFILL
  export SGLANG_MAX_RUNNING_REQUESTS SGLANG_MEM_FRACTION_STATIC
  export SGLANG_MAX_TOTAL_TOKENS SGLANG_CONFIG SGLANG_VERSION
  export SGLANG_CUSTOM_LOGIT_PROCESSOR_ENABLED
  export WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS WKVM_TOKEN_POOL_CAPACITY
  export WKVM_TOKEN_POOL_MAX_CONTEXT_LEN WKVM_MAX_COMPLETED_REQUESTS
  export WKVM_CONFIG WKVM_VERSION MAX_QUEUE
  export VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_BATCHED_TOKENS
  export VLLM_COMPILATION_CONFIG VLLM_CONFIG VLLM_VERSION
  export VLLM_USE_V2_MODEL_RUNNER_VALUE VLLM_MODEL_RUNNER_GENERATION
  export INCUMBENT_CONTEXT_LENGTH DRY_RUN
  export -f generate_uuid print_command quoted_command launch_profile
  export -f assert_port_unbound
  export -f blocking_gpu_processes refuse_parallel_gpu_run wait_for_gpu_clear
  export -f capture_prelaunch_memory start_server wait_for_health
  export -f capture_server_info stop_active_server run_client worker_on_exit
  export -f run_sglang run_wkvm run_vllm run_repeat worker_entrypoint
}

launch_repeat_worker() {
  local repeat="$1"
  local gpu="$2"
  local port="$3"
  local worker_log="$4"
  setsid bash -c 'worker_entrypoint "$1" "$2" "$3"' \
    wkvm-a800-worker "$repeat" "$gpu" "$port" \
    >"$worker_log" 2>&1 &
  local worker_pid=$!
  WORKER_PIDS+=("$worker_pid")
  printf 'worker active repeat=r%s pid=%s pgid=%s\n' \
    "$repeat" "$worker_pid" "$worker_pid"
}

main_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  stop_worker_groups || true
  exit "$status"
}

print_path_manifest() {
  local repeat
  printf '# campaign_id=%s\n' "$CAMPAIGN_ID"
  printf '# model_manifest_sha256=%s\n' "$MODEL_MANIFEST_SHA256"
  printf '# workload=%s\n' "$WORKLOAD_TAG"
  printf '# trace_source_engine=%s\n' "$TRACE_SOURCE_ENGINE"
  printf 'kind\trepeat\tpath\n'
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    printf 'trace\t%s\t%s\n' "$repeat" \
      "$TRACE_DIR/$WORKLOAD_TAG-r${repeat}.trace.json"
    if [[ "$TRACE_SOURCE_ENGINE" == sglang ]]; then
      printf 'sglang-source\t%s\t%s\n' "$repeat" \
        "$ARTIFACT_DIR/sglang-source-$WORKLOAD_TAG-r${repeat}.json"
      printf 'vllm-replay\t%s\t%s\n' "$repeat" \
        "$ARTIFACT_DIR/vllm-replay-$WORKLOAD_TAG-r${repeat}.json"
    else
      printf 'vllm-source\t%s\t%s\n' "$repeat" \
        "$ARTIFACT_DIR/vllm-source-$WORKLOAD_TAG-r${repeat}.json"
      printf 'sglang-replay\t%s\t%s\n' "$repeat" \
        "$ARTIFACT_DIR/sglang-replay-$WORKLOAD_TAG-r${repeat}.json"
    fi
    printf 'wkvm-replay\t%s\t%s\n' "$repeat" \
      "$ARTIFACT_DIR/wkvm-replay-$WORKLOAD_TAG-r${repeat}.json"
  done
  printf 'report\t-\t%s\n' "$MARKDOWN"
  printf 'summary\t-\t%s\n' "$SUMMARY_JSON"
  printf 'model-manifest-pre\t-\t%s\n' "$MODEL_FILE_MANIFEST"
  printf 'model-manifest-post\t-\t%s\n' "$MODEL_FILE_MANIFEST_POST"
  if [[ "$TRACE_SOURCE_ENGINE" == vllm ]]; then
    printf 'sglang-teacher-forcing-processor\t-\t%s\n' "$SGLANG_PROCESSOR_FILE"
  fi
}

validate_positive_integer REPEATS "$REPEATS"
validate_positive_integer SESSIONS "$SESSIONS"
validate_positive_integer TURNS "$TURNS"
validate_positive_integer INITIAL_CONTEXT_TOKENS "$INITIAL_CONTEXT_TOKENS"
validate_positive_integer TURN_INPUT_TOKENS "$TURN_INPUT_TOKENS"
validate_positive_integer OUTPUT_TOKENS_PER_TURN "$OUTPUT_TOKENS_PER_TURN"
validate_positive_integer MEMORY_CEILING_MIB "$MEMORY_CEILING_MIB"
validate_positive_integer MAX_IDLE_BASELINE_MIB "$MAX_IDLE_BASELINE_MIB"
validate_positive_integer REQUEST_TIMEOUT_S "$REQUEST_TIMEOUT_S"
validate_positive_integer WORKER_TERM_TIMEOUT_S "$WORKER_TERM_TIMEOUT_S"
validate_positive_integer WORKER_KILL_TIMEOUT_S "$WORKER_KILL_TIMEOUT_S"
validate_positive_integer SGLANG_CHUNKED_PREFILL_SIZE "$SGLANG_CHUNKED_PREFILL_SIZE"
validate_positive_integer SGLANG_MAX_RUNNING_REQUESTS "$SGLANG_MAX_RUNNING_REQUESTS"
validate_positive_integer VLLM_MAX_NUM_BATCHED_TOKENS "$VLLM_MAX_NUM_BATCHED_TOKENS"
validate_positive_integer WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS \
  "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS"
validate_positive_integer WKVM_TOKEN_POOL_CAPACITY "$WKVM_TOKEN_POOL_CAPACITY"
validate_positive_integer WKVM_TOKEN_POOL_MAX_CONTEXT_LEN "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN"
validate_fraction SGLANG_MEM_FRACTION_STATIC "$SGLANG_MEM_FRACTION_STATIC"
validate_fraction VLLM_GPU_MEMORY_UTILIZATION "$VLLM_GPU_MEMORY_UTILIZATION"
TRACE_SOURCE_ENGINE="${TRACE_SOURCE_ENGINE,,}"
case "$TRACE_SOURCE_ENGINE" in
  sglang|vllm) ;;
  *)
    printf 'TRACE_SOURCE_ENGINE must be sglang or vllm: %s\n' \
      "$TRACE_SOURCE_ENGINE" >&2
    exit 1
    ;;
esac
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
VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE^^}"
case "$VLLM_CUDAGRAPH_MODE" in
  NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
  *)
    printf '%s is not a supported vLLM cudagraph mode: %s\n' \
      VLLM_CUDAGRAPH_MODE "$VLLM_CUDAGRAPH_MODE" >&2
    exit 1
    ;;
esac
if ((REPEATS < 3)); then
  printf 'Strict publication campaigns require at least 3 repeats\n' >&2
  exit 1
fi
case "$REPORT_CLAIM_SCOPE" in
  continuation|full-session) ;;
  *)
    printf 'REPORT_CLAIM_SCOPE must be continuation or full-session: %s\n' \
      "$REPORT_CLAIM_SCOPE" >&2
    exit 1
    ;;
esac
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  printf 'DRY_RUN must be 0 or 1\n' >&2
  exit 1
fi
if [[ "$ALLOW_FAIL" != "0" && "$ALLOW_FAIL" != "1" ]]; then
  printf 'ALLOW_FAIL must be 0 or 1\n' >&2
  exit 1
fi

IFS=',' read -r -a GPU_DEVICES <<<"$REPEAT_GPU_DEVICES"
IFS=',' read -r -a PORTS <<<"$REPEAT_PORTS"
if ((${#GPU_DEVICES[@]} != REPEATS || ${#PORTS[@]} != REPEATS)); then
  printf 'REPEAT_GPU_DEVICES and REPEAT_PORTS must each contain %s values\n' \
    "$REPEATS" >&2
  exit 1
fi
declare -A seen_selectors=()
declare -A seen_ports=()
for ((index = 0; index < REPEATS; index++)); do
  GPU_DEVICES[index]="$(trim "${GPU_DEVICES[index]}")"
  PORTS[index]="$(trim "${PORTS[index]}")"
  if [[ -z "${GPU_DEVICES[index]}" || -n "${seen_selectors[${GPU_DEVICES[index]}]:-}" ]]; then
    printf 'GPU selectors must be nonempty and unique: %s\n' \
      "$REPEAT_GPU_DEVICES" >&2
    exit 1
  fi
  seen_selectors[${GPU_DEVICES[index]}]=1
  validate_port "repeat port" "${PORTS[index]}"
  if [[ -n "${seen_ports[${PORTS[index]}]:-}" ]]; then
    printf 'Repeat ports must be unique: %s\n' "$REPEAT_PORTS" >&2
    exit 1
  fi
  seen_ports[${PORTS[index]}]=1
done

REQUIRED_MODEL_LEN=$((
  INITIAL_CONTEXT_TOKENS
  + TURNS * OUTPUT_TOKENS_PER_TURN
  + (TURNS - 1) * TURN_INPUT_TOKENS
))
ALIGNED_REQUIRED_MODEL_LEN=$((((REQUIRED_MODEL_LEN + 15) / 16) * 16))
INCUMBENT_CONTEXT_LENGTH=$((ALIGNED_REQUIRED_MODEL_LEN + 16))
SGLANG_MAX_TOTAL_TOKENS=$((SESSIONS * (INCUMBENT_CONTEXT_LENGTH + 400)))
MAX_QUEUE=$((SESSIONS * 2))
WKVM_MAX_COMPLETED_REQUESTS=$((SESSIONS * TURNS + SESSIONS))
if ((WKVM_TOKEN_POOL_MAX_CONTEXT_LEN < REQUIRED_MODEL_LEN)); then
  printf 'WKVM token-pool max context %s is below required length %s\n' \
    "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN" "$REQUIRED_MODEL_LEN" >&2
  exit 1
fi
WORKLOAD_TAG="b${SESSIONS}_ctx${INITIAL_CONTEXT_TOKENS}_d${TURN_INPUT_TOKENS}_t${TURNS}_o${OUTPUT_TOKENS_PER_TURN}"

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

ROOT_REAL="$(realpath -e -- "$ROOT")"
MODEL_PATH="$(realpath -e -- "$MODEL_PATH")"
OUT_DIR="$(realpath -m -- "$OUT_DIR")"
if [[ "$OUT_DIR" == "$ROOT_REAL" || "$OUT_DIR" == "$ROOT_REAL/"* ]]; then
  printf 'OUT_DIR must be outside the source checkout: %s\n' "$OUT_DIR" >&2
  exit 1
fi
TRACE_DIR="$OUT_DIR/traces"
ARTIFACT_DIR="$OUT_DIR/artifacts"
LOG_DIR="$OUT_DIR/logs"
SERVER_INFO_DIR="$OUT_DIR/server-info"
MARKDOWN="$OUT_DIR/provider_http_10x_report.md"
SUMMARY_JSON="$OUT_DIR/provider_http_10x_summary.json"
PATH_MANIFEST="$OUT_DIR/artifact_paths.tsv"
MODEL_FILE_MANIFEST="$OUT_DIR/model_files.sha256"
MODEL_FILE_MANIFEST_POST="$OUT_DIR/model_files.post.sha256"
GPU_POOL_MANIFEST="$OUT_DIR/gpu_pool.tsv"
SGLANG_PROCESSOR_FILE="$OUT_DIR/sglang_teacher_forcing_processor.txt"

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
WKVM_VERSION="git:$GIT_COMMIT"
if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="wkvm-http-a800-$(generate_uuid)"
fi
if [[ "$DRY_RUN" == "1" ]]; then
  MODEL_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
  VLLM_VERSION="${VLLM_VERSION:-dry-run-vllm-version}"
  SGLANG_VERSION="${SGLANG_VERSION:-dry-run-sglang-version}"
else
  assert_clean_tree
  for command_name in bash cmp flock nvidia-smi curl ps setsid sha256sum find sort tee; do
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
  write_model_manifest "$MODEL_FILE_MANIFEST"
  MODEL_MANIFEST_SHA256="$(sha256sum "$MODEL_FILE_MANIFEST" | awk '{print $1}')"
  detected_vllm_version="$("$VLLM_PY" -c 'import vllm; print(vllm.__version__)')"
  detected_sglang_version="$("$SGLANG_PY" -c 'import sglang; print(sglang.__version__)')"
  if [[ -n "$VLLM_VERSION" && "$VLLM_VERSION" != "$detected_vllm_version" ]]; then
    printf 'VLLM_VERSION override does not match imported version: override=%s imported=%s\n' \
      "$VLLM_VERSION" "$detected_vllm_version" >&2
    exit 1
  fi
  if [[ -n "$SGLANG_VERSION" && "$SGLANG_VERSION" != "$detected_sglang_version" ]]; then
    printf 'SGLANG_VERSION override does not match imported version: override=%s imported=%s\n' \
      "$SGLANG_VERSION" "$detected_sglang_version" >&2
    exit 1
  fi
  VLLM_VERSION="$detected_vllm_version"
  SGLANG_VERSION="$detected_sglang_version"
  for port in "${PORTS[@]}"; do
    assert_port_unbound 127.0.0.1 "$port"
  done
  if [[ "$TRACE_SOURCE_ENGINE" == vllm ]]; then
    if [[ ! -f "$SGLANG_PROCESSOR_SOURCE" ]]; then
      printf 'SGLang teacher-forcing processor source not found: %s\n' \
        "$SGLANG_PROCESSOR_SOURCE" >&2
      exit 1
    fi
    PYTHONPATH="$ROOT" "$SGLANG_PY" "$SGLANG_PROCESSOR_SOURCE" \
      >"$SGLANG_PROCESSOR_FILE"
    if [[ ! -s "$SGLANG_PROCESSOR_FILE" ]]; then
      printf 'Failed to serialize the SGLang teacher-forcing processor\n' >&2
      exit 1
    fi
  fi
  check_gpu_pool
fi

MODEL_IDENTITY_JSON="$(
  "$WKVM_PY" -c \
    'import json,sys; print(json.dumps({"manifest_sha256":sys.argv[2],"path":sys.argv[1],"served_name":sys.argv[3]},sort_keys=True,separators=(",",":")))' \
    "$MODEL_PATH" "$MODEL_MANIFEST_SHA256" "$SERVED_MODEL_NAME"
)"
if [[ "$TRACE_SOURCE_ENGINE" == vllm ]]; then
  VLLM_USE_V2_MODEL_RUNNER_VALUE=1
  VLLM_MODEL_RUNNER_GENERATION=v2
  VLLM_KV_SHARING_FAST_PREFILL_ENABLED=false
  VLLM_LOGITS_PROCESSOR_ENABLED=false
  SGLANG_CUSTOM_LOGIT_PROCESSOR_ENABLED=true
else
  VLLM_USE_V2_MODEL_RUNNER_VALUE=0
  VLLM_MODEL_RUNNER_GENERATION=v1
  VLLM_KV_SHARING_FAST_PREFILL_ENABLED=true
  VLLM_LOGITS_PROCESSOR_ENABLED=true
  SGLANG_CUSTOM_LOGIT_PROCESSOR_ENABLED=false
fi
SGLANG_CONFIG="$(
  "$WKVM_PY" -c \
    'import json,sys; print(json.dumps({"attention_backend":"triton","chunked_prefill_size":int(sys.argv[2]),"context_length":int(sys.argv[3]),"cuda_graph_backend_decode":"full","cuda_graph_backend_prefill":sys.argv[7],"dtype":"bfloat16","enable_cache_report":True,"enable_custom_logit_processor":sys.argv[8] == "true","enable_multimodal":False,"max_running_requests":int(sys.argv[4]),"max_total_tokens":int(sys.argv[5]),"mem_fraction_static":float(sys.argv[6]),"model_identity":json.loads(sys.argv[1]),"sampling_defaults":"openai","skip_tokenizer_init":True,"trace_role":sys.argv[9],"trace_source_engine":sys.argv[10]},sort_keys=True,separators=(",",":")))' \
    "$MODEL_IDENTITY_JSON" "$SGLANG_CHUNKED_PREFILL_SIZE" \
    "$INCUMBENT_CONTEXT_LENGTH" "$SGLANG_MAX_RUNNING_REQUESTS" \
    "$SGLANG_MAX_TOTAL_TOKENS" "$SGLANG_MEM_FRACTION_STATIC" \
    "$SGLANG_CUDA_GRAPH_BACKEND_PREFILL" \
    "$SGLANG_CUSTOM_LOGIT_PROCESSOR_ENABLED" \
    "$(if [[ "$TRACE_SOURCE_ENGINE" == sglang ]]; then printf source; else printf replay; fi)" \
    "$TRACE_SOURCE_ENGINE"
)"
WKVM_CONFIG="$(
  "$WKVM_PY" -c \
    'import json,sys; print(json.dumps({"batch_wait_s":0.01,"continuation_prefill_microbatch_rows":int(sys.argv[8]),"decode_microbatch_rows":int(sys.argv[2]),"dtype":"bfloat16","enable_token_pool_attention":True,"max_completed_requests":int(sys.argv[3]),"max_queue":int(sys.argv[4]),"model_identity":json.loads(sys.argv[1]),"native_gemma_attention_backend":"triton_dense_gqa","native_gemma_kv_sharing_fast_prefill":True,"native_gemma_projection_backend":"separate","persistent_padded_decode_steps":int(sys.argv[5]),"prefill_chunk":2048,"prefill_microbatch_rows":2,"route_chunk":2048,"slots":int(sys.argv[2]),"token_pool_capacity":int(sys.argv[6]),"token_pool_max_context_len":int(sys.argv[7]),"token_pool_paged_block_size":16},sort_keys=True,separators=(",",":")))' \
    "$MODEL_IDENTITY_JSON" "$SESSIONS" "$WKVM_MAX_COMPLETED_REQUESTS" \
    "$MAX_QUEUE" "$OUTPUT_TOKENS_PER_TURN" "$WKVM_TOKEN_POOL_CAPACITY" \
    "$WKVM_TOKEN_POOL_MAX_CONTEXT_LEN" \
    "$WKVM_CONTINUATION_PREFILL_MICROBATCH_ROWS"
)"
VLLM_COMPILATION_CONFIG="$(
  "$WKVM_PY" -c \
    'import json,sys; sizes=[1,2,4,8,16,32]; sizes=[x for x in sizes if x <= int(sys.argv[1])]; print(json.dumps({"cudagraph_capture_sizes":sizes,"cudagraph_mode":sys.argv[3],"max_cudagraph_capture_size":max(sizes),"mode":int(sys.argv[2])},sort_keys=True,separators=(",",":")))' \
    "$SESSIONS" "$VLLM_COMPILE_MODE" "$VLLM_CUDAGRAPH_MODE"
)"
VLLM_CONFIG="$(
  "$WKVM_PY" -c \
    'import json,sys; print(json.dumps({"compilation_config":json.loads(sys.argv[2]),"dtype":"bfloat16","enable_chunked_prefill":True,"enable_prefix_caching":True,"gpu_memory_utilization":float(sys.argv[3]),"kv_sharing_fast_prefill":sys.argv[7] == "true","language_model_only":True,"logits_processor_enabled":sys.argv[8] == "true","max_model_len":int(sys.argv[4]),"max_num_batched_tokens":int(sys.argv[5]),"max_num_seqs":int(sys.argv[6]),"model_identity":json.loads(sys.argv[1]),"model_runner_generation":sys.argv[9],"return_tokens_as_token_ids":sys.argv[10] == "true","trace_role":sys.argv[11],"trace_source_engine":sys.argv[12],"use_v2_model_runner":sys.argv[13] == "1"},sort_keys=True,separators=(",",":")))' \
    "$MODEL_IDENTITY_JSON" "$VLLM_COMPILATION_CONFIG" \
    "$VLLM_GPU_MEMORY_UTILIZATION" "$INCUMBENT_CONTEXT_LENGTH" \
    "$VLLM_MAX_NUM_BATCHED_TOKENS" "$SESSIONS" \
    "$VLLM_KV_SHARING_FAST_PREFILL_ENABLED" "$VLLM_LOGITS_PROCESSOR_ENABLED" \
    "$VLLM_MODEL_RUNNER_GENERATION" \
    "$(if [[ "$TRACE_SOURCE_ENGINE" == vllm ]]; then printf true; else printf false; fi)" \
    "$(if [[ "$TRACE_SOURCE_ENGINE" == vllm ]]; then printf source; else printf replay; fi)" \
    "$TRACE_SOURCE_ENGINE" "$VLLM_USE_V2_MODEL_RUNNER_VALUE"
)"

declare -a REPORT_ARTIFACTS=()
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  if [[ "$TRACE_SOURCE_ENGINE" == sglang ]]; then
    REPORT_ARTIFACTS+=(
      "$ARTIFACT_DIR/sglang-source-$WORKLOAD_TAG-r${repeat}.json"
      "$ARTIFACT_DIR/wkvm-replay-$WORKLOAD_TAG-r${repeat}.json"
      "$ARTIFACT_DIR/vllm-replay-$WORKLOAD_TAG-r${repeat}.json"
    )
  else
    REPORT_ARTIFACTS+=(
      "$ARTIFACT_DIR/vllm-source-$WORKLOAD_TAG-r${repeat}.json"
      "$ARTIFACT_DIR/wkvm-replay-$WORKLOAD_TAG-r${repeat}.json"
      "$ARTIFACT_DIR/sglang-replay-$WORKLOAD_TAG-r${repeat}.json"
    )
  fi
done

printf 'campaign_id=%s workload=%s trace_source=%s commit=%s model_manifest_sha256=%s\n' \
  "$CAMPAIGN_ID" "$WORKLOAD_TAG" "$TRACE_SOURCE_ENGINE" "$GIT_COMMIT" \
  "$MODEL_MANIFEST_SHA256"
printf 'versions wkvm=%s vllm=%s sglang=%s\n' \
  "$WKVM_VERSION" "$VLLM_VERSION" "$SGLANG_VERSION"
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'output_dir=%s\n' "$OUT_DIR"
  print_path_manifest
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    printf 'worker launch repeat=r%s gpu=%s port=%s mode=background\n' \
      "$repeat" "${GPU_DEVICES[repeat - 1]}" "${PORTS[repeat - 1]}"
    run_repeat "$repeat" "${GPU_DEVICES[repeat - 1]}" "${PORTS[repeat - 1]}"
  done
else
  print_path_manifest >"$PATH_MANIFEST"
  export_worker_context
  trap main_on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    worker_log="$LOG_DIR/worker-r${repeat}.log"
    printf 'worker launch repeat=r%s gpu=%s port=%s mode=background log=%s\n' \
      "$repeat" "${GPU_DEVICES[repeat - 1]}" "${PORTS[repeat - 1]}" \
      "$worker_log"
    launch_repeat_worker \
      "$repeat" "${GPU_DEVICES[repeat - 1]}" "${PORTS[repeat - 1]}" \
      "$worker_log"
  done
  worker_status=0
  while ((${#WORKER_PIDS[@]} > 0)); do
    reaped_pid=""
    if wait -n -p reaped_pid "${WORKER_PIDS[@]}"; then
      reaped_status=0
    else
      reaped_status=$?
    fi
    if [[ -z "$reaped_pid" ]]; then
      worker_status=${reaped_status:-125}
      printf 'Could not identify the reaped repeat worker\n' >&2
      break
    fi
    remove_worker_pid "$reaped_pid"
    printf 'worker reaped pid=%s status=%s active=%s\n' \
      "$reaped_pid" "$reaped_status" "${#WORKER_PIDS[@]}"
    if ((reaped_status != 0)); then
      worker_status=$reaped_status
      break
    fi
  done
  if ((worker_status != 0)); then
    printf 'A repeat worker failed with status %s; stopping peers\n' \
      "$worker_status" >&2
    exit "$worker_status"
  fi
  WORKER_PIDS=()
  trap - EXIT INT TERM
  verify_model_manifest_unchanged
  assert_clean_tree
fi

report_command=(
  "$WKVM_PY" "$REPORT"
  "${REPORT_ARTIFACTS[@]}"
  --strict
  --gpu-policy homogeneous-pool
  --min-repeats "$REPEATS"
  --whole-device-memory-ceiling-mib "$MEMORY_CEILING_MIB"
  --claim-scope "$REPORT_CLAIM_SCOPE"
  --markdown "$MARKDOWN"
  --summary-json "$SUMMARY_JSON"
)
if [[ "$ALLOW_FAIL" == "1" ]]; then
  report_command+=(--allow-fail)
fi
printf 'report artifacts=%s markdown=%s summary=%s ceiling_mib=%s\n' \
  "${#REPORT_ARTIFACTS[@]}" "$MARKDOWN" "$SUMMARY_JSON" \
  "$MEMORY_CEILING_MIB"
print_command "${report_command[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
set +e
"${report_command[@]}" 2>&1 | tee "$LOG_DIR/report.log"
report_status=${PIPESTATUS[0]}
set -e
if [[ ! -s "$MARKDOWN" || ! -s "$SUMMARY_JSON" ]]; then
  printf 'Strict provider-HTTP report outputs are missing\n' >&2
  exit 1
fi
printf 'report=%s\nsummary=%s\n' "$MARKDOWN" "$SUMMARY_JSON"
exit "$report_status"
