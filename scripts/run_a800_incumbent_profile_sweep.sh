#!/usr/bin/env bash
set -euo pipefail

# Exploratory incumbent capability sweep. Replay profiles consume one frozen
# external trace, while native profiles emit their own autonomous trace.
# Winners still need a sequential, repeated confirmation run.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/aiuser/X/models/gemma-4-E4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-E4B-it}"
WKVM_PY="${WKVM_PY:-/home/aiuser/X/.venv-wkvm/bin/python}"
VLLM_PY="${VLLM_PY:-/home/aiuser/X/.venv-vllm/bin/python}"
SGLANG_PY="${SGLANG_PY:-/home/aiuser/X/.venv-sglang/bin/python}"
BENCHMARK="${BENCHMARK:-$ROOT/experiments/gemma_multiturn_http_bench.py}"
SGLANG_PROCESSOR_SOURCE="${SGLANG_PROCESSOR_SOURCE:-$ROOT/experiments/sglang_shared_history_logits.py}"
VALIDATOR="${VALIDATOR:-$ROOT/scripts/a800_incumbent_profile_validation.py}"
OUT_DIR="${OUT_DIR:-$ROOT/../results/a800/incumbent_profile_sweep_$(date +%Y%m%d_%H%M%S)}"
TRACE_JSON="${TRACE_JSON:-}"

GPU_DEVICES="${GPU_DEVICES:-5,6}"
PORTS="${PORTS:-8310,8311}"
PROFILE_BASES="${PROFILE_BASES:-vllm-inductor-full-and-piecewise,vllm-v2-native,vllm-mode0-control,sglang-auto-breakable,sglang-triton-breakable,sglang-auto-tc-piecewise,sglang-triton-tc-piecewise,sglang-auto-disabled,sglang-triton-disabled}"
MAX_PROFILE_RUNS="${MAX_PROFILE_RUNS:-24}"

SESSIONS="${SESSIONS:-32}"
TURNS="${TURNS:-4}"
INITIAL_CONTEXT_TOKENS="${INITIAL_CONTEXT_TOKENS:-98304}"
TURN_INPUT_TOKENS="${TURN_INPUT_TOKENS:-32}"
OUTPUT_TOKENS_PER_TURN="${OUTPUT_TOKENS_PER_TURN:-32}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-3600}"
MEMORY_CEILING_MIB="${MEMORY_CEILING_MIB:-77824}"
MAX_IDLE_BASELINE_MIB="${MAX_IDLE_BASELINE_MIB:-1024}"
GPU_MEMORY_SAMPLE_INTERVAL_S="${GPU_MEMORY_SAMPLE_INTERVAL_S:-0.1}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
SERVER_STOP_TIMEOUT_S="${SERVER_STOP_TIMEOUT_S:-60}"
GPU_CLEAR_TIMEOUT_S="${GPU_CLEAR_TIMEOUT_S:-90}"
HEALTH_POLL_INTERVAL_S="${HEALTH_POLL_INTERVAL_S:-2}"

# Lists expand into a bounded token/memory matrix for the selected base profiles.
VLLM_MAX_NUM_BATCHED_TOKENS_LIST="${VLLM_MAX_NUM_BATCHED_TOKENS_LIST:-16384}"
VLLM_GPU_MEMORY_UTILIZATION_LIST="${VLLM_GPU_MEMORY_UTILIZATION_LIST:-0.92}"
SGLANG_CHUNKED_PREFILL_SIZE_LIST="${SGLANG_CHUNKED_PREFILL_SIZE_LIST:-8192}"
SGLANG_MEM_FRACTION_STATIC_LIST="${SGLANG_MEM_FRACTION_STATIC_LIST:-0.92}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-}"

VLLM_REQUIRED_VERSION="${VLLM_REQUIRED_VERSION:-0.25.1}"
SGLANG_REQUIRED_VERSION="${SGLANG_REQUIRED_VERSION:-0.5.15.post1}"
VLLM_VERSION="${VLLM_VERSION:-}"
SGLANG_VERSION="${SGLANG_VERSION:-}"
ALLOW_VERSION_MISMATCH="${ALLOW_VERSION_MISMATCH:-0}"
FAIL_ON_PROFILE_ERROR="${FAIL_ON_PROFILE_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-${TMPDIR:-/tmp}}"

ARTIFACT_DIR="$OUT_DIR/artifacts"
LOG_DIR="$OUT_DIR/logs"
SERVER_INFO_DIR="$OUT_DIR/server-info"
TRACE_DIR="$OUT_DIR/autonomous-traces"
STATUS_DIR="$OUT_DIR/status"
MATRIX_PATH="$OUT_DIR/profile_matrix.tsv"
STATUS_PATH="$OUT_DIR/profile_status.tsv"
GPU_POOL_PATH="$OUT_DIR/gpu_pool.tsv"
SGLANG_PROCESSOR_FILE="$OUT_DIR/sglang_teacher_forcing_processor.txt"
FROZEN_TRACE_JSON="$OUT_DIR/shared_history_trace.json"
TRACE_CONTRACT_SHA256=""

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_PROFILE=""
ACTIVE_SERVER_GPU=""
ACTIVE_SERVER_LOG=""
declare -a WORKER_PIDS=()

SGLANG_PROGRAM='import json,sys; from incumbent_gemma_bench import sglang_language_model_override; from sglang.srt.entrypoints.http_server import launch_server; from sglang.srt.server_args import ServerArgs; model,served,host,port,chunk,max_running,prefill_graph,context_len,max_total,mem_fraction,backend=sys.argv[1:12]; kwargs=dict(model_path=model,served_model_name=served,host=host,port=int(port),dtype="bfloat16",context_length=int(context_len),max_total_tokens=int(max_total),mem_fraction_static=float(mem_fraction),chunked_prefill_size=int(chunk),max_running_requests=int(max_running),json_model_override_args=json.dumps(sglang_language_model_override(model),separators=(",",":"),sort_keys=True),cuda_graph_backend_decode="full",cuda_graph_backend_prefill=prefill_graph,disable_cuda_graph=False,disable_decode_cuda_graph=False,disable_prefill_cuda_graph=False,disable_radix_cache=False,disable_chunked_prefix_cache=False,disable_overlap_schedule=False,enable_cache_report=True,enable_custom_logit_processor=True,enable_multimodal=False,enable_torch_compile=False,enable_two_batch_overlap=False,enable_single_batch_overlap=False,skip_tokenizer_init=True,sampling_defaults="openai",log_level="warning"); kwargs.update({} if backend == "auto" else {"attention_backend":backend}); launch_server(ServerArgs(**kwargs))'

declare -a GPU_SELECTORS=()
declare -a PORT_VALUES=()
declare -a BASE_PROFILES=()
declare -a VLLM_TOKEN_VALUES=()
declare -a VLLM_MEMORY_VALUES=()
declare -a SGLANG_CHUNK_VALUES=()
declare -a SGLANG_MEMORY_VALUES=()
declare -A GPU_UUID_BY_SELECTOR=()

declare -a RUN_IDS=()
declare -a RUN_BASES=()
declare -a RUN_ENGINES=()
declare -a RUN_ROLES=()
declare -a RUN_COMPILE_MODES=()
declare -a RUN_CUDAGRAPH_MODES=()
declare -a RUN_ATTENTION_BACKENDS=()
declare -a RUN_PREFILL_GRAPHS=()
declare -a RUN_TOKEN_CHUNKS=()
declare -a RUN_MEMORY_FRACTIONS=()
declare -a RUN_MODEL_RUNNERS=()
declare -a RUN_TRACE_MODES=()

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_csv() {
  local raw="$1"
  local -n target="$2"
  local -a parsed=()
  local value
  IFS=',' read -r -a parsed <<<"$raw"
  target=()
  if ((${#parsed[@]} == 0)); then
    printf 'Comma-separated setting cannot be empty\n' >&2
    exit 1
  fi
  for value in "${parsed[@]}"; do
    value="$(trim "$value")"
    if [[ -z "$value" ]]; then
      printf 'Empty value in comma-separated setting: %s\n' "$raw" >&2
      exit 1
    fi
    target+=("$value")
  done
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
  if ! "$WKVM_PY" -c \
    'import math,sys; x=float(sys.argv[1]); raise SystemExit(not (math.isfinite(x) and 0 < x < 1))' \
    "$value"; then
    printf '%s must be finite and between 0 and 1: %s\n' "$name" "$value" >&2
    exit 1
  fi
}

validate_port() {
  local value="$1"
  validate_positive_integer port "$value"
  if ((value > 65535)); then
    printf 'port must be <= 65535: %s\n' "$value" >&2
    exit 1
  fi
}

fraction_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

generate_uuid() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    tr -d '\n' </proc/sys/kernel/random/uuid
  else
    "$WKVM_PY" -c 'import uuid; print(uuid.uuid4())'
  fi
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

quoted_command() {
  printf '%q ' "$@"
}

canonical_launch() {
  local gpu="$1"
  local port="$2"
  shift 2
  local rendered
  rendered="$(quoted_command setsid "$@")"
  rendered="${rendered//CUDA_VISIBLE_DEVICES=$gpu/CUDA_VISIBLE_DEVICES=GPU_DEVICE}"
  rendered="${rendered// $port / PORT }"
  printf '%s' "$rendered"
}

assert_port_unbound() {
  local port="$1"
  "$WKVM_PY" "$VALIDATOR" port-unbound --host 127.0.0.1 --port "$port"
}

prove_owned_listener() {
  local profile="$1"
  local port="$2"
  printf 'listener-proof profile=%s port=%s process_group=%s\n' \
    "$profile" "$port" "$ACTIVE_SERVER_PID"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$WKVM_PY" "$VALIDATOR" listener-owned \
      --port "$port" --process-group '<owned-server-pgid>'
    return 0
  fi
  "$WKVM_PY" "$VALIDATOR" listener-owned \
    --port "$port" --process-group "$ACTIVE_SERVER_PID" >/dev/null
}

validate_trace_contract() {
  local path="$1"
  "$WKVM_PY" "$VALIDATOR" trace \
    --path "$path" \
    --sessions "$SESSIONS" \
    --turns "$TURNS" \
    --initial-context-tokens "$INITIAL_CONTEXT_TOKENS" \
    --turn-input-tokens "$TURN_INPUT_TOKENS" \
    --output-tokens-per-turn "$OUTPUT_TOKENS_PER_TURN"
}

validate_server_info() {
  local profile="$1"
  local engine="$2"
  local phase="$3"
  local path="$4"
  local config="$5"
  local version="$6"
  printf 'server-info-validate profile=%s engine=%s phase=%s path=%s\n' \
    "$profile" "$engine" "$phase" "$path"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$WKVM_PY" "$VALIDATOR" server-info \
    --path "$path" \
    --engine "$engine" \
    --config-json "$config" \
    --version "$version" \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME"
}

validate_benchmark_artifact() {
  local profile="$1"
  local engine="$2"
  local artifact="$3"
  local run_id="$4"
  local config="$5"
  local version="$6"
  local gpu="$7"
  local trace_mode="$8"
  local trace_path="$9"
  local trace_sha256="$TRACE_CONTRACT_SHA256"
  if [[ "$DRY_RUN" != "1" && "$trace_mode" == autonomous_source ]]; then
    trace_sha256="$(validate_trace_contract "$trace_path")"
  fi
  printf 'artifact-validate profile=%s engine=%s artifact=%s trace_mode=%s trace_sha256=%s\n' \
    "$profile" "$engine" "$artifact" "$trace_mode" "$trace_sha256"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$WKVM_PY" "$VALIDATOR" artifact \
    --path "$artifact" \
    --engine "$engine" \
    --profile "$profile" \
    --campaign-id "$CAMPAIGN_ID" \
    --run-id "$run_id" \
    --trace-mode "$trace_mode" \
    --trace-sha256 "$trace_sha256" \
    --trace-path "$trace_path" \
    --version "$version" \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --gpu-selector "$gpu" \
    --sessions "$SESSIONS" \
    --turns "$TURNS" \
    --initial-context-tokens "$INITIAL_CONTEXT_TOKENS" \
    --turn-input-tokens "$TURN_INPUT_TOKENS" \
    --output-tokens-per-turn "$OUTPUT_TOKENS_PER_TURN" \
    --config-json "$config"
}

add_run() {
  RUN_IDS+=("$1")
  RUN_BASES+=("$2")
  RUN_ENGINES+=("$3")
  RUN_ROLES+=("$4")
  RUN_COMPILE_MODES+=("$5")
  RUN_CUDAGRAPH_MODES+=("$6")
  RUN_ATTENTION_BACKENDS+=("$7")
  RUN_PREFILL_GRAPHS+=("$8")
  RUN_TOKEN_CHUNKS+=("$9")
  RUN_MEMORY_FRACTIONS+=("${10}")
  RUN_MODEL_RUNNERS+=("${11}")
  RUN_TRACE_MODES+=("${12}")
}

expand_profile_matrix() {
  local base token memory backend graph role mode cudagraph run_id
  local -A seen_bases=()
  local -A seen_runs=()
  for base in "${BASE_PROFILES[@]}"; do
    if [[ -n "${seen_bases[$base]:-}" ]]; then
      printf 'Duplicate base profile: %s\n' "$base" >&2
      exit 1
    fi
    seen_bases[$base]=1
    case "$base" in
      vllm-inductor-full-and-piecewise)
        role=candidate
        mode=3
        cudagraph=FULL_AND_PIECEWISE
        for token in "${VLLM_TOKEN_VALUES[@]}"; do
          for memory in "${VLLM_MEMORY_VALUES[@]}"; do
            run_id="$base-bt$token-mem$(fraction_tag "$memory")"
            add_run "$run_id" "$base" vllm "$role" "$mode" "$cudagraph" - - "$token" "$memory" v1 teacher_forced_replay
          done
        done
        ;;
      vllm-v2-native)
        role=native
        mode=3
        cudagraph=FULL_AND_PIECEWISE
        for token in "${VLLM_TOKEN_VALUES[@]}"; do
          for memory in "${VLLM_MEMORY_VALUES[@]}"; do
            run_id="$base-bt$token-mem$(fraction_tag "$memory")"
            add_run "$run_id" "$base" vllm "$role" "$mode" "$cudagraph" - - "$token" "$memory" v2 autonomous_source
          done
        done
        ;;
      vllm-mode0-control)
        role=control
        mode=0
        cudagraph=FULL_DECODE_ONLY
        for token in "${VLLM_TOKEN_VALUES[@]}"; do
          for memory in "${VLLM_MEMORY_VALUES[@]}"; do
            run_id="$base-bt$token-mem$(fraction_tag "$memory")"
            add_run "$run_id" "$base" vllm "$role" "$mode" "$cudagraph" - - "$token" "$memory" v1 teacher_forced_replay
          done
        done
        ;;
      sglang-auto-breakable|sglang-auto-tc-piecewise|sglang-auto-disabled|sglang-triton-breakable|sglang-triton-tc-piecewise|sglang-triton-disabled)
        if [[ "$base" == sglang-auto-* ]]; then
          backend=auto
        else
          backend=triton
        fi
        graph="${base##*-}"
        if [[ "$base" == *-tc-piecewise ]]; then
          graph=tc_piecewise
        fi
        if [[ "$graph" == disabled ]]; then
          role=control
        else
          role=capability
        fi
        for token in "${SGLANG_CHUNK_VALUES[@]}"; do
          for memory in "${SGLANG_MEMORY_VALUES[@]}"; do
            run_id="$base-cp$token-mem$(fraction_tag "$memory")"
            add_run "$run_id" "$base" sglang "$role" - - "$backend" "$graph" "$token" "$memory" - teacher_forced_replay
          done
        done
        ;;
      *)
        printf 'Unknown profile base: %s\n' "$base" >&2
        exit 1
        ;;
    esac
  done
  if ((${#RUN_IDS[@]} > MAX_PROFILE_RUNS)); then
    printf 'Expanded profile count %s exceeds MAX_PROFILE_RUNS=%s\n' \
      "${#RUN_IDS[@]}" "$MAX_PROFILE_RUNS" >&2
    exit 1
  fi
  for run_id in "${RUN_IDS[@]}"; do
    if [[ -n "${seen_runs[$run_id]:-}" ]]; then
      printf 'Duplicate expanded profile: %s\n' "$run_id" >&2
      exit 1
    fi
    seen_runs[$run_id]=1
  done
}

blocking_gpu_processes() {
  local gpu="$1"
  local output
  if ! output="$(
    nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>&1
  )"; then
    printf 'Could not inspect compute processes on GPU %s: %s\n' \
      "$gpu" "$output" >&2
    return 2
  fi
  printf '%s\n' "$output" | awk -F',' '
    {
      pid = $1
      name = $2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (pid ~ /^[0-9]+$/) print pid ":" name
    }
  '
}

refuse_occupied_gpu() {
  local gpu="$1"
  local blocking
  if ! blocking="$(blocking_gpu_processes "$gpu")"; then
    return 2
  fi
  if [[ -n "$blocking" ]]; then
    printf 'GPU %s has active compute process(es); refusing without terminating them: %s\n' \
      "$gpu" "$(printf '%s' "$blocking" | tr '\n' ',')" >&2
    return 2
  fi
}

capture_idle_baseline() {
  local gpu="$1"
  local used
  if ! refuse_occupied_gpu "$gpu"; then
    return 2
  fi
  used="$(
    nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -n 1
  )"
  if [[ ! "$used" =~ ^[0-9]+$ || "$used" -gt "$MAX_IDLE_BASELINE_MIB" ]]; then
    printf 'GPU %s is not idle: used=%s MiB limit=%s MiB\n' \
      "$gpu" "${used:-unknown}" "$MAX_IDLE_BASELINE_MIB" >&2
    return 2
  fi
  printf '%s' "$used"
}

check_gpu_pool() {
  local reference_name=""
  local reference_total=""
  local reference_driver=""
  local gpu row name uuid driver total used
  local -A seen_uuids=()
  printf 'selector\tuuid\tname\tmemory_total_mib\tdriver_version\n'
  for gpu in "${GPU_SELECTORS[@]}"; do
    if ! refuse_occupied_gpu "$gpu"; then
      return 2
    fi
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
    if [[ "$name" != *A800* || -z "$uuid" || -z "$driver" || ! "$total" =~ ^[0-9]+$ ]]; then
      printf 'GPU selector %s is not a valid A800: %s\n' "$gpu" "$row" >&2
      return 2
    fi
    if [[ -n "${seen_uuids[$uuid]:-}" ]]; then
      printf 'GPU selectors resolve to the same UUID: %s\n' "$uuid" >&2
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
      printf 'GPU %s is not idle: used=%s MiB limit=%s MiB\n' \
        "$gpu" "${used:-unknown}" "$MAX_IDLE_BASELINE_MIB" >&2
      return 2
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$gpu" "$uuid" "$name" "$total" "$driver"
  done
}

wait_for_gpu_clear() {
  local gpu="$1"
  local deadline=$((SECONDS + GPU_CLEAR_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if refuse_occupied_gpu "$gpu" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  printf 'GPU %s did not clear after owned-server cleanup; no other process was terminated\n' \
    "$gpu" >&2
  return 2
}

start_server() {
  local profile="$1"
  local gpu="$2"
  local port="$3"
  local log_path="$4"
  shift 4
  printf 'server start profile=%s gpu=%s port=%s log=%s\n' \
    "$profile" "$gpu" "$port" "$log_path"
  print_command setsid "$@"
  ACTIVE_SERVER_PROFILE="$profile"
  ACTIVE_SERVER_GPU="$gpu"
  ACTIVE_SERVER_LOG="$log_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    ACTIVE_SERVER_PID=dry-run
    return 0
  fi
  setsid "$@" >"$log_path" 2>&1 &
  ACTIVE_SERVER_PID=$!
}

wait_for_health() {
  local url="$1"
  printf 'health profile=%s url=%s\n' "$ACTIVE_SERVER_PROFILE" "$url"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl -fsS --max-time 5 "$url"
    return 0
  fi
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$ACTIVE_SERVER_PID" 2>/dev/null; then
      printf 'Profile %s server exited before health check; log tail follows\n' \
        "$ACTIVE_SERVER_PROFILE" >&2
      tail -n 100 "$ACTIVE_SERVER_LOG" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$HEALTH_POLL_INTERVAL_S"
  done
  printf 'Profile %s server did not become healthy in %s seconds\n' \
    "$ACTIVE_SERVER_PROFILE" "$SERVER_READY_TIMEOUT_S" >&2
  tail -n 100 "$ACTIVE_SERVER_LOG" >&2 || true
  return 1
}

stop_owned_server() {
  if [[ -z "$ACTIVE_SERVER_PID" ]]; then
    return 0
  fi
  printf 'server stop profile=%s gpu=%s pid=%s ownership=runner-child-pgid\n' \
    "$ACTIVE_SERVER_PROFILE" "$ACTIVE_SERVER_GPU" "$ACTIVE_SERVER_PID"
  if [[ "$DRY_RUN" == "1" ]]; then
    ACTIVE_SERVER_PID=""
    return 0
  fi
  local pid="$ACTIVE_SERVER_PID"
  local gpu="$ACTIVE_SERVER_GPU"
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  if [[ -n "$pgid" && "$pgid" != "$pid" ]]; then
    printf 'Refusing cleanup for profile %s: child PID %s owns unexpected PGID %s\n' \
      "$ACTIVE_SERVER_PROFILE" "$pid" "$pgid" >&2
    return 2
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
  local deadline=$((SECONDS + SERVER_STOP_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf 'Force-stopping only owned process group %s for profile %s\n' \
      "$pid" "$ACTIVE_SERVER_PROFILE" >&2
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_PROFILE=""
  ACTIVE_SERVER_GPU=""
  ACTIVE_SERVER_LOG=""
  wait_for_gpu_clear "$gpu"
}

capture_server_info() {
  local profile="$1"
  local phase="$2"
  local url="$3"
  local output="$4"
  printf 'server-info profile=%s phase=%s url=%s output=%s\n' \
    "$profile" "$phase" "$url" "$output"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl -fsS --max-time 60 "$url" -o "$output"
    return 0
  fi
  curl -fsS --max-time 60 "$url" -o "$output"
}

run_client() {
  local profile="$1"
  local engine="$2"
  local artifact="$3"
  local log_path="$4"
  local run_id="$5"
  local config="$6"
  local version="$7"
  local gpu="$8"
  local trace_mode="$9"
  local trace_path="${10}"
  shift 10
  printf 'run profile=%s artifact=%s client_log=%s\n' \
    "$profile" "$artifact" "$log_path"
  print_command "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    validate_benchmark_artifact \
      "$profile" "$engine" "$artifact" "$run_id" "$config" "$version" \
      "$gpu" "$trace_mode" "$trace_path"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ! -s "$artifact" ]]; then
    printf 'Profile %s did not produce an artifact: %s\n' "$profile" "$artifact" >&2
    return 1
  fi
  if ((status != 0)); then
    return "$status"
  fi
  validate_benchmark_artifact \
    "$profile" "$engine" "$artifact" "$run_id" "$config" "$version" \
    "$gpu" "$trace_mode" "$trace_path"
}

vllm_compilation_json() {
  local mode="$1"
  local graph="$2"
  "$WKVM_PY" -c \
    'import json,sys; print(json.dumps({"cudagraph_capture_sizes":[1,2,4,8,16,32],"cudagraph_mode":sys.argv[2],"max_cudagraph_capture_size":32,"mode":int(sys.argv[1])},sort_keys=True,separators=(",",":")))' \
    "$mode" "$graph"
}

vllm_config_json() {
  local profile="$1"
  local mode="$2"
  local graph="$3"
  local tokens="$4"
  local memory="$5"
  local model_runner="$6"
  local compilation
  compilation="$(vllm_compilation_json "$mode" "$graph")"
  "$WKVM_PY" -c \
    'import json,sys; v2=sys.argv[7] == "v2"; print(json.dumps({"VLLM_USE_V2_MODEL_RUNNER":v2,"compilation_config":json.loads(sys.argv[2]),"custom_logits_processors_enabled":not v2,"dtype":"bfloat16","enable_chunked_prefill":True,"enable_prefix_caching":True,"gpu_memory_utilization":float(sys.argv[4]),"kv_sharing_fast_prefill":not v2,"language_model_only":True,"max_model_len":int(sys.argv[5]),"max_num_batched_tokens":int(sys.argv[3]),"max_num_seqs":int(sys.argv[6]),"model_runner_generation":sys.argv[7],"profile_id":sys.argv[1],"use_v2_model_runner":v2},sort_keys=True,separators=(",",":")))' \
    "$profile" "$compilation" "$tokens" "$memory" \
    "$INCUMBENT_CONTEXT_LENGTH" "$SESSIONS" "$model_runner"
}

sglang_config_json() {
  local profile="$1"
  local backend="$2"
  local graph="$3"
  local chunk="$4"
  local memory="$5"
  "$WKVM_PY" -c \
    'import json,sys; print(json.dumps({"attention_backend_requested":sys.argv[2],"chunked_prefill_size":int(sys.argv[4]),"context_length":int(sys.argv[6]),"cuda_graph_backend_decode":"full","cuda_graph_backend_prefill":sys.argv[3],"disable_chunked_prefix_cache":False,"disable_cuda_graph":False,"disable_decode_cuda_graph":False,"disable_overlap_schedule":False,"disable_prefill_cuda_graph":False,"disable_radix_cache":False,"dtype":"bfloat16","enable_cache_report":True,"enable_custom_logit_processor":True,"enable_multimodal":False,"enable_single_batch_overlap":False,"enable_torch_compile":False,"enable_two_batch_overlap":False,"max_running_requests":int(sys.argv[7]),"max_total_tokens":int(sys.argv[8]),"mem_fraction_static":float(sys.argv[5]),"profile_id":sys.argv[1],"sampling_defaults":"openai","skip_tokenizer_init":True,"untested_supported_optimizations":["enable_torch_compile","enable_two_batch_overlap","enable_single_batch_overlap"]},sort_keys=True,separators=(",",":")))' \
    "$profile" "$backend" "$graph" "$chunk" "$memory" \
    "$INCUMBENT_CONTEXT_LENGTH" "$SGLANG_MAX_RUNNING_REQUESTS" \
    "$SGLANG_MAX_TOTAL_TOKENS"
}

write_profile_status() {
  local profile="$1"
  local engine="$2"
  local gpu="$3"
  local outcome="$4"
  local exit_code="$5"
  local artifact="$6"
  local server_log="$7"
  local client_log="$8"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'status profile=%s engine=%s gpu=%s outcome=%s exit_code=%s artifact=%s\n' \
      "$profile" "$engine" "$gpu" "$outcome" "$exit_code" "$artifact"
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$profile" "$engine" "$gpu" "$outcome" "$exit_code" \
    "$artifact" "$server_log" "$client_log" \
    >"$STATUS_DIR/$profile.tsv"
}

profile_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  stop_owned_server || true
  exit "$status"
}

run_profile() {
  local index="$1"
  local gpu="$2"
  local port="$3"
  local profile="${RUN_IDS[index]}"
  local engine="${RUN_ENGINES[index]}"
  local role="${RUN_ROLES[index]}"
  local mode="${RUN_COMPILE_MODES[index]}"
  local graph="${RUN_CUDAGRAPH_MODES[index]}"
  local backend="${RUN_ATTENTION_BACKENDS[index]}"
  local prefill_graph="${RUN_PREFILL_GRAPHS[index]}"
  local token_chunk="${RUN_TOKEN_CHUNKS[index]}"
  local memory="${RUN_MEMORY_FRACTIONS[index]}"
  local model_runner="${RUN_MODEL_RUNNERS[index]}"
  local trace_mode="${RUN_TRACE_MODES[index]}"
  local artifact="$ARTIFACT_DIR/$profile.json"
  local autonomous_trace="$TRACE_DIR/$profile.json"
  local server_log="$LOG_DIR/$profile.server.log"
  local client_log="$LOG_DIR/$profile.client.log"
  local pre_info="$SERVER_INFO_DIR/$profile.pre.json"
  local post_info="$SERVER_INFO_DIR/$profile.post.json"
  local gpu_identity="${GPU_UUID_BY_SELECTOR[$gpu]:-$gpu}"
  local lock_name="${gpu_identity//[^[:alnum:]_.-]/_}"
  local lock_fd
  local port_lock_fd
  local baseline='<prelaunch-nvidia-smi>'
  local outcome=success
  local profile_status=0
  local cleanup_status=0
  local config compilation launch_text launch_profile run_id health_url info_url
  local engine_version runner_flag trace_path
  local -a server_command=()
  local -a client_command=()

  printf 'profile start index=%s profile=%s engine=%s role=%s model_runner=%s trace_mode=%s gpu=%s port=%s token_chunk=%s memory_fraction=%s\n' \
    "$index" "$profile" "$engine" "$role" "$model_runner" "$trace_mode" \
    "$gpu" "$port" "$token_chunk" "$memory"
  if [[ "$DRY_RUN" != "1" ]]; then
    trap profile_on_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    exec {lock_fd}>"$GPU_LOCK_DIR/wkvm-a800-incumbent-profile-$lock_name.lock"
    if ! flock -n "$lock_fd"; then
      write_profile_status "$profile" "$engine" "$gpu" gpu_lock_busy 2 \
        "$artifact" "$server_log" "$client_log"
      return 2
    fi
    exec {port_lock_fd}>"$GPU_LOCK_DIR/wkvm-a800-incumbent-port-$port.lock"
    if ! flock -n "$port_lock_fd"; then
      write_profile_status "$profile" "$engine" "$gpu" port_lock_busy 2 \
        "$artifact" "$server_log" "$client_log"
      return 2
    fi
    if ! assert_port_unbound "$port"; then
      write_profile_status "$profile" "$engine" "$gpu" port_already_bound 2 \
        "$artifact" "$server_log" "$client_log"
      return 2
    fi
    if ! baseline="$(capture_idle_baseline "$gpu")"; then
      write_profile_status "$profile" "$engine" "$gpu" gpu_not_idle 2 \
        "$artifact" "$server_log" "$client_log"
      return 2
    fi
  fi

  if [[ "$engine" == vllm ]]; then
    engine_version="$VLLM_VERSION"
    config="$(vllm_config_json "$profile" "$mode" "$graph" "$token_chunk" "$memory" "$model_runner")"
    compilation="$(vllm_compilation_json "$mode" "$graph")"
    if [[ "$model_runner" == v2 ]]; then
      runner_flag=1
    else
      runner_flag=0
    fi
    server_command=(
      env
      "CUDA_VISIBLE_DEVICES=$gpu"
      "PATH=$(dirname "$VLLM_PY"):$PATH"
      HF_HUB_OFFLINE=1
      TOKENIZERS_PARALLELISM=false
      VLLM_SERVER_DEV_MODE=1
      "VLLM_USE_V2_MODEL_RUNNER=$runner_flag"
      "PYTHONPATH=$ROOT"
      "$VLLM_PY" -m vllm.entrypoints.openai.api_server
      --model "$MODEL_PATH"
      --served-model-name "$SERVED_MODEL_NAME"
      --host 127.0.0.1
      --port "$port"
      --dtype bfloat16
      --max-model-len "$INCUMBENT_CONTEXT_LENGTH"
      --max-num-seqs "$SESSIONS"
      --gpu-memory-utilization "$memory"
      --max-num-batched-tokens "$token_chunk"
      --enable-chunked-prefill
      --enable-prefix-caching
      --language-model-only
      --limit-mm-per-prompt '{"image":0,"audio":0}'
      --compilation-config "$compilation"
      --enable-prompt-tokens-details
    )
    if [[ "$model_runner" == v1 ]]; then
      server_command+=(
        --kv-sharing-fast-prefill
        --logits-processors experiments.vllm_shared_history_logits:SharedHistoryLogitsProcessor
      )
    fi
    health_url="http://127.0.0.1:$port/health"
    info_url="http://127.0.0.1:$port/server_info?config_format=json"
    client_command=(
      env "CUDA_VISIBLE_DEVICES=$gpu" "PYTHONPATH=$ROOT"
      "$WKVM_PY" "$BENCHMARK" --engine vllm
      --base-url "http://127.0.0.1:$port"
      --endpoint /v1/completions
    )
  else
    engine_version="$SGLANG_VERSION"
    config="$(sglang_config_json "$profile" "$backend" "$prefill_graph" "$token_chunk" "$memory")"
    server_command=(
      env
      "CUDA_VISIBLE_DEVICES=$gpu"
      HF_HUB_OFFLINE=1
      TOKENIZERS_PARALLELISM=false
      "PYTHONPATH=$ROOT/experiments:$ROOT"
      "$SGLANG_PY" -c "$SGLANG_PROGRAM"
      "$MODEL_PATH" "$SERVED_MODEL_NAME" 127.0.0.1 "$port"
      "$token_chunk" "$SGLANG_MAX_RUNNING_REQUESTS" "$prefill_graph"
      "$INCUMBENT_CONTEXT_LENGTH" "$SGLANG_MAX_TOTAL_TOKENS" "$memory"
      "$backend"
    )
    health_url="http://127.0.0.1:$port/health"
    info_url="http://127.0.0.1:$port/server_info"
    client_command=(
      env "CUDA_VISIBLE_DEVICES=$gpu" "PYTHONPATH=$ROOT"
      "$WKVM_PY" "$BENCHMARK" --engine sglang
      --base-url "http://127.0.0.1:$port"
      --endpoint /v1/completions
      --teacher-forcing-processor "@$SGLANG_PROCESSOR_FILE"
    )
  fi

  launch_text="$(quoted_command setsid "${server_command[@]}")"
  launch_profile="$(canonical_launch "$gpu" "$port" "${server_command[@]}")"
  start_server "$profile" "$gpu" "$port" "$server_log" "${server_command[@]}"
  if ! wait_for_health "$health_url"; then
    outcome=startup_failed
    profile_status=1
  elif ! prove_owned_listener "$profile" "$port"; then
    outcome=server_identity_failed
    profile_status=1
  elif ! capture_server_info "$profile" pre "$info_url" "$pre_info"; then
    outcome=server_info_failed
    profile_status=1
  elif ! validate_server_info \
    "$profile" "$engine" pre "$pre_info" "$config" "$engine_version"; then
    outcome=server_info_invalid
    profile_status=1
  else
    run_id="$(generate_uuid)"
    client_command+=(
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
      --gpu-memory-baseline-used-mib "$baseline"
      --memory-ceiling-mib "$MEMORY_CEILING_MIB"
      --semantics full_kv
      --engine-version "$engine_version"
      --engine-version-source runtime_import
      --target-server-launch-command "$launch_text"
      --target-server-launch-profile "$launch_profile"
      --target-server-config-json "$config"
      --server-metrics-url "$info_url"
      --campaign-id "$CAMPAIGN_ID"
      --repeat-id "$profile"
      --run-id "$run_id"
      --json "$artifact"
    )
    if [[ "$trace_mode" == autonomous_source ]]; then
      trace_path="$autonomous_trace"
      client_command+=(
        --teacher-forcing-field none
        --write-shared-history-trace-json "$trace_path"
      )
    else
      trace_path="$TRACE_JSON"
      client_command+=(--shared-history-trace-json "$trace_path")
    fi
    if ! run_client "$profile" "$engine" "$artifact" "$client_log" \
      "$run_id" "$config" "$engine_version" "$gpu" "$trace_mode" \
      "$trace_path" "${client_command[@]}"; then
      outcome=benchmark_failed
      profile_status=1
    elif ! capture_server_info "$profile" post "$info_url" "$post_info"; then
      outcome=server_info_failed
      profile_status=1
    elif ! validate_server_info \
      "$profile" "$engine" post "$post_info" "$config" "$engine_version"; then
      outcome=server_info_invalid
      profile_status=1
    fi
  fi

  if ! stop_owned_server; then
    cleanup_status=2
    outcome=cleanup_failed
  fi
  write_profile_status "$profile" "$engine" "$gpu" "$outcome" \
    "$profile_status" "$artifact" "$server_log" "$client_log"
  printf 'profile complete profile=%s outcome=%s profile_exit=%s cleanup_exit=%s\n' \
    "$profile" "$outcome" "$profile_status" "$cleanup_status"
  if [[ "$DRY_RUN" != "1" ]]; then
    trap - EXIT INT TERM
  fi
  return "$cleanup_status"
}

print_profile_matrix() {
  local index gpu_index gpu port profile engine
  printf '# campaign_id=%s\n' "$CAMPAIGN_ID"
  printf '# workload=b%s_ctx%s_d%s_t%s_o%s\n' \
    "$SESSIONS" "$INITIAL_CONTEXT_TOKENS" "$TURN_INPUT_TOKENS" \
    "$TURNS" "$OUTPUT_TOKENS_PER_TURN"
  printf '# trace=%s\n' "$TRACE_JSON"
  printf 'index\tprofile\tbase_profile\tengine\trole\tmodel_runner\ttrace_mode\tgpu\tport\tcompile_mode\tcudagraph_mode\tattention_backend\tprefill_graph\ttoken_or_chunk_size\tmemory_fraction\tartifact\tautonomous_trace\tserver_log\tclient_log\tstatus\n'
  for ((index = 0; index < ${#RUN_IDS[@]}; index++)); do
    gpu_index=$((index % ${#GPU_SELECTORS[@]}))
    gpu="${GPU_SELECTORS[gpu_index]}"
    port="${PORT_VALUES[gpu_index]}"
    profile="${RUN_IDS[index]}"
    engine="${RUN_ENGINES[index]}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$index" "$profile" "${RUN_BASES[index]}" "$engine" \
      "${RUN_ROLES[index]}" "${RUN_MODEL_RUNNERS[index]}" \
      "${RUN_TRACE_MODES[index]}" "$gpu" "$port" \
      "${RUN_COMPILE_MODES[index]}" "${RUN_CUDAGRAPH_MODES[index]}" \
      "${RUN_ATTENTION_BACKENDS[index]}" "${RUN_PREFILL_GRAPHS[index]}" \
      "${RUN_TOKEN_CHUNKS[index]}" "${RUN_MEMORY_FRACTIONS[index]}" \
      "$ARTIFACT_DIR/$profile.json" "$TRACE_DIR/$profile.json" \
      "$LOG_DIR/$profile.server.log" \
      "$LOG_DIR/$profile.client.log" "$STATUS_DIR/$profile.tsv"
  done
}

aggregate_status() {
  local profile outcome failures=0
  printf 'profile\tengine\tgpu\toutcome\texit_code\tartifact\tserver_log\tclient_log\n' \
    >"$STATUS_PATH"
  for profile in "${RUN_IDS[@]}"; do
    if [[ ! -s "$STATUS_DIR/$profile.tsv" ]]; then
      printf 'Missing status record for profile %s\n' "$profile" >&2
      return 2
    fi
    cat "$STATUS_DIR/$profile.tsv" >>"$STATUS_PATH"
    outcome="$(cut -f4 "$STATUS_DIR/$profile.tsv")"
    if [[ "$outcome" != success ]]; then
      ((failures += 1))
    fi
  done
  printf 'profile_status=%s failures=%s total=%s\n' \
    "$STATUS_PATH" "$failures" "${#RUN_IDS[@]}"
  if [[ "$FAIL_ON_PROFILE_ERROR" == "1" && "$failures" -gt 0 ]]; then
    return 1
  fi
}

for python_path in "$WKVM_PY" "$VLLM_PY" "$SGLANG_PY"; do
  if [[ ! -x "$python_path" ]]; then
    printf 'Python executable not found: %s\n' "$python_path" >&2
    exit 1
  fi
done
for required_file in "$BENCHMARK" "$SGLANG_PROCESSOR_SOURCE" "$VALIDATOR"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Required file not found: %s\n' "$required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory not found: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

for pair in \
  "SESSIONS:$SESSIONS" \
  "TURNS:$TURNS" \
  "INITIAL_CONTEXT_TOKENS:$INITIAL_CONTEXT_TOKENS" \
  "TURN_INPUT_TOKENS:$TURN_INPUT_TOKENS" \
  "OUTPUT_TOKENS_PER_TURN:$OUTPUT_TOKENS_PER_TURN" \
  "REQUEST_TIMEOUT_S:$REQUEST_TIMEOUT_S" \
  "MEMORY_CEILING_MIB:$MEMORY_CEILING_MIB" \
  "MAX_IDLE_BASELINE_MIB:$MAX_IDLE_BASELINE_MIB" \
  "SGLANG_MAX_RUNNING_REQUESTS:$SGLANG_MAX_RUNNING_REQUESTS" \
  "MAX_PROFILE_RUNS:$MAX_PROFILE_RUNS"; do
  validate_positive_integer "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "DRY_RUN:$DRY_RUN" \
  "PREFLIGHT_ONLY:$PREFLIGHT_ONLY" \
  "ALLOW_VERSION_MISMATCH:$ALLOW_VERSION_MISMATCH" \
  "FAIL_ON_PROFILE_ERROR:$FAIL_ON_PROFILE_ERROR"; do
  validate_boolean "${pair%%:*}" "${pair#*:}"
done
if ((MAX_PROFILE_RUNS > 32)); then
  printf 'MAX_PROFILE_RUNS cannot exceed the hard safety bound of 32\n' >&2
  exit 1
fi

parse_csv "$GPU_DEVICES" GPU_SELECTORS
parse_csv "$PORTS" PORT_VALUES
parse_csv "$PROFILE_BASES" BASE_PROFILES
parse_csv "$VLLM_MAX_NUM_BATCHED_TOKENS_LIST" VLLM_TOKEN_VALUES
parse_csv "$VLLM_GPU_MEMORY_UTILIZATION_LIST" VLLM_MEMORY_VALUES
parse_csv "$SGLANG_CHUNKED_PREFILL_SIZE_LIST" SGLANG_CHUNK_VALUES
parse_csv "$SGLANG_MEM_FRACTION_STATIC_LIST" SGLANG_MEMORY_VALUES
if ((${#GPU_SELECTORS[@]} != ${#PORT_VALUES[@]})); then
  printf 'GPU_DEVICES and PORTS must contain the same number of values\n' >&2
  exit 1
fi
declare -A seen_selectors=()
declare -A seen_ports=()
for ((index = 0; index < ${#GPU_SELECTORS[@]}; index++)); do
  gpu="${GPU_SELECTORS[index]}"
  port="${PORT_VALUES[index]}"
  if [[ -n "${seen_selectors[$gpu]:-}" ]]; then
    printf 'GPU selectors must be unique: %s\n' "$GPU_DEVICES" >&2
    exit 1
  fi
  seen_selectors[$gpu]=1
  validate_port "$port"
  if [[ -n "${seen_ports[$port]:-}" ]]; then
    printf 'Ports must be unique: %s\n' "$PORTS" >&2
    exit 1
  fi
  seen_ports[$port]=1
done
for value in "${VLLM_TOKEN_VALUES[@]}" "${SGLANG_CHUNK_VALUES[@]}"; do
  validate_positive_integer token_or_chunk_size "$value"
done
for value in "${VLLM_MEMORY_VALUES[@]}" "${SGLANG_MEMORY_VALUES[@]}"; do
  validate_fraction memory_fraction "$value"
done

REQUIRED_MODEL_LEN=$((
  INITIAL_CONTEXT_TOKENS
  + TURNS * OUTPUT_TOKENS_PER_TURN
  + (TURNS - 1) * TURN_INPUT_TOKENS
))
ALIGNED_REQUIRED_MODEL_LEN=$((((REQUIRED_MODEL_LEN + 15) / 16) * 16))
INCUMBENT_CONTEXT_LENGTH=$((ALIGNED_REQUIRED_MODEL_LEN + 16))
if [[ -z "$SGLANG_MAX_TOTAL_TOKENS" ]]; then
  SGLANG_MAX_TOTAL_TOKENS=$((SESSIONS * (INCUMBENT_CONTEXT_LENGTH + 400)))
fi
validate_positive_integer SGLANG_MAX_TOTAL_TOKENS "$SGLANG_MAX_TOTAL_TOKENS"
expand_profile_matrix

ROOT_REAL="$(realpath -e -- "$ROOT")"
MODEL_PATH="$(realpath -e -- "$MODEL_PATH")"
OUT_DIR="$(realpath -m -- "$OUT_DIR")"
if [[ "$OUT_DIR" == "$ROOT_REAL" || "$OUT_DIR" == "$ROOT_REAL/"* ]]; then
  printf 'OUT_DIR must be outside the source checkout: %s\n' "$OUT_DIR" >&2
  exit 1
fi
ARTIFACT_DIR="$OUT_DIR/artifacts"
LOG_DIR="$OUT_DIR/logs"
SERVER_INFO_DIR="$OUT_DIR/server-info"
TRACE_DIR="$OUT_DIR/autonomous-traces"
STATUS_DIR="$OUT_DIR/status"
MATRIX_PATH="$OUT_DIR/profile_matrix.tsv"
STATUS_PATH="$OUT_DIR/profile_status.tsv"
GPU_POOL_PATH="$OUT_DIR/gpu_pool.tsv"
SGLANG_PROCESSOR_FILE="$OUT_DIR/sglang_teacher_forcing_processor.txt"
FROZEN_TRACE_JSON="$OUT_DIR/shared_history_trace.json"

if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="a800-incumbent-sweep-$(generate_uuid)"
fi
if [[ "$DRY_RUN" == "1" ]]; then
  TRACE_SOURCE_JSON="${TRACE_JSON:-<required-shared-history-trace.json>}"
  TRACE_JSON="$FROZEN_TRACE_JSON"
  TRACE_CONTRACT_SHA256="$(printf '0%.0s' {1..64})"
  VLLM_VERSION="${VLLM_VERSION:-$VLLM_REQUIRED_VERSION}"
  SGLANG_VERSION="${SGLANG_VERSION:-$SGLANG_REQUIRED_VERSION}"
  printf 'output_dir=%s matrix=%s status=%s profile_count=%s gpu_count=%s\n' \
    "$OUT_DIR" "$MATRIX_PATH" "$STATUS_PATH" "${#RUN_IDS[@]}" \
    "${#GPU_SELECTORS[@]}"
  print_profile_matrix
  for ((wave = 0; wave < ${#RUN_IDS[@]}; wave += ${#GPU_SELECTORS[@]})); do
    wave_end=$((wave + ${#GPU_SELECTORS[@]}))
    if ((wave_end > ${#RUN_IDS[@]})); then
      wave_end=${#RUN_IDS[@]}
    fi
    printf 'wave start first=%s end_exclusive=%s mode=parallel-different-gpus\n' \
      "$wave" "$wave_end"
    for ((index = wave; index < wave_end; index++)); do
      gpu_index=$((index - wave))
      run_profile "$index" "${GPU_SELECTORS[gpu_index]}" "${PORT_VALUES[gpu_index]}"
    done
  done
  exit 0
fi

for command_name in nvidia-smi flock curl setsid ps tee realpath cp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$command_name" >&2
    exit 1
  fi
done
if [[ -z "$TRACE_JSON" || ! -s "$TRACE_JSON" ]]; then
  printf 'TRACE_JSON must name a nonempty shared-history trace for an actual sweep\n' >&2
  exit 1
fi
TRACE_JSON="$(realpath -e -- "$TRACE_JSON")"
TRACE_SOURCE_JSON="$TRACE_JSON"
if [[ -e "$OUT_DIR" && ! -d "$OUT_DIR" ]]; then
  printf 'OUT_DIR exists and is not a directory: %s\n' "$OUT_DIR" >&2
  exit 1
fi
if [[ -d "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit)" ]]; then
  printf 'OUT_DIR must be empty before a sweep: %s\n' "$OUT_DIR" >&2
  exit 1
fi

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
if [[ "$ALLOW_VERSION_MISMATCH" != "1" && "$VLLM_VERSION" != "$VLLM_REQUIRED_VERSION" ]]; then
  printf 'vLLM version mismatch: required=%s actual=%s\n' \
    "$VLLM_REQUIRED_VERSION" "$VLLM_VERSION" >&2
  exit 1
fi
if [[ "$ALLOW_VERSION_MISMATCH" != "1" && "$SGLANG_VERSION" != "$SGLANG_REQUIRED_VERSION" ]]; then
  printf 'SGLang version mismatch: required=%s actual=%s\n' \
    "$SGLANG_REQUIRED_VERSION" "$SGLANG_VERSION" >&2
  exit 1
fi

for port in "${PORT_VALUES[@]}"; do
  assert_port_unbound "$port"
done
TRACE_CONTRACT_SHA256="$(validate_trace_contract "$TRACE_SOURCE_JSON")"

if ! GPU_POOL_CONTENT="$(check_gpu_pool)"; then
  exit 2
fi
while IFS=$'\t' read -r selector uuid _; do
  if [[ "$selector" != selector && -n "$selector" && -n "$uuid" ]]; then
    GPU_UUID_BY_SELECTOR["$selector"]="$uuid"
  fi
done <<<"$GPU_POOL_CONTENT"
mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$SERVER_INFO_DIR" "$TRACE_DIR" "$STATUS_DIR"
cp -- "$TRACE_SOURCE_JSON" "$FROZEN_TRACE_JSON"
frozen_trace_sha256="$(validate_trace_contract "$FROZEN_TRACE_JSON")"
if [[ "$frozen_trace_sha256" != "$TRACE_CONTRACT_SHA256" ]]; then
  printf 'Frozen shared-history trace changed while it was copied\n' >&2
  exit 1
fi
TRACE_JSON="$FROZEN_TRACE_JSON"
printf '%s\n' "$GPU_POOL_CONTENT" >"$GPU_POOL_PATH"
print_profile_matrix >"$MATRIX_PATH"
printf 'campaign_id=%s versions=vllm:%s,sglang:%s matrix=%s gpu_pool=%s\n' \
  "$CAMPAIGN_ID" "$VLLM_VERSION" "$SGLANG_VERSION" "$MATRIX_PATH" \
  "$GPU_POOL_PATH"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  printf 'preflight=passed profiles=%s gpus=%s\n' \
    "${#RUN_IDS[@]}" "${#GPU_SELECTORS[@]}"
  exit 0
fi

if printf '%s\n' "${RUN_ENGINES[@]}" | grep -qx sglang; then
  PYTHONPATH="$ROOT" "$SGLANG_PY" "$SGLANG_PROCESSOR_SOURCE" \
    >"$SGLANG_PROCESSOR_FILE"
  if [[ ! -s "$SGLANG_PROCESSOR_FILE" ]]; then
    printf 'Failed to serialize the SGLang trace processor\n' >&2
    exit 1
  fi
fi

main_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  WORKER_PIDS=()
  exit "$status"
}

remove_worker_pid() {
  local completed="$1"
  local pid
  local -a remaining=()
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ "$pid" != "$completed" ]]; then
      remaining+=("$pid")
    fi
  done
  WORKER_PIDS=("${remaining[@]}")
}

trap main_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((wave = 0; wave < ${#RUN_IDS[@]}; wave += ${#GPU_SELECTORS[@]})); do
  wave_end=$((wave + ${#GPU_SELECTORS[@]}))
  if ((wave_end > ${#RUN_IDS[@]})); then
    wave_end=${#RUN_IDS[@]}
  fi
  printf 'wave start first=%s end_exclusive=%s mode=parallel-different-gpus\n' \
    "$wave" "$wave_end"
  declare -a wave_pids=()
  for ((index = wave; index < wave_end; index++)); do
    gpu_index=$((index - wave))
    worker_log="$LOG_DIR/${RUN_IDS[index]}.worker.log"
    (
      run_profile "$index" "${GPU_SELECTORS[gpu_index]}" \
        "${PORT_VALUES[gpu_index]}"
    ) >"$worker_log" 2>&1 &
    wave_pids+=("$!")
    printf 'profile launch profile=%s gpu=%s port=%s worker_log=%s\n' \
      "${RUN_IDS[index]}" "${GPU_SELECTORS[gpu_index]}" \
      "${PORT_VALUES[gpu_index]}" "$worker_log"
  done
  WORKER_PIDS=("${wave_pids[@]}")
  wave_status=0
  for pid in "${wave_pids[@]}"; do
    if wait "$pid"; then
      :
    else
      child_status=$?
      wave_status="$child_status"
    fi
    remove_worker_pid "$pid"
  done
  if ((wave_status != 0)); then
    printf 'Sweep infrastructure failed in wave starting at index %s\n' "$wave" >&2
    exit 2
  fi
done

aggregate_status
printf 'sweep complete matrix=%s status=%s artifacts=%s logs=%s\n' \
  "$MATRIX_PATH" "$STATUS_PATH" "$ARTIFACT_DIR" "$LOG_DIR"
trap - EXIT INT TERM
