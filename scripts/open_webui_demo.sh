#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

WKVM_DEMO_HOME="${WKVM_DEMO_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/wkvm-open-webui-demo}"
WKVM_MODEL_DIR="${WKVM_MODEL_DIR:-}"
WKVM_PORT="${WKVM_PORT:-8000}"
OPEN_WEBUI_PORT="${OPEN_WEBUI_PORT:-3000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-wkvm-gemma-4-e4b-it}"
WKVM_VENV="${WKVM_VENV:-$WKVM_DEMO_HOME/wkvm-venv}"
WKVM_PYTHON="${WKVM_PYTHON:-$WKVM_VENV/bin/python}"
OPEN_WEBUI_BIN="${OPEN_WEBUI_BIN:-${UV_TOOL_BIN_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}/open-webui}"
DRY_RUN="${DRY_RUN:-0}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-30}"
LOG_LINES="${LOG_LINES:-100}"

RUN_DIR="$WKVM_DEMO_HOME/run"
LOG_DIR="$WKVM_DEMO_HOME/logs"
OPEN_WEBUI_DATA_DIR="$WKVM_DEMO_HOME/open-webui-data"
SECRET_FILE="$WKVM_DEMO_HOME/open-webui-secret"
WKVM_PID_FILE="$RUN_DIR/wkvm.pid"
OPEN_WEBUI_PID_FILE="$RUN_DIR/open-webui.pid"
WKVM_LOG="$LOG_DIR/wkvm.log"
OPEN_WEBUI_LOG="$LOG_DIR/open-webui.log"

usage() {
  cat <<'EOF'
Usage: scripts/open_webui_demo.sh COMMAND

Commands:
  install  Install WKVM and Open WebUI in isolated environments
  start    Start WKVM and Open WebUI on 127.0.0.1
  stop     Stop the managed Open WebUI and WKVM processes
  status   Show managed process status and local URLs
  logs     Follow both managed process logs
  smoke    Exercise WKVM's OpenAI API and the Open WebUI health endpoint
  doctor   Check the local prerequisites and configuration

Required for start:
  WKVM_MODEL_DIR=/path/to/gemma-4-E4B-it

Set DRY_RUN=1 to print commands without executing or waiting.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_command() {
  print_command "$@"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

validate_common_config() {
  if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    fail "DRY_RUN must be 0 or 1"
  fi
  if [[ ! "$WKVM_PORT" =~ ^[0-9]+$ ]] || ((WKVM_PORT < 1 || WKVM_PORT > 65535)); then
    fail "WKVM_PORT must be an integer from 1 to 65535"
  fi
  if [[ ! "$OPEN_WEBUI_PORT" =~ ^[0-9]+$ ]] || ((OPEN_WEBUI_PORT < 1 || OPEN_WEBUI_PORT > 65535)); then
    fail "OPEN_WEBUI_PORT must be an integer from 1 to 65535"
  fi
  if [[ "$WKVM_PORT" == "$OPEN_WEBUI_PORT" ]]; then
    fail "WKVM_PORT and OPEN_WEBUI_PORT must be different"
  fi
  if [[ ! "$SERVED_MODEL_NAME" =~ ^[[:alnum:]_.:/-]+$ ]]; then
    fail "SERVED_MODEL_NAME may contain only letters, numbers, '.', '_', ':', '/', and '-'"
  fi
  if [[ ! "$STARTUP_TIMEOUT_S" =~ ^[0-9]+$ ]] || ((STARTUP_TIMEOUT_S < 1)); then
    fail "STARTUP_TIMEOUT_S must be a positive integer"
  fi
  if [[ ! "$STOP_TIMEOUT_S" =~ ^[0-9]+$ ]] || ((STOP_TIMEOUT_S < 1)); then
    fail "STOP_TIMEOUT_S must be a positive integer"
  fi
  if [[ ! "$LOG_LINES" =~ ^[0-9]+$ ]] || ((LOG_LINES < 1)); then
    fail "LOG_LINES must be a positive integer"
  fi
}

require_command() {
  local command_name="$1"
  local install_hint="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    fail "$command_name is required; $install_hint"
  fi
}

require_executable() {
  local executable="$1"
  local install_hint="$2"
  if [[ ! -x "$executable" ]]; then
    fail "executable not found: $executable; $install_hint"
  fi
}

require_model() {
  if [[ -z "$WKVM_MODEL_DIR" ]]; then
    fail "WKVM_MODEL_DIR is required; set it to the local Gemma checkpoint directory"
  fi
  if [[ ! -d "$WKVM_MODEL_DIR" ]]; then
    fail "model directory not found: $WKVM_MODEL_DIR"
  fi
}

prepare_directories() {
  run_command mkdir -p "$RUN_DIR" "$LOG_DIR" "$OPEN_WEBUI_DATA_DIR"
  if [[ "$DRY_RUN" == "0" ]]; then
    chmod 700 "$WKVM_DEMO_HOME" "$RUN_DIR" "$OPEN_WEBUI_DATA_DIR"
  fi
}

read_live_pid() {
  local pid_file="$1"
  local pid
  [[ -f "$pid_file" ]] || return 1
  IFS= read -r pid < "$pid_file" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

remove_stale_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]] && ! read_live_pid "$pid_file" >/dev/null; then
    rm -f -- "$pid_file"
  fi
}

ensure_secret() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'ensure-secret path=%s\n' "$SECRET_FILE"
    return
  fi
  if [[ -s "$SECRET_FILE" ]]; then
    return
  fi
  local temporary_secret="$SECRET_FILE.tmp.$$"
  umask 077
  "$WKVM_PYTHON" -c 'import secrets; print(secrets.token_hex(32))' > "$temporary_secret"
  mv -f -- "$temporary_secret" "$SECRET_FILE"
}

launch_wkvm() {
  local -a environment=(
    "TOKENIZERS_PARALLELISM=false"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  )
  local -a command=(
    "$WKVM_PYTHON" -m wkvm.gemma_server
    --model "$WKVM_MODEL_DIR"
    --served-model-name "$SERVED_MODEL_NAME"
    --port "$WKVM_PORT"
    --enable-openai-chat
    --native-gemma-production-profile
    --slots 4
    --max-chat-sessions 4
    --max-queue 16
    --request-timeout-s 600
    --chat-session-ttl-s 1800
  )

  printf 'launch service=wkvm log=%s pid_file=%s\n' "$WKVM_LOG" "$WKVM_PID_FILE"
  print_command env "${environment[@]}" "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  nohup env "${environment[@]}" "${command[@]}" \
    >"$WKVM_LOG" 2>&1 </dev/null &
  printf '%s\n' "$!" > "$WKVM_PID_FILE"
}

launch_open_webui() {
  local webui_secret
  local printed_secret
  if [[ "$DRY_RUN" == "1" ]]; then
    webui_secret="dry-run-secret"
    printed_secret="<stored-in:$SECRET_FILE>"
  else
    IFS= read -r webui_secret < "$SECRET_FILE"
    printed_secret="<redacted>"
  fi
  local -a environment=(
    "DATA_DIR=$OPEN_WEBUI_DATA_DIR"
    "WEBUI_SECRET_KEY=$webui_secret"
    "WEBUI_AUTH=true"
    "ENABLE_OLLAMA_API=false"
    "ENABLE_OPENAI_API=true"
    "OPENAI_API_BASE_URLS=http://127.0.0.1:$WKVM_PORT/v1"
    "OPENAI_API_KEYS=wkvm-local"
    "ENABLE_FORWARD_USER_INFO_HEADERS=true"
    "ENABLE_WEBSOCKET_SUPPORT=true"
    "ENABLE_PERSISTENT_CONFIG=false"
    "DEFAULT_MODELS=$SERVED_MODEL_NAME"
    'DEFAULT_MODEL_PARAMS={"temperature":0,"top_p":1,"function_calling":"legacy","max_tokens":1152}'
    'DEFAULT_MODEL_METADATA={"capabilities":{"builtin_tools":false,"vision":false,"file_upload":false,"file_context":false,"web_search":false,"image_generation":false,"code_interpreter":false,"terminal":false,"memory":false}}'
    "ENABLE_TITLE_GENERATION=false"
    "ENABLE_TAGS_GENERATION=false"
    "ENABLE_FOLLOW_UP_GENERATION=false"
    "ENABLE_CONTEXT_COMPACTION=false"
    "ENABLE_REALTIME_CHAT_SAVE=false"
    "ENABLE_CODE_INTERPRETER=false"
    "ENABLE_MEMORIES=false"
    "ENABLE_WEB_SEARCH=false"
    "ENABLE_IMAGE_GENERATION=false"
  )
  local -a printed_environment=("${environment[@]}")
  printed_environment[1]="WEBUI_SECRET_KEY=$printed_secret"
  local -a command=(
    "$OPEN_WEBUI_BIN" serve --host 127.0.0.1 --port "$OPEN_WEBUI_PORT"
  )

  printf 'launch service=open-webui log=%s pid_file=%s\n' \
    "$OPEN_WEBUI_LOG" "$OPEN_WEBUI_PID_FILE"
  print_command env "${printed_environment[@]}" "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  nohup env "${environment[@]}" "${command[@]}" \
    >"$OPEN_WEBUI_LOG" 2>&1 </dev/null &
  printf '%s\n' "$!" > "$OPEN_WEBUI_PID_FILE"
}

wait_for_health() {
  local service_name="$1"
  local url="$2"
  local pid_file="$3"
  local log_file="$4"
  printf 'wait-for-health service=%s url=%s timeout_s=%s\n' \
    "$service_name" "$url" "$STARTUP_TIMEOUT_S"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl -fsS --max-time 5 "$url"
    return
  fi

  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if ! read_live_pid "$pid_file" >/dev/null; then
      printf 'error: %s exited before becoming healthy; log tail follows\n' \
        "$service_name" >&2
      tail -n 80 "$log_file" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      printf 'ready service=%s url=%s\n' "$service_name" "$url"
      return
    fi
    sleep 2
  done

  printf 'error: %s did not become healthy within %s seconds; log tail follows\n' \
    "$service_name" "$STARTUP_TIMEOUT_S" >&2
  tail -n 80 "$log_file" >&2 || true
  return 1
}

install_demo() {
  if [[ "$DRY_RUN" == "0" ]]; then
    require_command uv "install uv from https://docs.astral.sh/uv/getting-started/installation/"
  fi
  run_command mkdir -p "$WKVM_DEMO_HOME"
  run_command uv python install 3.12
  run_command uv venv --python 3.12 "$WKVM_VENV"
  run_command uv pip install --python "$WKVM_PYTHON" --editable "$ROOT[gemma-server]"
  run_command uv tool install --python 3.12 --torch-backend cpu \
    --with-executables-from huggingface_hub --force 'open-webui==0.10.2'

  if [[ "$DRY_RUN" == "0" && ! -x "$OPEN_WEBUI_BIN" ]]; then
    fail "Open WebUI installed, but $OPEN_WEBUI_BIN was not found; set OPEN_WEBUI_BIN to \"$(uv tool dir --bin)/open-webui\""
  fi
  printf 'installed wkvm_python=%s open_webui=%s\n' \
    "$WKVM_PYTHON" "$OPEN_WEBUI_BIN"
  printf 'next: WKVM_MODEL_DIR=/path/to/gemma-4-E4B-it %q start\n' "$0"
}

start_demo() {
  require_model
  if [[ "$DRY_RUN" == "0" ]]; then
    require_command curl "install curl with your system package manager"
    require_executable "$WKVM_PYTHON" "run '$0 install' first"
    require_executable "$OPEN_WEBUI_BIN" "run '$0 install' first or set OPEN_WEBUI_BIN"
  fi
  prepare_directories
  ensure_secret

  if [[ "$DRY_RUN" == "1" ]]; then
    launch_wkvm
    wait_for_health wkvm "http://127.0.0.1:$WKVM_PORT/health" \
      "$WKVM_PID_FILE" "$WKVM_LOG"
    launch_open_webui
    wait_for_health open-webui "http://127.0.0.1:$OPEN_WEBUI_PORT/health" \
      "$OPEN_WEBUI_PID_FILE" "$OPEN_WEBUI_LOG"
  else
    remove_stale_pid_file "$WKVM_PID_FILE"
    remove_stale_pid_file "$OPEN_WEBUI_PID_FILE"
    if read_live_pid "$WKVM_PID_FILE" >/dev/null; then
      printf 'already-running service=wkvm pid=%s\n' \
        "$(read_live_pid "$WKVM_PID_FILE")"
    else
      launch_wkvm
    fi
    wait_for_health wkvm "http://127.0.0.1:$WKVM_PORT/health" \
      "$WKVM_PID_FILE" "$WKVM_LOG"

    if read_live_pid "$OPEN_WEBUI_PID_FILE" >/dev/null; then
      printf 'already-running service=open-webui pid=%s\n' \
        "$(read_live_pid "$OPEN_WEBUI_PID_FILE")"
    else
      launch_open_webui
    fi
    wait_for_health open-webui "http://127.0.0.1:$OPEN_WEBUI_PORT/health" \
      "$OPEN_WEBUI_PID_FILE" "$OPEN_WEBUI_LOG"
  fi

  printf 'Open WebUI: http://127.0.0.1:%s\n' "$OPEN_WEBUI_PORT"
  printf 'WKVM API:  http://127.0.0.1:%s/v1\n' "$WKVM_PORT"
}

stop_process() {
  local service_name="$1"
  local pid_file="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'stop-process service=%s pid_file=%s timeout_s=%s\n' \
      "$service_name" "$pid_file" "$STOP_TIMEOUT_S"
    return
  fi

  local pid
  if ! pid="$(read_live_pid "$pid_file")"; then
    rm -f -- "$pid_file"
    printf 'not-running service=%s\n' "$service_name"
    return
  fi
  printf 'stopping service=%s pid=%s\n' "$service_name" "$pid"
  kill -TERM "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + STOP_TIMEOUT_S))
  while kill -0 "$pid" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf 'forcing-stop service=%s pid=%s\n' "$service_name" "$pid" >&2
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f -- "$pid_file"
}

stop_demo() {
  stop_process open-webui "$OPEN_WEBUI_PID_FILE"
  stop_process wkvm "$WKVM_PID_FILE"
}

show_process_status() {
  local service_name="$1"
  local pid_file="$2"
  local pid
  if pid="$(read_live_pid "$pid_file")"; then
    printf 'running service=%s pid=%s\n' "$service_name" "$pid"
    return 0
  fi
  printf 'stopped service=%s\n' "$service_name"
  return 1
}

status_demo() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'status-check service=wkvm pid_file=%s url=http://127.0.0.1:%s/health\n' \
      "$WKVM_PID_FILE" "$WKVM_PORT"
    printf 'status-check service=open-webui pid_file=%s url=http://127.0.0.1:%s/health\n' \
      "$OPEN_WEBUI_PID_FILE" "$OPEN_WEBUI_PORT"
    return
  fi
  local status_code=0
  show_process_status wkvm "$WKVM_PID_FILE" || status_code=1
  show_process_status open-webui "$OPEN_WEBUI_PID_FILE" || status_code=1
  printf 'Open WebUI: http://127.0.0.1:%s\n' "$OPEN_WEBUI_PORT"
  printf 'WKVM API:  http://127.0.0.1:%s/v1\n' "$WKVM_PORT"
  return "$status_code"
}

show_logs() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command tail -n "$LOG_LINES" -F "$WKVM_LOG" "$OPEN_WEBUI_LOG"
    return
  fi
  if [[ ! -f "$WKVM_LOG" && ! -f "$OPEN_WEBUI_LOG" ]]; then
    fail "no demo logs found under $LOG_DIR; run '$0 start' first"
  fi
  local -a logs=()
  [[ -f "$WKVM_LOG" ]] && logs+=("$WKVM_LOG")
  [[ -f "$OPEN_WEBUI_LOG" ]] && logs+=("$OPEN_WEBUI_LOG")
  tail -n "$LOG_LINES" -F "${logs[@]}"
}

run_smoke_request() {
  local label="$1"
  shift
  printf 'smoke request=%s\n' "$label"
  print_command "$@"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
    printf '\n'
  fi
}

smoke_demo() {
  if [[ "$DRY_RUN" == "0" ]]; then
    require_command curl "install curl with your system package manager"
  fi
  local wkvm_base="http://127.0.0.1:$WKVM_PORT"
  local webui_base="http://127.0.0.1:$OPEN_WEBUI_PORT"
  local chat_payload
  chat_payload="{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: WKVM demo ready\"}],\"temperature\":0,\"top_p\":1,\"max_tokens\":16,\"stream\":false}"

  run_smoke_request wkvm-health \
    curl -fsS --max-time 10 "$wkvm_base/health"
  run_smoke_request wkvm-models \
    curl -fsS --max-time 10 "$wkvm_base/v1/models"
  run_smoke_request wkvm-chat \
    curl -fsS --max-time 600 \
      -H 'Authorization: Bearer wkvm-local' \
      -H 'Content-Type: application/json' \
      -H 'X-OpenWebUI-User-Id: wkvm-demo-smoke' \
      -H 'X-OpenWebUI-Chat-Id: wkvm-demo-smoke' \
      --data "$chat_payload" \
      "$wkvm_base/v1/chat/completions"
  run_smoke_request open-webui-health \
    curl -fsS --max-time 10 "$webui_base/health"
  printf 'smoke passed\n'
}

doctor_demo() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'doctor-check command=uv\n'
    printf 'doctor-check command=curl\n'
    printf 'doctor-check command=nvidia-smi\n'
    printf 'doctor-check executable=%s\n' "$WKVM_PYTHON"
    printf 'doctor-check executable=%s\n' "$OPEN_WEBUI_BIN"
    printf 'doctor-check model_dir=%s\n' "${WKVM_MODEL_DIR:-<unset>}"
    return
  fi

  local failures=0
  local command_name
  for command_name in uv curl nvidia-smi; do
    if command -v "$command_name" >/dev/null 2>&1; then
      printf 'ok command=%s path=%s\n' "$command_name" "$(command -v "$command_name")"
    else
      printf 'missing command=%s\n' "$command_name" >&2
      failures=$((failures + 1))
    fi
  done
  if [[ -x "$WKVM_PYTHON" ]]; then
    printf 'ok wkvm_python=%s\n' "$WKVM_PYTHON"
  else
    printf 'missing wkvm_python=%s; run %q install\n' "$WKVM_PYTHON" "$0" >&2
    failures=$((failures + 1))
  fi
  if [[ -x "$OPEN_WEBUI_BIN" ]]; then
    printf 'ok open_webui=%s\n' "$OPEN_WEBUI_BIN"
  else
    printf 'missing open_webui=%s; run %q install\n' "$OPEN_WEBUI_BIN" "$0" >&2
    failures=$((failures + 1))
  fi
  if [[ -n "$WKVM_MODEL_DIR" && -d "$WKVM_MODEL_DIR" ]]; then
    printf 'ok model_dir=%s\n' "$WKVM_MODEL_DIR"
  else
    printf 'missing model_dir=%s; set WKVM_MODEL_DIR to the local checkpoint\n' \
      "${WKVM_MODEL_DIR:-<unset>}" >&2
    failures=$((failures + 1))
  fi
  if ((failures > 0)); then
    printf 'doctor found %s problem(s)\n' "$failures" >&2
    return 1
  fi
  printf 'doctor passed\n'
}

main() {
  validate_common_config
  local command_name="${1:-}"
  case "$command_name" in
    install) install_demo ;;
    start) start_demo ;;
    stop) stop_demo ;;
    status) status_demo ;;
    logs) show_logs ;;
    smoke) smoke_demo ;;
    doctor) doctor_demo ;;
    help|-h|--help) usage ;;
    "") usage >&2; return 2 ;;
    *) printf 'error: unknown command: %s\n' "$command_name" >&2; usage >&2; return 2 ;;
  esac
}

main "$@"
