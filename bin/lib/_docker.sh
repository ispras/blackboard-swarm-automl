#!/usr/bin/env bash
# Host-side Docker helpers shared by bin/*.sh wrappers.
# Source this file from a host wrapper; inside a container the wrappers exit
# before using these helpers (SWARM_IN_CONTAINER=1).

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
COMPOSE_FILE="$ROOT/compose.yaml"
GPU_COMPOSE_FILE="$ROOT/compose.gpu.yaml"
PROJECT="${SWARM_PROJECT:-blackboard-sisyphus-experiment}"

# Read a simple KEY=VALUE from .env without sourcing it. Compose also reads .env,
# but host wrappers need a few values before calling compose.
env_file_get() {
  local key="$1" file="$ROOT/.env" line val
  [ -f "$file" ] || return 1
  line="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" | tail -1)" || return 1
  val="${line#*=}"
  val="${val#"${val%%[![:space:]]*}"}"
  val="${val%"${val##*[![:space:]]}"}"
  case "$val" in
    \"*\") val="${val#\"}"; val="${val%\"}" ;;
    \'*\') val="${val#\'}"; val="${val%\'}" ;;
  esac
  printf '%s' "$val"
}

[ -n "${SWARM_TARGET:-}" ] || SWARM_TARGET="$(env_file_get SWARM_TARGET || true)"
[ -n "${SWARM_TOOLS_TARGET:-}" ] || SWARM_TOOLS_TARGET="$(env_file_get SWARM_TOOLS_TARGET || true)"
[ -n "${SWARM_AGENTS:-}" ] || SWARM_AGENTS="$(env_file_get SWARM_AGENTS || true)"
[ -n "${SWARM_HOURS:-}" ] || SWARM_HOURS="$(env_file_get SWARM_HOURS || true)"
[ -n "${SWARM_MODEL:-}" ] || SWARM_MODEL="$(env_file_get SWARM_MODEL || true)"
[ -n "${SWARM_GPU_COUNT:-}" ] || SWARM_GPU_COUNT="$(env_file_get SWARM_GPU_COUNT || true)"
[ -n "${SWARM_IMAGE:-}" ] || SWARM_IMAGE="$(env_file_get SWARM_IMAGE || true)"
[ -n "${SWARM_TOOLS_IMAGE:-}" ] || SWARM_TOOLS_IMAGE="$(env_file_get SWARM_TOOLS_IMAGE || true)"
SWARM_TARGET="${SWARM_TARGET:-swarm}"
SWARM_TOOLS_TARGET="${SWARM_TOOLS_TARGET:-tools}"
SWARM_AGENTS="${SWARM_AGENTS:-3}"
SWARM_HOURS="${SWARM_HOURS:-12}"
SWARM_MODEL="${SWARM_MODEL:-deepseek/deepseek-flash}"
SWARM_GPU_COUNT="${SWARM_GPU_COUNT:-1}"
SWARM_IMAGE="${SWARM_IMAGE:-blackboard-sisyphus-swarm:${SWARM_TARGET}}"
TOOLS_IMAGE="${SWARM_TOOLS_IMAGE:-blackboard-sisyphus-tools:${SWARM_TOOLS_TARGET}}"
export SWARM_TARGET SWARM_TOOLS_TARGET SWARM_AGENTS SWARM_HOURS SWARM_MODEL SWARM_GPU_COUNT SWARM_IMAGE TOOLS_IMAGE
# compose.yaml interpolates SWARM_IMAGE / SWARM_TOOLS_IMAGE.
export SWARM_TOOLS_IMAGE="$TOOLS_IMAGE"

in_container() { [ "${SWARM_IN_CONTAINER:-0}" = "1" ]; }

die() {
  echo "ERROR: $*" >&2
  exit 1
}

compose() {
  local -a files=(-f "$COMPOSE_FILE")
  case "${SWARM_GPU_COUNT:-1}" in
    ''|0) ;;
    *[!0-9]*) ;;
    *) files+=(-f "$GPU_COMPOSE_FILE") ;;
  esac
  docker compose "${files[@]}" -p "$PROJECT" "$@"
}

ensure_runtime_dirs() {
  mkdir -p \
    "$ROOT/sandbox/blackboard/agent_logs" \
    "$ROOT/sandbox/workspaces" \
    "$ROOT/sandbox/final_solution" \
    "$ROOT/sandbox/analysis" \
    "$ROOT/sandbox/data" \
    "$ROOT/sandbox/.prompts" \
    "$ROOT/private" \
    "$ROOT/results" \
    "$ROOT/costs" \
    "$ROOT/.cache/mle-bench" \
    "$ROOT/.kaggle" \
    "$ROOT/.opencode"
  if [ ! -e "$ROOT/sandbox/TASK.md" ]; then
    : > "$ROOT/sandbox/TASK.md"
  fi
}

ensure_image() {
  local image="$1" service="$2"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "building $image (compose service: $service)..." >&2
    compose build "$service"
  fi
}

ensure_swarm_image() { ensure_image "$SWARM_IMAGE" swarm; }
ensure_tools_image() { ensure_image "$TOOLS_IMAGE" grader; }

run_tools() {
  ensure_runtime_dirs
  ensure_tools_image
  compose run --rm -T tools "$@"
}

container_running() {
  compose ps --status running --services 2>/dev/null | grep -qx "$1"
}

wait_for_port_file() {
  local file="$ROOT/sandbox/blackboard/.swarm_port" i
  for i in $(seq 1 240); do
    if [ -s "$file" ]; then
      cat "$file"
      return 0
    fi
    sleep 0.5
  done
  return 1
}

exec_tools_script() {
  local script="$1"
  shift
  ensure_runtime_dirs
  ensure_tools_image
  set +e
  run_tools bash "/workspace/bin/$script" "$@"
  local rc=$?
  set -e
  exit "$rc"
}
