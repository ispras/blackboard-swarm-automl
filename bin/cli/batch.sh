#!/usr/bin/env bash
# Batch-run MLE-bench tasks listed in one or more subset files.
#
# Subset files contain one competition id per line; blank lines and lines
# starting with '#' are ignored. Files are resolved from subsets/
# first, then from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if in_container; then
  echo "ERROR: batch is a host-side command" >&2
  exit 1
fi

SUBSET_DIR="$ROOT/subsets"
TASK_FILES=()
AGENT_COUNT="${SWARM_AGENTS:-3}"
HOURS="${SWARM_HOURS:-12}"
MODEL="${SWARM_MODEL:-deepseek/deepseek-flash}"
REASONING="${SWARM_REASONING:-max}"
PROMPT_FILE="${SWARM_PROMPT:-prompts/PROMPT_SWARM_v2_internet_usage_rules.md}"
GPU_COUNT="${SWARM_GPU_COUNT:-1}"
BOARD_INTERVAL="${SWARM_BOARD_INTERVAL:-300}"
DO_CLEAN=1
DO_PREPARE=1
DO_SAVE=1
FAIL_FAST=0
DRY_RUN=0
LIST=0

usage() {
  cat >&2 <<'USAGE'
usage:
  bin/cli/swarm.sh batch --subset FILE [--subset FILE ...] [OPTIONS]
  bin/cli/swarm.sh batch FILE [FILE ...] [OPTIONS]
  bin/cli/swarm.sh batch --list

options:
  --subset FILE         subset file (repeatable); a bare name resolves under
                        subsets/ (with or without .txt)
  --agents N            parallel agent sessions per task (default: 3)
  --hours H             wall-clock budget per task in hours (default: 12)
  --model M             opencode model (default: deepseek/deepseek-flash)
  --reasoning LEVEL     model reasoning variant (default: max)
  --prompt FILE         prompt file (default: prompts/PROMPT_SWARM_v2_internet_usage_rules.md)
  --gpus N              GPUs for the swarm container (default: 1)
  --board-interval SEC  board digest period (default: 300; 0 disables)
  --no-clean            keep sandbox artifacts between tasks
  --no-prepare          do not run prepare_data; the active task must already match
  --no-save             disable automatic saving to results/
  --fail-fast           stop after the first failed task
  --dry-run             print the resolved task list and exit
  --list                list available files in subsets/
  -h, --help            show this help

The batch runs tasks sequentially: clean -> prepare_data -> swarm up ->
wait for grader -> stop. Each run is graded and saved like a normal run.
USAGE
  exit "${1:-1}"
}

die() { echo "ERROR: $*" >&2; exit 1; }

need_int() {
  local name="$1" value="$2" min="$3"
  case "$value" in ''|*[!0-9]*) die "$name: integer expected" ;; esac
  [ "$value" -ge "$min" ] || die "$name: must be >= $min"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --subset) [ $# -ge 2 ] || usage; TASK_FILES+=("$2"); shift 2 ;;
    --agents) [ $# -ge 2 ] || usage; AGENT_COUNT="$2"; shift 2 ;;
    --hours) [ $# -ge 2 ] || usage; HOURS="$2"; shift 2 ;;
    --model) [ $# -ge 2 ] || usage; MODEL="$2"; shift 2 ;;
    --reasoning) [ $# -ge 2 ] || usage; REASONING="$2"; shift 2 ;;
    --prompt) [ $# -ge 2 ] || usage; PROMPT_FILE="$2"; shift 2 ;;
    --gpus) [ $# -ge 2 ] || usage; GPU_COUNT="$2"; shift 2 ;;
    --board-interval) [ $# -ge 2 ] || usage; BOARD_INTERVAL="$2"; shift 2 ;;
    --no-clean) DO_CLEAN=0; shift ;;
    --no-prepare) DO_PREPARE=0; shift ;;
    --no-save) DO_SAVE=0; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) usage 0 ;;
    -*) die "unknown option: $1" ;;
    *) TASK_FILES+=("$1"); shift ;;
  esac
done

need_int --agents "$AGENT_COUNT" 1
need_int --hours "$HOURS" 1
need_int --gpus "$GPU_COUNT" 0
need_int --board-interval "$BOARD_INTERVAL" 0
[ -n "$REASONING" ] || die "--reasoning: value expected"

if [ "$LIST" = 1 ]; then
  [ -d "$SUBSET_DIR" ] || die "subset directory not found: $SUBSET_DIR"
  echo "subsets in ${SUBSET_DIR#$ROOT/}:"
  find "$SUBSET_DIR" -maxdepth 1 -type f -name '*.txt' -printf '%f\n' | sort
  exit 0
fi

[ "${#TASK_FILES[@]}" -gt 0 ] || usage

resolve_subset() {
  local raw="$1" candidate
  for candidate in \
    "$raw" \
    "$SUBSET_DIR/$raw" \
    "$SUBSET_DIR/$raw.txt" \
    "$ROOT/$raw" \
    "$ROOT/$raw.txt"; do
    if [ -f "$candidate" ]; then
      readlink -f "$candidate"
      return 0
    fi
  done
  die "subset file not found: $raw (looked in subsets/ and repo root)"
}

TASKS=()
for raw in "${TASK_FILES[@]}"; do
  file="$(resolve_subset "$raw")"
  echo "[batch] subset: ${file#$ROOT/}"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    TASKS+=("$line")
  done < "$file"
done

[ "${#TASKS[@]}" -gt 0 ] || die "no tasks found in the supplied subset files"

if [ "$DRY_RUN" = 1 ]; then
  echo "[batch] dry run: ${#TASKS[@]} task(s)"
  printf '  %s\n' "${TASKS[@]}"
  exit 0
fi

RUN_TAG="batch-$(date +%F_%H-%M-%S)"
LOG_DIR="$ROOT/results"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_TAG}.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

export SWARM_AGENTS="$AGENT_COUNT"
export SWARM_HOURS="$HOURS"
export SWARM_MODEL="$MODEL"
export SWARM_REASONING="$REASONING"
export SWARM_GPU_COUNT="$GPU_COUNT"
export SWARM_BOARD_INTERVAL="$BOARD_INTERVAL"
if [ "$DO_SAVE" = 1 ]; then
  export SWARM_AUTO_SAVE=1
else
  export SWARM_AUTO_SAVE=0
fi

echo "[batch] log: ${LOG#$ROOT/}"
echo "[batch] tasks: ${#TASKS[@]}, agents: $AGENT_COUNT, hours: $HOURS, model: $MODEL, reasoning: $REASONING, gpus: $GPU_COUNT"
echo "[batch] clean: $DO_CLEAN, prepare: $DO_PREPARE, save: $DO_SAVE"

stop_swarm() {
  bash "$ROOT/bin/cli/swarm.sh" stop >/dev/null 2>&1 || true
}

cleanup() {
  stop_swarm
}
trap cleanup EXIT INT TERM

wait_for_grader() {
  local start_ts now running started=0 limit
  start_ts="$(date +%s)"
  limit=$(( start_ts + HOURS * 3600 + 7200 ))
  while :; do
    running="$(compose ps --status running --services 2>/dev/null || true)"
    if printf '%s\n' "$running" | grep -qx grader; then
      started=1
    elif [ "$started" = 1 ]; then
      return 0
    fi
    if [ -f "$ROOT/sandbox/final_solution/grade.json" ]; then
      return 0
    fi
    now="$(date +%s)"
    if [ "$started" = 0 ] && [ $((now - start_ts)) -ge 300 ]; then
      echo "[batch] ERROR: grader did not start" >&2
      return 1
    fi
    if [ "$now" -ge "$limit" ]; then
      echo "[batch] ERROR: timeout waiting for the run to finish" >&2
      return 1
    fi
    sleep 10
  done
}

run_one() {
  local task="$1" index="$2" total="$3" started_at score grade_json active
  started_at="$(date +%s)"
  echo
  echo "===== [batch $index/$total] $task ====="

  if [ "$DO_CLEAN" = 1 ]; then
    echo "[batch] clean"
    if ! bash "$ROOT/bin/cli/clean.sh" --yes; then
      echo "[batch] FAILED: clean"
      stop_swarm
      return 1
    fi
  fi

  if [ "$DO_PREPARE" = 1 ]; then
    echo "[batch] prepare_data install -c $task"
    if ! bash "$ROOT/bin/cli/prepare_data.sh" install -c "$task"; then
      echo "[batch] FAILED: prepare_data"
      stop_swarm
      return 1
    fi
  else
    active="$(sed -n 's/.*"competition":[[:space:]]*"\([^"]*\)".*/\1/p' "$ROOT/private/competition.json" 2>/dev/null | head -1)"
    if [ "$active" != "$task" ]; then
      echo "[batch] FAILED: --no-prepare, but active task is '${active:-none}', expected '$task'"
      return 1
    fi
  fi

  echo "[batch] swarm up"
  if ! bash "$ROOT/bin/cli/swarm.sh" up \
      --agents "$AGENT_COUNT" \
      --hours "$HOURS" \
      --model "$MODEL" \
      --reasoning "$REASONING" \
      --prompt "$PROMPT_FILE" \
      --gpus "$GPU_COUNT" \
      --board-interval "$BOARD_INTERVAL" \
      --no-build; then
    echo "[batch] FAILED: swarm up"
    stop_swarm
    return 1
  fi

  if ! wait_for_grader; then
    echo "[batch] FAILED: wait_for_grader"
    stop_swarm
    return 1
  fi
  stop_swarm

  grade_json="$ROOT/sandbox/final_solution/grade.json"
  if [ ! -f "$grade_json" ]; then
    echo "[batch] FAILED: no grade.json for $task"
    return 1
  fi

  score="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("score", "?"))' "$grade_json" 2>/dev/null || echo "?")"
  echo "[batch] score: $score"

  if [ "$DO_SAVE" = 1 ]; then
    if ! find "$ROOT/results" -maxdepth 1 -type d -name "${task}_run-*" -newermt "@$started_at" -print -quit | grep -q .; then
      echo "[batch] auto-save not found; saving manually"
      bash "$ROOT/bin/cli/save.sh" "batch-${RUN_TAG}-${index}" \
        || echo "[batch] WARN: save failed"
    fi
  fi

  echo "[batch] done: $task"
  return 0
}

ok_count=0
fail_count=0
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  if run_one "$task" "$((i + 1))" "${#TASKS[@]}"; then
    ok_count=$((ok_count + 1))
  else
    fail_count=$((fail_count + 1))
    if [ "$FAIL_FAST" = 1 ]; then
      echo "[batch] fail-fast: stopping"
      break
    fi
  fi
done

echo
echo "[batch] summary: ok=$ok_count failed=$fail_count total=${#TASKS[@]}"
echo "[batch] log: ${LOG#$ROOT/}"
[ "$fail_count" -eq 0 ]
