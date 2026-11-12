#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

usage() {
  cat >&2 <<'USAGE'
usage:
  bin/cli/swarm.sh up [--agents N] [--hours H] [--model MODEL] --prompt FILE [OPTIONS]
  bin/cli/swarm.sh batch --subset FILE [OPTIONS]
  bin/cli/swarm.sh server
  bin/cli/swarm.sh final-phase
  bin/cli/swarm.sh stop
  bin/cli/swarm.sh status
  bin/cli/swarm.sh logs [session]
  bin/cli/swarm.sh attach
  bin/cli/swarm.sh shell
  bin/cli/swarm.sh build

up:
  --agents N            число параллельных сессий, целое >= 1 (default: 3)
  --hours H             бюджет работы в часах, целое >= 1 (default: 12)
                        work_end = start + H - 30 мин, hard_end = start + H
  --model M             модель opencode (default: deepseek/deepseek-flash)
  --prompt FILE         файл промпта (prompts/PROMPT_SWARM.md, prompts/PROMPT_RUS.md, ...)
  --isolation M         принимается для совместимости; в Docker всегда включена изоляция
  --gpus N              число GPU для swarm-контейнера (0 = без GPU; нужен nvidia runtime; default: 1)
  --board-interval SEC  период дайджеста доски (0 = выключить; default: 300)
  --build               пересобрать образы перед запуском
  --no-build            не собирать образы (они должны уже существовать)

batch:
  --subset FILE         subset file (repeatable); bare names resolve under
                        experiments/subsets/
  --list                list available subset files
  all other up flags (--agents, --hours, --model, --prompt, --gpus) apply per task;
  see bin/cli/batch.sh --help for --no-clean/--no-prepare/--no-save/--fail-fast/--dry-run.

Всё работает в Docker: swarm-контейнер видит только sandbox/ (data, blackboard,
workspaces, final_solution), но не private/, results/ и кэш MLE-bench.
Подготовка данных и оценка выполняются в отдельных tools/grader-контейнерах.
Оценка запускается автоматически после hard_end.
USAGE
  exit 1
}

die() { echo "ERROR: $*" >&2; exit 1; }

need_int() {
  local name="$1" value="$2" min="$3"
  case "$value" in ''|*[!0-9]*) die "$name: нужно целое число" ;; esac
  [ "$value" -ge "$min" ] || die "$name: должно быть >= $min"
}

resolve_prompt() {
  local raw="$1" p="" c
  for c in "$raw" "$ROOT/$raw" "$ROOT/prompts/$raw" "$ROOT/sandbox/$raw"; do
    if [ -f "$c" ]; then p="$(readlink -f "$c")"; break; fi
  done
  [ -n "$p" ] || die "нет файла промпта: $raw (искал также prompts/ и sandbox/)"
  [ -s "$p" ] || die "файл промпта пуст: $p"
  case "$p" in
    "$ROOT/prompts/"*)
      PROMPT_REL="prompts/${p#"$ROOT/prompts/"}"
      ;;
    *)
      # Arbitrary prompt files are copied into a read-only sandbox mount.
      mkdir -p "$ROOT/sandbox/.prompts"
      local base hash
      base="$(basename "$p")"
      hash="$(sha1sum "$p" | cut -c1-8)"
      cp -f "$p" "$ROOT/sandbox/.prompts/${hash}-${base}"
      PROMPT_REL="sandbox/.prompts/${hash}-${base}"
      ;;
  esac
}

active_competition() {
  if [ -f "$ROOT/private/competition.json" ]; then
    sed -n 's/.*"competition": *"\([^"]*\)".*/\1/p' "$ROOT/private/competition.json" | head -1
  fi
}

print_agent_map() {
  local f="$ROOT/sandbox/blackboard/.agent_ids" line out=""
  [ -f "$f" ] || { echo "-"; return; }
  while IFS=$'\t' read -r unit name; do
    [ -n "${name:-}" ] && out+="$unit=$name "
  done < "$f"
  echo "${out:- -}"
}

parse_up() {
  AGENT_COUNT="${SWARM_AGENTS:-3}"
  HOURS="${SWARM_HOURS:-12}"
  MODEL="${SWARM_MODEL:-deepseek/deepseek-flash}"
  PROMPT_FILE=""
  ISOLATION="docker"
  GPU_COUNT="${SWARM_GPU_COUNT:-1}"
  BOARD_INTERVAL="${SWARM_BOARD_INTERVAL:-300}"
  DO_BUILD=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --agents) [ $# -ge 2 ] || usage; AGENT_COUNT="$2"; shift 2 ;;
      --hours) [ $# -ge 2 ] || usage; HOURS="$2"; shift 2 ;;
      --model) [ $# -ge 2 ] || usage; MODEL="$2"; shift 2 ;;
      --prompt) [ $# -ge 2 ] || usage; PROMPT_FILE="$2"; shift 2 ;;
      --isolation) [ $# -ge 2 ] || usage; ISOLATION="$2"; shift 2 ;;
      --gpus) [ $# -ge 2 ] || usage; GPU_COUNT="$2"; shift 2 ;;
      --board-interval) [ $# -ge 2 ] || usage; BOARD_INTERVAL="$2"; shift 2 ;;
      --build) DO_BUILD=1; shift ;;
      --no-build) DO_BUILD=0; shift ;;
      *) die "неизвестный параметр: $1" ;;
    esac
  done
  [ -n "$AGENT_COUNT" ] || die "не задан --agents"
  [ -n "$HOURS" ] || die "не задан --hours"
  [ -n "$MODEL" ] || die "не задан --model"
  [ -n "$PROMPT_FILE" ] || die "не задан --prompt"
  need_int --agents "$AGENT_COUNT" 1
  need_int --hours "$HOURS" 1
  need_int --gpus "$GPU_COUNT" 0
  need_int --board-interval "$BOARD_INTERVAL" 0
  case "$ISOLATION" in
    systemd|none|docker) ;;
    *) die "--isolation: ожидается systemd, none или docker" ;;
  esac
  [ "$ISOLATION" = "docker" ] || echo "NOTE: --isolation=$ISOLATION принимается для совместимости; в Docker изоляция всегда включена" >&2
  resolve_prompt "$PROMPT_FILE"
}

cmd_up() {
  parse_up "$@"
  ensure_runtime_dirs
  if container_running swarm; then
    die "swarm уже запущен; сначала bin/cli/swarm.sh stop (или bin/cli/swarm.sh status)"
  fi
  export SWARM_AGENTS="$AGENT_COUNT"
  export SWARM_HOURS="$HOURS"
  export SWARM_MODEL="$MODEL"
  export SWARM_PROMPT="$PROMPT_REL"
  export SWARM_GPU_COUNT="$GPU_COUNT"
  export SWARM_BOARD_INTERVAL="$BOARD_INTERVAL"
  export SWARM_SERVER_ONLY=0

  ensure_swarm_image
  ensure_tools_image
  if [ "$DO_BUILD" = "1" ]; then
    compose build swarm grader
  fi
  rm -f \
    "$ROOT/sandbox/blackboard/.swarm_running" \
    "$ROOT/sandbox/blackboard/.swarm_port" \
    "$ROOT/sandbox/blackboard/.swarm_hard_end" \
    "$ROOT/sandbox/blackboard/.swarm_final_phase" \
    "$ROOT/sandbox/blackboard/.agent_ids" \
    "$ROOT/sandbox/blackboard/.swarm_agents" \
    "$ROOT/sandbox/blackboard/.swarm_hours" \
    "$ROOT/sandbox/blackboard/deadline.json" \
    "$ROOT/sandbox/analysis/.board_seen.json"
  # Derived grader artifacts are test feedback; never let a new swarm see them.
  rm -f "$ROOT/sandbox/analysis/grade.json" "$ROOT/sandbox/final_solution/grade.json"
  if compgen -G "$ROOT/sandbox/blackboard/*_SYSTEM_grade.md" >/dev/null; then
    echo "ВНИМАНИЕ: на доске есть SYSTEM_grade от прошлого запуска (test feedback)." >&2
    echo "           для чистого старта: bin/cli/clean.sh --yes" >&2
  fi
  compose up -d --remove-orphans swarm grader

  local port
  if ! port="$(wait_for_port_file)"; then
    echo "ERROR: opencode serve не поднялся; логи swarm:" >&2
    compose logs --tail=80 swarm >&2 || true
    exit 1
  fi

  # The supervisor writes .agent_ids right after starting the sessions.
  local i
  for i in $(seq 1 40); do
    [ -s "$ROOT/sandbox/blackboard/.agent_ids" ] && break
    sleep 0.25
  done

  local comp
  comp="$(active_competition)"
  cat <<SUMMARY
Сессий: $AGENT_COUNT, бюджет: ${HOURS}ч, модель: $MODEL, изоляция: docker
Агенты: $(print_agent_map)
Задача: ${comp:-(не выбрана — bin/cli/prepare_data.sh install -c <competition-id>)}
Промпт: $PROMPT_FILE
Дедлайн: $(cat "$ROOT/sandbox/blackboard/deadline.json" 2>/dev/null || echo 'ещё не записан')
Server:  http://127.0.0.1:$port (внутри контейнера)
Live:    bin/cli/swarm.sh logs            (systemd journal swarm.service)
         tail -f sandbox/blackboard/agent_logs/swarm-1.log
Подключиться: bin/cli/swarm.sh attach
Остановить:   bin/cli/swarm.sh stop
Оценка:       grader выполнит bin/cli/grade.sh --post --force после hard_end
SUMMARY
  if [ -z "$comp" ] || [ ! -d "$ROOT/sandbox/data" ] || [ -z "$(ls -A "$ROOT/sandbox/data" 2>/dev/null)" ]; then
    echo "ВНИМАНИЕ: задача не установлена. Сначала: bin/cli/prepare_data.sh install -c <competition-id>" >&2
  fi
  if [ -f "$ROOT/sandbox/final_solution/submission.csv" ]; then
    echo "ВНИМАНИЕ: в final_solution/ уже лежит submission.csv (возможно, от прошлого запуска);" >&2
    echo "           для чистого старта: bin/cli/clean.sh --yes" >&2
  fi
}

cmd_server() {
  ensure_runtime_dirs
  export SWARM_AGENTS=0
  export SWARM_SERVER_ONLY=1
  export SWARM_HOURS="${SWARM_HOURS:-12}"
  export SWARM_PROMPT="${SWARM_PROMPT:-prompts/PROMPT_SWARM.md}"
  export SWARM_MODEL="${SWARM_MODEL:-deepseek/deepseek-flash}"
  ensure_swarm_image
  rm -f "$ROOT/sandbox/blackboard/.swarm_port"
  compose up -d --remove-orphans swarm
  local port
  port="$(wait_for_port_file)" || die "opencode serve не поднялся; смотри bin/cli/swarm.sh logs"
  echo "server: http://127.0.0.1:$port (внутри swarm-контейнера)"
}

cmd_stop() {
  compose down --remove-orphans 2>/dev/null || true
  rm -f "$ROOT/sandbox/blackboard/.swarm_running"
  echo "остановлено"
}

cmd_logs() {
  local session="${1:-}"
  if [ -n "$session" ]; then
    tail -n 100 -f "$ROOT/sandbox/blackboard/agent_logs/$session.log"
  else
    compose exec -T swarm journalctl -u swarm.service -f -n 100
  fi
}

cmd_attach() {
  local port
  port="$(cat "$ROOT/sandbox/blackboard/.swarm_port" 2>/dev/null || true)"
  [ -n "$port" ] || die "нет sandbox/blackboard/.swarm_port: сначала bin/cli/swarm.sh up"
  compose exec -it swarm opencode attach "http://127.0.0.1:$port"
}

cmd_final_phase() { exec bash "$ROOT/bin/cli/final_phase.sh"; }
cmd_status() { exec bash "$ROOT/bin/cli/status.sh" "$@"; }
cmd_shell() { compose exec swarm bash; }
cmd_build() {
  ensure_runtime_dirs
  compose build swarm grader
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  batch) shift; exec bash "$ROOT/bin/cli/batch.sh" "$@" ;;
  server) shift; cmd_server "$@" ;;
  stop|down) shift; cmd_stop "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  attach) shift; cmd_attach "$@" ;;
  shell) shift; cmd_shell "$@" ;;
  build) shift; cmd_build "$@" ;;
  final-phase) shift; cmd_final_phase "$@" ;;
  clean) shift; exec bash "$ROOT/bin/cli/clean.sh" "$@" ;;
  *) usage ;;
esac
