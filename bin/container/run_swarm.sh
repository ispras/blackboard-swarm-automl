#!/usr/bin/env bash
# In-container supervisor: starts `opencode serve`, launches N agent sessions,
# publishes the final-phase message at work_end and stops everything at hard_end.
# This replaces the systemd units used by the original host harness.
set -euo pipefail
shopt -s nullglob

export TZ="${SWARM_TZ:-UTC}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="$ROOT/sandbox"
LOGDIR="$SANDBOX/blackboard/agent_logs"
PORT_FILE="$SANDBOX/blackboard/.swarm_port"
mkdir -p "$LOGDIR" "$SANDBOX/workspaces" "$SANDBOX/final_solution" "$SANDBOX/analysis"

AGENTS="${SWARM_AGENTS:-3}"
HOURS="${SWARM_HOURS:-12}"
MODEL="${SWARM_MODEL:-deepseek/deepseek-flash}"
REASONING="${SWARM_REASONING:-max}"
SERVER_ONLY="${SWARM_SERVER_ONLY:-0}"
OPENCODE_BIN="${OPENCODE_BIN:-$(command -v opencode)}"
GPU_COUNT="${SWARM_GPU_COUNT:-1}"
BOARD_INTERVAL="${SWARM_BOARD_INTERVAL:-300}"
ANALYSIS_DIR="$SANDBOX/analysis"
RESUME_MESSAGE="${SWARM_RESUME_MESSAGE:-Continue work until hard end.}"
RESUME_DELAY="${SWARM_RESUME_DELAY:-15}"
RESUME_DELAY_MAX="${SWARM_RESUME_DELAY_MAX:-300}"

case "$AGENTS" in ''|*[!0-9]*) AGENTS=3 ;; esac
case "$HOURS" in ''|*[!0-9]*) HOURS=12 ;; esac
case "$GPU_COUNT" in ''|*[!0-9]*) GPU_COUNT=1 ;; esac
case "$BOARD_INTERVAL" in ''|*[!0-9]*) BOARD_INTERVAL=300 ;; esac

PROMPT="${SWARM_PROMPT:-prompts/PROMPT_SWARM_v2_internet_usage_rules.md}"
case "$PROMPT" in
  /*) ;;
  *)
    if [ -f "$ROOT/$PROMPT" ]; then
      PROMPT="$ROOT/$PROMPT"
    else
      PROMPT="$SANDBOX/$PROMPT"
    fi
    ;;
esac
if [ "$SERVER_ONLY" != "1" ] && [ ! -s "$PROMPT" ]; then
  echo "run_swarm: промпт не найден или пуст: $PROMPT" >&2
  exit 1
fi

# The server and every agent session must start inside sandbox/, exactly like
# the host harness (`--working-directory=$SANDBOX`).
cd "$SANDBOX"

# opencode data-dir lives outside sandbox/ so agents cannot corrupt the DB.
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
mkdir -p "$XDG_DATA_HOME/opencode" "$HOME/.config/opencode"

# Optional auth/config injection (useful for CI and for .env-only setups).
if [ -n "${OPENCODE_AUTH_JSON:-}" ]; then
  printf '%s' "$OPENCODE_AUTH_JSON" > "$XDG_DATA_HOME/opencode/auth.json"
  printf '%s' "$OPENCODE_AUTH_JSON" > "$HOME/.local/share/opencode/auth.json"
  chmod 600 "$XDG_DATA_HOME/opencode/auth.json" 2>/dev/null || true
fi
if [ -n "${OPENCODE_CONFIG_JSON:-}" ]; then
  printf '%s' "$OPENCODE_CONFIG_JSON" > "$HOME/.config/opencode/opencode.json"
  echo "opencode config injected: $HOME/.config/opencode/opencode.json"
fi

# --- deadline (same semantics as the host harness) -------------------------
DEADLINE_JSON="$SANDBOX/blackboard/deadline.json"
if [ "$SERVER_ONLY" != "1" ]; then
  if [ ! -e "$DEADLINE_JSON" ]; then
    printf '{"start":"%s","work_end":"%s","hard_end":"%s"}\n' \
      "$(date -Is)" \
      "$(date -Is -d "+$((HOURS * 60 - 30)) minutes")" \
      "$(date -Is -d "+${HOURS} hours")" > "$DEADLINE_JSON"
  fi
  WORK_END="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_end"])' "$DEADLINE_JSON")"
  HARD_END="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hard_end"])' "$DEADLINE_JSON")"
  printf '%s\n' "$HOURS" > "$SANDBOX/blackboard/.swarm_hours"
fi
: > "$SANDBOX/blackboard/.swarm_running"

publish_final_phase() {
  local stamp msg
  stamp="$(date +%F_%H-%M-%S)"
  msg="$SANDBOX/blackboard/${stamp}_SYSTEM_final_phase.md"
  [ -e "$msg" ] && return 0
  cat > "$msg" <<EOF
# SYSTEM: ФИНАЛЬНАЯ ФАЗА ($(date +%H:%M))

До жёсткого стопа 30 минут (hard_end: $HARD_END).

1. Немедленно прекратите запускать новые эксперименты и обучения.
2. Соберите лучшее согласованное решение.
3. Загрузите финальный submission в \`final_solution/\`.
4. Подтвердите на доске постом.

После $HARD_END харнесс принудительно остановит все сессии.
Оценка сабмита выполняется автоматически grader-контейнером.
EOF
  echo "final phase published: $msg"
}

deadline_watcher() {
  local we he now
  we="$(date -d "$WORK_END" +%s)"
  he="$(date -d "$HARD_END" +%s)"
  while :; do
    now="$(date +%s)"
    if [ "$now" -ge "$he" ]; then
      touch "$SANDBOX/blackboard/.swarm_hard_end"
      return 0
    fi
    if [ "$now" -ge "$we" ] && [ ! -e "$SANDBOX/blackboard/.swarm_final_phase" ]; then
      publish_final_phase
      touch "$SANDBOX/blackboard/.swarm_final_phase"
    fi
    sleep 15
  done
}

save_mtimes() {
  python3 "$ROOT/bin/py/blackboard_mtime.py" save --bb "$SANDBOX/blackboard" || true
}

update_costs() {
  if [ -f "$ROOT/bin/cli/costs.sh" ]; then
    bash "$ROOT/bin/cli/costs.sh" --quiet || echo "[costs] WARN: costs.sh rc=$?" >&2
    if [ -f "$ROOT/costs/COSTS.md" ]; then
      cp -f "$ROOT/costs/COSTS.md" "$ANALYSIS_DIR/COSTS.md" || true
    fi
  fi
  return 0
}

board_tick() {
  local cycle="$1"
  local comp_args=()
  [ -n "${SWARM_COMPETITION:-}" ] && comp_args=(--competition "$SWARM_COMPETITION")
  python3 "$ROOT/bin/py/board_report.py" \
    --bb "$SANDBOX/blackboard" --analysis "$ANALYSIS_DIR" \
    "${comp_args[@]+"${comp_args[@]}"}" \
    --print-new --cycle "$cycle" \
    || echo "[board] WARN: board_report rc=$?" >&2
}

board_loop() {
  local cycle=0
  while :; do
    BOARD_LAST_CYCLE=$cycle
    board_tick "$cycle"
    cycle=$((cycle + 1))
    [ "${BOARD_INTERVAL:-0}" -gt 0 ] 2>/dev/null || return 0
    sleep "$BOARD_INTERVAL" || return 0
  done
}

AGENT_PIDS=()
AGENT_START=()
AGENT_NEXT=()
AGENT_DELAY=()
SERVER_PID=""
WATCH_PID=""
BOARD_PID=""
BOARD_LAST_CYCLE=0

stop_all() {
  local p
  for p in ${AGENT_PIDS[@]+"${AGENT_PIDS[@]}"}; do
    kill -TERM -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null || true
  done
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null || true
  [ -n "$BOARD_PID" ] && kill "$BOARD_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}

finalize() {
  trap - EXIT TERM INT
  stop_all
  if [ "${BOARD_INTERVAL:-0}" -gt 0 ] 2>/dev/null && [ -f "$ROOT/bin/py/board_report.py" ]; then
    echo "final board dump:"
    board_tick "$BOARD_LAST_CYCLE" || true
  fi
  update_costs
  save_mtimes
  rm -f "$SANDBOX/blackboard/.swarm_running"
}
trap finalize EXIT
trap 'exit 143' TERM INT

echo "== Docker swarm =="
echo "root:      $ROOT"
echo "cwd:       $(pwd)"
echo "agents:    $AGENTS"
echo "hours:     $HOURS"
echo "model:     ${MODEL:-(default)}"
echo "reasoning: $REASONING"
if [ "$SERVER_ONLY" = "1" ]; then echo "prompt:    (server-only)"; else echo "prompt:    $PROMPT"; fi
echo "gpus:      $GPU_COUNT"
if [ "$SERVER_ONLY" = "1" ]; then
  echo "deadline:  (server-only)"
else
  echo "deadline:  $DEADLINE_JSON"
  cat "$DEADLINE_JSON"
fi

if [ "$SERVER_ONLY" != "1" ]; then
  deadline_watcher &
  WATCH_PID=$!
fi

# --- pick a free port ------------------------------------------------------
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
port_free() { ! port_open "$1"; }
PORT="${SWARM_PORT:-4096}"
if ! port_free "$PORT"; then
  for i in $(seq 1 200); do
    if port_free $((4096 + i)); then PORT=$((4096 + i)); break; fi
  done
fi
echo "starting opencode serve on 127.0.0.1:$PORT"
"$OPENCODE_BIN" serve --port "$PORT" --hostname 127.0.0.1 >>"$LOGDIR/opencode-serve.log" 2>&1 &
SERVER_PID=$!

server_ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "opencode serve упал:" >&2
    tail -n 30 "$LOGDIR/opencode-serve.log" >&2 || true
    exit 1
  fi
  if port_open "$PORT"; then
    server_ready=1
    break
  fi
  sleep 0.5
done
if [ "$server_ready" != "1" ]; then
  echo "opencode serve не поднялся за 120с" >&2
  tail -n 30 "$LOGDIR/opencode-serve.log" >&2 || true
  exit 1
fi
printf '%s\n' "$PORT" > "$PORT_FILE"
URL="http://127.0.0.1:$PORT"

if [ "$SERVER_ONLY" = "1" ]; then
  echo "server-only mode: $URL"
  wait "$SERVER_PID"
  exit 0
fi

if [ "${BOARD_INTERVAL:-0}" -gt 0 ] 2>/dev/null && [ -f "$ROOT/bin/py/board_report.py" ]; then
  echo "board digest: каждые ${BOARD_INTERVAL}s"
  board_loop &
  BOARD_PID=$!
fi

AGENT_IDS_FILE="$SANDBOX/blackboard/.agent_ids"
mapfile -t AGENT_NAMES < <(bash "$ROOT/bin/lib/agent_names.sh" "$AGENTS" "$AGENT_IDS_FILE")
if [ "${#AGENT_NAMES[@]}" -lt "$AGENTS" ]; then
  echo "ERROR: не удалось сгенерировать уникальные имена агентов (нужно $AGENTS)" >&2
  exit 1
fi

agent_session_id() {
  local title="$1"
  "$OPENCODE_BIN" db \
    "SELECT id FROM session WHERE title = '$title' ORDER BY time_created DESC LIMIT 1" \
    --format json 2>/dev/null | python3 -c '
import json
import sys

try:
    rows = json.load(sys.stdin)
except Exception:
    rows = []
print(rows[0].get("id", "") if rows else "")
' 2>/dev/null || true
}

start_agent() {
  local i="$1" resume="${2:-0}" title="swarm-$i" sid="" gpu_env=()
  if [ "$GPU_COUNT" -gt 0 ]; then
    gpu_env=(CUDA_VISIBLE_DEVICES="$(( (i - 1) % GPU_COUNT ))")
  fi
  if [ "$resume" = "1" ]; then
    sid="$(agent_session_id "$title")"
    if [ -n "$sid" ]; then
      MODEL="$MODEL" REASONING="$REASONING" OPENCODE_BIN="$OPENCODE_BIN" AGENT_ID="${AGENT_NAMES[$((i - 1))]}" \
        OPENCODE_AGENT_LOG_ONLY=1 \
        OPENCODE_AGENT_SESSION="$sid" OPENCODE_AGENT_MESSAGE="$RESUME_MESSAGE" \
        "${gpu_env[@]}" \
        setsid bash "$ROOT/bin/container/opencode_agent.sh" "$URL" "$title" &
      AGENT_PIDS[$((i - 1))]="$!"
      echo "[agent $title] сессия $sid продолжена: $RESUME_MESSAGE"
      return 0
    fi
    echo "[agent $title] сессия не найдена (opencode.db) — стартую заново с промптом" >&2
  fi
  MODEL="$MODEL" REASONING="$REASONING" PROMPT_FILE="$PROMPT" OPENCODE_BIN="$OPENCODE_BIN" AGENT_ID="${AGENT_NAMES[$((i - 1))]}" \
    OPENCODE_AGENT_LOG_ONLY=1 \
    "${gpu_env[@]}" \
    setsid bash "$ROOT/bin/container/opencode_agent.sh" "$URL" "$title" &
  AGENT_PIDS[$((i - 1))]="$!"
  return 0
}

now="$(date +%s)"
for i in $(seq 1 "$AGENTS"); do
  start_agent "$i" 0
  AGENT_START[$((i - 1))]="$now"
  AGENT_NEXT[$((i - 1))]=0
  AGENT_DELAY[$((i - 1))]="$RESUME_DELAY"
  printf 'swarm-%s\t%s\n' "$i" "${AGENT_NAMES[$((i - 1))]}" >> "$AGENT_IDS_FILE"
done
printf '%s\n' "$AGENTS" > "$SANDBOX/blackboard/.swarm_agents"
AGENT_MAP=""
for i in $(seq 1 "$AGENTS"); do AGENT_MAP+="swarm-$i=${AGENT_NAMES[$((i - 1))]} "; done
echo "started $AGENTS agents (gpus: $GPU_COUNT, round-robin CUDA_VISIBLE_DEVICES)"
echo "agent ids: ${AGENT_MAP% }"
echo "supervision: агент, завершившийся до hard_end, продолжается в своей сессии сообщением: $RESUME_MESSAGE (пауза ${RESUME_DELAY}s, максимум ${RESUME_DELAY_MAX}s)"

while :; do
  if [ -e "$SANDBOX/blackboard/.swarm_hard_end" ]; then
    echo "hard_end reached, stopping agents"
    break
  fi
  now="$(date +%s)"
  for i in $(seq 1 "$AGENTS"); do
    idx=$((i - 1))
    p="${AGENT_PIDS[$idx]:-}"
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      continue
    fi
    if [ -e "$SANDBOX/blackboard/.swarm_hard_end" ]; then
      continue
    fi
    started="${AGENT_START[$idx]:-0}"
    delay="${AGENT_DELAY[$idx]:-$RESUME_DELAY}"
    ran=$(( now - started ))
    if [ "$started" -gt 0 ] && [ "$ran" -ge 60 ]; then
      delay="$RESUME_DELAY"
      AGENT_DELAY[$idx]="$delay"
    fi
    if [ "$now" -lt "${AGENT_NEXT[$idx]:-0}" ]; then
      continue
    fi
    echo "[agent swarm-$i] run завершился за ${ran}s — перезапуск (пауза ${delay}s)"
    start_agent "$i" 1
    AGENT_START[$idx]="$now"
    AGENT_NEXT[$idx]="$(( now + delay ))"
    if [ "$started" -gt 0 ] && [ "$ran" -lt 60 ]; then
      delay=$(( delay * 2 ))
      [ "$delay" -gt "$RESUME_DELAY_MAX" ] && delay="$RESUME_DELAY_MAX"
      AGENT_DELAY[$idx]="$delay"
    fi
  done
  sleep 10
done

exit 0
