#!/usr/bin/env bash
# Swarm state from inside the tools container: task, deadline, server, agents,
# latest messages, and session costs from opencode.db.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="$ROOT/sandbox"
cd "$ROOT"

now=$(date +%s)

human_age() {
  local s=$1
  if [ "$s" -lt 60 ]; then echo "${s}s назад"
  elif [ "$s" -lt 3600 ]; then echo "$((s / 60))m назад"
  else echo "$((s / 3600))h $((s % 3600 / 60))m назад"
  fi
}

remaining() {
  local t d
  t=$(date -d "$1" +%s 2>/dev/null) || { echo "-"; return; }
  d=$((t - now))
  if [ "$d" -le 0 ]; then echo "истёк"
  else printf '%dh%02dm' $((d / 3600)) $((d % 3600 / 60)); fi
}

echo "== Задача =="
if [ -f "$ROOT/private/competition.json" ]; then
  comp="$(sed -n 's/.*"competition": *"\([^"]*\)".*/\1/p' "$ROOT/private/competition.json" | head -1)"
  cname="$(sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' "$ROOT/private/competition.json" | head -1)"
  echo "активная: ${comp:-?} (${cname:-?})"
else
  echo "активная задача не выбрана (private/competition.json)"
fi

echo
echo "== Дедлайн =="
if [ -f "$SANDBOX/blackboard/deadline.json" ]; then
  we="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_end"])' "$SANDBOX/blackboard/deadline.json")"
  he="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hard_end"])' "$SANDBOX/blackboard/deadline.json")"
  echo "work_end: $we (осталось $(remaining "$we"))"
  echo "hard_end: $he (осталось $(remaining "$he"))"
else
  echo "нет sandbox/blackboard/deadline.json"
fi

echo
echo "== Сервер =="
if [ -f "$SANDBOX/blackboard/.swarm_port" ] && [ -s "$SANDBOX/blackboard/.swarm_port" ]; then
  echo "opencode-serve: порт $(cat "$SANDBOX/blackboard/.swarm_port") (процесс живёт в swarm-контейнере)"
else
  echo "opencode-serve: порт неизвестен (swarm не запущен?)"
fi
if [ -f "$SANDBOX/blackboard/.swarm_running" ]; then
  echo "swarm: marker .swarm_running присутствует"
else
  echo "swarm: marker .swarm_running отсутствует"
fi

echo
echo "== Сессии =="

n_expected=""
[ -f "$SANDBOX/blackboard/.swarm_agents" ] && n_expected="$(cat "$SANDBOX/blackboard/.swarm_agents")"
if [ -z "$n_expected" ] && [ -f "$SANDBOX/blackboard/.agent_ids" ]; then
  n_expected="$(grep -c . "$SANDBOX/blackboard/.agent_ids" 2>/dev/null || true)"
fi
if [ -z "$n_expected" ]; then
  mapfile -t logs < <(ls "$SANDBOX"/blackboard/agent_logs/swarm-*.log 2>/dev/null)
  n_expected="${#logs[@]}"
fi

printf '%-8s %-9s %-24s %-8s %-12s %s\n' "юнит" "статус" "последняя запись" "лог" "стоимость" "сессия"

DB="${XDG_DATA_HOME:-$HOME/.local/share}/opencode/opencode.db"
session_cost() {
  local sid="$1"
  [ -n "$sid" ] || { echo "-"; return; }
  local q="SELECT printf('%.4f', COALESCE(SUM(json_extract(data,'\$.cost')),0)) FROM message WHERE session_id='$sid' AND json_extract(data,'\$.role')='assistant';"
  if [ -f "$DB" ] && command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$DB" "$q" 2>/dev/null
  elif [ -f "$DB" ]; then
    opencode db "$q" --format tsv 2>/dev/null | tail -1
  else
    echo "-"
  fi
}

if [ "${n_expected:-0}" -eq 0 ]; then
  echo "нет данных о запущенных сессиях (sandbox/blackboard/.swarm_agents отсутствует)"
  exit 0
fi

total_cost="0.0000"
if [ -f "$DB" ]; then sessions_list="$(opencode session list 2>/dev/null || true)"; else sessions_list=""; fi
for i in $(seq 1 "$n_expected"); do
  unit="swarm-$i"
  log="$SANDBOX/blackboard/agent_logs/$unit.log"
  state="нет лога"
  act="-"
  size="-"
  if [ -f "$log" ]; then
    age=$((now - $(stat -c %Y "$log")))
    [ "$age" -lt 0 ] && age=0
    act="$(human_age "$age")"
    size="$(du -h "$log" | cut -f1)"
    if [ -f "$SANDBOX/blackboard/.swarm_running" ] && [ "$age" -le 600 ]; then
      state="active"
    elif [ -f "$SANDBOX/blackboard/.swarm_running" ]; then
      state="тишина"
    else
      state="остановлен"
    fi
  fi
  sid="$(printf '%s\n' "$sessions_list" | awk -v t="$unit" '$2 == t {print $1}' | head -1)"
  cost="$(session_cost "$sid")"
  [ -n "$cost" ] || cost="-"
  if [ "$cost" != "-" ]; then
    total_cost="$(awk -v a="$total_cost" -v b="$cost" 'BEGIN {printf "%.4f", a + b}')"
  fi
  printf '%-8s %-9s %-24s %-8s %-12s %s\n' "$unit" "$state" "$act" "$size" "\$$cost" "${sid:--}"
done

echo
echo "итого по сессиям: \$$total_cost"
