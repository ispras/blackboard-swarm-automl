#!/usr/bin/env bash
# Command: save the swarm artifacts (board and final solution) to results/ at the repo root.
# Folder name: <task>_<snapshot_name> (default name: timestamp).
# Run manually after a run, for example before bin/cli/clean.sh.
# usage: bin/cli/save.sh [snapshot_name]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  echo "usage: bin/cli/save.sh [имя_снимка]" >&2
  exit 1
}

[ $# -le 1 ] || usage
NAME="${1:-$(date +%F_%H-%M-%S)}"
case "$NAME" in
  */*|.*) usage ;;
esac

# Task name: private/competition.json -> first backtick in data/README.md
TASK="$(sed -n 's/.*"competition"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' private/competition.json 2>/dev/null | head -1)"
if [ -z "$TASK" ] && [ -f sandbox/data/README.md ]; then
  TASK="$(sed -n '1s/.*`\([^`]*\)`.*/\1/p' sandbox/data/README.md)"
fi
TASK="${TASK//[^A-Za-z0-9._-]/_}"

OUT="results/${TASK:+${TASK}_}${NAME}"
if [ -e "$OUT" ]; then
  echo "уже существует: $OUT (выбери другое имя)" >&2
  exit 1
fi

[ -d sandbox/blackboard ] || { echo "нет sandbox/blackboard — нечего сохранять" >&2; exit 1; }
if [ ! -d sandbox/final_solution ]; then
  echo "WARN: нет sandbox/final_solution — сохраню только доску" >&2
fi

mkdir -p "$OUT"
cp -a sandbox/blackboard "$OUT/blackboard"
if [ -d sandbox/final_solution ]; then
  cp -a sandbox/final_solution "$OUT/final_solution"
fi
if [ -d sandbox/analysis ]; then
  cp -a sandbox/analysis "$OUT/analysis"
fi

echo "сохранено: $OUT"
du -sh "$OUT"/* 2>/dev/null | sed 's/^/  /'
