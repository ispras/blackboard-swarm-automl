#!/usr/bin/env bash
# Builds the analytics folder for the board: sandbox/analysis/{graph.html,graph.json,
# metrics.csv,grade.json,COSTS.md}. Called from bin/cli/grade.sh --post;
# can also be run manually after a run.
# usage: bin/cli/analysis.sh [--bb DIR] [--out DIR] [--competition ID] [--no-extra]
#   --competition ID  pins the task for metric/direction (needed for archived
#                     boards in results/, otherwise the active task is used);
#   --no-extra        do not copy grade.json/COSTS.md from sandbox (for archives).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if ! in_container; then exec_tools_script cli/analysis.sh "$@"; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BB="$ROOT/sandbox/blackboard"
OUT="$ROOT/sandbox/analysis"
COMPETITION=""
EXTRA=1
while [ $# -gt 0 ]; do
  case "$1" in
    --bb) [ $# -ge 2 ] || { echo "--bb: нужен аргумент" >&2; exit 2; }; BB="$2"; shift 2 ;;
    --out) [ $# -ge 2 ] || { echo "--out: нужен аргумент" >&2; exit 2; }; OUT="$2"; shift 2 ;;
    --competition) [ $# -ge 2 ] || { echo "--competition: нужен аргумент" >&2; exit 2; }; COMPETITION="$2"; shift 2 ;;
    --no-extra) EXTRA=0; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "неизвестный параметр: $1" >&2; exit 2 ;;
  esac
done

[ -d "$BB" ] || { echo "нет доски: $BB" >&2; exit 1; }

mkdir -p "$OUT"
ARGS=(--bb "$BB" --analysis "$OUT")
[ -n "$COMPETITION" ] && ARGS+=(--competition "$COMPETITION")
python3 "$ROOT/bin/py/board_report.py" "${ARGS[@]}"

if [ "$EXTRA" = "1" ]; then
  if [ -f "$ROOT/sandbox/final_solution/grade.json" ]; then
    cp -f "$ROOT/sandbox/final_solution/grade.json" "$OUT/grade.json"
  fi
  if [ -f "$ROOT/costs/COSTS.md" ]; then
    cp -f "$ROOT/costs/COSTS.md" "$OUT/COSTS.md"
  fi
fi

echo "analysis: $OUT"
ls -1 "$OUT" 2>/dev/null | sed 's/^/  /' || true
