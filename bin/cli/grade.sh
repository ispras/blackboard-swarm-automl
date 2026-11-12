#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if ! in_container; then
  ensure_runtime_dirs
  ensure_tools_image
  set +e
  run_tools bash /workspace/bin/cli/grade.sh "$@"
  rc=$?
  set -e
  exit "$rc"
fi

cd "$ROOT"

want_post=0
comp=""
prev=""
for arg in "$@"; do
  [ "$arg" = "--post" ] && want_post=1
  case "$prev" in -c|--competition) comp="$arg" ;; esac
  prev="$arg"
done

# python with mlebench installed: MLEBENCH_PY -> python3 (tools image)
run_grade() {
  if [ -n "${MLEBENCH_PY:-}" ] && [ -x "${MLEBENCH_PY}" ]; then
    "$MLEBENCH_PY" "$ROOT/bin/py/grade.py" "$@"; return
  fi
  python3 "$ROOT/bin/py/grade.py" "$@"
}

rc=0
run_grade "$@" || rc=$?

# After the final grade (--post), automatically rebuild the board graph
# and collect sandbox/analysis/ (graph, metrics, grade.json, costs).
if [ "$rc" -eq 0 ] && [ "$want_post" -eq 1 ] && [ -x "$ROOT/bin/cli/graph.sh" ]; then
  echo
  echo "строю граф доски: bin/cli/graph.sh${comp:+ (--competition $comp)}"
  # Pin the task: otherwise analysis uses the metric of the currently active task (bleed)
  if [ -n "$comp" ]; then
    "$ROOT/bin/cli/graph.sh" --competition "$comp" || echo "WARN: граф построить не удалось (rc=$?)" >&2
    if [ -x "$ROOT/bin/cli/analysis.sh" ]; then
      bash "$ROOT/bin/cli/analysis.sh" --competition "$comp" || echo "WARN: analysis собрать не удалось (rc=$?)" >&2
    fi
  else
    "$ROOT/bin/cli/graph.sh" || echo "WARN: граф построить не удалось (rc=$?)" >&2
    if [ -x "$ROOT/bin/cli/analysis.sh" ]; then
      bash "$ROOT/bin/cli/analysis.sh" || echo "WARN: analysis собрать не удалось (rc=$?)" >&2
    fi
  fi
fi
exit "$rc"
