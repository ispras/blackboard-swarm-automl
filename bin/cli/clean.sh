#!/usr/bin/env bash
# Clean the swarm artifacts in sandbox/. Stops the swarm/grader containers.
# usage: bin/cli/clean.sh [--yes]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"
cd "$ROOT"
ensure_runtime_dirs

usage() {
  echo "usage: bin/cli/clean.sh [--yes]" >&2
  exit 1
}

[ $# -le 1 ] || usage
if [ -n "${1:-}" ] && [ "${1:-}" != "--yes" ]; then usage; fi

echo "Будет удалено: всё в sandbox/, что создали агенты:"
echo "  blackboard/, workspaces/, final_solution/, analysis/, shared/, catboost_info/"
echo "  и любые прочие их файлы и папки в sandbox/."
echo "Не тронет: sandbox/data/, sandbox/PROMPT.md, sandbox/TASK.md, private/, bin/, results/, costs/"
echo "История sandbox/blackboard/ остаётся в git, если она была закоммичена."

if [ "${1:-}" != "--yes" ]; then
  read -r -p "Продолжить? [y/N] " ans || { echo "нет ввода — отменено"; exit 1; }
  case "$ans" in y|Y|yes|YES) ;; *) echo "отменено"; exit 1 ;; esac
fi

if ! in_container; then
  compose down --remove-orphans 2>/dev/null || true
fi

find sandbox -mindepth 1 -maxdepth 1 \
  ! -name data ! -name PROMPT.md ! -name TASK.md \
  -exec rm -rf -- {} +

mkdir -p sandbox/blackboard/agent_logs sandbox/workspaces sandbox/final_solution

echo "готово: sandbox/ очищен, созданы пустые sandbox/blackboard/, sandbox/workspaces/, sandbox/final_solution/"
