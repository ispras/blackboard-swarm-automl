#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if in_container; then
  exec bash "$ROOT/bin/container/status.sh" "$@"
fi

ensure_runtime_dirs
echo "== Docker =="
compose ps -a 2>/dev/null || true
echo
if ! docker image inspect "$TOOLS_IMAGE" >/dev/null 2>&1; then
  echo "tools-образ $TOOLS_IMAGE ещё не собран; выполни bin/cli/swarm.sh build" 
  exit 0
fi
exec_tools_script container/status.sh "$@"
