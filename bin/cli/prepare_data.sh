#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if in_container; then
  exec python3 "$ROOT/bin/py/prepare_data.py" "$@"
fi

ensure_runtime_dirs
ensure_tools_image
set +e
run_tools python3 /workspace/bin/py/prepare_data.py "$@"
rc=$?
set -e
exit "$rc"
