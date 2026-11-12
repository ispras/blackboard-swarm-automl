#!/usr/bin/env bash
# Command: build an interactive graph of the swarm board (sandbox/blackboard/*.md).
# Run manually or automatically from bin/cli/grade.sh --post after the final grade.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if ! in_container; then exec_tools_script cli/graph.sh "$@"; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python3 "$ROOT/bin/py/blackboard_graph.py" "$@"
