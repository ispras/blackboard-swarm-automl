#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="$ROOT/sandbox"
cd "$SANDBOX"

URL="$1"
TITLE="$2"
OPENCODE_BIN="${OPENCODE_BIN:-$(command -v opencode)}"

mkdir -p "$SANDBOX/blackboard/agent_logs"
LOG_FILE="$SANDBOX/blackboard/agent_logs/$TITLE.log"
extra=()
[ -n "${MODEL:-}" ] && extra+=(-m "$MODEL")
[ -n "${REASONING:-}" ] && extra+=(--variant "$REASONING")

# OPENCODE_AGENT_LOG_ONLY=1 (Docker supervisor): agent output is not streamed to stdout
# (otherwise parallel agents produce a mess) — it goes only to the agent's log file.
# In a local swarm output still goes to journal/stdout (via tee).
run_agent_cmd() {
  local -a cmd=("$@")
  if [ "${OPENCODE_AGENT_LOG_ONLY:-0}" = "1" ]; then
    exec >>"$LOG_FILE" 2>&1
    exec "${cmd[@]}"
  fi
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
}

# Resume mode: the agent exited before hard_end; the supervisor (run_swarm.sh)
# sends a new message to the SAME opencode session (--session), without the prompt file.
if [ -n "${OPENCODE_AGENT_SESSION:-}" ]; then
  MESSAGE="${OPENCODE_AGENT_MESSAGE:-Continue work until hard end.}"
  echo "[agent $TITLE] resume session ${OPENCODE_AGENT_SESSION} ($(date -u +%FT%TZ)): $MESSAGE"
  printf '\n===== resume %s at %s =====\n' "$OPENCODE_AGENT_SESSION" "$(date -u +%FT%TZ)" >>"$LOG_FILE"
  run_agent_cmd "$OPENCODE_BIN" run --attach "$URL" --session "$OPENCODE_AGENT_SESSION" \
    --agent build --auto "${extra[@]}" "$MESSAGE"
  exit $?
fi

PROMPT_FILE="${PROMPT_FILE:-$SANDBOX/PROMPT.md}"
if [ ! -f "$PROMPT_FILE" ] && [ -f "$SANDBOX/$PROMPT_FILE" ]; then
  PROMPT_FILE="$SANDBOX/$PROMPT_FILE"
fi
if [ ! -s "$PROMPT_FILE" ]; then
  echo "opencode_agent: промпт не найден или пуст: $PROMPT_FILE" >&2
  exit 1
fi

AGENT_ID="${AGENT_ID:-}"
if [ -z "$AGENT_ID" ] && [ -x "$ROOT/bin/lib/agent_names.sh" ]; then
  AGENT_ID="$(bash "$ROOT/bin/lib/agent_names.sh" 1)" || AGENT_ID=""
fi
export AGENT_ID

PROMPT_TEXT="$(cat "$PROMPT_FILE")"
if [ -n "$AGENT_ID" ]; then
  if grep -qF '{{AGENT_ID}}' "$PROMPT_FILE"; then
    PROMPT_TEXT="${PROMPT_TEXT//\{\{AGENT_ID\}\}/$AGENT_ID}"
  else
    PROMPT_TEXT="AGENT_ID: $AGENT_ID

$PROMPT_TEXT"
  fi
fi

run_agent_cmd "$OPENCODE_BIN" run --attach "$URL" --title "$TITLE" --agent build --auto "${extra[@]}" "$PROMPT_TEXT"
