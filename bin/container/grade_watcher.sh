#!/usr/bin/env bash
# Waits for hard_end, then grades sandbox/final_solution/submission.csv and
# writes the SYSTEM_grade message, analysis/ and (optionally) results/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DEADLINE="sandbox/blackboard/deadline.json"
HOURS="${SWARM_HOURS:-12}"
case "$HOURS" in ''|*[!0-9]*) HOURS=12 ;; esac

# The swarm container writes deadline.json at startup. Wait for it briefly.
for _ in $(seq 1 600); do
  [ -f "$DEADLINE" ] && break
  sleep 0.5
done

if [ -f "$DEADLINE" ]; then
  HARD_END="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hard_end"])' "$DEADLINE")"
else
  HARD_END="$(date -Is -d "+${HOURS} hours")"
  echo "WARN: нет $DEADLINE; использую hard_end=$HARD_END" >&2
fi

echo "grader: hard_end=$HARD_END"
while :; do
  now="$(date +%s)"
  he="$(date -d "$HARD_END" +%s)"
  [ "$now" -ge "$he" ] && break
  sleep 10
done

echo "grader: hard_end reached, grading"
if [ -f "$ROOT/bin/cli/costs.sh" ]; then
  bash "$ROOT/bin/cli/costs.sh" --quiet || echo "WARN: costs.sh rc=$?" >&2
fi
if [ -f "$ROOT/bin/py/blackboard_mtime.py" ]; then
  python3 "$ROOT/bin/py/blackboard_mtime.py" save --bb "$ROOT/sandbox/blackboard" || true
fi

rc=0
bash "$ROOT/bin/cli/grade.sh" --post --force || rc=$?
if [ "$rc" -ne 0 ]; then
  echo "WARN: grade rc=$rc (см. sandbox/final_solution/ и логи)" >&2
fi

if [ "${SWARM_AUTO_SAVE:-1}" = "1" ]; then
  bash "$ROOT/bin/cli/save.sh" "run-$(date +%F_%H-%M-%S)" \
    || echo "WARN: save.sh rc=$?" >&2
fi

echo "grader: done (grade rc=$rc)"
exit 0
