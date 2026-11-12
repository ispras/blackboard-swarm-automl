#!/usr/bin/env bash
# Publishes the SYSTEM final-phase message. Submission scoring is performed
# by a separate grader container after hard_end.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DEADLINE="sandbox/blackboard/deadline.json"
[ -f "$DEADLINE" ] || { echo "нет $DEADLINE" >&2; exit 1; }
HARD_END="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hard_end"])' "$DEADLINE")"
STAMP="$(date +%F_%H-%M-%S)"
MSG="sandbox/blackboard/${STAMP}_SYSTEM_final_phase.md"

# Snapshot of agent costs (best effort): costs/COSTS.md
if [ -f "$ROOT/bin/cli/costs.sh" ]; then
  bash "$ROOT/bin/cli/costs.sh" --quiet || echo "[costs] WARN: costs.sh rc=$?" >&2
fi

cat > "$MSG" <<EOF
# SYSTEM: ФИНАЛЬНАЯ ФАЗА

До жёсткого стопа 30 минут (hard_end: $HARD_END).

1. Немедленно прекратите запускать новые эксперименты и обучения.
2. Соберите лучшее согласованное решение.
3. Загрузите финальный submission в \`final_solution/\`.
4. Подтвердите на доске постом.

После $HARD_END харнесс принудительно остановит все сессии.
Оценка сабмита выполняется автоматически grader-контейнером.
EOF

touch sandbox/blackboard/.swarm_final_phase
echo "опубликовано: $MSG"
echo "оценка: grader выполнит bin/cli/grade.sh --post --force после hard_end"
