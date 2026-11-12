#!/usr/bin/env bash
# Agent spend accounting: aggregates cost/tokens from opencode.db
# (session table). Runs in the tools/grader container (or locally
# when SWARM_IN_CONTAINER=1).
#
#   bin/cli/costs.sh [--dir DIR] [--since TS] [--write FILE] [--no-write] [--json] [--quiet]
#
# By default uses only sandbox/ sessions (directory=...) and, if
# blackboard/deadline.json exists, only sessions since the run start (start).
# The report is written to <repo>/costs/COSTS.md (outside sandbox and outside the agent board).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/bin/lib/_docker.sh"

if ! in_container; then exec_tools_script cli/costs.sh "$@"; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="$ROOT/sandbox"
DIR="$SANDBOX"
SINCE=""
# service report outside sandbox: agents cannot see it and it never reaches the board
OUT="$ROOT/costs/COSTS.md"
WRITE=1
JSON_MODE=0
QUIET=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) [ $# -ge 2 ] || { echo "--dir: нужен аргумент" >&2; exit 2; }; DIR="$2"; shift 2 ;;
    --since) [ $# -ge 2 ] || { echo "--since: нужен аргумент" >&2; exit 2; }; SINCE="$2"; shift 2 ;;
    --write) [ $# -ge 2 ] || { echo "--write: нужен аргумент" >&2; exit 2; }; OUT="$2"; shift 2 ;;
    --no-write) WRITE=0; shift ;;
    --json) JSON_MODE=1; shift ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "неизвестный параметр: $1" >&2; exit 2 ;;
  esac
done

OPENCODE_BIN="${OPENCODE_BIN:-$(command -v opencode 2>/dev/null || true)}"
if [ -z "$OPENCODE_BIN" ]; then
  echo "opencode не найден (нужен для 'opencode db')" >&2
  exit 1
fi

if [ -z "$SINCE" ] && [ -f "$SANDBOX/blackboard/deadline.json" ]; then
  start_iso="$(sed -n 's/.*"start":"\([^"]*\)".*/\1/p' "$SANDBOX/blackboard/deadline.json")"
  if [ -n "$start_iso" ]; then
    SINCE="$(date -d "$start_iso" +%s 2>/dev/null || true)"
  fi
fi

dir_sql="${DIR//\'/\'\'}"
sql="SELECT title, model, ROUND(SUM(cost), 4) AS cost, SUM(tokens_input) AS tin, SUM(tokens_output) AS tout, SUM(tokens_reasoning) AS treason, SUM(tokens_cache_read) AS tcache_r, SUM(tokens_cache_write) AS tcache_w, COUNT(*) AS sessions, MIN(time_created) AS first_ts, MAX(time_updated) AS last_ts FROM session WHERE directory = '$dir_sql'"
if [ -n "$SINCE" ]; then
  sql="$sql AND time_created >= $SINCE"
fi
sql="$sql GROUP BY title, model ORDER BY cost DESC"

# Do not open a nonexistent DB: `opencode db` creates it and runs
# migrations, which races with `opencode serve` migrations on a fresh data-dir
# (symptoms: "database is locked" / "CREATE TABLE workspace").
OC_DB="${XDG_DATA_HOME:-$HOME/.local/share}/opencode/opencode.db"
if [ -f "$OC_DB" ]; then
  data="$("$OPENCODE_BIN" db "$sql" --format json 2>/dev/null || true)"
else
  data="[]"
fi

DATA="$data" OUT="$OUT" WRITE="$WRITE" JSON_MODE="$JSON_MODE" QUIET="$QUIET" DIR="$DIR" SINCE="$SINCE" python3 - <<'PY'
import json
import os
import sys
from collections import defaultdict

rows = json.loads(os.environ.get("DATA") or "[]")
out_path = os.environ["OUT"]
write = os.environ.get("WRITE", "1") == "1"
json_mode = os.environ.get("JSON_MODE", "0") == "1"
quiet = os.environ.get("QUIET", "0") == "1"


def model_label(raw):
    try:
        m = json.loads(raw) if raw else {}
    except Exception:
        return raw or "?"
    provider = m.get("providerID", "?")
    name = m.get("id", "?")
    return f"{provider}/{name}"


def fmt_int(n):
    return f"{int(n or 0):,}".replace(",", " ")


by_agent = defaultdict(lambda: {"cost": 0.0, "tin": 0, "tout": 0, "treason": 0, "tcache_r": 0, "tcache_w": 0, "sessions": 0, "models": set()})
by_model = defaultdict(lambda: {"cost": 0.0, "tin": 0, "tout": 0, "treason": 0, "tcache_r": 0, "sessions": 0})
total = {"cost": 0.0, "tin": 0, "tout": 0, "treason": 0, "tcache_r": 0, "tcache_w": 0, "sessions": 0}

for r in rows:
    agent = r.get("title") or "?"
    model = model_label(r.get("model"))
    cost = float(r.get("cost") or 0)
    a = by_agent[agent]
    a["cost"] += cost
    a["tin"] += int(r.get("tin") or 0)
    a["tout"] += int(r.get("tout") or 0)
    a["treason"] += int(r.get("treason") or 0)
    a["tcache_r"] += int(r.get("tcache_r") or 0)
    a["tcache_w"] += int(r.get("tcache_w") or 0)
    a["sessions"] += int(r.get("sessions") or 0)
    a["models"].add(model)
    m = by_model[model]
    m["cost"] += cost
    m["tin"] += int(r.get("tin") or 0)
    m["tout"] += int(r.get("tout") or 0)
    m["treason"] += int(r.get("treason") or 0)
    m["tcache_r"] += int(r.get("tcache_r") or 0)
    m["sessions"] += int(r.get("sessions") or 0)
    total["cost"] += cost
    total["tin"] += int(r.get("tin") or 0)
    total["tout"] += int(r.get("tout") or 0)
    total["treason"] += int(r.get("treason") or 0)
    total["tcache_r"] += int(r.get("tcache_r") or 0)
    total["tcache_w"] += int(r.get("tcache_w") or 0)
    total["sessions"] += int(r.get("sessions") or 0)

report = {
    "directory": os.environ["DIR"],
    "since": os.environ.get("SINCE", ""),
    "total": total,
    "by_agent": {k: {**v, "models": sorted(v["models"])} for k, v in by_agent.items()},
    "by_model": dict(by_model),
}

if json_mode:
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    lines = []
    lines.append("# COSTS")
    lines.append("")
    lines.append(f"- каталог сессий: `{report['directory']}`")
    if report["since"]:
        lines.append(f"- считаются сессии с момента старта прогона (since={report['since']})")
    lines.append(f"- **итого: ${total['cost']:.4f}**")
    lines.append(
        f"- токены: input {fmt_int(total['tin'])}, output {fmt_int(total['tout'])}, "
        f"reasoning {fmt_int(total['treason'])}, cache_read {fmt_int(total['tcache_r'])}"
    )
    lines.append("")
    lines.append("| Agent | Cost | Input | Output | Reasoning | Cache read | Model |")
    lines.append("|---|---|---|---|---|---|---|")
    for agent, a in sorted(by_agent.items(), key=lambda kv: -kv[1]["cost"]):
        lines.append(
            f"| {agent} | ${a['cost']:.4f} | {fmt_int(a['tin'])} | {fmt_int(a['tout'])} | "
            f"{fmt_int(a['treason'])} | {fmt_int(a['tcache_r'])} | {', '.join(sorted(a['models']))} |"
        )
    lines.append("")
    lines.append("| Model | Cost | Input | Output | Cache read | Sessions |")
    lines.append("|---|---|---|---|---|---|")
    for model, m in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"]):
        lines.append(
            f"| {model} | ${m['cost']:.4f} | {fmt_int(m['tin'])} | {fmt_int(m['tout'])} | "
            f"{fmt_int(m['tcache_r'])} | {m['sessions']} |"
        )
    text = "\n".join(lines) + "\n"

    if write:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, out_path)
    if not quiet:
        print(text, end="")
        if write:
            print(f"\nсохранено: {out_path}")
PY
