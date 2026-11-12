#!/usr/bin/env python3
"""Board digest for job logs and artifacts: new messages + metrics + graph.

Every N minutes (bin/container/run_swarm.sh, --board-interval) reads sandbox/blackboard:
  * --print-new: prints new messages to stdout (the job log shows the board
    instead of the interleaved log of parallel agents), tracks state in a file;
  * collects metrics from <metric> tags into <analysis>/metrics.csv;
  * rebuilds the board graph in <analysis>/graph.html + graph.json;
  * --upload / --log-metric: MLM-compatible hooks (unused in the Docker harness).

On the host, the same thing (without --print-new/--upload) is called by bin/cli/analysis.sh,
which adds analysis/ grade.json and costs/COSTS.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blackboard_graph as bg  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bb", type=Path, required=True, help="каталог доски (sandbox/blackboard)")
    ap.add_argument("--analysis", type=Path, required=True, help="куда писать metrics.csv/graph (sandbox/analysis)")
    ap.add_argument("--competition", default=None,
                    help="id задачи MLE-bench для метрики/направления (default: активная из private/)")
    ap.add_argument("--state", type=Path, default=None,
                    help="state напечатанных сообщений (default: <analysis>/.board_seen.json)")
    ap.add_argument("--print-new", action="store_true", help="печатать новые сообщения в stdout")
    ap.add_argument("--upload", action="store_true", help="MLM-совместимый log_artifact(<analysis>)")
    ap.add_argument("--log-metric", action="store_true", help="MLM-совместимый log_metric лучшего значения")
    ap.add_argument("--cycle", type=int, default=0, help="номер цикла (step для log_metric)")
    return ap.parse_args()


def board_files(bb: Path) -> list[Path]:
    return sorted(bb.glob("*.md"), key=lambda p: p.stat().st_mtime)


def author_of(name: str) -> str:
    return "system" if bg.NAME_TS_RE.match(name) else name.split("_", 1)[0]


def print_new(files: list[Path], state_path: Path) -> int:
    seen: set[str] = set()
    try:
        seen = set(json.loads(state_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    new = [p for p in files if p.name not in seen]
    for p in new:
        print(f"===== {p.name} =====", flush=True)
        print(p.read_text(errors="replace").strip(), flush=True)
        print(flush=True)
    if new:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(sorted(seen | {p.name for p in new}), ensure_ascii=False), encoding="utf-8"
        )
    return len(new)


def write_metrics(
    bb: Path, files: list[Path], csv_path: Path, comp_id: str | None = None
) -> tuple[list[dict], str | None, str | None]:
    label, direction = bg.load_task_metric(bb, comp_id)
    mtimes = bg.load_mtime_manifest(bb)
    rows: list[dict] = []
    for p in files:
        text = p.read_text(errors="replace")
        try:
            ts = datetime.fromtimestamp(mtimes[p.name], tz=timezone.utc).isoformat(timespec="seconds")
        except KeyError:
            ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        for m in bg.extract_metrics(text, label=(label or "metric"), direction=direction):
            rows.append({
                "time": ts, "file": p.name, "author": author_of(p.name),
                "label": m["label"], "value": m["v"], "dir": m["dir"] or "",
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "file", "author", "label", "value", "dir"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)
    return rows, label, direction


def best_metric(rows: list[dict], direction: str | None) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: r["value"]) if direction == "low" else max(rows, key=lambda r: r["value"])


def build_graph(bb: Path, analysis: Path, comp_id: str | None = None) -> None:
    graph = bg.build_graph(bb, comp_id)
    (analysis / "graph.html").write_text(bg.render_html(graph), encoding="utf-8")
    (analysis / "graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def upload_analysis(analysis: Path) -> None:
    cwd = os.getcwd()
    try:
        os.chdir(analysis.parent)
        from ML_management import mlmanagement

        mlmanagement.log_artifact(analysis.name)
        print(f"[board] artifact uploaded: {analysis.name}/", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[board] WARN: log_artifact: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        os.chdir(cwd)


def log_best(all_rows: list[dict], direction: str | None, step: int) -> None:
    best = best_metric(all_rows, direction)
    if not best:
        return
    key = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", str(best["label"] or "metric"))[:250] or "metric"
    try:
        from ML_management import mlmanagement

        mlmanagement.log_metric(key, float(best["value"]), step=step, no_wait=True)
        print(f"[board] metric: {key} = {best['value']} (step {step})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[board] WARN: log_metric: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    a = parse_args()
    bb = a.bb.resolve()
    analysis = a.analysis.resolve()
    analysis.mkdir(parents=True, exist_ok=True)
    if not bb.is_dir():
        print(f"[board] нет доски: {bb}", file=sys.stderr)
        return
    files = board_files(bb)
    printed = print_new(files, a.state or (analysis / ".board_seen.json")) if a.print_new else 0
    rows, label, direction = write_metrics(bb, files, analysis / "metrics.csv", a.competition)
    build_graph(bb, analysis, a.competition)
    print(
        f"[board] сообщений {len(files)}, новых {printed}, метрик {len(rows)}"
        + (f", метрика задачи: {label} ({direction or '?'})" if label else ""),
        flush=True,
    )
    if a.upload:
        upload_analysis(analysis)
    if a.log_metric:
        log_best(rows, direction, a.cycle)


if __name__ == "__main__":
    main()
