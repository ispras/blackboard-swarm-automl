#!/usr/bin/env python3
"""Average per-task metrics across several runs into one combined set."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def write(p, rows):
    if not rows:
        return
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def avg_rows(rows, id_keys=("task", "short")):
    out = {}
    for r in rows:
        key = tuple(r[k] for k in id_keys)
        out.setdefault(key, []).append(r)
    res = []
    for key, rs in out.items():
        base = {k: key[i] for i, k in enumerate(id_keys)}
        for k in rs[0]:
            if k in id_keys:
                continue
            vals = []
            for r in rs:
                try:
                    vals.append(float(r[k]))
                except (ValueError, TypeError):
                    pass
            base[k] = statistics.mean(vals) if vals else rs[0][k]
        res.append(base)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("paper/analysis/data"))
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        name, path = spec.split("=", 1)
        runs.append((name, Path(path), a.data / name))
    out = a.data / "combined"; out.mkdir(parents=True, exist_ok=True)

    write(out / "coordination.csv", avg_rows([x for _, _, d in runs for x in load(d / "coordination.csv")]))
    write(out / "citation_lags.csv",
          [dict(x, run=name) for name, _, d in runs for x in load(d / "citation_lags.csv")])
    write(out / "improvement.csv", avg_rows([x for _, _, d in runs for x in load(d / "improvement.csv")]))
    write(out / "improvement_by_run.csv",
          [dict(x, run=name) for name, _, d in runs for x in load(d / "improvement.csv")])
    write(out / "internet_tasks.csv", avg_rows([x for _, _, d in runs for x in load(d / "internet_tasks.csv")]))

    tag = defaultdict(lambda: {"messages": 0, "cited_peer_k": 0, "cited_same_k": 0})
    # Recompute pooled rates over both runs.
    for _, _, d in runs:
        for x in load(d / "tag_citations.csv"):
            n = int(x["messages"]); t = x["tag"]
            tag[t]["messages"] += n
            tag[t]["cited_peer_k"] += round(float(x["cited_peer_rate"]) * n)
            if "cited_same_rate" in x:
                tag[t]["cited_same_k"] += round(float(x["cited_same_rate"]) * n)
    tag_rows = []
    for t, v in tag.items():
        n = v["messages"]
        tag_rows.append({"tag": t, "messages": n, "cited_peer_k": v["cited_peer_k"],
                         "cited_peer_rate": v["cited_peer_k"] / n if n else 0,
                         "cited_same_rate": v["cited_same_k"] / n if n else 0})
    write(out / "tag_citations.csv", tag_rows)

    # Roles: keep the run so the plot can average normalized trajectories.
    roles = []
    for name, _, d in runs:
        for x in load(d / "roles.csv"):
            x = dict(x); x["run"] = name; roles.append(x)
    write(out / "roles.csv", roles)

    for fn in ("reading_metrics.csv", "log_composition.csv", "context_usage_by_task.csv",
               "forgetting_read_depth.csv"):
        rows = []
        for _, root, _ in runs:
            rows += load(root / "analysis" / fn)
        write(out / fn, avg_rows(rows))


if __name__ == "__main__":
    main()
