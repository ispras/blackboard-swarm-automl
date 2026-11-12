#!/usr/bin/env python3
"""Figures under the names used by paper/tex/access.tex.

Reuses the panels from paper_figures_combined.py and writes:
  fig_swarm_coordination.png  (A board size, B chain length, C chain duration)
  fig_swarm_tags.png          (via fig_tags_compact)
  fig_swarm_roles.png         (roles over time)
  fig_swarm_improvement.png   (A improvements, B best time per run)
  fig_swarm_context.png       (A context composition, B context at end)
  fig_swarm_board.png         (A coverage, B reads per hour)
  fig_swarm_read_depth.png    (depth with Median row)
  fig_swarm_board_depth.png   (3 panels: coverage / reads per hour / depth,
                               Median row of C on top)
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_figures_combined as pfc

C = pfc.C


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def fig_board_depth(rd, depth, out):
    rows = sorted(rd, key=lambda x: x["short"])
    ys = np.arange(len(rows))
    cov = [float(x["content_coverage"]) for x in rows]
    cov2 = [float(x["coverage_2plus"]) for x in rows]
    rph = [float(x["reads_per_hour"]) for x in rows]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.35, 0.95, 1.35]})
    ax = axes[0]
    ax.barh(ys - 0.18, cov, height=0.34, color=C[2], label="at least one agent")
    ax.barh(ys + 0.18, cov2, height=0.34, color=C[0], label="at least two agents")
    ax.axvline(statistics.median(cov), color=C[2], ls=":", lw=0.9)
    ax.axvline(statistics.median(cov2), color=C[0], ls=":", lw=0.9)
    ax.set_yticks(ys, [x["short"] for x in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("share of messages"); ax.set_title("A. Content coverage")
    ax = axes[1]
    ax.barh(ys, rph, height=0.55, color=C[2])
    ax.axvline(statistics.median(rph), color="#111827", ls=":", lw=0.9)
    ax.set_yticks(ys, ["" for _ in rows]); ax.invert_yaxis()
    ax.set_xlabel("board reads per agent-hour"); ax.set_title("B. Board reads per hour")
    ax = axes[2]
    d_rows = sorted(depth, key=lambda x: x["short"])
    shallow = [float(x["shallow_share"]) for x in d_rows]
    partial = [float(x["partial_share"]) for x in d_rows]
    full = [float(x["full_share"]) for x in d_rows]
    med_s, med_p, med_f = statistics.median(shallow), statistics.median(partial), statistics.median(full)
    labels = ["Median"] + [x["short"] for x in d_rows]
    yy = np.arange(len(labels))
    ss = [med_s] + shallow; pp = [med_p] + partial; ff = [med_f] + full
    ax.barh(yy[1:], shallow, color="#ef4444", label="<=20%")
    ax.barh(yy[1:], partial, left=shallow, color="#f59e0b", label="20-80%")
    ax.barh(yy[1:], full, left=np.array(shallow) + np.array(partial), color="#15803d", label=">=80%")
    ax.barh(yy[0], med_s, color="#ef4444")
    ax.barh(yy[0], med_p, left=med_s, color="#f59e0b")
    ax.barh(yy[0], med_f, left=med_s + med_p, color="#15803d")
    for y, s, f in zip(yy, ss, ff):
        ax.text(1.02, y, f"full {100*f:.0f}% / <=20% {100*s:.0f}%", va="center",
                fontsize=5.5, fontweight="bold" if y == yy[0] else "normal")
    ax.axhline(0.5, color="#6b7280", ls="--", lw=0.6)
    ax.set_yticks(yy, labels); ax.invert_yaxis()
    for tick in ax.get_yticklabels():
        if tick.get_text() == "Median":
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("share of deep read events"); ax.set_title("C. Read depth of board messages")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[2]),
               plt.Rectangle((0, 0), 1, 1, color=C[0]),
               plt.Rectangle((0, 0), 1, 1, color="#ef4444"),
               plt.Rectangle((0, 0), 1, 1, color="#f59e0b"),
               plt.Rectangle((0, 0), 1, 1, color="#15803d")]
    fig.legend(handles, ["at least one agent", "at least two agents",
                         "<=20%", "20-80%", ">=80%"],
               loc="lower center", ncol=5, frameon=False, fontsize=6)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    pfc.save(fig, out / "fig_swarm_board_depth.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("paper/analysis/data/combined"))
    ap.add_argument("--out", type=Path, default=Path("paper/tex"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = a.data
    pfc.fig_coordination(load(data / "coordination.csv"), a.out)
    pfc.fig_roles(load(data / "roles.csv"), a.out)
    pfc.fig_improvement(load(data / "improvement.csv"), load(data / "improvement_by_run.csv"), a.out)
    pfc.fig_context(load(data / "log_composition.csv"), load(data / "context_usage_by_task.csv"), a.out)
    pfc.fig_board(load(data / "reading_metrics.csv"), a.out)
    pfc.fig_depth(load(data / "forgetting_read_depth.csv"), a.out)
    fig_board_depth(load(data / "reading_metrics.csv"), load(data / "forgetting_read_depth.csv"), a.out)
    for src, dst in [("fig01_coordination.png", "fig_swarm_coordination.png"),
                     ("fig03_roles.png", "fig_swarm_roles.png"),
                     ("fig04_improvement.png", "fig_swarm_improvement.png"),
                     ("fig05_context.png", "fig_swarm_context.png"),
                     ("fig06_board.png", "fig_swarm_board.png"),
                     ("fig07_read_depth.png", "fig_swarm_read_depth.png")]:
        (a.out / src).replace(a.out / dst)
    print("tex figures ->", a.out)


if __name__ == "__main__":
    main()
