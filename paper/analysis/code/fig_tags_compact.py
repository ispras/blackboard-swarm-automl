#!/usr/bin/env python3
"""Compact two-panel tag-citation figure for the paper (fig_swarm_tags.png).

Panel A is ordered by peer-citation rate; panel B is ordered by its own
same-author citation rate.
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

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paper").is_dir())
DATA = ROOT / "paper" / "analysis" / "data" / "combined" / "tag_citations.csv"
OUT = ROOT / "paper" / "tex" / "fig_swarm_tags.png"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.data, encoding="utf-8")))
    rows = [r for r in rows if int(r["messages"]) >= 10]
    rows.sort(key=lambda r: -float(r["cited_peer_rate"]))
    top = rows

    panels = [
        ("A. Cited by another agent", "cited_peer_rate", "#2563eb"),
        ("B. Cited by the same author", "cited_same_rate", "#f59e0b"),
    ]
    height = 4.08 + (len(top) - 12) * 0.25
    fig, axes = plt.subplots(1, 2, figsize=(7.98, height))
    fig.subplots_adjust(left=0.135, right=0.985,
                        bottom=0.534 / height, top=1 - 0.302 / height, wspace=0.55)
    for ax, (title, key, color) in zip(axes, panels):
        order = sorted(top, key=lambda r: -float(r[key]))
        vals = [float(r[key]) for r in order]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.8, color=color)
        for yi, r in zip(y, order):
            v = float(r[key])
            ax.text(v + 0.025, yi, f"{v:.2f}  n={r['messages']}",
                    va="center", fontsize=6)
        ax.axvline(statistics.median(vals), color="#6b7280", ls=":", lw=0.8)
        ax.set_xlim(0, 1.28)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(y, [r["tag"] for r in order])
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("citation probability")
        ax.set_title(title)
    fig.savefig(a.out, bbox_inches="tight")
    plt.close(fig)
    print("written", a.out)


if __name__ == "__main__":
    main()
