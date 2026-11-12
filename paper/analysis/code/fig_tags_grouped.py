#!/usr/bin/env python3
"""Grouped tag-citation chart: peer and same-author rates side by side."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paper").is_dir())
DEFAULT_DATA = ROOT / "paper" / "analysis" / "data" / "combined" / "tag_citations.csv"
DEFAULT_OUT = ROOT / "paper" / "analysis" / "figures" / "fig_swarm_tags_grouped.png"

C_PEER = "#2563eb"
C_SAME = "#f59e0b"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-messages", type=int, default=10)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.data, encoding="utf-8"))
            if int(r["messages"]) >= a.min_messages]
    rows.sort(key=lambda r: (-float(r["cited_peer_rate"]), r["tag"]))

    height = 5.0 + (len(rows) - 12) * 0.27
    fig, ax = plt.subplots(figsize=(7.6, height))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.08, top=0.93)

    y = np.arange(len(rows))
    bar_h = 0.38
    peer = np.array([float(r["cited_peer_rate"]) for r in rows])
    same = np.array([float(r["cited_same_rate"]) for r in rows])

    ax.barh(y - bar_h / 2, peer, height=bar_h, color=C_PEER,
            label="cited by another agent")
    ax.barh(y + bar_h / 2, same, height=bar_h, color=C_SAME,
            label="cited by the same author")

    for yi, v in zip(y - bar_h / 2, peer):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6.3,
                color="#111827")
    for yi, v in zip(y + bar_h / 2, same):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6.3,
                color="#111827")

    ax.set_yticks(y, [r["tag"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("citation probability")
    ax.set_title("Citation probability by hashtag: another agent vs. same author",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    fig.savefig(a.out, bbox_inches="tight")
    plt.close(fig)
    print("written", a.out)


if __name__ == "__main__":
    main()
