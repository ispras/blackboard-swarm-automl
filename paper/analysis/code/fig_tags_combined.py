#!/usr/bin/env python3
"""Single-panel tag-citation chart combining peer and same-author citations.

Peer and same-author citations overlap: one message can be cited by both its
author and another agent. The bar is therefore split into disjoint categories
(peer only, both, same-author only), and its length is the probability that a
message is cited at least once.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_metrics import load_task

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paper").is_dir())
DEFAULT_RUNS = [
    ROOT / "paper" / "analysis" / "runs" / name
    for name in (
        "1-lite10-blackboard-mlebench",
        "2-lite10-blackboard-mlebench",
        "3-lite10-blackboard-mlebench",
    )
]
DEFAULT_OUT = ROOT / "paper" / "analysis" / "figures" / "fig_swarm_tags_combined.png"
DEFAULT_CSV = ROOT / "paper" / "analysis" / "data" / "combined" / "tag_citation_composition.csv"

C_PEER = "#2563eb"
C_BOTH = "#7c3aed"
C_SAME = "#f59e0b"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})


def collect(runs: list[Path], min_messages: int) -> list[dict]:
    counts = defaultdict(lambda: {
        "messages": 0, "peer_only": 0, "both": 0, "same_only": 0, "neither": 0,
    })
    for run in runs:
        for d in sorted(run.iterdir()):
            if not (d.is_dir() and (d / "blackboard").exists()):
                continue
            task = load_task(d)
            by = {m["name"]: m for m in task["msgs"]}
            inc_peer = Counter()
            inc_same = Counter()
            for citing, cited in task["edges"]:
                if citing not in by or cited not in by:
                    continue
                if by[citing]["agent"] != by[cited]["agent"]:
                    inc_peer[cited] += 1
                else:
                    inc_same[cited] += 1
            for msg in task["msgs"]:
                peer = inc_peer[msg["name"]] > 0
                same = inc_same[msg["name"]] > 0
                for tag in set(msg["tags"]):
                    c = counts[tag]
                    c["messages"] += 1
                    if peer and same:
                        c["both"] += 1
                    elif peer:
                        c["peer_only"] += 1
                    elif same:
                        c["same_only"] += 1
                    else:
                        c["neither"] += 1
    rows = []
    for tag, c in counts.items():
        n = c["messages"]
        if n < min_messages:
            continue
        rows.append({
            "tag": tag,
            "messages": n,
            "cited_any_rate": (c["peer_only"] + c["both"] + c["same_only"]) / n,
            "cited_peer_rate": (c["peer_only"] + c["both"]) / n,
            "cited_same_rate": (c["both"] + c["same_only"]) / n,
            "both_rate": c["both"] / n,
            "peer_only_rate": c["peer_only"] / n,
            "same_only_rate": c["same_only"] / n,
            "neither_rate": c["neither"] / n,
        })
    rows.sort(key=lambda r: (-r["cited_any_rate"], -r["cited_peer_rate"], r["tag"]))
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("written", out)


def plot(rows: list[dict], out: Path) -> None:
    height = 4.3 + (len(rows) - 12) * 0.23
    fig, ax = plt.subplots(figsize=(7.6, height))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.09, top=0.86)

    y = np.arange(len(rows))
    peer_only = np.array([r["peer_only_rate"] for r in rows])
    both = np.array([r["both_rate"] for r in rows])
    same_only = np.array([r["same_only_rate"] for r in rows])
    total = peer_only + both + same_only

    ax.barh(y, peer_only, height=0.78, color=C_PEER, label="another agent only")
    ax.barh(y, both, left=peer_only, height=0.78, color=C_BOTH, label="both")
    ax.barh(y, same_only, left=peer_only + both, height=0.78,
            color=C_SAME, label="same author only")

    for yi, v in zip(y, total):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6.5,
                color="#111827")

    med = statistics.median(total)
    ax.axvline(med, color="#111827", ls=":", lw=1)
    ax.text(med + 0.008, len(rows) - 0.45, f"median {med:.2f}",
            fontsize=6.5, color="#111827", va="bottom")

    ax.set_yticks(y, [r["tag"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("probability a message is cited at least once")
    ax.set_title("Citation probability by hashtag: peer and same-author citations combined",
                 fontsize=10, pad=24)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              frameon=False, fontsize=7.5, borderaxespad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("written", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--min-messages", type=int, default=10)
    a = ap.parse_args()
    rows = collect(a.runs, a.min_messages)
    write_csv(rows, a.csv)
    plot(rows, a.out)


if __name__ == "__main__":
    main()
