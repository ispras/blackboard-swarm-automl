#!/usr/bin/env python3
"""Comparison figure: original hashtags vs LLM-assigned prompt tags."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "comparison_hashtags_vs_llm.csv").open(encoding="utf-8")))
data = {row["metric"]: row for row in rows}


def value(metric: str, column: str) -> float:
    return float(data[metric][column])


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)

agents = ["rigel", "deneb", "arcturus"]
display = ["rigel_xi", "deneb_upsilon", "arcturus_alpha"]
colors = ["#e11d48", "#2563eb", "#059669"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.30)

ax = axes[0]
positions = np.arange(len(agents))
width = 0.34
original = [value(f"SI_{agent}", "hashtags_original") for agent in agents]
llm = [value(f"SI_{agent}", "llm_tags") for agent in agents]
ax.bar(positions - width / 2, original, width, color="#cbd5e1", edgecolor="white", label="original hashtags")
ax.bar(positions + width / 2, llm, width, color=[colors[i] for i in range(len(agents))], edgecolor="white", label="LLM tags")
for i in range(len(agents)):
    ax.text(i - width / 2, original[i] + 0.008, f"{original[i]:.3f}", ha="center", fontsize=6.5)
    ax.text(i + width / 2, llm[i] + 0.008, f"{llm[i]:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(positions)
ax.set_xticklabels(display, rotation=15, ha="right")
ax.set_ylim(0, 0.30)
ax.set_ylabel("Specialization Index")
ax.set_title("A. Absolute specialization is unchanged", fontsize=9, fontweight="bold", loc="left")
ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
pairs = [("JSD_rigel_deneb", "rigel–deneb"), ("JSD_rigel_arcturus", "rigel–arcturus"), ("JSD_deneb_arcturus", "deneb–arcturus"), ("mean_JSD_bits", "mean")]
positions = np.arange(len(pairs))
original = [value(key, "hashtags_original") for key, _ in pairs]
llm = [value(key, "llm_tags") for key, _ in pairs]
ax.bar(positions - width / 2, original, width, color="#cbd5e1", edgecolor="white", label="original hashtags")
ax.bar(positions + width / 2, llm, width, color="#7c3aed", edgecolor="white", label="LLM tags")
for i in range(len(pairs)):
    ax.text(i - width / 2, original[i] + 0.006, f"{original[i]:.3f}", ha="center", fontsize=6.5)
    ax.text(i + width / 2, llm[i] + 0.006, f"{llm[i]:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(positions)
ax.set_xticklabels([label for _, label in pairs], rotation=15, ha="right")
ax.set_ylim(0, 0.24)
ax.set_ylabel("Jensen–Shannon divergence (bits)")
ax.set_title("B. Divergence improves slightly, still non-significant", fontsize=9, fontweight="bold", loc="left")
ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(
    0.98,
    0.94,
    f"mean JSD p: {value('mean_JSD_p', 'hashtags_original'):.3f} → {value('mean_JSD_p', 'llm_tags'):.3f}\n"
    f"rigel–arcturus p: {value('JSD_rigel_arcturus_p', 'hashtags_original'):.3f} → {value('JSD_rigel_arcturus_p', 'llm_tags'):.3f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=7,
)
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#cbd5e1", edgecolor="white", label="original hashtags"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#7c3aed", edgecolor="white", label="LLM tags"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
fig.suptitle("Hashtags vs LLM-assigned prompt tags", fontsize=11, fontweight="bold", y=0.98)
fig.savefig(HERE / "figures/fig09_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(HERE / "figures/fig09_comparison.pdf", bbox_inches="tight", facecolor="white")
print(HERE / "figures/fig09_comparison.png")
