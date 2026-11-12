#!/usr/bin/env python3
"""Compare four tag representations and show role shares."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)

representations = [
    "Hashtags\n(22 base)",
    "LLM base\n(22)",
    "LLM rollup\n(13 used)",
    "LLM specific\n(27 used)",
    "Roles\n(7)",
]
mean_jsd = [0.0922, 0.1345, 0.1479, 0.2603, 0.3028]
p_values = [0.1186, 0.0980, 0.0013, 0.0007, 0.0020]
colors = ["#cbd5e1", "#94a3b8", "#60a5fa", "#7c3aed", "#111827"]

fig = plt.figure(figsize=(11.5, 4.6))
grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)
fig.subplots_adjust(top=0.80, bottom=0.20)

ax = fig.add_subplot(grid[0, 0])
positions = np.arange(len(representations))
ax.bar(positions, mean_jsd, color=colors, edgecolor="white", width=0.66)
for i, (value, pvalue) in enumerate(zip(mean_jsd, p_values)):
    ax.text(i, value + 0.008, f"{value:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax.text(i, 0.012, f"p={pvalue:.4f}", ha="center", fontsize=6.2, color="white" if i >= 2 else "#111827")
ax.set_xticks(positions)
ax.set_xticklabels(representations, fontsize=6.8)
ax.set_ylim(0, 0.36)
ax.set_ylabel("mean pairwise JSD (bits)")
ax.set_title("A. Vocabulary expansion recovers behavioral divergence", fontsize=9, fontweight="bold", loc="left")
ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = fig.add_subplot(grid[0, 1])
roles = ["role_integrator", "role_base_model_producer", "role_training_technique_specialist", "role_validation_owner", "role_coordinator", "role_analyst", "role_releaser"]
role_labels = ["Integrator", "Base-model producer", "Training-tech specialist", "Validation owner", "Coordinator", "Analyst", "Releaser"]
role_colors = ["#7c3aed", "#2563eb", "#ea580c", "#0891b2", "#64748b", "#059669", "#111827"]
counts = {
    "rigel_xi": [19, 3, 1, 0, 5, 0, 2],
    "deneb_upsilon": [12, 9, 3, 1, 4, 1, 2],
    "arcturus_alpha": [0, 7, 1, 0, 1, 1, 1],
}
agents = list(counts)
totals = {agent: sum(counts[agent]) for agent in agents}
left = np.zeros(len(agents))
for index, (role, label, color) in enumerate(zip(roles, role_labels, role_colors)):
    values = np.array([counts[agent][index] / totals[agent] for agent in agents])
    ax.barh(np.arange(len(agents)), values, left=left, color=color, edgecolor="white", linewidth=0.8, label=label)
    for row, value in enumerate(values):
        if value >= 0.08:
            ax.text(left[row] + value / 2, row, f"{value:.0%}", ha="center", va="center", fontsize=6.5, color="white")
    left += values
ax.set_yticks(range(len(agents)))
ax.set_yticklabels(agents, fontsize=7)
ax.set_xlim(0, 1)
ax.set_xlabel("share of messages (primary role)")
ax.set_title("B. Role structure is qualitative, not just quantitative", fontsize=9, fontweight="bold", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=6.2)
ax.grid(True, axis="x", alpha=0.18, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("How the tag representation changes the emergence measurement", fontsize=11, fontweight="bold", y=0.98)
fig.savefig(HERE / "figures/fig10_representation_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(HERE / "figures/fig10_representation_comparison.pdf", bbox_inches="tight", facecolor="white")
print(HERE / "figures/fig10_representation_comparison.png")
