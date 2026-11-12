#!/usr/bin/env python3
"""Per-task plots from combined data (average of two runs), 300 dpi."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2563eb", "#0ea5e9", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#d1d5db"]
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def med_iqr(rows, key):
    vals = [float(r[key]) for r in rows if r[key] not in ("", None, "None")]
    return (statistics.median(vals), float(np.percentile(vals, 25)), float(np.percentile(vals, 75))) if vals else (None, None, None)


def save(fig, p):
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_coordination(co, out):
    rs = sorted(co, key=lambda x: x["short"]); ys = np.arange(len(rs))
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    msgs = [float(x["messages"]) for x in rs]; edges = [float(x["edges"]) for x in rs]
    ax.barh(ys-0.18, msgs, 0.36, color=C[0], label="messages")
    ax.barh(ys+0.18, edges, 0.36, color=C[1], label="cite edges")
    for vals, col in ((msgs, C[0]), (edges, C[1])):
        ax.axvline(statistics.median(vals), color=col, ls=":", lw=0.9)
    for y, x in zip(ys, rs):
        ax.text(max(float(x["messages"]), float(x["edges"]))+4, y, f"{float(x['messages_per_hour']):.1f}/h",
                va="center", fontsize=5.5)
    ax.set_yticks(ys, [x["short"] for x in rs]); ax.invert_yaxis(); ax.set_xlabel("count")
    ax.set_title("A. Size of the board (averaged runs)"); ax.legend(frameon=False, fontsize=6)
    ax = fig.add_subplot(gs[1, 0])
    vals = [float(x["depth_median"]) for x in rs]
    q1 = [float(x["depth_q1"]) for x in rs]; q3 = [float(x["depth_q3"]) for x in rs]
    ax.barh(ys, vals, xerr=[np.array(vals)-np.array(q1), np.array(q3)-np.array(vals)],
            color=C[5], error_kw={"lw": 0.6, "capsize": 1.5})
    ax.axvline(statistics.median(vals), color="#374151", ls=":", lw=0.8)
    ax.set_yticks(ys, [x["short"] for x in rs]); ax.invert_yaxis()
    ax.set_xlabel("depth, edges (median, IQR)"); ax.set_title("B. Chain length (citation edges)")
    ax = fig.add_subplot(gs[1, 1])
    vals = [float(x["mean_chain_span_h"]) for x in rs]
    q1 = [float(x["chain_span_q1"]) for x in rs]; q3 = [float(x["chain_span_q3"]) for x in rs]
    ax.barh(ys, vals, xerr=[np.array(vals)-np.array(q1), np.array(q3)-np.array(vals)],
            color=C[3], error_kw={"lw": 0.6, "capsize": 1.5})
    ax.axvline(statistics.median(vals), color="#374151", ls=":", lw=0.8)
    ax.set_yticks(ys, ["" for _ in rs]); ax.set_xlabel("mean time first->last, hours (IQR)")
    ax.set_title("C. Mean chain duration")
    fig.tight_layout(); save(fig, out/"fig01_coordination.png")


def fig_tags(tag, out):
    rows = [x for x in tag if int(x["messages"]) >= 10]
    for key, fn, lab in [("cited_peer_rate", "fig02a_tag_citation_peer.png", "peer"),
                         ("cited_same_rate", "fig02b_tag_citation_same.png", "same agent")]:
        rws = sorted(rows, key=lambda x: float(x[key])); ys = np.arange(len(rws))
        fig, ax = plt.subplots(figsize=(6.8, 0.31*len(rws)+1.3))
        vals = [float(x[key]) for x in rws]
        ax.barh(ys, vals, color=C[0] if key == "cited_peer_rate" else C[3])
        for y, x, v in zip(ys, rws, vals):
            ax.text(min(1.12, v+0.02), y, f"{v:.2f} (n={x['messages']})", va="center", fontsize=5.5)
        ax.axvline(statistics.median(vals), color="#111827", ls=":", lw=0.8)
        ax.set_yticks(ys, [x["tag"] for x in rws]); ax.set_xlim(0, 1.35)
        ax.set_xlabel(f"P(message cited by {lab} | hashtag)")
        ax.set_title(f"Citation probability by hashtag: {lab} (averaged runs)")
        fig.tight_layout(); save(fig, out/fn)


def role_time(roles):
    names = ["explorer", "producer", "integrator", "validator", "coordinator", "communicator"]
    bins, bin_h = 8, 1.5
    acc = np.zeros((bins, len(names))); valid = np.zeros(bins)
    groups = defaultdict(list)
    for r in roles:
        groups[(r.get("run", ""), r["task"])].append(r)
    for rr in groups.values():
        t0 = min(datetime.fromisoformat(x["time"]) for x in rr)
        mat = np.zeros((bins, len(names)))
        for x in rr:
            h = (datetime.fromisoformat(x["time"])-t0).total_seconds()/3600
            b = min(int(h/bin_h), bins-1); mat[b, names.index(x["role"])] += 1
        rs = mat.sum(axis=1, keepdims=True); valid += (rs[:, 0] > 0)
        acc += np.divide(mat, np.maximum(rs, 1))
    return acc/np.maximum(valid[:, None], 1)


def fig_citation_lag(lags, out):
    xs = [float(x["citing_h"]) for x in lags]
    ys = [float(x["lag_h"]) for x in lags]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    hb = ax.hexbin(xs, ys, gridsize=35, yscale="log", mincnt=1, bins="log", cmap="viridis")
    med = statistics.median(ys); p95 = float(np.percentile(ys, 95))
    ax.axhline(med, color="#ef4444", ls=":", lw=0.9, label=f"median {med:.2f} h")
    ax.axhline(p95, color="#f59e0b", ls=":", lw=0.9, label=f"p95 {p95:.2f} h")
    ax.set_xlabel("time of citing message, h"); ax.set_ylabel("citation lag, h (log scale)")
    ax.set_title("A. Citation time versus lag"); ax.set_xlim(0, 12); ax.set_ylim(0.01, 12)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    cbar = fig.colorbar(hb, ax=ax, pad=0.01); cbar.set_label("citation edges", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax = axes[1]
    edges = np.arange(0, 13, 2); centers = (edges[:-1] + edges[1:]) / 2
    p3 = []; p6 = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [y for x, y in zip(xs, ys) if lo <= x < hi]
        p3.append(100 * sum(v > 3 for v in vals) / len(vals) if vals else 0)
        p6.append(100 * sum(v > 6 for v in vals) / len(vals) if vals else 0)
    ax.plot(centers, p3, marker="o", color=C[0], label="lag > 3 h")
    ax.plot(centers, p6, marker="o", color=C[3], label="lag > 6 h")
    ax.set_xticks(centers, [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])])
    ax.set_xlabel("time of citing message, h"); ax.set_ylabel("% of citations in bin")
    ax.set_title("B. Long-range citations by time"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, out/"fig02_citation_lag.png")


def fig_roles(roles, out):
    names = ["explorer", "producer", "integrator", "validator", "coordinator", "communicator"]
    stack = role_time(roles); x = np.arange(8)*1.5+0.75; bottom = np.zeros(8)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, (name, col) in enumerate(zip(names, C[:6])):
        ax.bar(x, stack[:, i], bottom=bottom, width=1.35, color=col, label=name)
        bottom += stack[:, i]
    ax.set_xticks(np.arange(0, 12.1, 1.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 1)
    ax.set_xlabel("hours since start"); ax.set_ylabel("share of messages")
    ax.set_title("LLM-assigned roles over time (averaged runs)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout(); save(fig, out/"fig03_roles.png")


def fig_improvement(imp, imp_run, out):
    rows = [x for x in imp if x["new_best_share"] not in ("", None)]
    rows.sort(key=lambda x: x["short"]); new = [float(x["new_best_share"]) for x in rows]
    beat = [float(x["improve_cited_share"]) if x["improve_cited_share"] else float("nan") for x in rows]
    labels = [x["short"] for x in rows]; ys = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax = axes[0]
    ax.barh(ys-0.18, new, 0.36, color=C[0], label="new board best")
    ax.barh(ys+0.18, beat, 0.36, color=C[2], label="beats cited best")
    for y, x in enumerate(rows):
        if math.isnan(beat[y]): ax.text(0.02, y+0.18, "n/a", va="center", fontsize=5.5)
    ax.axvline(statistics.median(new), color=C[0], ls=":", lw=1)
    ax.axvline(statistics.median([v for v in beat if not math.isnan(v)]), color=C[2], ls=":", lw=1)
    ax.set_yticks(ys, labels); ax.set_xlim(0, 1)
    ax.set_xlabel("share of metric-bearing messages"); ax.set_title("A. Messages that improve the metric")
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    by_task = defaultdict(lambda: defaultdict(list)); run_names = sorted({x["run"] for x in imp_run})
    for x in imp_run:
        v = x.get("best_time_h")
        if v not in ("", None, "None"):
            by_task[x["short"]][x["run"]].append(float(v))
    complete = [t for t in by_task if all(name in by_task[t] for name in run_names)]
    complete.sort(key=lambda t: statistics.mean(v for name in run_names for v in by_task[t][name]))
    yy = np.arange(len(complete)); offsets = np.linspace(-0.14, 0.14, len(run_names))
    palette = {name: [C[0], C[3], C[5], C[2]][i % 4] for i, name in enumerate(run_names)}
    for name, off in zip(run_names, offsets):
        vals = [statistics.mean(by_task[t][name]) for t in complete]
        ax.scatter(vals, yy+off, color=palette[name], s=20, edgecolors="white", linewidths=0.4,
                   label=name, zorder=3)
    all_vals = [statistics.mean(by_task[t][name]) for t in complete for name in run_names]
    ax.axvline(statistics.median(all_vals), color="#111827", ls=":", lw=1,
               label=f"median {statistics.median(all_vals):.1f} h")
    ax.set_yticks(yy, complete); ax.invert_yaxis()
    ax.set_xlabel("time to best internal result, h"); ax.set_title("B. When best result is reached")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(); save(fig, out/"fig04_improvement.png")


def fig_context(comp, tok, out):
    parts = [("action_chars_share", "agent actions"), ("docs_share", "board"),
             ("code_share", "code"), ("artifacts_share", "data/artifacts"),
             ("peer_log_share", "peer logs"), ("execution_share", "execution"),
             ("environment_share", "environment"), ("other_share", "other")]
    tasks = sorted(comp, key=lambda x: x["short"]); rows = []
    for r in tasks:
        rows.append((r["short"], [float(r[k]) for k, _ in parts]))
    vals = np.array([r[1] for r in rows])
    med = np.median(vals, axis=0); med = med / med.sum()
    rows.append(("Median", list(med)))
    labels = [r[0] for r in rows]; ys = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]; left = np.zeros(len(rows))
    for j, ((k, lab), col) in enumerate(zip(parts, C)):
        v = np.array([r[1][j] for r in rows]); ax.barh(ys, v, left=left, color=col, label=lab); left += v
    ax.set_yticks(ys, labels); ax.set_xlim(0, 1); ax.axhline(len(tasks)-0.5, color="#6b7280", ls="--", lw=0.6)
    ax.set_xlabel("median share of visible transcript"); ax.set_title("A. Context composition")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=5.5)
    ax = axes[1]
    tok_rows = sorted(tok, key=lambda x: x["short"])
    vals = [float(x["end_context_mean"])/1000 for x in tok_rows]
    ax.barh(np.arange(len(tok_rows)), vals, color=C[0])
    for y, v in zip(np.arange(len(tok_rows)), vals): ax.text(v+3, y, f"{v:.0f}k", va="center", fontsize=5.5)
    ax.axvline(statistics.median(vals), color="#111827", ls=":", lw=0.8)
    ax.set_yticks(np.arange(len(tok_rows)), [x["short"] for x in tok_rows]); ax.set_xlabel("k tokens")
    ax.set_title("B. Context at session end")
    fig.tight_layout(); save(fig, out/"fig05_context.png")


def fig_board(rd, out):
    rows = sorted(rd, key=lambda x: x["short"]); ys = np.arange(len(rows))
    cov = [float(x["content_coverage"]) for x in rows]
    cov2 = [float(x["coverage_2plus"]) for x in rows]
    rph = [float(x["reads_per_hour"]) for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), gridspec_kw={"width_ratios": [1.35, 1]})
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
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[2]), plt.Rectangle((0, 0), 1, 1, color=C[0])]
    fig.legend(handles, ["at least one agent", "at least two agents"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.42, -0.01), fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1]); save(fig, out/"fig06_board.png")


def fig_depth(depth, out):
    rows = sorted(depth, key=lambda x: x["short"])
    shallow = [float(x["shallow_share"]) for x in rows]
    partial = [float(x["partial_share"]) for x in rows]
    full = [float(x["full_share"]) for x in rows]
    med_s, med_p, med_f = statistics.median(shallow), statistics.median(partial), statistics.median(full)
    labels = [x["short"] for x in rows] + ["Median"]
    ys = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(ys[:-1], shallow, color="#ef4444", label="<=20%")
    ax.barh(ys[:-1], partial, left=shallow, color="#f59e0b", label="20-80%")
    ax.barh(ys[:-1], full, left=np.array(shallow) + np.array(partial), color="#22c55e", label=">=80%")
    ax.barh(ys[-1], med_s, color="#ef4444")
    ax.barh(ys[-1], med_p, left=med_s, color="#f59e0b")
    ax.barh(ys[-1], med_f, left=med_s + med_p, color="#22c55e")
    for y, s, f in zip(ys[:-1], shallow, full):
        ax.text(1.02, y, f"full {100*f:.0f}% / <=20% {100*s:.0f}%", va="center", fontsize=5.5)
    ax.text(1.02, ys[-1], f"full {100*med_f:.0f}% / <=20% {100*med_s:.0f}%", va="center",
            fontsize=5.5, fontweight="bold")
    ax.axhline(len(rows) - 0.5, color="#6b7280", ls="--", lw=0.6)
    ax.set_yticks(ys, labels)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "Median":
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1.3); ax.set_ylim(-0.6, len(rows) + 0.6)
    ax.set_xlabel("share of deep read events"); ax.set_title("Read depth of board messages (averaged runs)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.tight_layout(); save(fig, out/"fig07_read_depth.png")


def fig_internet(it, out):
    rows = sorted(it, key=lambda x: x["short"]); ys = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    axes[0].barh(ys, [float(x["weight_gb"]) for x in rows], color=C[3])
    axes[0].set_yticks(ys, [x["short"] for x in rows]); axes[0].set_xlabel("weights, GB")
    axes[0].set_title("A. Pretrained weights")
    axes[1].barh(ys, [float(x["webfetch_calls"]) for x in rows], color=C[0])
    axes[1].set_yticks(ys, ["" for _ in rows]); axes[1].set_xlabel("WebFetch calls")
    axes[1].set_title("B. Web access")
    ax = axes[2]
    xs = [float(x["weight_gb"]) for x in rows]; lad = [float(x["ladder"]) for x in rows]
    ax.scatter(xs, lad, color=C[2], s=18)
    for x, y, r in zip(xs, lad, rows): ax.annotate(r["short"], (x, y), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("weights, GB"); ax.set_ylabel("medal ladder"); ax.set_title("C. Internet vs outcome")
    fig.tight_layout(); save(fig, out/"fig08_internet.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, default=Path("paper/analysis/data/combined")); ap.add_argument("--out", type=Path, default=Path("paper/analysis/figures"))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    fig_coordination(load(a.data/"coordination.csv"), a.out)
    fig_citation_lag(load(a.data/"citation_lags.csv"), a.out)
    fig_tags(load(a.data/"tag_citations.csv"), a.out)
    fig_roles(load(a.data/"roles.csv"), a.out)
    fig_improvement(load(a.data/"improvement.csv"), load(a.data/"improvement_by_run.csv"), a.out)
    fig_context(load(a.data/"log_composition.csv"), load(a.data/"context_usage_by_task.csv"), a.out)
    fig_board(load(a.data/"reading_metrics.csv"), a.out)
    fig_depth(load(a.data/"forgetting_read_depth.csv"), a.out)
    fig_internet(load(a.data/"internet_tasks.csv"), a.out)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
