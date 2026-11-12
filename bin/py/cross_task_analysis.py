#!/usr/bin/env python3
"""Cross-task aggregation of swarm analysis across all experiment tasks.

Reads per-task artifacts:
  <root>/<task>/analysis/swarm/{summary.json,synergy.csv,tta.csv,recovery.csv}
  <root>/<task>/analysis/{grade.json,COSTS.md,metrics.csv,graph.json}

Writes to --out:
  cross_task_metrics.csv, cross_task_summary.json, SUMMARY.md, figures/*.png

Run:
  python3 bin/py/cross_task_analysis.py \
      --root results/1-lie10-blackboard-mlebench \
      --out  results/1-lie10-blackboard-mlebench/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

MODALITY = {
    "billion-word-imputation": "NLP",
    "bms-molecular-translation": "Vision+Text",
    "cassava-leaf-disease-classification": "Image",
    "mlsp-2013-birds": "Audio",
    "new-york-city-taxi-fare-prediction": "Tabular",
    "nfl-player-contact-detection": "Tabular/Temporal",
    "spooky-author-identification": "Text",
    "stanford-covid-vaccine": "Bio-sequence",
    "uw-madison-gi-tract-image-segmentation": "Image seg.",
    "ventilator-pressure-prediction": "Time series",
}
SHORT = {
    "billion-word-imputation": "billion-word",
    "bms-molecular-translation": "bms-mol",
    "cassava-leaf-disease-classification": "cassava",
    "mlsp-2013-birds": "mlsp-birds",
    "new-york-city-taxi-fare-prediction": "taxi",
    "nfl-player-contact-detection": "nfl-contact",
    "spooky-author-identification": "spooky",
    "stanford-covid-vaccine": "stanford-covid",
    "uw-madison-gi-tract-image-segmentation": "uw-gi",
    "ventilator-pressure-prediction": "ventilator",
}
COLORS = {
    "gold": "#c9a227",
    "silver": "#9ca3af",
    "bronze": "#b45309",
    "above_median": "#3b82f6",
    "below_median": "#ef4444",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def medal_tier(flags: dict) -> str:
    if flags.get("gold_medal"):
        return "gold"
    if flags.get("silver_medal"):
        return "silver"
    if flags.get("bronze_medal"):
        return "bronze"
    return "above_median" if flags.get("above_median") else "below_median"


def ladder(score: float, th: dict, lower: bool) -> float:
    points = [th["median"], th["bronze"], th["silver"], th["gold"]]
    s = score
    for i in range(3):
        a, b = points[i], points[i + 1]
        inside = (a >= s >= b) if lower else (a <= s <= b)
        if inside:
            if abs(a - b) < 1e-12:
                return float(i)
            frac = (a - s) / (a - b) if lower else (s - a) / (b - a)
            return float(i + frac)
    beyond = (s < points[-1]) if lower else (s > points[-1])
    if beyond:
        band = abs(points[-1] - points[-2])
        if band < 1e-12:
            return 3.0
        extra = (points[-1] - s) / band if lower else (s - points[-1]) / band
        return min(3.0 + extra, 4.0)
    return 0.0


def gold_gap(score: float, th: dict, lower: bool) -> float:
    g = th["gold"]
    return (g - score) / abs(g) if lower else (score - g) / abs(g)


def parse_cost(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"итого:\s*\$([0-9]+(?:\.[0-9]+)?)", path.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def load_curve(task_dir: Path, lower: bool, t0: datetime, t1: datetime) -> dict:
    rows = []
    with open(task_dir / "analysis" / "metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["author"] == "system":
                continue
            try:
                rows.append((iso(r["time"]), float(r["value"]), r["author"]))
            except (ValueError, KeyError):
                continue
    if not rows:
        return {"points": [], "best": None, "first": None, "n_best": 0,
                "last_improvement_frac": None, "new_best_by_agent": {}}
    rows.sort(key=lambda x: x[0])
    span = max((t1 - t0).total_seconds(), 1.0)
    best = None
    points = []
    n_best = 0
    last_improve = None
    by_agent: dict[str, int] = {}
    for ts, val, author in rows:
        better = best is None or (
            (val < best - 1e-12) if lower else (val > best + 1e-12)
        )
        if better:
            if best is not None:
                n_best += 1
                by_agent[author] = by_agent.get(author, 0) + 1
            best = val
            last_improve = ts
        x = min(max((ts - t0).total_seconds() / span, 0.0), 1.0)
        points.append((x, best))
    last_frac = None
    if last_improve is not None:
        last_frac = min(max((last_improve - t0).total_seconds() / span, 0.0), 1.0)
    return {
        "points": points,
        "best": best,
        "first": rows[0][1],
        "n_best": n_best,
        "last_improvement_frac": last_frac,
        "new_best_by_agent": by_agent,
    }


def graph_stats(task_dir: Path) -> dict:
    p = task_dir / "analysis" / "graph.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in d.get("nodes", []) if n.get("kind") == "doc"}
    agents = {k for k, n in nodes.items() if n.get("author") not in (None, "system")}
    cite = [
        e for e in d.get("edges", [])
        if e.get("kind") == "cite" and e.get("s") in agents and e.get("t") in agents and e["s"] != e["t"]
    ]
    out_cross = {k: 0 for k in agents}
    in_cross = {k: 0 for k in agents}
    for e in cite:
        if nodes[e["s"]].get("author") != nodes[e["t"]].get("author"):
            out_cross[e["s"]] += 1
            in_cross[e["t"]] += 1
    n_msgs = len(agents)
    with_peer = sum(1 for k in agents if out_cross[k] > 0)
    cross_edges = sum(out_cross.values())
    self_cites = len(cite) - cross_edges
    self_links = sum(1 for e in cite if e["s"] == e["t"])
    return {
        "graph_messages": n_msgs,
        "cite_edges": len(cite),
        "reactivity": with_peer / n_msgs if n_msgs else None,
        "reactivity_n": n_msgs,
        "reactivity_k": with_peer,
        "cross_edge_share": cross_edges / len(cite) if cite else None,
        "cross_edges": cross_edges,
        "self_cites": self_cites,
        "self_cite_share": self_cites / len(cite) if cite else None,
        "self_links": self_links,
        "external_indegree_per_msg": cross_edges / n_msgs if n_msgs else None,
    }


def load_tag_tta(task_dir: Path) -> list[dict]:
    swarm = task_dir / "analysis" / "swarm"
    msgs = []
    with open(swarm / "messages.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "agent":
                continue
            msgs.append({
                "name": r["name"],
                "agent": r["agent"],
                "idx": len(msgs),
                "time": iso(r["time"]),
                "tags": [t for t in r["tags"].split() if t],
                "metric": r["metric_best"].strip(),
            })
    by_name = {m["name"]: m for m in msgs}
    citers: dict[str, set[str]] = {}
    with open(swarm / "edges.csv", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e["src"] in by_name and e["tgt"] in by_name:
                citers.setdefault(e["tgt"], set()).add(e["src"])
    rows = []
    for m in msgs:
        if not m["metric"]:
            continue
        for tag in sorted(set(m["tags"])):
            cand = [
                c for c in citers.get(m["name"], ())
                if by_name[c]["idx"] > m["idx"] and by_name[c]["agent"] != m["agent"]
            ]
            adopter = min(cand, key=lambda c: by_name[c]["idx"]) if cand else None
            rows.append({
                "task": task_dir.name,
                "tag": tag,
                "event": m["name"],
                "agent": m["agent"],
                "metric": m["metric"],
                "adopter": adopter or "",
                "msgs": (by_name[adopter]["idx"] - m["idx"]) if adopter else None,
                "minutes": ((by_name[adopter]["time"] - m["time"]).total_seconds() / 60)
                if adopter else None,
            })
    return rows


def load_task(task_dir: Path) -> dict | None:
    swarm = task_dir / "analysis" / "swarm"
    if not (swarm / "summary.json").exists():
        return None
    s = json.loads((swarm / "summary.json").read_text(encoding="utf-8"))
    g = json.loads((task_dir / "analysis" / "grade.json").read_text(encoding="utf-8"))
    th = g["thresholds"]
    lower = bool(g["is_lower_better"])
    name = task_dir.name

    with open(swarm / "messages.csv", encoding="utf-8") as f:
        msgs = [r for r in csv.DictReader(f) if r["kind"] == "agent"]
    times = sorted(iso(r["time"]) for r in msgs if r["time"])
    t0, t1 = times[0], times[-1]
    hours = (t1 - t0).total_seconds() / 3600

    with open(swarm / "synergy.csv", encoding="utf-8") as f:
        syn = list(csv.DictReader(f))
    linked = [float(r["delta_prev"]) for r in syn if r["has_links"] == "1" and r["delta_prev"]]
    unlinked = [float(r["delta_prev"]) for r in syn if r["has_links"] == "0" and r["delta_prev"]]

    with open(swarm / "tta.csv", encoding="utf-8") as f:
        tta = list(csv.DictReader(f))
    with open(swarm / "recovery.csv", encoding="utf-8") as f:
        rec = list(csv.DictReader(f))

    curve = load_curve(task_dir, lower, t0, t1)
    gs = graph_stats(task_dir)
    cost = parse_cost(task_dir / "analysis" / "COSTS.md")

    direct = s["synergy"]["direct"]
    board_gain = None
    if curve["first"] is not None and curve["best"] is not None:
        board_gain = abs(curve["first"] - curve["best"])
    gen_gap = None
    if curve["best"] not in (None, 0):
        gen_gap = ((curve["best"] - g["score"]) / abs(curve["best"]) if lower
                   else (g["score"] - curve["best"]) / abs(curve["best"]))
    best_owners = curve["new_best_by_agent"]
    top_producer_share = (max(best_owners.values()) / sum(best_owners.values())
                          if best_owners else None)

    row = {
        "task": name,
        "short": SHORT.get(name, name),
        "modality": MODALITY.get(name, ""),
        "metric": g["metric"],
        "lower_better": lower,
        "score": g["score"],
        "gold": th["gold"],
        "silver": th["silver"],
        "bronze": th["bronze"],
        "median_threshold": th["median"],
        "medal": medal_tier(g["flags"]),
        "above_median": bool(g["flags"].get("above_median")),
        "ladder": ladder(g["score"], th, lower),
        "gold_gap": gold_gap(g["score"], th, lower),
        "agents": len({r["agent"] for r in msgs}),
        "messages": s["messages"]["agent"],
        "system_messages": s["messages"]["system"],
        "duration_h": hours,
        "cost_usd": cost,
        "cost_per_msg": (cost / s["messages"]["agent"]) if cost and s["messages"]["agent"] else None,
        "edges": s["graph"]["edges"],
        "density": s["graph"]["edges"] / (s["graph"]["nodes"] * (s["graph"]["nodes"] - 1))
        if s["graph"]["nodes"] > 1 else None,
        "depth_max": s["graph"]["max_depth"],
        "depth_mean_all": s["graph"]["mean_depth_all"],
        "depth_mean_linked": s["graph"]["mean_depth_linked"],
        "max_indegree": s["graph"]["max_indegree"],
        "synergy_n": direct["n"],
        "synergy_mean": direct["mean"],
        "synergy_mean_rel": (direct["mean"] / board_gain)
        if direct["mean"] is not None and board_gain else None,
        "synergy_positive": direct["positive"],
        "linked_n": len(linked),
        "linked_positive": (sum(1 for x in linked if x > 1e-12) / len(linked)) if linked else None,
        "unlinked_n": len(unlinked),
        "unlinked_positive": (sum(1 for x in unlinked if x > 1e-12) / len(unlinked)) if unlinked else None,
        "linked_mean_rel": (statistics.mean(linked) / board_gain)
        if linked and board_gain else None,
        "unlinked_mean_rel": (statistics.mean(unlinked) / board_gain)
        if unlinked and board_gain else None,
        "tta_events": s["tta"]["broad"]["direct"]["events"],
        "tta_adopted": s["tta"]["broad"]["direct"]["adopted"],
        "tta_rate": s["tta"]["broad"]["direct"]["rate"],
        "tta_median_msgs": s["tta"]["broad"]["direct"]["median_msgs"],
        "tta_median_min": (s["tta"]["broad"]["direct"]["median_seconds"] or 0) / 60
        if s["tta"]["broad"]["direct"]["median_seconds"] is not None else None,
        "problems": s["recovery"]["problems"],
        "recovered": s["recovery"]["direct"]["recovered"],
        "recovery_rate": s["recovery"]["direct"]["rate"],
        "recovery_median_msgs": s["recovery"]["direct"]["median_msgs"],
        "recovery_different_author": s["recovery"]["direct"]["different_author"],
        "best_board": curve["best"],
        "first_board": curve["first"],
        "generalization_gap": gen_gap,
        "n_new_best": curve["n_best"],
        "last_improvement_frac": curve["last_improvement_frac"],
        "top_producer_share": top_producer_share,
        "agents_with_best": len(best_owners),
        "new_best_by_agent": json.dumps(best_owners, ensure_ascii=False),
        "curve_points": curve["points"],
        **gs,
    }
    return row


def describe(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0}
    xs2 = sorted(xs)
    q1 = float(np.percentile(xs2, 25))
    q3 = float(np.percentile(xs2, 75))
    return {
        "n": len(xs2),
        "mean": statistics.mean(xs2),
        "median": statistics.median(xs2),
        "q1": q1,
        "q3": q3,
        "min": min(xs2),
        "max": max(xs2),
    }


def spearman_table(rows: list[dict]) -> list[dict]:
    keys = [
        ("messages", "Messages"),
        ("duration_h", "Duration (h)"),
        ("cost_usd", "Cost"),
        ("reactivity", "Reactivity"),
        ("cross_edge_share", "Cross-agent edges"),
        ("synergy_positive", "Synergy: share improving"),
        ("tta_rate", "TTA adoption"),
        ("recovery_rate", "Recovery rate"),
        ("depth_max", "Max depth"),
        ("max_indegree", "Max in-degree"),
    ]
    out = []
    for key, label in keys:
        xs, ys = [], []
        for r in rows:
            x, y = r.get(key), r.get("ladder")
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if len(xs) >= 5 and len(set(xs)) > 1:
            rho, p = sps.spearmanr(xs, ys)
            out.append({"key": key, "label": label, "rho": float(rho), "p": float(p), "n": len(xs)})
    return out


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_outcome(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: r["ladder"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ys = np.arange(len(rs))
    vals = [min(r["ladder"], 4.0) for r in rs]
    ax.barh(ys, vals, color=[COLORS[r["medal"]] for r in rs], edgecolor="white")
    ax.set_yticks(ys, [r["short"] for r in rs])
    ax.axvline(3.0, color="#111827", ls="--", lw=1, label="gold")
    ax.axvline(0.0, color="#6b7280", ls=":", lw=1, label="median")
    for y, r in zip(ys, rs):
        tag = " (capped)" if r["ladder"] >= 4.0 else ""
        ax.text(min(r["ladder"], 4.0) + 0.04, y, f"{r['score']:.4g}{tag}", va="center", fontsize=8)
    ax.set_xlabel("Medal ladder (0 = median, 3 = gold; label = test score)")
    ax.set_title("A. Outcome by task")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    ax.set_xlim(0, 4.25)
    ax.invert_yaxis()

    ax = axes[1]
    vals = [100 * (r["generalization_gap"] or 0) for r in rs]
    clipped = [max(-100, min(100, v)) for v in vals]
    ax.barh(ys, clipped, color=["#16a34a" if v >= 0 else "#dc2626" for v in clipped],
            edgecolor="white")
    ax.set_yticks(ys, ["" for _ in rs])
    ax.axvline(0.0, color="#111827", lw=1)
    for y, v, c in zip(ys, vals, clipped):
        ax.text(c + (2 if c >= 0 else -2), y, f"{v:+.1f}%", va="center",
                ha="left" if c >= 0 else "right", fontsize=7.5)
    ax.set_xlim(-115, 115)
    ax.set_xlabel("Test vs best board (%, + = test better; clipped ±100)")
    ax.set_title("B. CV → test transfer")
    fig.suptitle("Final outcome across 10 tasks", fontsize=12)
    fig.tight_layout()
    save(fig, out / "fig01_outcome.png")
    plt.close("all")


def plot_progress(rows: list[dict], out: Path) -> None:
    n = len(rows)
    ncol = 5
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, sorted(rows, key=lambda x: x["short"])):
        pts = r["curve_points"]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=COLORS[r["medal"]], lw=1.8)
            positive = [v for v in ys if v > 0]
            if positive and max(positive) / max(min(positive), 1e-12) > 50:
                ax.set_yscale("log")
            ax.scatter([1.0], [r["score"]], marker="X", color="#111827", s=28, zorder=4,
                       label="test")
        ax.axhline(r["gold"], color="#c9a227", ls="--", lw=0.9)
        ax.axhline(r["median_threshold"], color="#9ca3af", ls=":", lw=0.7)
        if r["last_improvement_frac"] is not None:
            ax.axvline(r["last_improvement_frac"], color="#6b7280", ls="-.", lw=0.7)
        nb = r["n_new_best"]
        ax.set_title(f"{r['short']}\n{nb} new best" + ("" if nb == 1 else "s"), fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(0, 1.05)
    for ax in axes[n:]:
        ax.axis("off")
    fig.supxlabel("Fraction of run time (dash-dot = last board improvement)", fontsize=9)
    fig.supylabel("Metric (best so far; dashed = gold, dotted = median)", fontsize=9)
    fig.suptitle("Metric progress per task (X = final test score)", fontsize=12)
    fig.tight_layout()
    save(fig, out / "fig02_progress_curves.png")
    plt.close("all")


def plot_cost(rows: list[dict], out: Path) -> None:
    offsets = [(3, 4), (3, -10), (-8, 8), (3, 12), (-10, -12)]
    fixed = {
        "billion-word": (3, -14),
        "mlsp-birds": (3, 4),
        "stanford-covid": (3, 12),
        "nfl-contact": (-58, -6),
        "spooky": (3, 4),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, xkey, xlabel in [
        (axes[0], "cost_usd", "Run cost, USD (dashed = median)"),
        (axes[1], "messages", "Board messages (dashed = median)"),
    ]:
        pts = [r for r in rows if r.get(xkey) is not None]
        pts.sort(key=lambda r: r[xkey])
        xs = [r[xkey] for r in pts]
        ys = [r["ladder"] for r in pts]
        cs = [COLORS[r["medal"]] for r in pts]
        ss = [18 + 1.2 * r["messages"] for r in pts]
        ax.scatter(xs, ys, c=cs, s=ss, alpha=0.85, edgecolor="white", linewidth=0.8)
        for i, r in enumerate(pts):
            off = fixed.get(r["short"], offsets[i % len(offsets)])
            ax.annotate(r["short"], (r[xkey], r["ladder"]), fontsize=7,
                        xytext=off, textcoords="offset points")
        med_x = statistics.median(xs)
        ax.axvline(med_x, color="#6b7280", ls="--", lw=0.8)
        ax.axhline(3, color="#111827", ls="--", lw=0.8)
        ax.axhline(0, color="#6b7280", ls=":", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.5, 4.35)
    axes[0].set_ylabel("Medal ladder")
    axes[0].set_title("A. Cost vs outcome")
    axes[1].set_title("B. Board volume vs outcome")
    fig.tight_layout()
    save(fig, out / "fig03_cost_volume_outcome.png")
    plt.close("all")


def plot_coordination(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: (r.get("reactivity") or 0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, kkey, nkey, title, xlabel in [
        (axes[0], "reactivity", "reactivity_k", "reactivity_n",
         "A. Message reactivity", "Share of messages citing a peer"),
        (axes[1], "cross_edge_share", "cross_edges", "cite_edges",
         "B. Cross-agent citation share", "Share of citation edges across agents"),
    ]:
        ys = np.arange(len(rs))
        vals = [r.get(key) or 0 for r in rs]
        ax.barh(ys, vals, color="#0ea5e9", edgecolor="white")
        for y, r in zip(ys, rs):
            k, n = r.get(kkey), r.get(nkey)
            if k is not None and n:
                lo, hi = wilson(int(k), int(n))
                ax.plot([lo, hi], [y, y], color="#0f172a", lw=2)
                ax.text(max(hi, vals[y]) + 0.02, y, f"{vals[y]:.2f}", va="center", fontsize=7)
        ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    fig.tight_layout()
    save(fig, out / "fig04_coordination.png")
    plt.close("all")


def plot_structure(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: r["depth_max"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    panels = [
        ("edges", "A. Citation edges", "#6366f1"),
        ("depth_max", "B. Max lineage depth", "#8b5cf6"),
        ("max_indegree", "C. Max in-degree (hubs)", "#ec4899"),
        ("density", "D. Graph density", "#14b8a6"),
    ]
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        ys = np.arange(len(rs))
        vals = [r.get(key) or 0 for r in rs]
        ax.barh(ys, vals, color=color, edgecolor="white")
        for y, v in zip(ys, vals):
            ax.text(v, y, f" {v:.3g}", va="center", fontsize=7)
        ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
        ax.set_title(title)
        ax.margins(x=0.18)
    fig.tight_layout()
    save(fig, out / "fig05_graph_structure.png")
    plt.close("all")


def plot_synergy(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    rs = sorted(rows, key=lambda r: (r.get("synergy_positive") or 0))
    ys = np.arange(len(rs))
    ax = axes[0]
    vals = [r.get("synergy_positive") or 0 for r in rs]
    ax.barh(ys, vals, color="#f59e0b", edgecolor="white")
    for y, r in zip(ys, rs):
        if r["synergy_n"]:
            k = round(vals[y] * r["synergy_n"])
            lo, hi = wilson(k, r["synergy_n"])
            ax.plot([lo, hi], [y, y], color="#111827", lw=2)
            ax.text(min(1.28, max(hi, vals[y]) + 0.03), y,
                    f"{vals[y]:.2f} (n={r['synergy_n']})", va="center", fontsize=6.5)
        else:
            ax.text(0.03, y, "no metric links", va="center", fontsize=6.5)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 1.32)
    ax.set_xlabel("Share beating the best directly cited result")
    ax.set_title("A. Synergy (direct)")

    rs2 = sorted(rows, key=lambda r: (r.get("tta_rate") or 0))
    ys = np.arange(len(rs2))
    ax = axes[1]
    vals = [r.get("tta_rate") or 0 for r in rs2]
    ax.barh(ys, vals, color="#22c55e", edgecolor="white")
    for y, r in zip(ys, rs2):
        lo, hi = wilson(r["tta_adopted"], r["tta_events"]) if r["tta_events"] else (0, 0)
        ax.plot([lo, hi], [y, y], color="#111827", lw=2)
        ax.text(max(hi, vals[y]) + 0.02, y, f"{vals[y]:.2f} ({r['tta_adopted']}/{r['tta_events']})",
                va="center", fontsize=6.5)
    ax.set_yticks(ys, [r["short"] for r in rs2], fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Adoption rate")
    ax.set_title("B. Insight diffusion (TTA)")

    rs3 = sorted(rows, key=lambda r: (r.get("recovery_rate") or 0))
    ys = np.arange(len(rs3))
    ax = axes[2]
    vals = [r.get("recovery_rate") or 0 for r in rs3]
    ax.barh(ys, vals, color="#ef4444", edgecolor="white")
    for y, r in zip(ys, rs3):
        if r["problems"]:
            lo, hi = wilson(r["recovered"], r["problems"])
            ax.plot([lo, hi], [y, y], color="#111827", lw=2)
        label_x = max(vals[y], hi) + 0.03 if r["problems"] else vals[y] + 0.03
        ax.text(min(1.22, label_x), y, f"{r['recovered']}/{r['problems']}", va="center", fontsize=7)
    ax.set_yticks(ys, [r["short"] for r in rs3], fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Recovery rate")
    ax.set_title("C. Problem recovery")
    fig.tight_layout()
    save(fig, out / "fig06_synergy_diffusion_recovery.png")
    plt.close("all")


def plot_tag_tta(tag_agg: list[dict], out: Path) -> None:
    tags = [t for t in tag_agg if t["events"] >= 5]
    tags.sort(key=lambda t: t["rate"])
    if not tags:
        return
    ys = np.arange(len(tags))
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.42 * len(tags) + 2.2), sharey=True)
    panels = [
        ("rate", "Adoption rate (Wilson 95% CI)", "A. Adoption rate by hashtag", "#8b5cf6", ("ci",)),
        ("median_msgs", "Median messages to adoption [IQR]", "B. Diffusion lag: messages", "#0ea5e9",
         ("q1_msgs", "q3_msgs")),
        ("median_minutes", "Median minutes to adoption [IQR]", "C. Diffusion lag: minutes", "#f59e0b",
         ("q1_minutes", "q3_minutes")),
    ]
    for ax, (key, xlabel, title, color, bounds) in zip(axes, panels):
        vals = [t[key] or 0 for t in tags]
        highs = []
        for t in tags:
            hi = t["ci"][1] if bounds == ("ci",) else t[bounds[1]]
            highs.append(hi or 0)
        xmax = max(vals + highs + [1]) * 1.28
        ax.barh(ys, vals, color=color, edgecolor="white")
        for y, t, v in zip(ys, tags, vals):
            if bounds == ("ci",):
                lo, hi = t["ci"]
            else:
                lo, hi = t[bounds[0]], t[bounds[1]]
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y, y], color="#111827", lw=2)
            label = f"{v:.2f} (n={t['events']})" if key == "rate" else f"{v:.1f}"
            ax.text(max(hi or v, v) + 0.02 * xmax, y, label, va="center", fontsize=7, clip_on=True)
        ax.set_yticks(ys, [t["tag"] for t in tags], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(0, xmax)
    fig.suptitle("Per-hashtag insight diffusion (tags with >= 5 events)", fontsize=12)
    fig.tight_layout()
    save(fig, out / "fig08_tag_tta.png")
    plt.close("all")


def plot_corr(rows: list[dict], corr: list[dict], out: Path) -> None:
    labels = [c["label"] for c in corr]
    rhos = [c["rho"] for c in corr]
    ps = [c["p"] for c in corr]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(corr) + 1.6))
    ys = np.arange(len(corr))
    colors = ["#16a34a" if r >= 0 else "#dc2626" for r in rhos]
    ax.barh(ys, rhos, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(ys, labels, fontsize=8)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlim(-1, 1)
    for y, rho, p in zip(ys, rhos, ps):
        mark = "*" if p < 0.1 else ""
        ax.text(rho + (0.02 if rho >= 0 else -0.02), y, f"{rho:+.2f}{mark}", va="center",
                ha="left" if rho >= 0 else "right", fontsize=8)
    ax.set_xlabel("Spearman ρ with outcome ladder (n = 10, * p < 0.1, exploratory)")
    ax.set_title("A. Process metrics vs outcome")
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, out / "fig07_correlations.png")
    plt.close("all")


def fmt(x, digits=3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figdir = args.out / "figures"
    figdir.mkdir(exist_ok=True)

    rows = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        r = load_task(d)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit("не найдено ни одной задачи с analysis/swarm/summary.json")

    tag_rows = []
    for d in sorted(args.root.iterdir()):
        if d.is_dir() and (d / "analysis" / "swarm" / "messages.csv").exists():
            tag_rows.extend(load_tag_tta(d))
    by_tag: dict[str, list[dict]] = defaultdict(list)
    by_task_tag: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in tag_rows:
        by_tag[r["tag"]].append(r)
        by_task_tag[(r["task"], r["tag"])].append(r)

    def tag_stats(recs: list[dict]) -> dict:
        adopted = [r for r in recs if r["adopter"]]
        msgs = [r["msgs"] for r in adopted]
        mins = [r["minutes"] for r in adopted]
        return {
            "events": len(recs),
            "adopted": len(adopted),
            "rate": len(adopted) / len(recs) if recs else None,
            "ci": wilson(len(adopted), len(recs)) if recs else (0.0, 0.0),
            "median_msgs": statistics.median(msgs) if msgs else None,
            "mean_msgs": statistics.mean(msgs) if msgs else None,
            "q1_msgs": float(np.percentile(msgs, 25)) if msgs else None,
            "q3_msgs": float(np.percentile(msgs, 75)) if msgs else None,
            "median_minutes": statistics.median(mins) if mins else None,
            "mean_minutes": statistics.mean(mins) if mins else None,
            "q1_minutes": float(np.percentile(mins, 25)) if mins else None,
            "q3_minutes": float(np.percentile(mins, 75)) if mins else None,
        }

    tag_agg = []
    for tag, recs in by_tag.items():
        st = tag_stats(recs)
        tag_agg.append({"tag": tag, "tasks": len({r["task"] for r in recs}), **st})
    tag_agg.sort(key=lambda x: (-x["events"], x["tag"]))
    task_tag_agg = []
    for (task, tag), recs in sorted(by_task_tag.items()):
        st = tag_stats(recs)
        task_tag_agg.append({"task": task, "tag": tag, **st})

    fields = [k for k in rows[0] if k != "curve_points"]
    with open(args.out / "cross_task_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(args.out / "tag_tta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "tasks", "events", "adopted", "rate", "ci_low", "ci_high",
                    "median_msgs", "mean_msgs", "q1_msgs", "q3_msgs",
                    "median_minutes", "mean_minutes", "q1_minutes", "q3_minutes"])
        for t in tag_agg:
            w.writerow([t["tag"], t["tasks"], t["events"], t["adopted"], f"{t['rate']:.4f}",
                        f"{t['ci'][0]:.4f}", f"{t['ci'][1]:.4f}",
                        *[(f"{t[k]:.3f}" if t[k] is not None else "")
                          for k in ("median_msgs", "mean_msgs", "q1_msgs", "q3_msgs",
                                    "median_minutes", "mean_minutes", "q1_minutes", "q3_minutes")]])
    with open(args.out / "tag_tta_by_task.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "tag", "events", "adopted", "rate", "median_msgs", "median_minutes"])
        for t in task_tag_agg:
            w.writerow([t["task"], t["tag"], t["events"], t["adopted"], f"{t['rate']:.4f}",
                        f"{t['median_msgs']:.3f}" if t["median_msgs"] is not None else "",
                        f"{t['median_minutes']:.3f}" if t["median_minutes"] is not None else ""])

    pooled = {
        "n_tasks": len(rows),
        "medals": {k: sum(1 for r in rows if r["medal"] == k)
                   for k in ["gold", "silver", "bronze", "above_median", "below_median"]},
        "above_median": sum(1 for r in rows if r["above_median"]),
        "ladder": describe([r["ladder"] for r in rows]),
        "gold_gap": describe([100 * r["gold_gap"] for r in rows]),
        "messages": describe([r["messages"] for r in rows]),
        "duration_h": describe([r["duration_h"] for r in rows]),
        "cost_usd": describe([r["cost_usd"] for r in rows if r["cost_usd"] is not None]),
        "cost_total": sum(r["cost_usd"] for r in rows if r["cost_usd"] is not None),
        "cost_per_msg": describe([r["cost_per_msg"] for r in rows if r["cost_per_msg"] is not None]),
        "reactivity": describe([r["reactivity"] for r in rows]),
        "cross_edge_share": describe([r["cross_edge_share"] for r in rows]),
        "synergy_positive": describe([r["synergy_positive"] for r in rows]),
        "tta_rate": describe([r["tta_rate"] for r in rows]),
        "recovery_rate": describe([r["recovery_rate"] for r in rows]),
        "depth_max": describe([r["depth_max"] for r in rows]),
        "max_indegree": describe([r["max_indegree"] for r in rows]),
        "generalization_gap": describe([100 * r["generalization_gap"] for r in rows
                                        if r["generalization_gap"] is not None]),
        "n_new_best": describe([r["n_new_best"] for r in rows]),
        "last_improvement_frac": describe([r["last_improvement_frac"] for r in rows
                                           if r["last_improvement_frac"] is not None]),
        "top_producer_share": describe([r["top_producer_share"] for r in rows
                                        if r["top_producer_share"] is not None]),
        "agents_with_best": describe([r["agents_with_best"] for r in rows]),
    }
    pooled["reactivity_pooled"] = {
        "k": sum(r["reactivity_k"] for r in rows),
        "n": sum(r["reactivity_n"] for r in rows),
        "rate": sum(r["reactivity_k"] for r in rows) / max(1, sum(r["reactivity_n"] for r in rows)),
        "ci": wilson(sum(r["reactivity_k"] for r in rows), sum(r["reactivity_n"] for r in rows)),
    }
    pooled["cross_edges_pooled"] = {
        "k": sum(r["cross_edges"] for r in rows),
        "n": sum(r["cite_edges"] for r in rows),
        "rate": sum(r["cross_edges"] for r in rows) / max(1, sum(r["cite_edges"] for r in rows)),
        "ci": wilson(sum(r["cross_edges"] for r in rows), sum(r["cite_edges"] for r in rows)),
    }
    pooled["self_cites_pooled"] = {
        "k": sum(r["self_cites"] for r in rows),
        "n": sum(r["cite_edges"] for r in rows),
        "rate": sum(r["self_cites"] for r in rows) / max(1, sum(r["cite_edges"] for r in rows)),
        "ci": wilson(sum(r["self_cites"] for r in rows), sum(r["cite_edges"] for r in rows)),
        "self_links": sum(r["self_links"] for r in rows),
    }
    pooled["tta_pooled"] = {
        "k": sum(r["tta_adopted"] for r in rows),
        "n": sum(r["tta_events"] for r in rows),
        "rate": sum(r["tta_adopted"] for r in rows) / max(1, sum(r["tta_events"] for r in rows)),
        "ci": wilson(sum(r["tta_adopted"] for r in rows), sum(r["tta_events"] for r in rows)),
    }
    pooled["recovery_pooled"] = {
        "k": sum(r["recovered"] for r in rows),
        "n": sum(r["problems"] for r in rows),
        "rate": sum(r["recovered"] for r in rows) / max(1, sum(r["problems"] for r in rows)),
        "ci": wilson(sum(r["recovered"] for r in rows), sum(r["problems"] for r in rows)),
        "different_author": sum(r["recovery_different_author"] for r in rows),
    }
    syn_direct_k = sum(round(r["synergy_positive"] * r["synergy_n"]) for r in rows
                       if r["synergy_positive"] is not None)
    syn_direct_n = sum(r["synergy_n"] for r in rows)
    pooled["synergy_direct"] = {
        "k": syn_direct_k,
        "n": syn_direct_n,
        "rate": syn_direct_k / max(1, syn_direct_n),
        "ci": wilson(syn_direct_k, syn_direct_n),
    }
    delta_pct = [100 * r["synergy_mean_rel"] for r in rows if r["synergy_mean_rel"] is not None]
    pooled["tag_tta"] = tag_agg
    pooled["synergy_delta_pct"] = describe(delta_pct)
    pooled["synergy_delta_pct_weighted"] = (
        sum(100 * r["synergy_mean_rel"] * r["synergy_n"]
            for r in rows if r["synergy_mean_rel"] is not None)
        / max(1, sum(r["synergy_n"] for r in rows if r["synergy_mean_rel"] is not None))
    )
    pooled["board_improve"] = {
        "linked_k": sum(round(r["linked_positive"] * r["linked_n"]) for r in rows
                        if r["linked_positive"] is not None),
        "linked_n": sum(r["linked_n"] for r in rows),
        "unlinked_k": sum(round(r["unlinked_positive"] * r["unlinked_n"]) for r in rows
                          if r["unlinked_positive"] is not None),
        "unlinked_n": sum(r["unlinked_n"] for r in rows),
    }
    bi = pooled["board_improve"]
    bi["linked_rate"] = bi["linked_k"] / max(1, bi["linked_n"])
    bi["unlinked_rate"] = bi["unlinked_k"] / max(1, bi["unlinked_n"])
    bi["linked_ci"] = wilson(bi["linked_k"], bi["linked_n"])
    bi["unlinked_ci"] = wilson(bi["unlinked_k"], bi["unlinked_n"])

    both = [r for r in rows if r["linked_mean_rel"] is not None and r["unlinked_mean_rel"] is not None]
    pooled["synergy_task_comparison"] = {
        "n_tasks": len(both),
        "linked_gt_unlinked": sum(1 for r in both if r["linked_mean_rel"] > r["unlinked_mean_rel"]),
        "tasks": [r["short"] for r in both],
    }

    corr = spearman_table(rows)
    pooled["correlations"] = corr

    (args.out / "cross_task_summary.json").write_text(
        json.dumps(pooled, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_outcome(rows, figdir)
    plot_progress(rows, figdir)
    plot_cost(rows, figdir)
    plot_coordination(rows, figdir)
    plot_structure(rows, figdir)
    plot_synergy(rows, figdir)
    plot_tag_tta(tag_agg, figdir)
    plot_corr(rows, corr, figdir)

    med = pooled
    def ci_txt(d):
        return f"{d['rate']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"

    def col_mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    lines = []
    lines.append("# Кросс-задачная аналитика: 10 задач MLE-bench\n")
    lines.append("Эксперимент `1-lie10-blackboard-mlebench`: 3 однородных ReAct-агента, общая доска, "
                 "12-часовой лимит. Анализ Variant A: per-task `swarm_analysis.py` + агрегация.\n")
    lines.append("Использование GPU/CPU/RAM — отдельный отчёт: [`RESOURCES.md`](RESOURCES.md).\n")
    lines.append("""## 1. Метрики: определения и формулы

Обозначения: `lower` — направление метрики (`is_lower_better` из `grade.json`); `x` — скор задачи; `th_gold`, `th_silver`, `th_bronze`, `th_median` — пороги MLE-bench; `N` — число агентских сообщений (для графовых метрик — число узлов-документов агентов); `E` — число cite-рёбер.

### 1.1. Источники данных

Для каждой задачи агрегатор читает:

| Файл | Что берётся |
|---|---|
| `analysis/grade.json` | тестовый скор `score`, пороги `gold/silver/bronze/median`, флаги медалей, направление метрики |
| `analysis/metrics.csv` | все `<metric>` с доски: время, автор, значение; строки `system` (официальный grade) исключаются |
| `analysis/graph.json` | citation-граф: узлы-документы с авторами, рёбра `kind=cite` |
| `analysis/swarm/summary.json` | per-task агрегаты: синергия, TTA, recovery, граф, качество ссылок |
| `analysis/COSTS.md` | суммарная стоимость прогона (строка «итого: $X») |

### 1.2. Outcome: медали и перенос на тест

**`score`** — официальный скор MLE-bench на тестовой выборке. **`medal`** — официальный флаг (`gold`/`silver`/`bronze`/`above_median`/`below_median`), а не пересчёт по порогам.

**Medal ladder `L`** — кусочно-линейная позиция скора между порогами: 0 = median, 1 = bronze, 2 = silver, 3 = gold. Для lower-better:

```
L(x) = i + (p_i − x) / (p_i − p_{i+1}),   если p_i ≥ x ≥ p_{i+1}
p = [th_median, th_bronze, th_silver, th_gold]
```

для higher-better — зеркально (`p_i ≤ x ≤ p_{i+1}`). Выше gold: `L = 3 + (th_gold − x)/(th_gold − th_silver)` для lower-better (для higher-better зеркально), с потолком 4.0.

Зачем ladder, а не «сырой» medal margin: маржа `(th_median − x)/(th_median − th_gold)` неограничена и на этих данных даёт 15.3 для `billion-word` и 9.4 для `stanford-covid`, то есть одна задача доминирует в любом среднем. Ladder ограничивает вклад каждой задачи диапазоном 0–4 и сохраняет интерпретацию «сколько ступеней до gold».

**CV → test gap `G`** — направленная относительная разница между тестом и лучшим бортовым результатом:

```
G = (best_board − score) / |best_board|   для lower-better
G = (score − best_board) / |best_board|   для higher-better
```

`G > 0` — тест лучше лучшего поста на доске, `G < 0` — рой переоценил качество решения по внутренней валидации. Метрика отделяет качество финального сабмита от оптимизма CV и не зависит от порогов.

### 1.3. Прогресс по доске

Считается по `metrics.csv` без SYSTEM-строк:

- **`first_board`** — первый опубликованный агентом `<metric>` (стартовая точка).
- **`best_board`** — лучший metric за прогон: `min` для lower-better, `max` для higher-better.
- **New best** — событие, когда очередной metric строго улучшает best-so-far (порог `1e-12`); первый metric считается базовой точкой и событием не является.
- **`n_new_best`** — число таких событий за прогон.
- **`last_improvement_frac`** — время последнего new best, делённое на длительность прогона (от первого до последнего агентского сообщения), в диапазоне 0–1. Значение 0.95 означает, что последний рекорд поставлен в конце, 0.20 — что поиск фактически остановился рано.
- **`top_producer_share`** — максимальная доля new best, приходящаяся на одного агента; **`agents_with_best`** — сколько агентов дали хотя бы один new best. Это грубый индикатор концентрации «производства» лучших результатов.

Эти метрики безразмерны и поэтому сравнимы между задачами, в отличие от абсолютных значений metric: у `bms` бортовые значения падают с 73 до 2.3, у `spooky` — с 0.387 до 0.184. Если агенты использовали разные определения метрики, `first_board`/`best_board` могут быть несопоставимы — такие случаи перечислены в §9.

### 1.4. Координация и граф знаний

Считается по `graph.json`; учитываются только узлы-документы агентов (SYSTEM исключён), рёбра `kind=cite` между существующими документами, self-рёбра и `missing:*` отброшены.

- **Reactivity** = доля агентских сообщений, у которых есть хотя бы одно исходящее cite-ребро к сообщению другого автора:
  `reactivity = |{k : out_cross(k) > 0}| / N`.
- **Cross-agent edge share** = доля кросс-авторских cite-рёбер среди всех cite-рёбер:
  `cross_edges / E`.
- **Self-cites (self-citation)** — число cite-рёбер, где источник и цель принадлежат одному агенту (агент ссылается на свои же сообщения); **self-cite share** = `self_cites / E`. Вместе с cross-agent share даёт 100%.
- **Self-links** — рёбра `s = t`, где сообщение цитирует само себя; структурные петли, считаются отдельно.
- **External in-degree per message** = `cross_edges / N` — сколько кросс-агентских входящих ссылок в среднем приходится на сообщение.
- **Density** = `E / (N(N−1))` — плотность citation-графа.
- **Depth max** — максимальная длина цепочки «сообщение → процитированные предки» (из `swarm/summary.json`).

При пулинге по задачам доли агрегируются как `Σk / Σn` (event pooling), а не как среднее долей, и сопровождаются Wilson 95% CI.

### 1.5. Синергия

Для сообщения `k` со скором `M_k`, которое ссылается хотя бы на одно сообщение с metric:

```
ΔM_link(k) = min(M_j) − M_k      для lower-better, j — прямо процитированные
ΔM_link(k) = M_k − max(M_j)      для higher-better
```

Положительное значение = сообщение превзошло лучший непосредственно процитированный результат.

- **`synergy_n`** — число сообщений, у которых есть и свой metric, и хотя бы один процитированный metric.
- **`share improving`** — доля таких сообщений с `ΔM_link > 1e-12`.
- **`ΔM (% прогресса доски)`** — `mean(ΔM_link) / |first_board − best_board| × 100`: какую долю всего выигрыша роя даёт типичное сообщение. Нормировка на gold здесь не используется: у `bms` gold = 0.62 при первом бортовом значении 73, и деление на gold даёт бессмысленные сотни процентов. Знаменатель `|first − best_board|` — собственный масштаб прогресса доски.
- **Board improvement (linked vs unlinked)** — `ΔM_prev(k)` = улучшение относительно лучшего значения на доске *до* сообщения `k`, отдельно для сообщений со ссылками и без. Это вспомогательное сравнение: оно отвечает на вопрос «чаще ли сообщения со ссылками ставят новый рекорд», но `unlinked`-сообщений всего 18 на 10 задач, поэтому вывод по нему underpowered.

### 1.6. Диффузия инсайтов (TTA)

- **Insight event (broad)** — сообщение с тегом `#insight` или `#feature_engineering` и собственным metric.
- **Adoption (direct)** — первое более позднее сообщение **другого** агента, которое прямо цитирует event.
- **TTA** — число сообщений роя и минуты от event до adoption; **adoption rate** = adopted / events; **TTA med msgs** — медиана числа сообщений по adopted events.

В таблице используется broad-набор; в `swarm_analysis.py` также считаются strict (insight + новая лучшая метрика) и all_new_best (любая новая лучшая метрика) — они лежат в `summary.json`. В §5.1 те же события пересчитаны отдельно по каждому хэштегу: событие = сообщение с этим тегом и metric; сообщение с несколькими тегами входит в несколько строк.

### 1.7. Resilience

- **Problem event** — сообщение с тегом `#bug`, `#conflict` или `#problem`.
- **Recovery** — более позднее сообщение с тегом `#insight`, `#submission` или `#solution`, прямо цитирующее problem.
- Метрики: **recovery rate** = recovered / problems, **different_author** — сколько recovery сделано другим агентом, медиана сообщений и минут до recovery.

### 1.8. Стоимость

**`cost_usd`** — суммарная стоимость прогона из `COSTS.md` (строка «итого: $X»). **`cost_per_msg`** = `cost_usd / N`. Стоимость сравнивается только в абсолютных величинах и на задачу; для сравнения с конкурентами нужны их заявленные бюджеты, которых в репозитории нет.

### 1.9. Агрегация и статистика

- По 10 задачам: **медиана [Q1, Q3]** в итоговых абзацах — распределения скошены (`billion-word`, `taxi`); в каждой позадачной таблице последняя строка — **среднее по задачам** с равным весом (для несопоставимых raw-шкал — прочерк).
- Доли: пулинг событий `Σk/Σn` + **Wilson 95% CI**.
- Связь процессных метрик с outcome: **Spearman ρ** с medal ladder, n = 10, без поправки на множественные сравнения — только exploratory-гипотезы, не подтверждённые эффекты.
- Причинность не заявляется: для неё нужен ablation (например, отключение видимости чужих сообщений или рандомизация citation-графа).

### 1.10. Сводная таблица метрик

| Метрика | Определение | Интерпретация |
|---|---|---|
| `score` | официальный тестовый скор | качество финального сабмита |
| `medal` | официальный флаг MLE-bench | gold/silver/bronze/above_median/below_median |
| `L` (ladder) | позиция между порогами, 0–4 | 0 = median, 3 = gold; устойчива к выбросам |
| `G` (CV→test) | тест vs лучший борд, % | >0 — тест лучше внутренней валидации, <0 — переоценка CV |
| `first_board` | первый metric агента | стартовая точка прогресса |
| `best_board` | лучший metric за прогон | внутренний рекорд роя |
| `n_new_best` | число строгих улучшений best-so-far | интенсивность поиска |
| `last_improvement_frac` | время последнего new best / длительность | 1.0 — улучшения до конца, 0.2 — ранняя конвергенция |
| `top_producer_share` | доля new best у одного агента | концентрация «производства» лучших результатов |
| `reactivity` | доля сообщений с кросс-агентской ссылкой | реактивность на доску |
| `cross_edge_share` | доля кросс-агентских cite-рёбер | степень перемешивания агентов |
| `self_cites` | число ссылок агента на свои сообщения | объём внутренней преемственности |
| `self_cite_share` | `self_cites / E` | доля «замкнутых» ссылок; дополняет cross-agent share |
| `self_links` | рёбра `s = t` | структурные петли (во всех задачах 0) |
| `density` | `E/(N(N−1))` | плотность citation-графа |
| `depth_max` | длина самой длинной цепочки предков | глубина проработки идей |
| `ΔM_link` | прирост над лучшим процитированным | сила локальной синергии |
| `share improving` | доля `ΔM_link > 0` | как часто сообщение обгоняет базу |
| `ΔM %` | `mean(ΔM_link) / board_gain` | вклад типичного сообщения в общий прогресс |
| `TTA adoption` | adopted / insight-events | скорость принятия идей |
| `TTA med msgs` | медиана сообщений до adoption | лаг диффузии |
| `recovery rate` | recovered / problem-events | устойчивость к сбоям |
| `cost_usd` | сумма из `COSTS.md` | абсолютный бюджет прогона |
| `cost_per_msg` | `cost_usd / N` | удельная стоимость сообщения |

""")
    lines.append("## 2. Таблица 1. Outcome и перенос на тест\n")
    lines.append("| Задача | Модальность | Метрика | Тест | Лучший борд | CV→test | Gold | Медаль | Ladder |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|---:|")
    for r in sorted(rows, key=lambda x: -x["ladder"]):
        lines.append(f"| {r['short']} | {r['modality']} | {r['metric']} | {r['score']:.5g} | "
                     f"{fmt(r['best_board'],5)} | {100*(r['generalization_gap'] or 0):+.1f}% | "
                     f"{r['gold']:.5g} | {r['medal']} | {r['ladder']:.2f} |")
    m_cv, m_lad = 100*col_mean("generalization_gap"), col_mean("ladder")
    lines.append(f"| **Среднее** | — | — | — | — | {m_cv:+.1f}% | — | — | {m_lad:.2f} |")
    lines.append("")
    lines.append("Средние по столбцам «Тест», «Лучший борд» и «Gold» не приводятся: это разные "
                 "метрики с несопоставимыми шкалами. Сравнимы только нормализованные CV→test и Ladder; "
                 "среднее CV→test сильно смещено `billion-word` (−217.9%), робастная медиана — в «Итог» ниже.\n")
    lines.append(f"**Итог:** gold {med['medals']['gold']}/10, silver {med['medals']['silver']}/10, "
                 f"bronze {med['medals']['bronze']}/10, выше медианы {med['above_median']}/10. "
                 f"Ladder median = {med['ladder']['median']:.2f} "
                 f"[{med['ladder']['q1']:.2f}, {med['ladder']['q3']:.2f}]. "
                 f"CV→test gap median = {med['generalization_gap']['median']:+.1f}% "
                 f"[{med['generalization_gap']['q1']:+.1f}, {med['generalization_gap']['q3']:+.1f}] "
                 f"(n = {med['generalization_gap']['n']}).\n")

    lines.append("## 3. Таблица 2. Процесс, стоимость и сходимость\n")
    lines.append("| Задача | Сообщений | Длит., ч | Cost, $ | $/msg | Новых best | Последний best | Top producer | Рёбра | Depth max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x["task"]):
        lines.append(f"| {r['short']} | {r['messages']} | {r['duration_h']:.1f} | "
                     f"{fmt(r['cost_usd'],4)} | {fmt(r['cost_per_msg'],3)} | {r['n_new_best']} | "
                     f"{fmt(r['last_improvement_frac'],2)} | {fmt(r['top_producer_share'],2)} | "
                     f"{r['cite_edges']} | {r['depth_max']} |")
    lines.append(f"| **Среднее** | {col_mean('messages'):.1f} | {col_mean('duration_h'):.1f} | "
                 f"{col_mean('cost_usd'):.3f} | {col_mean('cost_per_msg'):.3f} | "
                 f"{col_mean('n_new_best'):.1f} | {col_mean('last_improvement_frac'):.2f} | "
                 f"{col_mean('top_producer_share'):.2f} | {col_mean('cite_edges'):.1f} | "
                 f"{col_mean('depth_max'):.1f} |")
    lines.append("")
    lines.append(f"**Итог:** суммарная стоимость ${med['cost_total']:.2f}; "
                 f"медиана на задачу ${med['cost_usd']['median']:.2f} "
                 f"[{med['cost_usd']['q1']:.2f}, {med['cost_usd']['q3']:.2f}]; "
                 f"медиана сообщений {med['messages']['median']:.0f} "
                 f"[{med['messages']['q1']:.0f}, {med['messages']['q3']:.0f}]; "
                 f"длительность {med['duration_h']['median']:.1f} ч "
                 f"[{med['duration_h']['q1']:.1f}, {med['duration_h']['q3']:.1f}]. "
                 f"Новых best: {med['n_new_best']['median']:.0f} [{med['n_new_best']['q1']:.0f}, "
                 f"{med['n_new_best']['q3']:.0f}]; последнее улучшение на "
                 f"{med['last_improvement_frac']['median']:.2f} "
                 f"[{med['last_improvement_frac']['q1']:.2f}, {med['last_improvement_frac']['q3']:.2f}] "
                 f"времени прогона; top producer share = {med['top_producer_share']['median']:.2f} "
                 f"[{med['top_producer_share']['q1']:.2f}, {med['top_producer_share']['q3']:.2f}].\n")

    lines.append("## 4. Таблица 3. Координация и граф знаний\n")
    lines.append("| Задача | Reactivity | Cross-agent edges | Self-cites | Self-cite share | Cite edges | External in-deg/msg | Density |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: -(x["reactivity"] or 0)):
        lines.append(f"| {r['short']} | {fmt(r['reactivity'],2)} | {fmt(r['cross_edge_share'],2)} | "
                     f"{r['self_cites']} | {fmt(r['self_cite_share'],2)} | {r['cite_edges']} | "
                     f"{fmt(r['external_indegree_per_msg'],2)} | {fmt(r['density'],4)} |")
    lines.append(f"| **Среднее** | {col_mean('reactivity'):.2f} | {col_mean('cross_edge_share'):.2f} | "
                 f"{col_mean('self_cites'):.1f} | {col_mean('self_cite_share'):.2f} | "
                 f"{col_mean('cite_edges'):.1f} | {col_mean('external_indegree_per_msg'):.2f} | "
                 f"{col_mean('density'):.4f} |")
    lines.append("")
    lines.append(f"**Пулинг:** reactivity {ci_txt(med['reactivity_pooled'])} "
                 f"({med['reactivity_pooled']['k']}/{med['reactivity_pooled']['n']} сообщений); "
                 f"cross-agent citations {ci_txt(med['cross_edges_pooled'])} "
                 f"({med['cross_edges_pooled']['k']}/{med['cross_edges_pooled']['n']} рёбер); "
                 f"self-citations (агент → свои сообщения) {ci_txt(med['self_cites_pooled'])} "
                 f"({med['self_cites_pooled']['k']}/{med['self_cites_pooled']['n']} рёбер). "
                 f"Self-links (сообщение → оно само): {med['self_cites_pooled']['self_links']} во всех 10 задачах. "
                 f"Медиана reactivity = {med['reactivity']['median']:.2f} "
                 f"[{med['reactivity']['q1']:.2f}, {med['reactivity']['q3']:.2f}].\n")

    lines.append("## 5. Таблица 4. Синергия, диффузия, resilience\n")
    lines.append("| Задача | Synergy N | Share improving | ΔM (% прогресса доски) | TTA adoption | TTA med msgs | Recovery |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: -(x["synergy_positive"] or 0)):
        dm = f"{100*r['synergy_mean_rel']:+.2f}%" if r["synergy_mean_rel"] is not None else "—"
        lines.append(f"| {r['short']} | {r['synergy_n']} | {fmt(r['synergy_positive'],2)} | {dm} | "
                     f"{r['tta_adopted']}/{r['tta_events']} | {fmt(r['tta_median_msgs'],1)} | "
                     f"{r['recovered']}/{r['problems']} |")
    lines.append(f"| **Среднее** | {col_mean('synergy_n'):.1f} | {col_mean('synergy_positive'):.2f} | "
                 f"{col_mean('synergy_mean_rel')*100:+.1f}% | {col_mean('tta_rate'):.2f} | "
                 f"{col_mean('tta_median_msgs'):.1f} | {col_mean('recovery_rate'):.2f} |")
    lines.append("")
    lines.append("В строке «Среднее» для TTA и Recovery — среднее по задачам от per-task rate "
                 "(pooled-значения — в тексте ниже); Synergy N — по всем 10 задачам, share improving "
                 "и ΔM — по задачам, где метрика определена.\n")
    sd = med["synergy_direct"]
    bi = med["board_improve"]
    cmp_ = med["synergy_task_comparison"]
    lines.append(f"**Пулинг:** сообщения превосходят лучший процитированный результат в "
                 f"{sd['k']}/{sd['n']} = {sd['rate']:.3f} [{sd['ci'][0]:.3f}, {sd['ci'][1]:.3f}] случаев. "
                 f"Новый лучший результат доски дают {bi['linked_k']}/{bi['linked_n']} = {bi['linked_rate']:.3f} "
                 f"сообщений со ссылками и {bi['unlinked_k']}/{bi['unlinked_n']} = {bi['unlinked_rate']:.3f} "
                 f"без ссылок; средний ΔM со ссылками выше, чем без ссылок, в {cmp_['linked_gt_unlinked']} "
                 f"из {cmp_['n_tasks']} задач, где есть обе категории — сравнение underpowered "
                 f"(unlinked N = {bi['unlinked_n']}). "
                 f"TTA adoption {ci_txt(med['tta_pooled'])}; recovery {ci_txt(med['recovery_pooled'])} "
                 f"({med['recovery_pooled']['different_author']} из {med['recovery_pooled']['k']} — другой агент).\n")
    dp = med["synergy_delta_pct"]
    lines.append("**ΔM (% прогресса доски), сводка по задачам:**\n")
    lines.append("| Показатель | Значение |")
    lines.append("|---|---:|")
    lines.append(f"| Среднее по задачам | {dp['mean']:+.1f}% |")
    lines.append(f"| Медиана [Q1, Q3] | {dp['median']:+.1f}% [{dp['q1']:+.1f}, {dp['q3']:+.1f}] |")
    lines.append(f"| Взвешенное по числу сообщений | {med['synergy_delta_pct_weighted']:+.1f}% |")
    lines.append(f"| Задач с отрицательным средним | "
                 f"{sum(1 for v in delta_pct if v < 0)}/{len(delta_pct)} |")
    lines.append(f"| Сообщений в расчёте | {sd['n']} |\n")

    lines.append("### 5.1. TTA по отдельным хэштегам\n")
    lines.append("Событие = сообщение с данным тегом и собственным metric; adoption = первое более позднее "
                 "сообщение другого агента, прямо цитирующее событие. Сообщение с несколькими тегами "
                 "входит в каждую строку, поэтому суммы по таблице не равны общему числу событий.\n")
    shown_tags = [t for t in tag_agg if t["events"] >= 5]
    lines.append("| Хэштег | Задач | Events | Adopted | Rate [95% CI] | Lag, msgs: med [IQR] | Lag, min: med [IQR] |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    for t in shown_tags:
        rate = f"{t['rate']:.2f} [{t['ci'][0]:.2f}, {t['ci'][1]:.2f}]"
        lag_m = (f"{t['median_msgs']:.1f} [{t['q1_msgs']:.1f}, {t['q3_msgs']:.1f}]"
                 if t["median_msgs"] is not None else "—")
        lag_t = (f"{t['median_minutes']:.1f} [{t['q1_minutes']:.1f}, {t['q3_minutes']:.1f}]"
                 if t["median_minutes"] is not None else "—")
        lines.append(f"| `{t['tag']}` | {t['tasks']} | {t['events']} | {t['adopted']} | {rate} | "
                     f"{lag_m} | {lag_t} |")
    lines.append("")
    lookup = {t["tag"]: t for t in tag_agg}
    ins = lookup.get("insight")
    fe = lookup.get("feature_engineering")
    if ins and fe:
        lines.append(f"Для сравнения: `insight` — {ins['adopted']}/{ins['events']} = {ins['rate']:.2f}, "
                     f"`feature_engineering` — {fe['adopted']}/{fe['events']} = {fe['rate']:.2f}; "
                     f"их объединение (broad) — {med['tta_pooled']['k']}/{med['tta_pooled']['n']} = "
                     f"{med['tta_pooled']['rate']:.2f}. "
                     "В таблицу включены теги с ≥5 событиями; теги с меньшим N оставлены только в "
                     "`tag_tta.csv`, полные per-task значения — в `tag_tta_by_task.csv`.\n")
    else:
        lines.append("В таблицу включены теги с ≥5 событиями; теги с меньшим N оставлены только в "
                     "`tag_tta.csv`, полные per-task значения — в `tag_tta_by_task.csv`.\n")

    lines.append("## 6. Сходимость и перенос на тест\n")
    lines.append(f"Рой делал в среднем {med['n_new_best']['median']:.0f} новых лучших результатов "
                 f"за прогон (IQR [{med['n_new_best']['q1']:.0f}, {med['n_new_best']['q3']:.0f}]). "
                 f"Последнее улучшение приходилось на {100*med['last_improvement_frac']['median']:.0f}% "
                 f"времени (IQR [{100*med['last_improvement_frac']['q1']:.0f}%, "
                 f"{100*med['last_improvement_frac']['q3']:.0f}%]) — то есть поиск не затухал задолго "
                 "до дедлайна. Медианный CV→test gap "
                 f"{med['generalization_gap']['median']:+.1f}%: типично финальный тест немного хуже "
                 "лучшего бортового скора, что указывает на осторожную трактовку внутренних метрик.\n")

    lines.append("## 7. Исследовательские корреляции (Spearman, n = 10)\n")
    lines.append("| Метрика | ρ с ladder | p |")
    lines.append("|---|---:|---:|")
    for c in sorted(corr, key=lambda x: -abs(x["rho"])):
        lines.append(f"| {c['label']} | {c['rho']:+.2f} | {c['p']:.3f} |")
    lines.append("")
    lines.append("Интерпретировать осторожно: n = 10, без поправки на множественные сравнения. "
                 "Это гипотезы для основного анализа (Variant B/C), не подтверждённые эффекты.\n")

    lines.append("## 8. Что можно утверждать в статье (черновик)\n")
    lines.append(f"1. **Outcome.** {med['above_median']}/10 задач выше медианы MLE-bench; "
                 f"{med['medals']['gold']} gold, {med['medals']['silver']} silver, "
                 f"{med['medals']['bronze']} bronze. Медианный ladder = {med['ladder']['median']:.2f}/3.\n")
    lines.append(f"2. **Координация.** {100*med['reactivity_pooled']['rate']:.1f}% сообщений цитируют "
                 f"хотя бы одного другого агента; {100*med['cross_edges_pooled']['rate']:.1f}% всех cite-рёбер "
                 "ведут к другому агенту. Это воспроизводится на всех 10 задачах, а не только на spooky.\n")
    delta_vals = [r["synergy_mean_rel"] for r in rows if r["synergy_mean_rel"] is not None]
    neg = sum(1 for v in delta_vals if v < 0)
    med_delta = statistics.median(delta_vals)
    lines.append(f"3. **Синергия ограничена.** {100*sd['rate']:.1f}% сообщений превосходят лучший "
                 f"процитированный результат, но средний ΔM = {med['synergy_delta_pct']['mean']:+.1f}%, "
                 f"медианный ΔM = {100*med_delta:+.1f}% прогресса доски, "
                 f"и в {neg}/{len(delta_vals)} задач средний ΔM отрицателен. Рой чаще дорабатывает "
                 "чужие артефакты, чем сразу их превосходит.\n")
    lines.append(f"4. **Диффузия и resilience.** TTA adoption {100*med['tta_pooled']['rate']:.0f}%, "
                 f"recovery {100*med['recovery_pooled']['rate']:.0f}%; {med['recovery_pooled']['different_author']} "
                 f"из {med['recovery_pooled']['k']} recovery сделаны другим агентом.\n")
    lines.append(f"5. **Поиск не затухает.** Медианно {med['n_new_best']['median']:.0f} новых лучших "
                 f"результатов; последнее улучшение на {100*med['last_improvement_frac']['median']:.0f}% "
                 "времени прогона. Это аргумент против «ранней конвергенции в локальный оптимум».\n")
    lines.append("6. **Стоимость.** Суммарно ~$"
                 f"{med['cost_total']:.1f} за 10 задач, медиана ${med['cost_usd']['median']:.2f}/задачу — "
                 "низкий абсолютный бюджет, но для сравнения с конкурентами нужны их заявленные cost-данные.\n")

    lines.append("## 9. Ограничения\n")
    lines.append("- Один прогон на задачу (N = 10), p-value корреляций exploratory.\n")
    lines.append("- Бортовые метрики не всегда сопоставимы между агентами и с порогами MLE-bench: "
                 "`bms` начинает с 73 (другое определение metric), `billion-word` — с 2.27 при gold 5.37, "
                 "`stanford-covid` — с 0.230 при gold 0.347. Поэтому прогресс и ΔM нормируются на "
                 "прогресс самой доски, а не на gold.\n")
    lines.append("- CV→test gap показывает, что финальный тест в медиане на "
                 f"{abs(med['generalization_gap']['median']):.1f}% хуже лучшего поста на доске; "
                 "причины — переобучение на CV, различия split и postprocessing.\n")
    lines.append("- `uw-madison` имеет 3 metric-сообщения, поэтому synergy N = 0 — задача выпадает "
                 "из synergy-агрегатов (но не из outcome/графа).\n")
    lines.append("- Сравнение linked vs unlinked underpowered: unlinked сообщений всего "
                 f"{bi['unlinked_n']} на 10 задач.\n")
    lines.append("- Reactivity считается по `graph.json` (cite-рёбра между агентскими документами); "
                 "это message-level приближение, а не полный emergence-анализ с null-моделями.\n")
    lines.append("- Workstream-роли и SI-тренды в Variant A не считаются — это следующий шаг (Variant B).\n")

    lines.append("## 10. Файлы\n")
    lines.append("```\nanalysis/\n├── cross_task_metrics.csv\n├── cross_task_summary.json\n├── SUMMARY.md\n"
                 "├── tag_tta.csv\n├── tag_tta_by_task.csv\n└── figures/\n"
                 "    ├── fig01_outcome.png\n    ├── fig02_progress_curves.png\n"
                 "    ├── fig03_cost_volume_outcome.png\n    ├── fig04_coordination.png\n"
                 "    ├── fig05_graph_structure.png\n    ├── fig06_synergy_diffusion_recovery.png\n"
                 "    ├── fig07_correlations.png\n    └── fig08_tag_tta.png\n```\n")

    (args.out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"готово: {args.out}")
    print(f"  задач: {len(rows)}; ladders: {[round(r['ladder'], 2) for r in rows]}")
    print(f"  pooled reactivity {pooled['reactivity_pooled']['rate']:.3f}; "
          f"cross-agent {pooled['cross_edges_pooled']['rate']:.3f}")


if __name__ == "__main__":
    main()
