#!/usr/bin/env python3
"""Swarm analysis over the blackboard: synergy, knowledge graph, insight diffusion, resilience.

Usage:
  python3 paper/analysis/code/swarm_analysis.py \
      --board results/<experiment>/<task>/blackboard \
      --out   results/<experiment>/<task>/analysis/swarm \
      [--label <task>] [--permutations 10000]

Computes the directions from the research proposal:
  1) ΔM_link — metric gain relative to the best cited note
     (direct and transitive links) + ΔM_prev relative to the best before the message;
  2) knowledge graph: improvement-line depth, in-degree (citations), hubs;
  3) TTA — time/messages until the first citation of an insight (direct/transitive);
  4) resilience — problem events (#bug/#conflict/#problem) and their recovery.

Writes *.csv, summary.json, and PNG plots to --out.
Task metric and direction come from <run>/metrics.json (or info.json);
if missing, assumes "lower is better" (logloss) with a warning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

BIN = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN))
import blackboard_graph as bg  # noqa: E402

INSIGHT_TAGS = {"insight", "feature_engineering"}
RECOVERY_TAGS = {"insight", "submission", "solution"}
PROBLEM_TAGS = {"bug", "conflict", "problem"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, required=True, help="каталог blackboard прогона")
    ap.add_argument("--out", type=Path, required=True, help="куда писать csv/png")
    ap.add_argument("--label", default=None, help="имя задачи для заголовка (default: имя каталога)")
    ap.add_argument("--permutations", type=int, default=10000, help="перестановок для perm-теста")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------------------------------------------------- data loading


def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{digits}f}"


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def load_meta(run_dir: Path) -> tuple[dict, bool, str]:
    meta: dict = {}
    for name in ("metrics.json", "info.json"):
        p = run_dir / name
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for k, v in data.items():
                meta.setdefault(k, v)
    lower = meta.get("is_lower_better")
    if lower is None:
        lower = True
        print("WARN: is_lower_better не найден — считаю, что меньше лучше", file=sys.stderr)
    return meta, bool(lower), str(meta.get("metric") or "metric")


def load_messages(board: Path, lower: bool, metric_name: str) -> tuple[list[dict], dict]:
    files = sorted(board.glob("*.md"))
    names = {p.name for p in files}
    mtimes = bg.load_mtime_manifest(board)
    direction = "low" if lower else "high"
    msgs: list[dict] = []
    quality = {
        "unresolved_links": 0,
        "self_links": 0,
        "forward_links": 0,
        "multiple_metrics": 0,
        "without_metric": 0,
        "system_links": 0,
    }
    for p in files:
        text = p.read_text(errors="replace")
        name = p.name
        system = bool(bg.NAME_TS_RE.match(name))
        agent = "system" if system else name.split("_", 1)[0]
        ts = mtimes.get(name)
        if ts is None:
            ts = p.stat().st_mtime
        tags = {t.lower() for t in bg.TAG_RE.findall(text)}
        values = [m["v"] for m in bg.extract_metrics(text, label=metric_name, direction=direction)]
        if len(values) > 1:
            quality["multiple_metrics"] += 1
        if not values and not system:
            quality["without_metric"] += 1
        metric = (min(values) if lower else max(values)) if values else None
        links: list[str] = []
        for raw in bg.LINK_RE.findall(text):
            target = raw.strip()
            cands = [target] if target.endswith(".md") else [target + ".md", target]
            hit = next((c for c in cands if c in names), None)
            if hit is None:
                quality["unresolved_links"] += 1
            elif hit == name:
                quality["self_links"] += 1
            elif bg.NAME_TS_RE.match(hit):
                quality["system_links"] += 1
            else:
                links.append(hit)
        msgs.append({
            "name": name,
            "agent": agent,
            "system": system,
            "time": float(ts),
            "time_iso": fmt_ts(float(ts)),
            "tags": sorted(tags),
            "metric_values": [round(v, 6) for v in values],
            "metric": metric,
            "links": links,
        })
    msgs.sort(key=lambda m: (m["time"], m["name"]))
    for i, m in enumerate(msgs):
        m["idx_all"] = i
    agents = [m for m in msgs if not m["system"]]
    for i, m in enumerate(agents):
        m["idx"] = i
    name_to_msg = {m["name"]: m for m in agents}
    # link edges only between agent messages; links "into the future" are counted separately
    for m in agents:
        m["links"] = [n for n in m["links"] if n in name_to_msg]
        m["links_time_ok"] = [n for n in m["links"] if name_to_msg[n]["time"] <= m["time"]]
        quality["forward_links"] += len(m["links"]) - len(m["links_time_ok"])
    return msgs, quality


def build_graph(agents: list[dict], time_ok: bool = True) -> tuple[nx.DiGraph, set[tuple[str, str]]]:
    G = nx.DiGraph()
    G.add_nodes_from(m["name"] for m in agents)
    edges: set[tuple[str, str]] = set()
    for m in agents:
        for target in m["links_time_ok"] if time_ok else m["links"]:
            edges.add((m["name"], target))
    G.add_edges_from(edges)
    return G, edges


# ------------------------------------------------------------------- analyses


def lineage_depth(G: nx.DiGraph) -> tuple[dict[str, int], list[str], list[list[str]]]:
    """Length of the longest "message -> cited ancestors" chain.
    Returns depth, the longest path, and found cycles (SCC > 1)."""
    if nx.is_directed_acyclic_graph(G):
        depth = {n: 0 for n in G}
        for n in reversed(list(nx.topological_sort(G))):
            for s in G.successors(n):
                depth[n] = max(depth[n], depth[s] + 1)
        longest = nx.dag_longest_path(G) if G.number_of_edges() else []
        return depth, longest, []
    # cycles: reduce to condensation, compute depth per component
    C = nx.condensation(G)
    mapping = C.graph["mapping"]
    cdepth = {c: 0 for c in C}
    for c in reversed(list(nx.topological_sort(C))):
        for s in C.successors(c):
            cdepth[c] = max(cdepth[c], cdepth[s] + 1)
    depth = {n: cdepth[mapping[n]] for n in G}
    cycles = [sorted(s) for s in nx.strongly_connected_components(G) if len(s) > 1]
    return depth, [], cycles


def analyze_graph(agents: list[dict], G_full: nx.DiGraph, G_time: nx.DiGraph, out: Path) -> dict:
    depth, longest, cycles_time = lineage_depth(G_time)
    _, _, cycles_full = lineage_depth(G_full)
    indeg = dict(G_time.in_degree())
    outdeg = dict(G_time.out_degree())
    linked = [m for m in agents if m["links_time_ok"]]
    depths = [depth[n] for n in depth]
    submission = [m for m in agents if "submission" in m["tags"]]
    final_sub = submission[-1] if submission else None
    hubs = sorted(indeg.items(), key=lambda kv: -kv[1])[:7]
    by_name = {m["name"]: m for m in agents}

    rows = []
    for m in agents:
        rows.append({
            "name": m["name"],
            "agent": m["agent"],
            "time": m["time_iso"],
            "tags": " ".join(m["tags"]),
            "in_degree": indeg.get(m["name"], 0),
            "out_degree": outdeg.get(m["name"], 0),
            "depth": depth.get(m["name"], 0),
            "is_submission": int("submission" in m["tags"]),
        })
    write_csv(out / "graph_nodes.csv", rows, ["name", "agent", "time", "tags", "in_degree", "out_degree", "depth", "is_submission"])

    stats = {
        "nodes": G_time.number_of_nodes(),
        "edges": G_time.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(G_time),
        "cycles_full": cycles_full,
        "cycles_time": cycles_time,
        "max_depth": max(depths) if depths else 0,
        "mean_depth_all": statistics.mean(depths) if depths else 0.0,
        "mean_depth_linked": statistics.mean(depth[m["name"]] for m in linked) if linked else 0.0,
        "longest_path": longest,
        "max_indegree": max(indeg.values()) if indeg else 0,
        "hubs": [
            {
                "name": n,
                "agent": by_name[n]["agent"],
                "in_degree": d,
                "depth": depth.get(n, 0),
                "tags": by_name[n]["tags"],
                "time": by_name[n]["time_iso"],
            }
            for n, d in hubs
        ],
        "submission_count": len(submission),
        "submission_depths": sorted(depth.get(m["name"], 0) for m in submission),
        "final_submission": None,
    }
    if final_sub is not None:
        stats["final_submission"] = {
            "name": final_sub["name"],
            "time": final_sub["time_iso"],
            "depth": depth.get(final_sub["name"], 0),
            "in_degree": indeg.get(final_sub["name"], 0),
        }
    return stats


def improvement(ref: float | None, value: float | None, lower: bool) -> float | None:
    """A positive value means the result is better than ref."""
    if ref is None or value is None:
        return None
    return (ref - value) if lower else (value - ref)


def analyze_synergy(agents: list[dict], lower: bool, permutations: int, seed: int) -> tuple[dict, list[dict]]:
    by_name = {m["name"]: m for m in agents}
    best_before: float | None = None
    rows = []
    for m in agents:
        prev_best = best_before
        delta_prev = improvement(prev_best, m["metric"], lower)
        # direct links with a metric
        direct_vals = [by_name[t]["metric"] for t in m["links_time_ok"] if by_name[t]["metric"] is not None]
        ref_direct = (min(direct_vals) if lower else max(direct_vals)) if direct_vals else None
        # transitive ancestors
        anc: set[str] = set()
        frontier = list(m["links_time_ok"])
        while frontier:
            nxt: list[str] = []
            for n in frontier:
                if n in anc or n == m["name"]:
                    continue
                anc.add(n)
                nxt.extend(by_name[n]["links_time_ok"])
            frontier = nxt
        anc_vals = [by_name[n]["metric"] for n in anc if by_name[n]["metric"] is not None]
        ref_anc = (min(anc_vals) if lower else max(anc_vals)) if anc_vals else None
        row = {
            "name": m["name"],
            "agent": m["agent"],
            "time": m["time_iso"],
            "metric": m["metric"],
            "has_links": int(bool(m["links_time_ok"])),
            "n_links": len(m["links_time_ok"]),
            "n_ancestors": len(anc),
            "prev_best": prev_best,
            "delta_prev": delta_prev,
            "ref_direct": ref_direct,
            "delta_direct": improvement(ref_direct, m["metric"], lower),
            "ref_anc": ref_anc,
            "delta_anc": improvement(ref_anc, m["metric"], lower),
            "new_best": 0,
        }
        rows.append(row)
        if m["metric"] is not None:
            if best_before is None or improvement(best_before, m["metric"], lower) > 1e-12:
                row["new_best"] = 1
                best_before = m["metric"]

    def vals(key: str, pred=None) -> list[float]:
        return [r[key] for r in rows if r[key] is not None and (pred is None or pred(r))]

    def describe(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "median": None, "positive": None, "max": None}
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "positive": sum(1 for x in xs if x > 1e-12) / len(xs),
            "max": max(xs),
        }

    direct = vals("delta_direct")
    anc = vals("delta_anc")
    prev_linked = vals("delta_prev", lambda r: r["has_links"] == 1)
    prev_unlinked = vals("delta_prev", lambda r: r["has_links"] == 0)

    def perm_test(a: list[float], b: list[float]) -> dict | None:
        if len(a) < 2 or len(b) < 2:
            return None
        diff = statistics.mean(a) - statistics.mean(b)
        pool = list(a) + list(b)
        rng = random.Random(seed)
        hits = 0
        for _ in range(permutations):
            rng.shuffle(pool)
            d = statistics.mean(pool[: len(a)]) - statistics.mean(pool[len(a):])
            if abs(d) >= abs(diff) - 1e-15:
                hits += 1
        return {"diff": diff, "p": (hits + 1) / (permutations + 1)}

    stats = {
        "direct": describe(direct),
        "transitive": describe(anc),
        "prev_linked": describe(prev_linked),
        "prev_unlinked": describe(prev_unlinked),
        "perm_prev": perm_test(prev_linked, prev_unlinked),
    }
    return stats, rows


def adopters(event: str, citers: dict[str, set[str]], idx: dict[str, int], agents_by_name: dict[str, dict], direct: bool) -> str | None:
    seen = {event}
    frontier = {event}
    if direct:
        candidates = citers.get(event, set())
    else:
        while frontier:
            nxt: set[str] = set()
            for n in frontier:
                for c in citers.get(n, ()):
                    if c not in seen:
                        seen.add(c)
                        nxt.add(c)
            frontier = nxt
        candidates = seen - {event}
    later = [
        c for c in candidates
        if idx[c] > idx[event] and agents_by_name[c]["agent"] != agents_by_name[event]["agent"]
    ]
    return min(later, key=lambda c: idx[c]) if later else None


def analyze_tta(agents: list[dict], synergy_rows: list[dict], G_time: nx.DiGraph) -> tuple[dict, list[dict]]:
    by_name = {m["name"]: m for m in agents}
    idx = {m["name"]: m["idx"] for m in agents}
    citers: dict[str, set[str]] = defaultdict(set)
    for u, v in G_time.edges():
        citers[v].add(u)
    rows = {m["name"]: m for m in synergy_rows}
    strict, broad, all_best = [], [], []
    for m in agents:
        r = rows[m["name"]]
        tags = set(m["tags"])
        if m["metric"] is not None and r["new_best"]:
            all_best.append(m)
            if tags & INSIGHT_TAGS:
                strict.append(m)
        if m["metric"] is not None and tags & INSIGHT_TAGS:
            broad.append(m)
    sets = {"strict": strict, "broad": broad, "all_new_best": all_best}

    def measure(events: list[dict]) -> list[dict]:
        result = []
        for e in events:
            d = adopters(e["name"], citers, idx, by_name, direct=True)
            t = adopters(e["name"], citers, idx, by_name, direct=False)
            result.append({
                "event": e["name"],
                "agent": e["agent"],
                "time": e["time_iso"],
                "metric": e["metric"],
                "tags": " ".join(sorted(set(e["tags"]) & (INSIGHT_TAGS | {"submission"}))),
                "direct_adopter": d or "",
                "direct_msgs": (idx[d] - idx[e["name"]]) if d else None,
                "direct_seconds": (by_name[d]["time"] - e["time"]) if d else None,
                "trans_adopter": t or "",
                "trans_msgs": (idx[t] - idx[e["name"]]) if t else None,
                "trans_seconds": (by_name[t]["time"] - e["time"]) if t else None,
            })
        return result

    def describe(recs: list[dict], key: str) -> dict:
        vals = [r[key] for r in recs if r[key] is not None]
        return {
            "events": len(recs),
            "adopted": len(vals),
            "rate": (len(vals) / len(recs)) if recs else None,
            "median_msgs": statistics.median(vals) if vals else None,
            "mean_msgs": statistics.mean(vals) if vals else None,
            "median_seconds": statistics.median([r[key.replace("_msgs", "_seconds")] for r in recs if r[key] is not None]) if vals else None,
        }

    all_stats: dict[str, dict] = {}
    primary_rows: list[dict] = []
    for name, events in sets.items():
        recs = measure(events)
        all_stats[name] = {
            "n_events": len(events),
            "direct": describe(recs, "direct_msgs"),
            "transitive": describe(recs, "trans_msgs"),
        }
        if name == "broad":
            primary_rows = recs
    return all_stats, primary_rows


def analyze_recovery(agents: list[dict], G_time: nx.DiGraph) -> tuple[dict, list[dict]]:
    by_name = {m["name"]: m for m in agents}
    idx = {m["name"]: m["idx"] for m in agents}
    citers: dict[str, set[str]] = defaultdict(set)
    for u, v in G_time.edges():
        citers[v].add(u)
    problems = [m for m in agents if set(m["tags"]) & PROBLEM_TAGS]

    def find(problem: dict, direct: bool) -> str | None:
        seen = {problem["name"]}
        frontier = {problem["name"]}
        if direct:
            candidates = set(citers.get(problem["name"], set()))
        else:
            while frontier:
                nxt: set[str] = set()
                for n in frontier:
                    for c in citers.get(n, ()):
                        if c not in seen:
                            seen.add(c)
                            nxt.add(c)
                frontier = nxt
            candidates = seen - {problem["name"]}
        ok = [c for c in candidates if idx[c] > idx[problem["name"]] and set(by_name[c]["tags"]) & RECOVERY_TAGS]
        return min(ok, key=lambda c: idx[c]) if ok else None

    recs = []
    for p in problems:
        d = find(p, True)
        t = find(p, False)
        recs.append({
            "problem": p["name"],
            "agent": p["agent"],
            "time": p["time_iso"],
            "tags": " ".join(sorted(set(p["tags"]) & PROBLEM_TAGS)),
            "recovery": d or "",
            "recovery_agent": by_name[d]["agent"] if d else "",
            "same_author": int(bool(d) and by_name[d]["agent"] == p["agent"]),
            "msgs": (idx[d] - idx[p["name"]]) if d else None,
            "seconds": (by_name[d]["time"] - p["time"]) if d else None,
            "trans_recovery": t or "",
            "trans_msgs": (idx[t] - idx[p["name"]]) if t else None,
            "trans_seconds": (by_name[t]["time"] - p["time"]) if t else None,
        })

    def rate(key: str) -> dict:
        vals = [r for r in recs if r[key]]
        msgs = [r["msgs" if key == "recovery" else "trans_msgs"] for r in vals]
        secs = [r["seconds" if key == "recovery" else "trans_seconds"] for r in vals]
        return {
            "events": len(recs),
            "recovered": len(vals),
            "rate": (len(vals) / len(recs)) if recs else None,
            "median_msgs": statistics.median(msgs) if msgs else None,
            "median_seconds": statistics.median(secs) if secs else None,
            "different_author": sum(1 for r in vals if not r["same_author"]),
        }

    stats = {
        "problems": len(problems),
        "direct": rate("recovery"),
        "transitive": rate("trans_recovery"),
        "problem_authors": Counter(r["agent"] for r in recs),
        "recovery_authors": Counter(r["recovery_agent"] for r in recs if r["recovery"]),
    }
    return stats, recs


# ---------------------------------------------------------------------- plots


def plot_progress(agents: list[dict], lower: bool, score_final: float | None, out: Path) -> None:
    pts = [(m["time"], m["metric"], m["agent"]) for m in agents if m["metric"] is not None]
    if not pts:
        return
    t0 = min(m["time"] for m in agents)
    x = [(t - t0) / 3600 for t, _, _ in pts]
    best = []
    cur = None
    for _, v, _ in pts:
        if cur is None or ((v < cur) if lower else (v > cur)):
            cur = v
        best.append(cur)
    agents_order = sorted({a for _, _, a in pts})
    colors = {a: c for a, c in zip(agents_order, plt.cm.tab10.colors)}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(x, best, drawstyle="steps-post", color="#333333", lw=1.6, label="лучший результат")
    for i, (_, v, a) in enumerate(pts):
        is_new = i == 0 or (v < best[i - 1] - 1e-12 if lower else v > best[i - 1] + 1e-12)
        if is_new:
            ax.scatter([x[i]], [v], s=36, color=colors[a], zorder=3, edgecolor="white", linewidth=0.6)
    for a in agents_order:
        ax.scatter([], [], color=colors[a], label=a)
    if score_final is not None:
        ax.axhline(score_final, ls="--", color="#d62728", lw=1.2, label=f"финальный grade {score_final:.5f}")
    ax.set_xlabel("часы от первого сообщения")
    ax.set_ylabel("метрика (лучше " + ("меньше" if lower else "больше") + ")")
    ax.set_title("Прогресс лучшей метрики")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "metric_progress.png", dpi=150)
    plt.close(fig)


def tag_class(tags: list[str]) -> str:
    t = set(tags)
    if t & {"bug", "problem", "conflict"}:
        return "problem"
    if "insight" in t:
        return "insight"
    if "submission" in t:
        return "submission"
    if t & {"ensembling", "model_selection"}:
        return "ensembling"
    if t & {"status", "communication", "plan"}:
        return "status"
    return "other"


def plot_graph(agents: list[dict], G: nx.DiGraph, depth: dict[str, int], out: Path, label: str) -> None:
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G, seed=42, k=1.6 / math.sqrt(max(G.number_of_nodes(), 2)), iterations=120)
    indeg = dict(G.in_degree())
    sizes = [60 + 110 * indeg.get(n, 0) for n in G.nodes()]
    by_name = {m["name"]: m for m in agents}
    top = sorted(indeg.items(), key=lambda kv: -kv[1])[:5]
    label_nodes = {n for n, _ in top}
    agents_order = sorted({m["agent"] for m in agents})
    agent_colors = {a: c for a, c in zip(agents_order, plt.cm.tab10.colors)}
    class_colors = {
        "submission": "#d62728", "problem": "#ff7f0e", "insight": "#2ca02c",
        "ensembling": "#1f77b4", "status": "#8c8c8c", "other": "#bcbd22",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, mode in ((axes[0], "agent"), (axes[1], "tag")):
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=6, alpha=0.22, width=0.7)
        colors = [
            agent_colors[by_name[n]["agent"]] if mode == "agent" else class_colors[tag_class(by_name[n]["tags"])]
            for n in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, edgecolors="white", linewidths=0.6, alpha=0.95)
        nx.draw_networkx_labels(G, pos, ax=ax, labels={n: (n[:26] if n in label_nodes else "") for n in G.nodes()}, font_size=6)
        ax.set_title("по агентам" if mode == "agent" else "по тегам (submission/insight/problem/…)")
        ax.axis("off")
        ax.margins(0.06)
    for a in agents_order:
        axes[0].scatter([], [], color=agent_colors[a], label=a, s=30)
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    for cls, color in class_colors.items():
        axes[1].scatter([], [], color=color, label=cls, s=30)
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle(f"Граф знаний: {label} (размер узла = цитируемость)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "graph.png", dpi=150)
    plt.close(fig)

    # depth and in-degree distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(list(depth.values()), bins=range(0, max(depth.values(), default=0) + 2), color="#1f77b4", alpha=0.85)
    axes[0].set_title("Глубина линии улучшений (рёбра)")
    axes[0].set_xlabel("depth")
    axes[0].set_ylabel("сообщений")
    degs = [d for _, d in indeg.items()]
    axes[1].hist(degs, bins=range(0, max(degs, default=0) + 2), color="#ff7f0e", alpha=0.85)
    axes[1].set_title("In-degree (сколько раз цитируют)")
    axes[1].set_xlabel("входящих ссылок")
    fig.tight_layout()
    fig.savefig(out / "graph_depth_indegree.png", dpi=150)
    plt.close(fig)


def plot_synergy(rows: list[dict], out: Path) -> None:
    direct = [r["delta_direct"] for r in rows if r["delta_direct"] is not None]
    linked = [r["delta_prev"] for r in rows if r["delta_prev"] is not None and r["has_links"]]
    unlinked = [r["delta_prev"] for r in rows if r["delta_prev"] is not None and not r["has_links"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if direct:
        lo = min(direct + [0.0])
        hi = max(direct + [0.0])
        pad = (hi - lo) * 0.08 or 1e-3
        axes[0].hist(direct, bins=min(20, max(5, len(direct) // 2)), color="#2ca02c", alpha=0.85, range=(lo - pad, hi + pad))
        axes[0].axvline(0, color="#333", lw=1, ls="--")
        axes[0].set_title("ΔM по прямым ссылкам (>0 — лучше процитированного)")
        axes[0].set_xlabel("прирост метрики")
    else:
        axes[0].text(0.5, 0.5, "нет сообщений со ссылками и метрикой", ha="center", va="center", transform=axes[0].transAxes)
    data = [x for x in (linked, unlinked) if x]
    labels = [lab for lab, x in (("со ссылками", linked), ("без ссылок", unlinked)) if x]
    if data:
        bp = axes[1].boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#2ca02c", "#8c8c8c"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1].axhline(0, color="#333", lw=1, ls="--")
        axes[1].set_title("ΔM относительно лучшего до сообщения")
        axes[1].set_ylabel("прирост метрики")
    fig.tight_layout()
    fig.savefig(out / "synergy.png", dpi=150)
    plt.close(fig)


def plot_tta(recs: list[dict], out: Path) -> None:
    if not recs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, label, color in (("direct", "прямая ссылка", "#1f77b4"), ("trans", "транзитивная", "#2ca02c")):
        vals = sorted(r[f"{key}_msgs"] for r in recs if r[f"{key}_msgs"] is not None)
        if vals:
            y = np.arange(1, len(vals) + 1) / len(recs)
            axes[0].step(vals, y, where="post", label=label, color=color, lw=1.6)
        secs = sorted(r[f"{key}_seconds"] for r in recs if r[f"{key}_seconds"] is not None)
        if secs:
            y = np.arange(1, len(secs) + 1) / len(recs)
            axes[1].step([s / 60 for s in secs], y, where="post", label=label, color=color, lw=1.6)
    axes[0].set_title("CDF: TTA в сообщениях роя")
    axes[0].set_xlabel("сообщений до adoption")
    axes[0].set_ylabel("доля событий (из всех)")
    axes[1].set_title("CDF: TTA во времени")
    axes[1].set_xlabel("минут до adoption")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tta_cdf.png", dpi=150)
    plt.close(fig)


def plot_timeline(recs: list[dict], agents: list[dict], out: Path) -> None:
    if not recs:
        return
    t0 = min(m["time"] for m in agents)
    agent_names = sorted({m["agent"] for m in agents})
    y = {a: i for i, a in enumerate(agent_names)}
    fig, ax = plt.subplots(figsize=(9, max(2.6, 0.6 * len(agent_names) + 1.4)))
    for r in recs:
        t = (datetime.fromisoformat(r["time"]).timestamp() - t0) / 3600
        ax.scatter([t], [y[r["agent"]]], marker="X", s=90, color="#d62728", zorder=3)
        if r["recovery"] and r["seconds"] is not None:
            tr = t + r["seconds"] / 3600
            ax.plot([t, tr], [y[r["agent"]], y[r["recovery_agent"]]], color="#2ca02c", lw=1.1, alpha=0.7)
            ax.scatter([tr], [y[r["recovery_agent"]]], marker="o", s=60, color="#2ca02c", zorder=3)
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels(list(y.keys()))
    ax.set_xlabel("часы от первого сообщения")
    ax.set_title("Problem-события (X) и recovery-сообщения (●)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out / "recovery_timeline.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ io


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def write_report(path: Path, label: str, run_dir: Path, meta: dict, lower: bool, metric_name: str,
                 msgs: list[dict], quality: dict, graph_stats: dict, synergy: dict,
                 tta: dict, recovery: dict, recovery_rows: list[dict], agents: list[dict],
                 deadline: dict | None = None) -> None:
    agent_msgs = [m for m in msgs if not m["system"]]
    system_msgs = [m for m in msgs if m["system"]]
    # SYSTEM_grade appears on the host after pull, sometimes the next day —
    # do not include it in the run period
    fresh = [m for m in msgs if "_SYSTEM_grade" not in m["name"]]
    msg_lo = min((m["time"] for m in fresh), default=None)
    msg_hi = max((m["time"] for m in fresh), default=None)
    period = (
        (deadline or {}).get("start") or fmt_ts(msg_lo),
        (deadline or {}).get("hard_end") or fmt_ts(msg_hi),
    )
    tags = Counter(t for m in agent_msgs for t in m["tags"])
    best = [m for m in agent_msgs if m["metric"] is not None]
    first_metric = min((m["metric"] for m in best), key=lambda v: v if lower else -v) if best else None
    score = meta.get("score")
    thresholds = meta.get("thresholds") or {}
    medals = meta.get("flags") or {}

    L: list[str] = []
    L.append(f"# Анализ роя: {label}")
    L.append("")
    L.append(f"- **Прогон:** `{run_dir}`")
    L.append(f"- **Период (deadline):** {period[0]} — {period[1]}")
    if msg_hi is not None:
        L.append(f"- **Последнее сообщение на доске:** {fmt_ts(msg_hi)}")
    L.append(f"- **Агенты:** {len({m['agent'] for m in agent_msgs})} ({', '.join(sorted({m['agent'] for m in agent_msgs}))})")
    L.append(f"- **Сообщений:** {len(agent_msgs)} агентских + {len(system_msgs)} SYSTEM")
    L.append(f"- **Метрика:** `{metric_name}` ({'меньше лучше' if lower else 'больше лучше'}); финальный grade: **{fmt_num(score, 5)}**")
    if thresholds:
        L.append(f"- **Пороги:** gold {fmt_num(thresholds.get('gold'), 5)}, silver {fmt_num(thresholds.get('silver'), 5)}, "
                 f"bronze {fmt_num(thresholds.get('bronze'), 5)}, median {fmt_num(thresholds.get('median'), 5)}")
    if medals:
        L.append(f"- **Медали:** gold={medals.get('gold_medal')}, silver={medals.get('silver_medal')}, "
                 f"bronze={medals.get('bronze_medal')}, above_median={medals.get('above_median')}")
    L.append("")
    L.append("## Ключевые выводы")
    L.append("")
    d0 = synergy["direct"]
    g0 = graph_stats
    b0 = tta["broad"]
    r0 = recovery["direct"]
    syn_note = (
        "в среднем обгоняют лучший процитированный результат"
        if (d0["mean"] or 0) > 0
        else "в среднем не обгоняют лучший процитированный результат"
    )
    L.append(f"- Рой прошёл путь к финальному результату ({fmt_num(score, 5)}): **{g0['edges']} ссылок** между "
             f"{g0['nodes']} сообщениями, средняя глубина линии улучшений {fmt_num(g0['mean_depth_all'])}, "
             f"максимальная — {g0['max_depth']}.")
    if d0["n"]:
        L.append(f"- Синергия: {pct(d0['positive'])} сообщений с прямыми ссылками улучшают лучший из процитированных "
                 f"результатов; средний ΔM = {fmt_num(d0['mean'])} ({syn_note}).")
    else:
        L.append("- Синергия: недостаточно сообщений с метриками и прямыми ссылками для оценки.")
    tta_med = b0["direct"]["median_seconds"]
    L.append(f"- Диффузия: инсайты принимаются в {pct(b0['direct']['rate'])} случаев, медиана — "
             f"{fmt_num(b0['direct']['median_msgs'], 1)} сообщения / "
             f"{fmt_num(tta_med / 60, 1) if tta_med is not None else '—'} мин.")
    L.append(f"- Resilience: восстановление после {r0['recovered']} из {r0['events']} проблем "
             f"({pct(r0['rate'])}), медиана {fmt_num(r0['median_msgs'], 1)} сообщения; "
             f"{r0['different_author']} из {r0['recovered']} — силами другого агента.")
    L.append("- Ограничения: один прогон, малые N событий, метрика сообщения — лучшее из `<metric>`; "
             "выводы иллюстративные, не статистически значимые.")
    L.append("")
    L.append("## Данные")
    L.append("")
    L.append(f"- Ссылок (рёбер между сообщениями): **{graph_stats['edges']}**; неразрешённых: {quality['unresolved_links']}; "
             f"self: {quality['self_links']}; ссылок «в будущее»: {quality['forward_links']}")
    L.append(f"- Сообщений без `<metric>`: {quality['without_metric']}; сообщений с несколькими значениями метрики: "
             f"{quality['multiple_metrics']} (в расчёте берётся лучшее значение)")
    L.append(f"- Теги (топ): " + ", ".join(f"`#{t}`×{c}" for t, c in tags.most_common(10)))
    if graph_stats["cycles_time"]:
        L.append(f"- **Циклы в графе:** {graph_stats['cycles_time']} (учтены через конденсацию SCC)")
    elif graph_stats["cycles_full"]:
        L.append(f"- Циклы в полном графе: {graph_stats['cycles_full']}; после отбрасывания ссылок «в будущее» граф ацикличен")
    else:
        L.append("- Граф ацикличен (DAG)")
    L.append("")
    L.append("## Методика")
    L.append("")
    L.append("- Сообщение = `*.md` на доске; метрика сообщения = лучшее из тегов `<metric>` (min для logloss).")
    L.append("- Связи: `[[файл]]`; для анализа линий улучшений используются только ссылки на более ранние сообщения "
             f"(ссылок «в будущее» отброшено: {quality['forward_links']}).")
    dir_formula = "min(M_j) − M_k" if lower else "M_k − max(M_j)"
    L.append("- **ΔM_link** = улучшение метрики относительно лучшей метрики среди процитированных сообщений "
             f"(`{dir_formula}`; положительное = лучше). Транзитивный вариант — относительно лучшей "
             "метрики среди всех предков по ссылкам.")
    L.append("- **ΔM_prev** = улучшение относительно лучшего результата на доске до этого сообщения "
             "(сравнение сообщений со ссылками и без).")
    L.append("- **Глубина** = длина самой длинной цепочки «сообщение → процитированные предки»; "
             "**in-degree** = сколько раз сообщение цитируют.")
    L.append("- **TTA** = от сообщения-инсайта до первого сообщения другого агента, цитирующего его "
             "(прямо или транзитивно); считаются минуты и число сообщений роя.")
    L.append("- **Problem events** = сообщения с тегами `#bug`/`#conflict`/`#problem`; "
             "**recovery** = последующее сообщение с `#insight`/`#submission`/`#solution`, ссылающееся на проблему.")
    L.append("")
    L.append("## 1. Синергия (ΔM)")
    L.append("")
    d, t = synergy["direct"], synergy["transitive"]
    L.append("| вариант | N | среднее | медиана | доля улучшений | максимум |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| прямые ссылки | {d['n']} | {fmt_num(d['mean'])} | {fmt_num(d['median'])} | {pct(d['positive'])} | {fmt_num(d['max'])} |")
    L.append(f"| транзитивные предки | {t['n']} | {fmt_num(t['mean'])} | {fmt_num(t['median'])} | {pct(t['positive'])} | {fmt_num(t['max'])} |")
    L.append("")
    pl, pu, pm = synergy["prev_linked"], synergy["prev_unlinked"], synergy["perm_prev"]
    L.append(f"- Сообщения **со ссылками** (N={pl['n']}): средний ΔM_prev = {fmt_num(pl['mean'])}, медиана {fmt_num(pl['median'])}.")
    L.append(f"- **Без ссылок** (N={pu['n']}): средний ΔM_prev = {fmt_num(pu['mean'])}, медиана {fmt_num(pu['median'])}.")
    if pm:
        L.append(f"- Перестановочный тест разницы средних: разница {fmt_num(pm['diff'])}, p ≈ {pm['p']:.3f}.")
    L.append(f"- Интерпретация: {synergy_text(synergy, lower)}")
    L.append("")
    L.append("## 2. Граф знаний")
    L.append("")
    L.append(f"- Узлов: {graph_stats['nodes']}, рёбер: {graph_stats['edges']}, DAG: {graph_stats['is_dag']}")
    L.append(f"- Максимальная глубина линии улучшений: **{graph_stats['max_depth']}**; "
             f"средняя по всем: {fmt_num(graph_stats['mean_depth_all'])}; по сообщениям со ссылками: {fmt_num(graph_stats['mean_depth_linked'])}")
    if graph_stats["longest_path"]:
        L.append(f"- Самая длинная цепочка: " + " → ".join(graph_stats["longest_path"]))
    if graph_stats["final_submission"]:
        fs = graph_stats["final_submission"]
        L.append(f"- Финальный submission: `{fs['name']}` (depth {fs['depth']}, in-degree {fs['in_degree']})")
    L.append(f"- Максимальная цитируемость: {graph_stats['max_indegree']}")
    L.append("")
    L.append("| hub | агент | in-degree | depth | теги |")
    L.append("|---|---|---|---|---|")
    for h in graph_stats["hubs"]:
        L.append(f"| `{h['name']}` | {h['agent']} | {h['in_degree']} | {h['depth']} | {', '.join('#' + t for t in h['tags'][:5])} |")
    L.append("")
    L.append(f"- Интерпретация: {graph_text(graph_stats)}")
    L.append("")
    L.append("## 3. Диффузия инсайтов (TTA)")
    L.append("")
    L.append("| набор событий | N | adopted (прям.) | медиана, сообщений | adopted (транз.) | медиана, сообщений |")
    L.append("|---|---|---|---|---|---|")
    for name, label_ in (("broad", "тег `#insight`/`#feature_engineering` + метрика"),
                         ("strict", "+ новая лучшая метрика"),
                         ("all_new_best", "любая новая лучшая метрика")):
        s = tta[name]
        L.append(f"| {label_} | {s['n_events']} | {s['direct']['adopted']} ({pct(s['direct']['rate'])}) | "
                 f"{fmt_num(s['direct']['median_msgs'], 1)} | {s['transitive']['adopted']} ({pct(s['transitive']['rate'])}) | "
                 f"{fmt_num(s['transitive']['median_msgs'], 1)} |")
    L.append("")
    b = tta["broad"]
    if b["direct"]["median_seconds"] is not None:
        L.append(f"- Для основного набора (broad): медиана TTA ≈ {fmt_num(b['direct']['median_seconds'] / 60, 1)} мин / "
                 f"{fmt_num(b['direct']['median_msgs'], 1)} сообщений (прямые ссылки).")
    L.append(f"- Интерпретация: {tta_text(tta)}")
    L.append("")
    L.append("## 4. Resilience и разрешение конфликтов")
    L.append("")
    L.append(f"- Problem-событий: **{recovery['problems']}**")
    rd, rt = recovery["direct"], recovery["transitive"]
    L.append(f"- Восстановлено прямыми ссылками: {rd['recovered']}/{rd['events']} ({pct(rd['rate'])}), "
             f"медиана {fmt_num(rd['median_msgs'], 1)} сообщений; транзитивными: {rt['recovered']}/{rt['events']} ({pct(rt['rate'])})")
    if rd["median_seconds"] is not None:
        L.append(f"- Медианное время до recovery: {fmt_num(rd['median_seconds'] / 60, 1)} мин")
    L.append("")
    L.append("| проблема | автор | recovery | автор recovery | сообщений | минут |")
    L.append("|---|---|---|---|---|---|")
    for r in recovery_rows:
        L.append(f"| `{r['problem']}` | {r['agent']} | {('`' + r['recovery'] + '`') if r['recovery'] else '—'} | "
                 f"{r['recovery_agent'] or '—'} | {r['msgs'] if r['msgs'] is not None else '—'} | "
                 f"{fmt_num((r['seconds'] or 0) / 60, 1) if r['seconds'] is not None else '—'} |")
    L.append("")
    L.append(f"- Интерпретация: {recovery_text(recovery)}")
    L.append("")
    L.append("## Ограничения")
    L.append("")
    L.append("- Один прогон одной задачи: выводы иллюстративные, не статистически значимые (малые N событий).")
    L.append("- Метрика сообщения — лучшее из нескольких `<metric>` в тексте; часть сообщений (планы, статусы) без метрик.")
    L.append("- Времена берутся из `blackboard/.mtime.tsv` (mtime), а не из содержания сообщений.")
    L.append("- Транзитивные варианты могут завышать adoption: цепочка ссылок длины >1 засчитывается как принятие идеи.")
    L.append("")
    L.append("## Файлы")
    L.append("")
    for f in ("messages.csv", "edges.csv", "synergy.csv", "graph_nodes.csv", "tta.csv", "recovery.csv",
              "summary.json", "metric_progress.png", "graph.png", "graph_depth_indegree.png",
              "synergy.png", "tta_cdf.png", "recovery_timeline.png"):
        L.append(f"- `{f}`")
    L.append("")
    L.append("## Воспроизвести")
    L.append("")
    L.append("```bash")
    L.append(f"python3 paper/analysis/code/swarm_analysis.py --board {run_dir / 'blackboard'} --out {path.parent} --label {label}")
    L.append("```")
    L.append("")



def synergy_text(s: dict, lower: bool) -> str:
    d, pl, pu = s["direct"], s["prev_linked"], s["prev_unlinked"]
    parts = []
    if d["n"]:
        parts.append(f"сообщения с прямыми ссылками дают средний прирост {fmt_num(d['mean'])} "
                     f"(улучшение в {pct(d['positive'])} случаев)")
    if pl["n"] and pu["n"]:
        better = "выше" if (pl["mean"] or 0) > (pu["mean"] or 0) else "ниже"
        parts.append(f"средний ΔM_prev у сообщений со ссылками {better}, чем без ссылок "
                     f"({fmt_num(pl['mean'])} против {fmt_num(pu['mean'])})")
    if s["perm_prev"] and s["perm_prev"]["p"] < 0.05:
        parts.append("разница статистически значима на уровне 5%")
    elif s["perm_prev"]:
        parts.append("разница не значима (малая выборка)")
    return "; ".join(parts) if parts else "недостаточно сообщений с метриками и ссылками"


def graph_text(g: dict) -> str:
    hubs = ", ".join(f"`{h['name']}` ({h['in_degree']})" for h in g["hubs"][:3])
    tags = Counter(t for h in g["hubs"][:3] for t in h["tags"])
    tag_txt = ", ".join(f"#{t}" for t, _ in tags.most_common(3))
    return (f"рой строит цепочки глубиной до {g['max_depth']}; самые цитируемые сообщения — {hubs} "
            f"(теги {tag_txt}); это точки, на которых сходится работа агентов")


def tta_text(t: dict) -> str:
    b = t["broad"]
    if not b["direct"]["adopted"]:
        return "adoption не обнаружен"
    return (f"инсайты принимаются в {pct(b['direct']['rate'])} случаев; медиана — "
            f"{fmt_num(b['direct']['median_msgs'], 1)} сообщений, транзитивно — "
            f"{fmt_num(b['transitive']['median_msgs'], 1)}; малый N требует осторожной трактовки")


def recovery_text(r: dict) -> str:
    d = r["direct"]
    if not d["events"]:
        return "problem-события не найдены"
    return (f"рой восстанавливается после {d['recovered']} из {d['events']} проблем "
            f"(в {d['different_author']} случаях — другой агент), медиана {fmt_num(d['median_msgs'], 1)} сообщений; "
            f"часть проблем остаётся без явного ответа — это ограничение анализа")


# ---------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    board = args.board.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_dir = board.parent
    label = args.label or run_dir.name
    meta, lower, metric_name = load_meta(run_dir)
    msgs, quality = load_messages(board, lower, metric_name)
    agents = [m for m in msgs if not m["system"]]
    if not agents:
        raise SystemExit(f"на доске нет сообщений агентов: {board}")

    G_full, _ = build_graph(agents, time_ok=False)
    G_time, edges = build_graph(agents, time_ok=True)
    graph_stats = analyze_graph(agents, G_full, G_time, out)
    depth, _, _ = lineage_depth(G_time)

    synergy, synergy_rows = analyze_synergy(agents, lower, args.permutations, args.seed)
    tta, tta_rows = analyze_tta(agents, synergy_rows, G_time)
    recovery, recovery_rows = analyze_recovery(agents, G_time)

    # CSV
    write_csv(out / "messages.csv",
              [{
                  "name": m["name"], "agent": m["agent"], "kind": "system" if m["system"] else "agent",
                  "time": m["time_iso"], "tags": " ".join(m["tags"]),
                  "metric_values": " ".join(str(v) for v in m["metric_values"]),
                  "metric_best": m["metric"],
                  "links": " ".join(m["links"]),
                  "links_time_ok": " ".join(m.get("links_time_ok", [])),
              } for m in msgs],
              ["name", "agent", "kind", "time", "tags", "metric_values", "metric_best", "links", "links_time_ok"])
    write_csv(out / "edges.csv",
              [{"src": u, "tgt": v} for u, v in sorted(edges)],
              ["src", "tgt"])
    write_csv(out / "synergy.csv", synergy_rows, list(synergy_rows[0].keys()))
    write_csv(out / "tta.csv", tta_rows, list(tta_rows[0].keys()) if tta_rows else ["event"])
    write_csv(out / "recovery.csv", recovery_rows, list(recovery_rows[0].keys()) if recovery_rows else ["problem"])

    summary = {
        "run": str(run_dir),
        "label": label,
        "metric": metric_name,
        "is_lower_better": lower,
        "final_score": meta.get("score"),
        "thresholds": meta.get("thresholds"),
        "messages": {"agent": len(agents), "system": len(msgs) - len(agents)},
        "quality": quality,
        "graph": graph_stats,
        "synergy": synergy,
        "tta": tta,
        "recovery": {**recovery, "problem_authors": dict(recovery["problem_authors"]),
                     "recovery_authors": dict(recovery["recovery_authors"])},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # plots
    plot_progress(agents, lower, meta.get("score"), out)
    plot_graph(agents, G_time, depth, out, label)
    plot_synergy(synergy_rows, out)
    plot_tta(tta_rows, out)
    plot_timeline(recovery_rows, agents, out)

    deadline = {}
    dp = board / "deadline.json"
    if dp.is_file():
        try:
            deadline = json.loads(dp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            deadline = {}
    print(f"готово: {out}")
    print(f"  сообщений: {len(agents)} агентских, {len(msgs) - len(agents)} SYSTEM")
    print(f"  граф: depth_max={graph_stats['max_depth']}, edges={graph_stats['edges']}, hubs={len(graph_stats['hubs'])}")
    print(f"  синергия: N_direct={synergy['direct']['n']}, mean={fmt_num(synergy['direct']['mean'])}")
    print(f"  TTA(broad): adopted {tta['broad']['direct']['adopted']}/{tta['broad']['n_events']}")
    print(f"  recovery: {recovery['direct']['recovered']}/{recovery['problems']}")


if __name__ == "__main__":
    main()
