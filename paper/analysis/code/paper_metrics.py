#!/usr/bin/env python3
"""Metrics for the paper: coordination, roles, improvements, internet, resources.

Writes CSVs to paper/analysis/data and caches LLM role labeling.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from scipy.stats import spearmanr


ROLES = ["explorer", "producer", "integrator", "validator", "coordinator", "communicator"]
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


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "1-lie=results/1-lie10-blackboard-mlebench",
        "2-lite10=results/2-lite10-blackboard-mlebench",
    ])
    ap.add_argument("--out", type=Path, default=Path("paper/analysis"))
    ap.add_argument("--auth", type=Path, default=Path.home() / ".local/share/opencode/auth.json")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_task(d: Path) -> dict:
    msgs = []
    with open(d / "analysis" / "swarm" / "messages.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "agent":
                continue
            msgs.append({
                "task": d.name, "name": r["name"], "agent": r["agent"], "time": iso(r["time"]),
                "tags": [t for t in r["tags"].split() if t],
                "text": (d / "blackboard" / r["name"]).read_text(encoding="utf-8", errors="ignore"),
            })
    msgs.sort(key=lambda x: x["time"])
    graph = json.loads((d / "analysis" / "graph.json").read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in msgs}
    edges = [(e["s"], e["t"]) for e in graph.get("edges", [])
             if e.get("kind") == "cite" and e.get("s") in by_name and e.get("t") in by_name
             and by_name[e["s"]]["time"] > by_name[e["t"]]["time"]]
    grade = json.loads((d / "analysis" / "grade.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "analysis" / "swarm" / "summary.json").read_text(encoding="utf-8"))
    return {"task": d.name, "short": SHORT.get(d.name, d.name), "msgs": msgs,
            "edges": edges, "grade": grade, "summary": summary}


def coordination(tasks: list[dict]) -> list[dict]:
    rows = []
    for t in tasks:
        msgs = t["msgs"]; by = {m["name"]: m for m in msgs}
        edges = [(s, dst) for s, dst in t["edges"] if s in by and dst in by]
        out = Counter(); inc = Counter(); cross_out = Counter(); self_edges = 0
        for s, dst in edges:
            out[s] += 1; inc[dst] += 1
            if by[s]["agent"] != by[dst]["agent"]:
                cross_out[s] += 1
            if s == dst:
                self_edges += 1
        cite_any = sum(1 for m in msgs if out[m["name"]] > 0) / len(msgs)
        cross_any = sum(1 for m in msgs if cross_out[m["name"]] > 0) / len(msgs)
        cited_any = sum(1 for m in msgs if inc[m["name"]] > 0) / len(msgs)
        cross_edges = sum(1 for s, dst in edges if by[s]["agent"] != by[dst]["agent"])
        # chain span in time: earliest ancestor -> node (fixed-point relaxation, cycles safe)
        anc = {n: by[n]["time"] for n in by}
        for _ in range(len(by)):
            changed = False
            for s, dst in edges:
                if s in by and dst in by and anc[dst] < anc[s]:
                    anc[s] = anc[dst]; changed = True
            if not changed:
                break
        spans = [(by[n]["time"] - anc[n]).total_seconds() / 3600 for n in by]
        lags = [(by[s]["time"] - by[dst]["time"]).total_seconds() / 3600 for s, dst in edges]
        duration_h = max((max(by[n]["time"] for n in by) - min(by[n]["time"] for n in by)).total_seconds() / 3600, 1e-6)
        def _ci(vals):
            if len(vals) < 2:
                return (None, None)
            m = statistics.mean(vals); se = statistics.stdev(vals) / math.sqrt(len(vals))
            return (m - 1.96 * se, m + 1.96 * se)
        span_mean_ci = _ci(spans)
        depth_by = {n: 0 for n in by}
        for _ in range(len(by)):
            changed = False
            for s, dst in edges:
                if s in depth_by and dst in depth_by and depth_by[dst] + 1 > depth_by[s]:
                    depth_by[s] = depth_by[dst] + 1; changed = True
            if not changed:
                break
        depths = [float(x) for x in depth_by.values()]
        def mean_ci(vals):
            if len(vals) < 2:
                return (None, None)
            m = statistics.mean(vals); se = statistics.stdev(vals) / math.sqrt(len(vals))
            return (m - 1.96 * se, m + 1.96 * se)
        d_mean_ci = mean_ci(depths); l_mean_ci = mean_ci(lags)
        longest_path = t["summary"]["graph"].get("longest_path", [])
        path_span = 0.0
        if longest_path and all(n in by for n in longest_path):
            path_span = (by[longest_path[0]]["time"] - by[longest_path[-1]]["time"]).total_seconds() / 3600
        rows.append({
            "task": t["task"], "short": t["short"], "messages": len(msgs),
            "messages_per_hour": len(msgs) / duration_h,
            "agents": len({m["agent"] for m in msgs}), "edges": len(edges),
            "cross_edges": cross_edges, "same_author_edges": len(edges) - cross_edges,
            "same_author_edge_share": (len(edges) - cross_edges) / len(edges) if edges else None,
            "self_edges": self_edges,
            "self_edge_share": self_edges / len(edges) if edges else None,
            "cross_edge_share": cross_edges / len(edges) if edges else None,
            "cite_any_share": cite_any, "cross_agent_cite_share": cross_any,
            "cited_any_share": cited_any,
            "mean_depth_all": t["summary"]["graph"].get("mean_depth_all"),
            "mean_depth_linked": t["summary"]["graph"].get("mean_depth_linked"),
            "max_depth": t["summary"]["graph"].get("max_depth"),
            "depth_median": statistics.median(depths) if depths else None,
            "depth_q1": float(np.percentile(depths, 25)) if depths else None,
            "depth_q3": float(np.percentile(depths, 75)) if depths else None,
            "depth_mean_ci_low": d_mean_ci[0], "depth_mean_ci_high": d_mean_ci[1],
            "lag_q1": float(np.percentile(lags, 25)) if lags else None,
            "lag_q3": float(np.percentile(lags, 75)) if lags else None,
            "lag_mean_ci_low": l_mean_ci[0], "lag_mean_ci_high": l_mean_ci[1],
            "longest_path_len": len(longest_path),
            "mean_edge_lag_h": statistics.mean(lags) if lags else None,
            "median_edge_lag_h": statistics.median(lags) if lags else None,
            "mean_chain_span_h": statistics.mean(spans) if spans else None,
            "median_chain_span_h": statistics.median(spans) if spans else None,
            "chain_span_q1": float(np.percentile(spans, 25)) if spans else None,
            "chain_span_q3": float(np.percentile(spans, 75)) if spans else None,
            "chain_span_mean_ci_low": span_mean_ci[0], "chain_span_mean_ci_high": span_mean_ci[1],
            "max_chain_span_h": max(spans) if spans else None,
            "longest_path_span_h": path_span,
        })
    return rows


def citation_lags(tasks: list[dict]) -> list[dict]:
    rows = []
    for t in tasks:
        by = {m["name"]: m for m in t["msgs"]}
        if not by:
            continue
        t0 = min(m["time"] for m in t["msgs"])
        for src, dst in t["edges"]:
            if src not in by or dst not in by:
                continue
            citing, cited = by[src], by[dst]
            lag_h = (citing["time"] - cited["time"]).total_seconds() / 3600
            if lag_h <= 0:
                continue
            rows.append({
                "task": t["task"], "short": t["short"],
                "citing": src, "cited": dst,
                "citing_h": (citing["time"] - t0).total_seconds() / 3600,
                "cited_h": (cited["time"] - t0).total_seconds() / 3600,
                "lag_h": lag_h,
                "citing_agent": citing["agent"], "cited_agent": cited["agent"],
                "citing_tags": " ".join(citing["tags"]), "cited_tags": " ".join(cited["tags"]),
            })
    return rows


def tag_citations(tasks: list[dict]) -> list[dict]:
    agg = defaultdict(lambda: {"messages": 0, "cited_any": 0, "cited_peer": 0,
                               "cites_any": 0, "cites_peer": 0})
    for t in tasks:
        by = {m["name"]: m for m in t["msgs"]}
        inc_any = Counter(); inc_peer = Counter(); inc_same = Counter()
        out_any = Counter(); out_peer = Counter(); out_same = Counter()
        for s, dst in t["edges"]:
            if s not in by or dst not in by:
                continue
            inc_any[dst] += 1; out_any[s] += 1
            if by[s]["agent"] != by[dst]["agent"]:
                inc_peer[dst] += 1; out_peer[s] += 1
            else:
                inc_same[dst] += 1; out_same[s] += 1
        for m in t["msgs"]:
            tags = set(m["tags"])
            for tag in tags:
                a = agg[tag]; a["messages"] += 1
                a["cited_any"] += inc_any[m["name"]] > 0
                a["cited_peer"] += inc_peer[m["name"]] > 0
                a["cites_any"] += out_any[m["name"]] > 0
                a["cites_peer"] += out_peer[m["name"]] > 0
                if "cited_same" not in a:
                    a["cited_same"] = 0; a["cites_same"] = 0
                a["cited_same"] += inc_same[m["name"]] > 0
                a["cites_same"] += out_same[m["name"]] > 0
    out = []
    for tag, a in agg.items():
        n = a["messages"]
        out.append({"tag": tag, "messages": n,
                    "cited_any_rate": a["cited_any"] / n,
                    "cited_peer_rate": a["cited_peer"] / n,
                    "cites_any_rate": a["cites_any"] / n,
                    "cites_peer_rate": a["cites_peer"] / n,
                    "cited_peer_k": a["cited_peer"],
                    "cited_same_rate": a.get("cited_same", 0) / n,
                    "cites_same_rate": a.get("cites_same", 0) / n})
    out.sort(key=lambda x: (-x["messages"], x["tag"]))
    return out


def rolling(tasks: list[dict], window: int = 8, bins: int = 8) -> tuple[list[dict], list[dict]]:
    spec_rows = []
    div_raw = defaultdict(list)
    for t in tasks:
        t0 = min(m["time"] for m in t["msgs"])
        t1 = max(m["time"] for m in t["msgs"])
        span = max((t1 - t0).total_seconds(), 1)
        per_agent = defaultdict(list)
        for m in t["msgs"]:
            per_agent[m["agent"]].append(m)
        dists = defaultdict(list)  # (bin, agent) -> [(x,p)]
        for agent, ms in per_agent.items():
            ms.sort(key=lambda x: x["time"])
            for i in range(0, max(1, len(ms) - window + 1)):
                win = ms[i:i + window]
                if len(win) < 3:
                    continue
                cnt = Counter()
                for m in win:
                    tags = set(m["tags"])
                    if tags:
                        for tag in tags:
                            cnt[tag] += 1 / len(tags)
                total = sum(cnt.values())
                if total <= 0:
                    continue
                p = {k: v / total for k, v in cnt.items()}
                k = len(p)
                h = -sum(v * math.log(v) for v in p.values())
                spec = 1 - h / math.log(k) if k > 1 else 1.0
                x = statistics.mean((m["time"] - t0).total_seconds() / span for m in win)
                b = min(int(x * bins), bins - 1)
                spec_rows.append({"task": t["task"], "short": t["short"], "agent": agent,
                                  "bin": b, "x": x, "specialization": spec})
                dists[(b, agent)].append(p)
        for b in range(bins):
            agents = [a for (bb, a) in dists if bb == b]
            vals = []
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    p = np.mean([d for d in dists[(b, agents[i])]], axis=0) if False else None
            # pairwise JSD using mean distribution per agent in bin
            ad = {}
            for a in agents:
                keys = set().union(*[set(d) for d in dists[(b, a)]])
                vec = {k: statistics.mean(d.get(k, 0) for d in dists[(b, a)]) for k in keys}
                ad[a] = vec
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    p, q = ad[agents[i]], ad[agents[j]]
                    keys = set(p) | set(q)
                    pv = np.array([p.get(k, 0) for k in keys]); qv = np.array([q.get(k, 0) for k in keys])
                    m = (pv + qv) / 2
                    def kl(a, b):
                        mask = a > 0
                        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
                    vals.append(0.5 * kl(pv, m) + 0.5 * kl(qv, m))
            if vals:
                div_raw[b].append(statistics.mean(vals))
    spec_agg = []
    for b in range(bins):
        vals = [r["specialization"] for r in spec_rows if r["bin"] == b]
        spec_agg.append({"bin": b, "mean": statistics.mean(vals) if vals else None,
                         "q1": float(np.percentile(vals, 25)) if vals else None,
                         "q3": float(np.percentile(vals, 75)) if vals else None, "n": len(vals)})
    div_agg = []
    for b in range(bins):
        vals = div_raw.get(b, [])
        div_agg.append({"bin": b, "mean": statistics.mean(vals) if vals else None,
                        "q1": float(np.percentile(vals, 25)) if vals else None,
                        "q3": float(np.percentile(vals, 75)) if vals else None, "n": len(vals)})
    return spec_agg, div_agg


def improvement(tasks: list[dict], root: Path) -> list[dict]:
    rows = []
    for t in tasks:
        rows_csv = list(csv.DictReader(open(root / t["task"] / "analysis" / "metrics.csv")))
        lower = t["grade"]["is_lower_better"]
        vals = [(iso(r["time"]), r["file"], r["author"], float(r["value"])) for r in rows_csv if r["author"] != "system"]
        vals.sort()
        best = None; best_files = set(); metric_files = set()
        for tm, f, a, v in vals:
            metric_files.add(f)
            better = best is None or (v < best if lower else v > best)
            if better:
                best = v
        final_best = best
        first_best_time = None
        for tm, f, a, v in vals:
            if abs(v - final_best) < 1e-12 and first_best_time is None:
                first_best_time = tm
        new_best_files = set(); cur = None
        for tm, f, a, v in vals:
            better = cur is None or (v < cur if lower else v > cur)
            if better:
                cur = v; new_best_files.add(f)
        t0 = vals[0][0] if vals else min(m["time"] for m in t["msgs"])
        t1 = vals[-1][0] if vals else max(m["time"] for m in t["msgs"])
        span = max((t1 - t0).total_seconds(), 1)
        best_frac = (first_best_time - t0).total_seconds() / span if first_best_time else None
        sub = root / t["task"] / "final_solution" / "submission.csv"
        sub_frac = None; sub_h = None
        if sub.exists():
            mt = datetime.fromtimestamp(sub.stat().st_mtime, tz=t0.tzinfo)
            sub_frac = (mt - t0).total_seconds() / span
            sub_h = (mt - t0).total_seconds() / 3600
        rows.append({
            "task": t["task"], "short": t["short"],
            "metric_messages": len(metric_files), "new_best_messages": len(new_best_files),
            "new_best_share": len(new_best_files) / len(metric_files) if metric_files else None,
            "improve_cited_share": t["summary"]["synergy"]["direct"]["positive"],
            "synergy_n": t["summary"]["synergy"]["direct"]["n"],
            "improve_cited_k": (round((t["summary"]["synergy"]["direct"]["positive"] or 0)
                                      * t["summary"]["synergy"]["direct"]["n"])
                                if t["summary"]["synergy"]["direct"]["n"] else 0),
            "best_time_h": (first_best_time - t0).total_seconds() / 3600 if first_best_time else None,
            "best_time_frac": best_frac,
            "submission_time_h": sub_h, "submission_time_frac": sub_frac,
        })
    return rows


def improvement_second(root: Path) -> list[dict]:
    rows = []
    for d in sorted(root.iterdir()):
        mp = d / "analysis" / "metrics.csv"
        gp = d / "analysis" / "grade.json"
        if not mp.exists() or not gp.exists():
            continue
        grade = json.loads(gp.read_text(encoding="utf-8"))
        lower = grade["is_lower_better"]
        vals = []
        with open(mp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["author"] == "system":
                    continue
                try:
                    vals.append((iso(r["time"]), float(r["value"])))
                except ValueError:
                    pass
        if not vals:
            continue
        vals.sort()
        best = min(v for _, v in vals) if lower else max(v for _, v in vals)
        first_best = next((t for t, v in vals if abs(v - best) < 1e-12), None)
        t0 = vals[0][0]
        rows.append({
            "task": d.name, "short": SHORT.get(d.name, d.name),
            "best_time_h": (first_best - t0).total_seconds() / 3600 if first_best else None,
        })
    return rows


def internet_effect(tasks: list[dict], root: Path) -> tuple[list[dict], list[dict]]:
    base = root / "analysis"
    it = list(csv.DictReader(open(base / "internet_by_task.csv")))
    ia = list(csv.DictReader(open(base / "internet_by_agent.csv")))
    cmp_path = base / "cross_task_metrics.csv"
    cm = {}
    if cmp_path.exists():
        cm = {r["task"]: r for r in csv.DictReader(open(cmp_path))}
    rows = []
    for r in it:
        c = cm.get(r["task"], {})
        t = next((x for x in tasks if x["task"] == r["task"]), None)
        ladder = float(c.get("ladder") or 0)
        medal = c.get("medal", "")
        if t and not c:
            g = t["grade"]; th = g["thresholds"]; lower = g["is_lower_better"]; sc = g["score"]
            pts = [th["median"], th["bronze"], th["silver"], th["gold"]]
            pts = sorted(pts, reverse=not lower)
            ladder = 0.0
            for i in range(3):
                a, b = pts[i], pts[i+1]
                inside = (a >= sc >= b) if lower else (a <= sc <= b)
                if inside:
                    ladder = i + ((a - sc) / (a - b) if lower else (sc - a) / (b - a))
                    break
            else:
                ladder = 3.0 + ((pts[-1] - sc) / abs(pts[-1] - pts[-2]) if lower
                                else (sc - pts[-1]) / abs(pts[-1] - pts[-2]))
                ladder = min(4.0, ladder)
            medal = ("gold" if g["flags"].get("gold_medal") else "silver"
                     if g["flags"].get("silver_medal") else "bronze"
                     if g["flags"].get("bronze_medal") else "above_median"
                     if g["flags"].get("above_median") else "below_median")
        rows.append({"task": r["task"], "short": r["short"], "internet_any": r["internet_any"] == "True",
                     "webfetch_calls": int(r["webfetch_calls"]), "weight_gb": float(r["weight_gb"]),
                     "ladder": ladder, "medal": medal})
    agent_rows = []
    for r in ia:
        t = next((x for x in tasks if x["task"] == r["task"]), None)
        if not t:
            continue
        lower = t["grade"]["is_lower_better"]
        vals = [(x["author"], float(x["value"])) for x in csv.DictReader(
            open(root / r["task"] / "analysis" / "metrics.csv"))
            if x["author"] != "system"]
        if not vals:
            continue
        task_best = min(v for _, v in vals) if lower else max(v for _, v in vals)
        ag = r["agent"].replace("swarm-1", "1").replace("swarm-2", "2").replace("swarm-3", "3")
        # map swarm-N to actual short name via .agent_ids
        ids = {}
        fp = root / r["task"] / "blackboard" / ".agent_ids"
        if fp.exists():
            for line in fp.read_text().splitlines():
                p = line.split("\t")
                if len(p) == 2:
                    ids[p[0]] = p[1].split("_")[0]
        short = ids.get(r["agent"], r["agent"])
        av = [v for a, v in vals if a == short]
        if not av:
            continue
        ab = min(av) if lower else max(av)
        rel = ((task_best - ab) / abs(task_best)) if lower else ((ab - task_best) / abs(task_best))
        agent_rows.append({"task": r["task"], "short": r["short"], "agent": short,
                           "internet_any": r["internet_any"] == "True",
                           "webfetch_calls": int(r["webfetch_calls"]),
                           "weight_gb": float(r["weight_gb"]),
                           "agent_best_rel": rel})
    return rows, agent_rows


def roles_prompt(m: dict) -> str:
    tags = ", ".join(m["tags"]) if m["tags"] else "none"
    return f"""Assign ONE primary role to this message from an AI agent in a 3-agent ML swarm.
Roles:
- explorer: new hypotheses, models, features, exploratory analysis.
- producer: runs training/experiments and produces artifacts/predictions.
- integrator: ensembles, combines predictions, postprocessing.
- validator: verifies metrics, finds bugs/conflicts, checks reproducibility.
- coordinator: plans, status, task/role division, agreements.
- communicator: general updates, answers, social coordination.

Tags: {tags}
Message:
{m['text']}

Return JSON only: {{"role":"explorer|producer|integrator|validator|coordinator|communicator","confidence":0.0}}"""


def call_role(m: dict, key: str, model: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.post("https://api.deepseek.com/chat/completions",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "user", "content": roles_prompt(m)}],
                                    "temperature": 0, "max_tokens": 200}, timeout=90)
            r.raise_for_status()
            c = r.json()["choices"][0]["message"].get("content") or ""
            c = re.sub(r"^```(?:json)?\s*|\s*```$", "", c.strip(), flags=re.S)
            d = json.loads(c)
            role = str(d.get("role", "")).lower()
            if role not in ROLES:
                role = "communicator"
            return {"role": role, "confidence": float(d.get("confidence", 0) or 0)}
        except Exception:  # noqa: BLE001
            time.sleep(2 + attempt * 3)
    return {"role": "communicator", "confidence": 0.0}


def label_roles(tasks: list[dict], data_dir: Path, key: str, model: str, workers: int) -> list[dict]:
    cache_path = data_dir / "roles_raw.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = []
    for t in tasks:
        for m in t["msgs"]:
            k = f"{t['task']}/{m['name']}"
            if k not in cache:
                todo.append((k, m))
    if todo:
        def work(item):
            k, m = item
            return k, call_role(m, key, model)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(work, x) for x in todo]
            for i, f in enumerate(as_completed(futs), 1):
                k, v = f.result(); cache[k] = v
                if i % 25 == 0:
                    print(f"  roles {i}/{len(todo)}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for t in tasks:
        for m in t["msgs"]:
            k = f"{t['task']}/{m['name']}"
            v = cache.get(k, {"role": "communicator", "confidence": 0})
            rows.append({"task": t["task"], "short": t["short"], "agent": m["agent"],
                         "time": m["time"].isoformat(), "role": v["role"], "confidence": v["confidence"]})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    a = args_parse()
    a.out.mkdir(parents=True, exist_ok=True)
    key = json.loads(a.auth.read_text(encoding="utf-8"))["deepseek"]["key"]
    sources = {}
    for spec in a.runs:
        if "=" not in spec:
            raise SystemExit(f"run must be name=path: {spec}")
        name, raw = spec.split("=", 1)
        root = Path(raw)
        data_dir = a.out / "data" / name
        data_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "analysis" / "swarm" / "messages.csv").exists():
                tasks.append(load_task(d))
        if not tasks:
            print(f"skip {name}: no swarm messages")
            continue
        write_csv(data_dir / "coordination.csv", coordination(tasks))
        write_csv(data_dir / "citation_lags.csv", citation_lags(tasks))
        write_csv(data_dir / "tag_citations.csv", tag_citations(tasks))
        write_csv(data_dir / "improvement.csv", improvement(tasks, root))
        it, ia = internet_effect(tasks, root)
        write_csv(data_dir / "internet_tasks.csv", it)
        write_csv(data_dir / "internet_agents.csv", ia)
        roles = label_roles(tasks, data_dir, key, a.model, a.workers)
        write_csv(data_dir / "roles.csv", roles)
        sources[name] = {"path": str(root), "tasks": len(tasks)}
        print(f"готово {name}: {len(tasks)} задач")
    (a.out / "data" / "run_sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print("data ->", a.out / "data")


if __name__ == "__main__":
    main()
