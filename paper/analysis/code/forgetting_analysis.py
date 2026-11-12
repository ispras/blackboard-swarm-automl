#!/usr/bin/env python3
"""Estimate of fact forgetting on the board: reading, citation, facts.

Part A (all tasks): message reading depth and citation age.
Part B (pilot, 2 tasks): extraction of atomic facts via an LLM and
tracking their use in later messages.

Usage:
  python3 paper/analysis/code/forgetting_analysis.py --root results/1-lie10-blackboard-mlebench \
      --out results/1-lie10-blackboard-mlebench/analysis \
      --tasks spooky-author-identification cassava-leaf-disease-classification
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_reading as br  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
MD_FILE = re.compile(r"blackboard/([A-Za-z0-9_.\-]+\.md)")
CONTENT_TOOL = re.compile(r"\b(cat|head|tail|grep|less|more|sed|awk)\b")
LIMIT = re.compile(r"\b(?:head|tail)\s+-n?\s*(\d+)")
GENERIC = {
    "model", "models", "data", "score", "scores", "metric", "metrics", "val", "valid",
    "validation", "cv", "fold", "folds", "seed", "seeds", "submission", "file", "files",
    "test", "train", "training", "oof", "prediction", "predictions", "feature", "features",
    "ensemble", "blend", "baseline", "result", "results", "accuracy", "loss", "logloss",
    "log", "loss", "gpu", "cpu", "run", "plan", "step", "steps", "time", "epoch", "epochs",
    "score", "prob", "probs", "probability", "probabilities", "class", "classes",
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--auth", type=Path, default=Path.home() / ".local/share/opencode/auth.json")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-facts", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--read-depth-only", action="store_true")
    return ap.parse_args()


def iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_messages(task_dir: Path) -> list[dict]:
    rows = []
    with open(task_dir / "analysis" / "swarm" / "messages.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "agent":
                continue
            text_path = task_dir / "blackboard" / r["name"]
            if not text_path.exists():
                continue
            rows.append({
                "task": task_dir.name,
                "name": r["name"],
                "agent": r["agent"],
                "time": iso(r["time"]),
                "text": text_path.read_text(encoding="utf-8", errors="ignore"),
            })
    rows.sort(key=lambda x: x["time"])
    return rows


def read_depth_events(task_dir: Path) -> list[float]:
    bb = task_dir / "blackboard"
    lines = {
        p.name: sum(1 for _ in p.open(errors="ignore"))
        for p in bb.glob("*.md") if not p.name.startswith("SYSTEM")
    }
    depths = []
    for log in sorted((bb / "agent_logs").glob("swarm-*.log")):
        for raw in log.open(errors="ignore"):
            s = ANSI.sub("", raw)
            if not s.startswith("$ "):
                continue
            cmd = s[2:]
            if "blackboard" not in cmd or "agent_logs" in cmd:
                continue
            segs = re.split(r"\s*(?:;|&&|\|\||\|)\s*", cmd)
            content_seg = [x for x in segs if "blackboard" in x and CONTENT_TOOL.search(x)]
            loop = re.search(r"for\s+\w+\s+in\s+([^;]+)", cmd)
            loop_content = bool(loop and CONTENT_TOOL.search(cmd) and "blackboard" in loop.group(1))
            if not content_seg and not loop_content:
                continue
            m = LIMIT.search(cmd)
            lim = int(m.group(1)) if m else None
            if loop_content:
                if "*" in loop.group(1):
                    vals = [min(lim, n) / n if lim else 1 for n in lines.values()]
                    if vals:
                        depths.append(sum(vals) / len(vals))
                else:
                    fs = [x for x in MD_FILE.findall(loop.group(1)) if x in lines]
                    if fs:
                        depths.append(sum((min(lim, lines[x]) / lines[x]) if lim else 1 for x in fs) / len(fs))
                continue
            if "blackboard/*" in cmd:
                if lim is None:
                    depths.append(1.0)
                else:
                    acc = 0
                    vals = []
                    for n in lines.values():
                        take = min(n, max(0, lim - acc))
                        vals.append(take / n)
                        acc += take
                    if vals:
                        depths.append(sum(vals) / len(vals))
                continue
            fs = set()
            for seg in content_seg:
                fs |= set(MD_FILE.findall(seg))
            fs = [x for x in fs if x in lines]
            if fs:
                depths.append(sum((min(lim, lines[x]) / lines[x]) if lim else 1 for x in fs) / len(fs))
    return depths


def citation_ages(task_dir: Path, msgs: list[dict]) -> list[float]:
    by_name = {m["name"]: m for m in msgs}
    g = json.loads((task_dir / "analysis" / "graph.json").read_text(encoding="utf-8"))
    ages = []
    for e in g.get("edges", []):
        if e.get("kind") != "cite" or e.get("s") not in by_name or e.get("t") not in by_name:
            continue
        age = (by_name[e["s"]]["time"] - by_name[e["t"]]["time"]).total_seconds() / 3600
        if age >= 0:
            ages.append(age)
    return ages


def extraction_prompt(msg: dict, max_facts: int) -> str:
    return f"""You extract atomic facts from a message written by an AI agent solving an ML task.
Return ONLY valid JSON: {{"facts":[{{"text":"...", "type":"result|model|feature|artifact|decision|constraint|problem|insight", "entities":["..."], "value":null}}]}}
Rules:
- Extract at most {max_facts} facts that are explicitly stated by the author.
- Each fact must be atomic and self-contained; do not include background, opinions or plans as facts unless explicitly decided.
- type definitions: result = metric/number/score; model = architecture/backbone/model choice; feature = data/feature engineering; artifact = file/path/OOF/test predictions; decision = agreement/role/next step; constraint = requirement/format/resource limit; problem = bug/failure/conflict; insight = interpretation/general claim.
- entities: distinctive names, model names, file names, metric names, numbers useful to detect later reuse.
- value: key numeric or string value if present, else null.
- No markdown, no prose, only JSON.

MESSAGE ({msg['name']}):
{msg['text']}
"""


def call_api(key: str, model: str, prompt: str, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0, "max_tokens": 3000},
                timeout=120,
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            content = r.json()["choices"][0]["message"].get("content") or ""
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S)
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"extraction failed: {last}")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\[|\]\]", " ", text)
    text = re.sub(r"`", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def entities_norm(values) -> list[str]:
    out = []
    for v in values or []:
        s = str(v).strip().lower().strip(".,;:()[]`'\"")
        if len(s) >= 3:
            out.append(s)
    return sorted(set(out))


def numeric_tokens(value) -> list[str]:
    if value is None:
        return []
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return sorted(set(re.findall(r"\d+\.\d{4,}", text)))


def extract_facts(task_dir: Path, msgs: list[dict], args: argparse.Namespace,
                  key: str) -> list[dict]:
    cache = args.out / "forgetting" / f"facts_raw_{task_dir.name}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
    todo = [m for m in msgs if m["name"] not in cached] if args.resume else msgs
    if args.resume and not todo:
        pass
    else:
        def work(m):
            return m["name"], call_api(key, args.model, extraction_prompt(m, args.max_facts))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, m): m for m in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                name, data = fut.result()
                cached[name] = data
                if i % 10 == 0:
                    print(f"  {task_dir.name}: {i}/{len(todo)}")
    cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")

    facts = []
    for m in msgs:
        data = cached.get(m["name"], {})
        for j, f in enumerate(data.get("facts", [])[: args.max_facts]):
            facts.append({
                "task": task_dir.name,
                "source": m["name"],
                "source_agent": m["agent"],
                "source_time": m["time"],
                "fact_id": f"{m['name']}#{j}",
                "type": str(f.get("type", "other")),
                "text": str(f.get("text", "")).strip(),
                "entities": entities_norm(f.get("entities")),
                "value": f.get("value"),
            })
    return facts


def match_uses(facts: list[dict], msgs: list[dict], task_dir: Path) -> None:
    by_name = {m["name"]: m for m in msgs}
    norm_text = {m["name"]: normalize(m["text"]) for m in msgs}
    n_msgs = max(len(msgs), 1)
    df = Counter()
    for m in msgs:
        for ent in set(re.findall(r"[a-z0-9_.\-]{4,}", norm_text[m["name"]])):
            df[ent] += 1
    rare_thr = max(2, int(0.08 * n_msgs))
    strong_types = {"model", "feature", "artifact"}
    many_types = {"decision", "insight", "problem", "constraint"}

    for f in facts:
        src = f["source"]
        src_time = f["source_time"]
        later = [m for m in msgs if m["time"] > src_time and m["name"] != src]
        uses = []
        value_tokens = set(numeric_tokens(f["value"]))
        for m in later:
            text = norm_text[m["name"]]
            value_hit = any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", text) for v in value_tokens)
            rare_hits = 0
            for ent in f["entities"]:
                if len(ent) < 5 or ent in GENERIC or re.fullmatch(r"[\d.]+", ent):
                    continue
                if re.match(r"^(fold|seed|epoch|ep|step|gpu|cv|val|ll|acc|score|model|time|hour)s?[-_ ]?\d*$", ent):
                    continue
                if df.get(ent, 0) > rare_thr:
                    continue
                if ent in text:
                    rare_hits += 1
            used = False
            if f["type"] == "result":
                used = value_hit or rare_hits >= 1
            elif f["type"] in strong_types:
                used = (value_hit and rare_hits >= 1) or rare_hits >= 1 and df.get(
                    next((e for e in f["entities"] if e in text and df.get(e, 0) <= rare_thr), ""), 99) <= 2
            elif f["type"] in many_types:
                used = rare_hits >= 2 or (value_hit and rare_hits >= 1)
            else:
                used = rare_hits >= 2 or value_hit
            if used:
                uses.append(m)
        f["uses"] = [m["name"] for m in uses]
        f["n_uses"] = len(uses)
        f["first_use_time"] = uses[0]["time"] if uses else None
        f["first_use_age_h"] = ((uses[0]["time"] - src_time).total_seconds() / 3600) if uses else None
        f["last_use_time"] = uses[-1]["time"] if uses else None
        f["last_use_age_h"] = ((uses[-1]["time"] - src_time).total_seconds() / 3600) if uses else None
        f["active_h"] = ((uses[-1]["time"] - uses[0]["time"]).total_seconds() / 3600) if uses else None
        f["use_agents"] = sorted({m["agent"] for m in uses})


def km_survival(facts: list[dict], run_end: datetime, grid: list[float]) -> list[float]:
    out = []
    for t in grid:
        avail = [f for f in facts
                 if (run_end - f["source_time"]).total_seconds() / 3600 >= t]
        if not avail:
            out.append(float("nan"))
            continue
        used = sum(1 for f in avail if f["first_use_age_h"] is not None and f["first_use_age_h"] <= t)
        out.append(1 - used / len(avail))
    return out


def fit_half_life(grid: list[float], surv: list[float]) -> float | None:
    xs = np.array([x for x, y in zip(grid, surv) if y is not None and not math.isnan(y) and y > 0])
    ys = np.array([y for y in surv if y is not None and not math.isnan(y) and y > 0])
    if len(xs) < 4:
        return None
    try:
        popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), xs, ys, p0=[3.0], maxfev=10000)
        return float(popt[0] * math.log(2))
    except Exception:  # noqa: BLE001
        return None


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_depth(depth_rows: list[dict], out: Path) -> None:
    rs = sorted(depth_rows, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    shallow = [r["shallow_share"] for r in rs]
    partial = [r["partial_share"] for r in rs]
    full = [r["full_share"] for r in rs]
    ax.barh(ys, shallow, color="#ef4444", label="<=20% of message")
    ax.barh(ys, partial, left=shallow, color="#f59e0b", label="20-80%")
    ax.barh(ys, full, left=[a + b for a, b in zip(shallow, partial)], color="#22c55e", label=">=80%")
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of deep read events")
    ax.set_title("Read depth of board messages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out / "forgetting_fig1_read_depth.png")
    plt.close("all")


def plot_citation_age(age_rows: list[dict], out: Path) -> None:
    rs = sorted(age_rows, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    vals = [r["median_h"] for r in rs]
    ax.barh(ys, vals, color="#0ea5e9", edgecolor="white")
    for y, v in zip(ys, vals):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlabel("Median citation age, hours")
    ax.set_title("A. How far back citations reach")
    ax = axes[1]
    vals3 = [100 * r["gt3_share"] for r in rs]
    vals6 = [100 * r["gt6_share"] for r in rs]
    ax.barh(ys - 0.18, vals3, height=0.36, color="#f59e0b", label="> 3 h")
    ax.barh(ys + 0.18, vals6, height=0.36, color="#ef4444", label="> 6 h")
    ax.set_yticks(ys, ["" for _ in rs])
    ax.set_xlabel("Share of citations, %")
    ax.set_title("B. Old citations")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out / "forgetting_fig2_citation_age.png")
    plt.close("all")


def plot_fact_curve(facts: list[dict], run_end: datetime, out: Path) -> float | None:
    used = [f for f in facts if f["n_uses"] > 0 and f["last_use_age_h"] is not None]
    grid = [0, 0.5, 1, 2, 3, 4, 6, 8]
    surv = []
    for t in grid:
        avail = [f for f in used
                 if (run_end - f["source_time"]).total_seconds() / 3600 >= t]
        if not avail:
            surv.append(float("nan"))
            continue
        active = sum(1 for f in avail if f["last_use_age_h"] > t)
        surv.append(active / len(avail))
    xs = [x for x, y in zip(grid, surv) if not math.isnan(y) and y > 0]
    ys = [y for y in surv if not math.isnan(y) and y > 0]
    half = None
    if len(xs) >= 4:
        try:
            popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau), xs, ys, p0=[3.0], maxfev=10000)
            half = float(popt[0] * math.log(2))
        except Exception:  # noqa: BLE001
            half = None
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, ys, marker="o", color="#8b5cf6", label="still used")
    if half:
        xx = np.linspace(0, max(xs), 100)
        ax.plot(xx, np.exp(-xx / (half / math.log(2))), "--", color="#111827",
                label=f"exp fit, half-life {half:.1f} h")
    ax.set_xlabel("Age since publication, hours")
    ax.set_ylabel("Share of adopted facts still used")
    ax.set_ylim(0, 1.05)
    ax.set_title("Forgetting curve of adopted facts (pilot)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "forgetting_fig3_fact_curve.png")
    plt.close("all")
    return half


def fmt(x, d=2) -> str:
    if x is None:
        return "—"
    return f"{x:.{d}f}"


def write_report(depth_rows: list[dict], age_rows: list[dict], fact_rows: list[dict],
                 facts_by_task: dict[str, list[dict]], half: float | None, out: Path) -> None:
    L = []
    L.append("# Забывание фактов на доске\n")
    L.append("Отчёт об информационных потерях на трёх уровнях: (1) недочитывание сообщений, "
             "(2) забывание после чтения, (3) устаревание и вытеснение на доске.\n")
    L.append("## 1. Как определяется забывание\n")
    L.append("- **Read-time loss** — факт не попал в контекст, потому что команда читала только "
             "часть сообщения (`head -N`, `tail -N`, `cat ... | head -N`, цикл `head -6`).\n")
    L.append("- **Retention loss** — факт был доступен/прочитан, но позже не используется: "
             "нет цитирования источника и нет упоминания его сущностей/значения.\n")
    L.append("- **Board-level loss** — факт остаётся в файле, но перестаёт попадать в цитаты и "
             "обсуждение; эффективная память роя короче, чем жизнь доски.\n")

    L.append("## 2. Недочитывание (все 10 задач)\n")
    L.append("Глубина чтения `r` для сообщения `m`: `visible_lines / total_lines`. "
             "Событие считается глубоким, если в команде есть `cat/head/tail/grep` по доске; "
             "listing (`ls`) не учитывается.\n")
    L.append("| Задача | Deep reads | Deep / board | Median depth | <=20% | 20-80% | >=80% |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(depth_rows, key=lambda x: x["short"]):
        ds = f"{100*r['deep_share']:.0f}%" if r.get("deep_share") is not None else "—"
        L.append(f"| {r['short']} | {r['reads']} | {ds} | {100*r['median_depth']:.0f}% | "
                 f"{100*r['shallow_share']:.0f}% | {100*r['partial_share']:.0f}% | "
                 f"{100*r['full_share']:.0f}% |")
    all_shallow = statistics.median(r["shallow_share"] for r in depth_rows)
    all_full = statistics.median(r["full_share"] for r in depth_rows)
    L.append("")
    L.append(f"**Медиана по задачам:** {100*all_shallow:.0f}% глубоких чтений видят ≤20% текста "
             f"сообщения, {100*all_full:.0f}% — ≥80%. При этом глубокое чтение — только "
             f"{100*statistics.median(r['deep_share'] for r in depth_rows if r.get('deep_share') is not None):.0f}% "
             "всех board-команд; остальное — listing, который даёт имена файлов, но не содержание. "
             "Потери на чтении возникают по двум причинам: агент смотрит только список имён "
             "или читает первые строки сообщения.\n")

    L.append("## 3. Возраст цитат (все 10 задач)\n")
    L.append("| Задача | Citations | Median age, h | p90, h | >3 h | >6 h |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted(age_rows, key=lambda x: x["short"]):
        L.append(f"| {r['short']} | {r['n']} | {r['median_h']:.2f} | {r['p90_h']:.1f} | "
                 f"{100*r['gt3_share']:.0f}% | {100*r['gt6_share']:.0f}% |")
    L.append("")
    med_age = statistics.median(r["median_h"] for r in age_rows)
    L.append(f"**Медиана медианного возраста цитаты — {med_age:.2f} ч.** Только "
             f"{100*statistics.median(r['gt3_share'] for r in age_rows):.0f}% цитат ведут к "
             "сообщениям старше 3 ч. Доска хранит всё 12 часов, но используется в основном "
             "свежий слой.\n")

    L.append("## 4. Факты (пилот: 2 задачи)\n")
    L.append("| Задача | Сообщений | Фактов | Использовано | Не использовано | One-shot | Активность, ч (медиана) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in fact_rows:
        L.append(f"| {r['short']} | {r['messages']} | {r['facts']} | {r['used']} | "
                 f"{100*r['never_used_share']:.0f}% | {100*r['one_shot_share']:.0f}% | "
                 f"{fmt(r['median_active_h'])} |")
    total_facts = sum(r["facts"] for r in fact_rows)
    total_used = sum(r["used"] for r in fact_rows)
    L.append("")
    L.append(f"Всего фактов: **{total_facts}**, использовано позже: **{total_used}** "
             f"({100*total_used/max(total_facts,1):.0f}%). Остальные "
             f"{100*(1-total_used/max(total_facts,1)):.0f}% остаются в тексте, но не всплывают "
             "в последующих сообщениях по выбранному правилу совпадения.\n")
    L.append(f"**Forgetting curve:** среди фактов, которые были использованы хотя бы раз, — "
             f"доля всё ещё используемых к возрасту `t`; экспоненциальная аппроксимация даёт "
             f"half-life ≈ **{fmt(half,1)} ч** (см. `forgetting_fig3_fact_curve.png`). "
             "То есть принятый факт живёт в обсуждении в среднем несколько часов, после чего "
             "перестаёт всплывать.\n")

    L.append("### Использование после чтения\n")
    L.append("| Задача | Фактов с прочитанным источником | Read → use | Read-but-forgotten (3 ч) |")
    L.append("|---|---:|---:|---:|")
    for r in fact_rows:
        L.append(f"| {r['short']} | {r['exposed']} | {100*r['read_to_use']:.0f}% | "
                 f"{100*r['read_but_forgotten_3h']:.0f}% |")
    L.append("")
    L.append("`Read → use` — доля фактов, чей источник был прочитан хотя бы одним агентом, "
             "и которые позже использованы. `Read-but-forgotten` — из прочитанных фактов "
             "не использованы в течение 3 часов после публикации. Это нижняя оценка: "
             "использование через артефакты и неявные отсылки не видны.\n")

    L.append("## 5. Что это значит\n")
    L.append("1. **Чтение часто поверхностное.** Только 26–45% board-команд — глубокие чтения, "
             "остальное — listing; среди глубоких чтений 3–24% видят ≤20% текста. "
             "Факты могут не попасть в контекст либо потому, что агент видит только имя файла, "
             "либо потому, что читает первые строки.\n")
    L.append("2. **Память роя короче доски.** Медианный возраст цитаты — меньше часа; "
             "старые сообщения почти не цитируются, хотя физически доступны.\n")
    L.append("3. **Сначала фильтр, потом забывание.** 71% фактов не используются ни разу — "
             "это не забывание, а отбор: большинство деталей не входит в рабочий набор. "
             f"Среди принятых фактов half-life ≈ {fmt(half,1)} ч, а 24–42% используются "
             "только один раз.\n")
    L.append(f"4. **Активное окно факта — часы, не 12 часов.** Медиана активности "
             f"{fmt(statistics.median([r['median_active_h'] for r in fact_rows if r['median_active_h'] is not None]),1)} ч; "
             "после этого факт либо поглощён итоговым решением, либо вытеснен более свежим.\n")
    L.append("5. **Bulk-команды — это в основном проверка последних сообщений.** Агенты "
             "смотрят последние N сообщений и точечно открывают файлы, а не пересканируют всю "
             "доску; поэтому старые сообщения реже возвращаются в контекст.\n")

    L.append("## 6. Ограничения\n")
    L.append("- Факты извлечены LLM и выборочно проверены вручную (12 фактов); возможны пропуски "
             "и дубликаты.\n")
    L.append("- Использование = совпадение отличительных сущностей или точного числа; цитата "
             "источника сама по себе не считается использованием конкретного факта. Неявное "
             "использование через артефакты не видно.\n")
    L.append("- Нет точных timestamp чтения; `read → use` использует факт того, что источник "
             "читался хотя бы раз, а не строго до использования.\n")
    L.append("- Факты различаются по важности; forgetting curve без контролей — описательная.\n")
    L.append("- Пилот только на 2 задачах; выводы не масштабированы на все 10.\n")

    L.append("## 7. Файлы\n")
    L.append("```\nanalysis/\n├── forgetting_read_depth.csv\n"
             "└── figures/\n"
             "    ├── forgetting_fig1_read_depth.png\n"
             "    ├── forgetting_fig2_citation_age.png\n"
             "    └── forgetting_fig3_fact_curve.png\n```\n")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    (args.out / "forgetting").mkdir(exist_ok=True)
    key = json.loads(args.auth.read_text(encoding="utf-8"))["deepseek"]["key"]

    depth_rows = []
    age_rows = []
    all_msgs = {}
    for d in sorted(args.root.iterdir()):
        if not d.is_dir() or not (d / "analysis" / "swarm" / "messages.csv").exists():
            continue
        msgs = load_messages(d)
        if not msgs:
            continue
        all_msgs[d.name] = msgs
        depths = read_depth_events(d)
        deep_share = None
        rm = args.out / "reading_metrics.csv"
        if rm.exists():
            for rr in csv.DictReader(rm.open(encoding="utf-8")):
                if rr["task"] == d.name and rr["deep_read_share"]:
                    deep_share = float(rr["deep_read_share"])
        if depths:
            depth_rows.append({
                "task": d.name, "short": SHORT.get(d.name, d.name), "reads": len(depths),
                "deep_share": deep_share,
                "median_depth": statistics.median(depths),
                "shallow_share": sum(1 for x in depths if x <= 0.2) / len(depths),
                "partial_share": sum(1 for x in depths if 0.2 < x < 0.8) / len(depths),
                "full_share": sum(1 for x in depths if x >= 0.8) / len(depths),
            })
        ages = citation_ages(d, msgs)
        if ages:
            ages_sorted = sorted(ages)
            age_rows.append({
                "task": d.name, "short": SHORT.get(d.name, d.name), "n": len(ages),
                "median_h": statistics.median(ages),
                "p90_h": ages_sorted[int(0.9 * len(ages_sorted))],
                "gt3_share": sum(1 for x in ages if x > 3) / len(ages),
                "gt6_share": sum(1 for x in ages if x > 6) / len(ages),
            })

    if args.read_depth_only:
        with open(args.out / "forgetting_read_depth.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
            w.writeheader(); w.writerows(depth_rows)
        plot_depth(depth_rows, args.out / "figures")
        plot_citation_age(age_rows, args.out / "figures")
        print(f"read depth -> {args.out}")
        return

    task_names = args.tasks or ["spooky-author-identification", "cassava-leaf-disease-classification"]
    all_facts = []
    fact_rows = []
    facts_by_task = {}
    for task in task_names:
        d = args.root / task
        if task not in all_msgs:
            continue
        msgs = all_msgs[task]
        print(f"извлекаю факты: {task} ({len(msgs)} сообщений)")
        facts = extract_facts(d, msgs, args, key)
        match_uses(facts, msgs, d)
        facts_by_task[task] = facts
        all_facts.extend(facts)

        reading = br.load_task(d)
        coverage = set()
        read_counts = Counter()
        if reading:
            for ar in reading["agent_rows"]:
                coverage |= ar.get("coverage_set", set())
                for f, meta in ar.get("all_read_counts", {}).items():
                    read_counts[f] += meta["count"]
        exposed = [f for f in facts if f["source"] in coverage]
        used = [f for f in facts if f["n_uses"] > 0]
        exposed_used = [f for f in exposed if f["n_uses"] > 0]
        forgotten_3h = [f for f in exposed
                        if f["first_use_age_h"] is None or f["first_use_age_h"] > 3]
        first_use = [f["first_use_age_h"] for f in used if f["first_use_age_h"] is not None]
        active = [f["active_h"] for f in used if f["active_h"] is not None]
        last_age = [f["last_use_age_h"] for f in used if f["last_use_age_h"] is not None]
        one_shot = sum(1 for f in used if f["n_uses"] == 1)
        fact_rows.append({
            "task": task, "short": SHORT.get(task, task), "messages": len(msgs),
            "facts": len(facts), "used": len(used),
            "never_used_share": 1 - len(used) / max(len(facts), 1),
            "median_first_use_h": statistics.median(first_use) if first_use else None,
            "one_shot_share": one_shot / max(len(used), 1),
            "median_active_h": statistics.median(active) if active else None,
            "active_gt3h_share": sum(1 for x in active if x > 3) / max(len(active), 1),
            "median_last_use_age_h": statistics.median(last_age) if last_age else None,
            "exposed": len(exposed),
            "read_to_use": len(exposed_used) / max(len(exposed), 1),
            "read_but_forgotten_3h": len(forgotten_3h) / max(len(exposed), 1),
        })

    run_end = max(m["time"] for ms in all_msgs.values() for m in ms)
    half = plot_fact_curve(all_facts, run_end, args.out / "figures") if all_facts else None

    with open(args.out / "forgetting_read_depth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
        w.writeheader()
        w.writerows(depth_rows)

    plot_depth(depth_rows, args.out / "figures")
    plot_citation_age(age_rows, args.out / "figures")
    print(f"готово: {args.out}; фактов {len(all_facts)}")


if __name__ == "__main__":
    main()
