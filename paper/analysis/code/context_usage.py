#!/usr/bin/env python3
"""Estimate of agent context usage over the entire run (12 hours).

Sources:
  <task>/analysis/COSTS.md — input/output/reasoning/cache_read by agent;
  <task>/blackboard/agent_logs/swarm-*.log — transcript size and number of tool steps.

Usage:
  python3 paper/analysis/code/context_usage.py --root results/1-lie10-blackboard-mlebench \
      --out results/1-lie10-blackboard-mlebench/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
COST_ROW = re.compile(
    r"\| (swarm-\d+) \| \$([0-9.]+) \| ([\d ]+) \| ([\d ]+) \| ([\d ]+) \| ([\d ]+) \|"
)
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
SYSTEM_TOKENS = 8000
CHARS_PER_TOKEN = 3.5
WINDOWS = {"1m": 1_048_576, "262k": 262_144}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def analyze_task(task_dir: Path) -> dict | None:
    cost_path = task_dir / "analysis" / "COSTS.md"
    if not cost_path.exists():
        return None
    text = cost_path.read_text(encoding="utf-8")
    agents = []
    for name, cost, tin, tout, treason, tcache in COST_ROW.findall(text):
        tin = int(tin.replace(" ", ""))
        tout = int(tout.replace(" ", ""))
        treason = int(treason.replace(" ", ""))
        tcache = int(tcache.replace(" ", ""))
        log = task_dir / "blackboard" / "agent_logs" / f"{name}.log"
        if not log.exists():
            continue
        lines = [ANSI.sub("", s) for s in log.open(errors="ignore")]
        steps = sum(1 for s in lines if s.startswith("$ ") or s.startswith("← Write") or s.startswith("← Edit"))
        chars = sum(len(s) for s in lines)
        transcript = chars / CHARS_PER_TOKEN
        end_context = transcript + treason + SYSTEM_TOKENS
        prompt = tin + tcache
        agents.append({
            "task": task_dir.name,
            "short": SHORT.get(task_dir.name, task_dir.name),
            "agent": name,
            "cost_usd": float(cost),
            "input": tin,
            "output": tout,
            "reasoning": treason,
            "cache_read": tcache,
            "prompt_total": prompt,
            "steps": steps,
            "transcript_tokens": transcript,
            "end_context_tokens": end_context,
            "turnovers": prompt / end_context if end_context else None,
            "fill_1m": end_context / WINDOWS["1m"],
            "fill_262k": end_context / WINDOWS["262k"],
            "cache_share": tcache / prompt if prompt else None,
        })
    if not agents:
        return None
    return {
        "task": task_dir.name,
        "short": SHORT.get(task_dir.name, task_dir.name),
        "prompt_total": sum(a["prompt_total"] for a in agents),
        "input": sum(a["input"] for a in agents),
        "output": sum(a["output"] for a in agents),
        "reasoning": sum(a["reasoning"] for a in agents),
        "cache_read": sum(a["cache_read"] for a in agents),
        "steps": sum(a["steps"] for a in agents),
        "end_context_mean": statistics.mean(a["end_context_tokens"] for a in agents),
        "end_context_max": max(a["end_context_tokens"] for a in agents),
        "fill_1m": statistics.mean(a["fill_1m"] for a in agents),
        "fill_262k": statistics.mean(a["fill_262k"] for a in agents),
        "turnovers": statistics.mean(a["turnovers"] for a in agents),
        "agents": agents,
    }


def plot(tasks: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    vals = [r["prompt_total"] / 1e6 for r in rs]
    ax.barh(ys, vals, color="#0ea5e9", edgecolor="white")
    for y, v in zip(ys, vals):
        ax.text(v + 2, y, f"{v:.0f}M", va="center", fontsize=7)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlabel("Prompt tokens over 12 h, millions (input + cache_read)")
    ax.set_title("A. Cumulative prompt tokens")
    ax = axes[1]
    f1 = [100 * r["fill_1m"] for r in rs]
    f2 = [100 * r["fill_262k"] for r in rs]
    ax.barh(ys - 0.18, f1, height=0.36, color="#22c55e", label="1M window")
    ax.barh(ys + 0.18, f2, height=0.36, color="#ef4444", label="262k window")
    ax.axvline(100, color="#111827", ls="--", lw=0.8)
    ax.set_yticks(ys, ["" for _ in rs])
    ax.set_xlim(0, 115)
    ax.set_xlabel("Estimated context at the end, % of window")
    ax.set_title("B. Window fill at the end")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "context_fig1_usage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(n: float, d: int = 1) -> str:
    return f"{n:.{d}f}"


def write_report(tasks: list[dict], agents: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    total_prompt = sum(r["prompt_total"] for r in rs)
    med_end = statistics.median(r["end_context_mean"] for r in rs)
    med_1m = statistics.median(r["fill_1m"] for r in rs)
    med_262k = statistics.median(r["fill_262k"] for r in rs)
    med_agent_prompt = statistics.median(a["prompt_total"] for a in agents)
    L = []
    L.append("# Контекст агентов за 12 часов\n")
    L.append("Отдельный отчёт: сколько контекста агенты обрабатывают за весь прогон и насколько "
             "заполнено окно к концу. Не путать два разных числа: **суммарные prompt-токены за 12 ч** "
             "и **уникальное содержимое в окне** в конце.\n")
    L.append("## 1. Методика\n")
    L.append("- `prompt_total = input + cache_read` из `analysis/COSTS.md`: это все prompt-токены, "
             "отправленные модели за прогон, с учётом кэша.\n")
    L.append("- `steps` — число tool-шагов из лога (`$`-команды + `Write/Edit`); примерно столько "
             "LLM-вызовов сделал агент.\n")
    L.append("- `end_context ≈ transcript_tokens + reasoning + system`, где transcript_tokens = "
             "размер видимого лога / 3.5, reasoning — из `COSTS.md`, system — оценка 8k. "
             "Это оценка уникального содержимого к концу прогона, а не сумма по вызовам.\n")
    L.append("- `turnovers = prompt_total / end_context`: сколько раз содержимое окна было "
             "повторно отправлено модели за 12 ч.\n")
    L.append("- Окно: 1M (DeepSeek V4.1 Flash) и 262k (V4 Flash) — точная модель в `COSTS.md` "
             "не зафиксирована, поэтому даны оба варианта.\n")

    L.append("## 2. Суммарно за 12 часов\n")
    L.append("| Задача | Prompt, M | Шагов | Уникальный контекст в конце, k | Turnovers | Fill 1M | Fill 262k |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        L.append(f"| {r['short']} | {r['prompt_total']/1e6:.0f} | {r['steps']} | "
                 f"{r['end_context_mean']/1000:.0f} | {r['turnovers']:.0f}× | "
                 f"{100*r['fill_1m']:.0f}% | {100*r['fill_262k']:.0f}% |")
    L.append("")
    L.append(f"**Итого по 10 задачам:** {total_prompt/1e9:.2f} млрд prompt-токенов; "
             f"input {sum(r['input'] for r in rs)/1e6:.1f}M, output {sum(r['output'] for r in rs)/1e6:.1f}M, "
             f"reasoning {sum(r['reasoning'] for r in rs)/1e6:.1f}M, cache_read "
             f"{sum(r['cache_read'] for r in rs)/1e6:.0f}M. "
             f"Медиана на агента — {med_agent_prompt/1e6:.0f}M prompt-токенов за 12 ч.\n")
    L.append(f"**К концу прогона** уникальный контекст — медиана {med_end/1000:.0f}k токенов: "
             f"{100*med_1m:.0f}% окна 1M или {100*med_262k:.0f}% окна 262k. "
             "У отдельных агентов оценка доходит до 273k (104% окна 262k), что означало бы "
             "компакцию/усечение, если модель была 262k.\n")

    L.append("## 3. Что это значит\n")
    L.append(f"1. **За 12 часов через модель проходит очень много.** Медиана "
             f"{med_agent_prompt/1e6:.0f}M prompt-токенов на агента; вся серия — "
             f"{total_prompt/1e9:.1f} млрд. Это не размер контекста, а сумма по всем вызовам.\n")
    L.append(f"2. **Окно не растёт до бесконечности.** Уникального содержимого к концу — "
             f"медиана {med_end/1000:.0f}k токенов. При окне 1M это {100*med_1m:.0f}%, "
             f"при 262k — {100*med_262k:.0f}%.\n")
    L.append("3. **Контекст переиспользуется, а не пересчитывается.** Cache read — "
             f"{100*sum(r['cache_read'] for r in rs)/total_prompt:.1f}% всех prompt-токенов; "
             "нового (uncached) input всего несколько процентов. Поэтому 12-часовой прогон "
             "экономически возможен.\n")
    L.append("4. **Число «оборотов» контекста** — сотни: одно и то же окно многократно "
             "отправляется модели. Это важнее, чем конечная заполненность, для объяснения "
             "стоимости и латентности.\n")
    L.append("5. **Если модель была 262k**, часть агентов (billion-word, taxi, mlsp, uw-gi) "
             "подходила к пределу; в логах нет явных маркеров компакции, поэтому более вероятен "
             "вариант 1M. Для статьи это надо зафиксировать по `model` из session JSON.\n")

    L.append("## 4. Ограничения\n")
    L.append("- Нет per-request token usage в артефактах; `end_context` — оценка по размеру "
             "транскрипта, reasoning и system prompt, а не измерение окна.\n")
    L.append("- `steps` — приближение числа LLM-вызовов; один вызов может содержать несколько "
             "tool-call'ов, а ретраи не видны.\n")
    L.append("- Размер транскрипта в символах переводится в токены коэффициентом 3.5; для кода "
             "и логов это приближение.\n")
    L.append("- Точная модель и её context limit в `COSTS.md` не указаны; даны оба сценария.\n")

    L.append("## 5. Файлы\n")
    L.append("```\nanalysis/\n├── context_usage_by_task.csv\n"
             "└── figures/context_fig1_usage.png\n```\n")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    tasks = []
    agents = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        r = analyze_task(d)
        if r:
            agents.extend(r.pop("agents"))
            tasks.append(r)

    with open(args.out / "context_usage_by_task.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tasks[0].keys()))
        w.writeheader()
        w.writerows(tasks)

    plot(tasks, args.out / "figures")
    print(f"готово: {args.out}; задач {len(tasks)}, агентов {len(agents)}")


if __name__ == "__main__":
    main()
