#!/usr/bin/env python3
"""Context composition from agent logs: agent actions vs artifacts/documents vs environment output.

Usage:
  python3 paper/analysis/code/log_composition.py --root results/1-lie10-blackboard-mlebench \
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
CATS = ["docs", "code", "artifacts", "peer_log", "execution", "environment", "other"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def classify(cmd: str) -> str:
    c = cmd.lower()
    if "agent_logs" in c:
        return "peer_log"
    if any(x in c for x in ("blackboard/", "task.md", "prompt.md", "readme.md", ".md")):
        return "docs"
    if any(x in c for x in ("python", "pytest", "torchrun", "accelerate", "bash ", "nohup",
                            "make ", "rscript")):
        return "execution"
    if any(x in c for x in ("pip ", "nvidia-smi", "conda ", "apt ", "df -", "free -", "ps aux")):
        return "environment"
    if any(x in c for x in (".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg",
                            "requirements")):
        return "code"
    if any(x in c for x in (".npy", ".csv", ".pkl", ".pt", ".bin", ".parquet",
                            "workspaces/", "final_solution/", "ls ", "find ", "du ")):
        return "artifacts"
    return "other"


def parse_log(path: Path) -> dict:
    lines = [ANSI.sub("", s).rstrip("\n") for s in path.open(errors="ignore")]
    counts = {k: 0 for k in CATS}
    cmd_chars = tool_chars = prose_chars = 0
    i = 0
    while i < len(lines):
        s = lines[i]
        if s.startswith("$ "):
            cmd = s[2:]
            cmd_chars += len(s)
            i += 1
            while i < len(lines) and not lines[i].startswith("$ ") and not lines[i].startswith("← "):
                counts[classify(cmd)] += len(lines[i])
                i += 1
            continue
        if s.startswith("← "):
            tool_chars += len(s)
        elif not s.startswith("> "):
            prose_chars += len(s)
        i += 1
    output_chars = sum(counts.values())
    action_chars = cmd_chars + tool_chars + prose_chars
    total = action_chars + output_chars
    return {
        "cmd_chars": cmd_chars,
        "tool_chars": tool_chars,
        "prose_chars": prose_chars,
        "action_chars": action_chars,
        "output_chars": output_chars,
        "total_chars": total,
        **counts,
    }


def add_shares(row: dict) -> dict:
    total = max(row["total_chars"], 1)
    for k in ("action_chars", "cmd_chars", "tool_chars", "prose_chars", "output_chars", *CATS):
        row[f"{k}_share"] = row[k] / total
    row["artifacts_docs_chars"] = row["docs"] + row["code"] + row["artifacts"]
    row["artifacts_docs_share"] = row["artifacts_docs_chars"] / total
    row["artifacts_docs_peer_share"] = (row["artifacts_docs_chars"] + row["peer_log"]) / total
    row["exec_env_share"] = (row["execution"] + row["environment"] + row["other"]) / total
    row["tokens_est"] = total / 3.5
    return row


def analyze(root: Path) -> tuple[list[dict], list[dict]]:
    tasks = []
    agents = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        logs_dir = d / "blackboard" / "agent_logs"
        if not logs_dir.exists():
            continue
        agg = {k: 0 for k in ("cmd_chars", "tool_chars", "prose_chars", "action_chars",
                              "output_chars", "total_chars", *CATS)}
        for log in sorted(logs_dir.glob("swarm-*.log")):
            r = parse_log(log)
            r.update({"task": d.name, "short": SHORT.get(d.name, d.name), "log": log.name})
            agents.append(add_shares(r))
            for k in agg:
                agg[k] += r[k]
        if agg["total_chars"] == 0:
            continue
        agg.update({"task": d.name, "short": SHORT.get(d.name, d.name)})
        tasks.append(add_shares(agg))
    return tasks, agents


def plot(tasks: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    cats = [("docs", "docs/board", "#2563eb"), ("code", "code", "#7c3aed"),
            ("artifacts", "data/artifacts", "#0ea5e9"), ("peer_log", "peer logs", "#f97316"),
            ("execution", "execution", "#ef4444"), ("environment", "environment", "#9ca3af"),
            ("other", "other", "#d1d5db")]
    left = np.zeros(len(rs))
    for key, label, color in cats:
        vals = [r[f"{key}_share"] for r in rs]
        ax.barh(ys, vals, left=left, color=color, label=label, edgecolor="white", linewidth=0.4)
        left += np.array(vals)
    act = [r["action_chars_share"] for r in rs]
    ax.barh(ys, act, left=left, color="#22c55e", label="agent actions", edgecolor="white")
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of visible log characters")
    ax.set_title("Context composition in agent logs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "log_composition_fig1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(x, d=1) -> str:
    return f"{100*x:.{d}f}%"


def write_report(tasks: list[dict], agents: list[dict], out: Path) -> None:
    def sumcat(k: str) -> int:
        return sum(r[k] for r in tasks)
    total = sum(r["total_chars"] for r in tasks)
    actions = sum(r["action_chars"] for r in tasks)
    L = []
    L.append("# Состав контекста по agent logs\n")
    L.append("Вопрос: какая доля видимого контекста занята артефактами и документами, "
             "а какая — действиями самого агента. Считаны **символы лога** (≈ токены / 3.5), "
             "без system prompt, tool schemas и reasoning.\n")
    L.append("## 1. Как считаем\n")
    L.append("- **Действия агента** — строки команд `$ ...`, tool-вызовы `← Write/Edit/...` и "
             "assistant-проза (строки вне блоков вывода).\n")
    L.append("- **Вывод инструментов** — всё, что напечатала среда после команды до следующей "
             "команды/tool-вызова.\n")
    L.append("- Категория вывода определяется по команде: `docs/board` (blackboard, TASK/README, `*.md`), "
             "`code` (`.py/.sh/.json/...`), `data/artifacts` (`.npy/.csv/.pkl`, `workspaces/`, `ls/find`), "
             "`peer logs` (`agent_logs`), `execution` (python/training), `environment` (pip/nvidia-smi/...).\n")
    L.append(f"- Всего видимого текста: **{total/1e6:.2f}M символов ≈ {total/3.5/1e6:.2f}M токенов** "
             f"на 10 задач / 30 агентов.\n")

    L.append("## 2. Общий баланс\n")
    L.append("| Категория | Доля | Токены, M |")
    L.append("|---|---:|---:|")
    for k, label in [("action_chars", "**Действия агента**"), ("docs", "Документы / доска"),
                     ("code", "Код"), ("artifacts", "Данные / артефакты"),
                     ("peer_log", "Логи коллег"), ("execution", "Вывод исполнения"),
                     ("environment", "Окружение"), ("other", "Прочее")]:
        chars = actions if k == "action_chars" else sumcat(k)
        L.append(f"| {label} | {100*chars/total:.1f}% | {chars/3.5/1e6:.2f} |")
    L.append("")
    docs_art = sumcat("docs") + sumcat("code") + sumcat("artifacts")
    L.append(f"**Артефакты + документация + код = {100*docs_art/total:.1f}%**; "
             f"с логами коллег — {100*(docs_art+sumcat('peer_log'))/total:.1f}%. "
             f"**Действия агента = {100*actions/total:.1f}%** "
             f"(из них команды {100*sum(r['cmd_chars'] for r in tasks)/total:.1f}%, "
             f"tool-вызовы {100*sum(r['tool_chars'] for r in tasks)/total:.1f}%, "
             f"проза {100*sum(r['prose_chars'] for r in tasks)/total:.1f}%). "
             f"Остальное — вывод исполнения и окружения ({100*(sumcat('execution')+sumcat('environment')+sumcat('other'))/total:.1f}%).\n")

    L.append("## 3. По задачам\n")
    L.append("| Задача | Действия | Docs | Code | Data/artifacts | Peer logs | Execution+env |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(tasks, key=lambda x: x["short"]):
        L.append(f"| {r['short']} | {fmt(r['action_chars_share'])} | {fmt(r['docs_share'])} | "
                 f"{fmt(r['code_share'])} | {fmt(r['artifacts_share'])} | {fmt(r['peer_log_share'])} | "
                 f"{fmt(r['exec_env_share'])} |")
    L.append("")
    L.append(f"Разброс: действия {100*min(r['action_chars_share'] for r in tasks):.0f}–"
             f"{100*max(r['action_chars_share'] for r in tasks):.0f}%; docs+code+artifacts "
             f"{100*min(r['artifacts_docs_share'] for r in tasks):.0f}–"
             f"{100*max(r['artifacts_docs_share'] for r in tasks):.0f}%.\n")

    L.append("## 4. Что это значит\n")
    L.append(f"1. **Видимый контекст на три четверти — не действия агента.** Tool-выводы занимают "
             f"{100*(1-actions/total):.0f}%; агент в основном читает среду, а не говорит.\n")
    L.append(f"2. **Самая большая содержательная категория — документы и доска** "
             f"({100*sumcat('docs')/total:.0f}%). Это согласуется с идеей якорей: агенты много "
             "перечитывают планы, статусы и board-сообщения.\n")
    L.append(f"3. **Код и данные — меньше, чем кажется.** Код {100*sumcat('code')/total:.0f}%, "
             f"данные/артефакты {100*sumcat('artifacts')/total:.0f}%; исполнение и окружение — "
             f"{100*(sumcat('execution')+sumcat('environment')+sumcat('other'))/total:.0f}%.\n")
    L.append("4. **Если добавить reasoning**, доля «внутреннего» агента вырастет: по `COSTS.md` "
             "reasoning-токенов ~2.6M, это ещё ~50% от видимого транскрипта. Тогда действия+reasoning "
             "занимают около 60% оценённого контекста, а артефакты/доки — около 20%. Но это уже "
             "другая метрика: reasoning не виден в логе.\n")

    L.append("## 5. Ограничения\n")
    L.append("- Считается видимый транскрипт, а не фактическое окно: system prompt, tool schemas "
             "и reasoning не входят.\n")
    L.append("- Категория определяется по тексту команды; смешанные команды "
             "(`python ... && cat board.md`) относятся к одной категории по приоритету.\n")
    L.append("- Строки, не попавшие в блок вывода, считаются assistant-прозой; возможны "
             "служебные сообщения.\n")
    L.append("- Токены оценены как символы / 3.5; для кода и логов это приближение.\n")

    L.append("## 6. Файлы\n")
    L.append("```\nanalysis/\n├── log_composition.csv\n"
             "└── figures/log_composition_fig1.png\n```\n")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    tasks, agents = analyze(args.root)
    fields_t = [k for k in tasks[0] if k.endswith("_share") or k in
                ("task", "short", "total_chars", "tokens_est", "action_chars", "docs", "code",
                 "artifacts", "peer_log", "execution", "environment", "other")]
    with open(args.out / "log_composition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_t)
        w.writeheader()
        for r in tasks:
            w.writerow({k: r[k] for k in fields_t})
    plot(tasks, args.out / "figures")
    print(f"готово: {args.out}; задач {len(tasks)}, агентов {len(agents)}")


if __name__ == "__main__":
    main()
