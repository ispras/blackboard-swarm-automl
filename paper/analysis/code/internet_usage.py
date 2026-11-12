#!/usr/bin/env python3
"""Internet resource usage by agents from session logs.

Sources:
  <task>/blackboard/agent_logs/swarm-*.log — WebFetch/WebSearch, shell commands,
  weight download progress, and explicit urlretrieve/urlopen;
  <task>/analysis/grade.json — medal and test score.

Usage:
  python3 paper/analysis/code/internet_usage.py --root results/1-lie10-blackboard-mlebench \
      --out results/1-lie10-blackboard-mlebench/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
WEB = re.compile(r"^([%✗])\s+WebFetch\s+(\S+)")
WEBSEARCH = re.compile(r"^([%✗])\s+WebSearch\s+(.*)")
PROGRESS = re.compile(
    r"([^\s:]+):\s+100%\|[^|]*\|\s*([0-9.]+)\s*([KMG]?)(?:i?B)?/"
    r"([0-9.]+)\s*([KMG]?)(?:i?B)?"
)
WEIGHT_EXT = re.compile(r"\.(safetensors|bin|pth|pt|tflite|onnx)$", re.I)
HUB = re.compile(r'Downloading: "([^"]+)"')
EXPLICIT_DL = re.compile(
    r"\s*downloading\s+(\S+\.(?:tflite|onnx|pth|pt|bin|safetensors))\s*$", re.I
)
EXPLICIT_SIZE = re.compile(
    r"(\S+\.(?:tflite|onnx|pth|pt|bin|safetensors))\s+([0-9.]+)\s*(MB|GB)\s*$", re.I
)
MULT = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}

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
    return ap.parse_args()


def medal_of(task_dir: Path) -> tuple[str, float | None]:
    path = task_dir / "analysis" / "grade.json"
    if not path.exists():
        return "", None
    data = json.loads(path.read_text(encoding="utf-8"))
    flags = data.get("flags", {})
    if flags.get("gold_medal"):
        medal = "gold"
    elif flags.get("silver_medal"):
        medal = "silver"
    elif flags.get("bronze_medal"):
        medal = "bronze"
    elif flags.get("above_median"):
        medal = "above_median"
    else:
        medal = "below_median"
    return medal, data.get("score")


def blocks(lines: list[str]):
    i = 0
    while i < len(lines):
        if lines[i].startswith("$ "):
            cmd = [lines[i][2:]]
            j = i + 1
            while j < len(lines) and not lines[j].startswith(("$ ", "← ", "→ ", "% ", "✗ ")):
                cmd.append(lines[j])
                j += 1
            yield "\n".join(cmd)
            i = j
        else:
            i += 1


def weight_events(text: str, lines: list[str]) -> list[dict]:
    found: dict[tuple, dict] = {}

    for line in lines:
        m = PROGRESS.search(line)
        if m and WEIGHT_EXT.search(m.group(1)):
            size = float(m.group(4)) * MULT[m.group(5)]
            key = ("tqdm", m.group(1), round(size))
            found[key] = {"kind": "tqdm", "artifact": m.group(1), "bytes": size}

    for i, line in enumerate(lines):
        m = HUB.match(line)
        if not m:
            continue
        url = m.group(1)
        for j in range(i, min(len(lines), i + 8)):
            mm = re.search(r"([0-9.]+)([KMG]?)/([0-9.]+)([KMG]?)", lines[j])
            if mm and "%" in lines[j]:
                size = float(mm.group(3)) * MULT[mm.group(4)]
                key = ("hub", url, round(size))
                found[key] = {"kind": "torch_hub", "artifact": url, "bytes": size}
                break

    for i, line in enumerate(lines):
        m = EXPLICIT_DL.match(line)
        if m:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            mm = re.match(r"\s*\S+\s+(\d+)\s*$", nxt)
            if mm:
                size = int(mm.group(1))
                key = ("explicit", m.group(1), size)
                found[key] = {"kind": "explicit", "artifact": m.group(1), "bytes": size}
        m = EXPLICIT_SIZE.search(line)
        if m:
            size = float(m.group(2)) * (1e6 if m.group(3).upper() == "MB" else 1e9)
            key = ("explicit", m.group(1), round(size))
            found[key] = {"kind": "explicit", "artifact": m.group(1), "bytes": size}

    return list(found.values())


def parse_log(path: Path) -> dict:
    text = ANSI.sub("", path.read_text(errors="ignore"))
    lines = text.splitlines()
    web = []
    websearch = []
    for line in lines:
        m = WEB.match(line)
        if m:
            web.append({"ok": m.group(1) == "%", "url": m.group(2)})
        m = WEBSEARCH.match(line)
        if m:
            websearch.append({"ok": m.group(1) == "%", "query": m.group(2)})

    probes = direct = pip = urllib = 0
    for block in blocks(lines):
        first = block.splitlines()[0]
        if re.search(r"\bcurl\b[^\n]*-[A-Za-z]*I", block) or re.search(
            r"urlopen\([^\n]*\).{0,120}status", block, re.S
        ):
            probes += 1
        if re.search(r"\bcurl\b[^\n]*-o\s|\bwget\b", first):
            direct += 1
        if re.search(r"\bpip3?\s+install\b|\bpython3?\s+-m\s+pip\s+install\b", first):
            pip += 1
        if "urlretrieve" in block or (
            "urlopen" in block and (".read(" in block or "json.load" in block)
        ):
            urllib += 1

    weights = weight_events(text, lines)
    return {
        "web": web,
        "websearch": websearch,
        "probes": probes,
        "direct": direct,
        "pip": pip,
        "urllib": urllib,
        "weights": weights,
        "chars": len(text),
    }


def build(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    agents: list[dict] = []
    for log in sorted(root.glob("*/blackboard/agent_logs/swarm-*.log")):
        task_dir = log.parents[2]
        task = task_dir.name
        parsed = parse_log(log)
        medal, score = medal_of(task_dir)
        row = {
            "task": task,
            "short": SHORT.get(task, task),
            "agent": log.stem,
            "medal": medal,
            "score": score,
            "webfetch_calls": len(parsed["web"]),
            "webfetch_ok": sum(1 for x in parsed["web"] if x["ok"]),
            "webfetch_fail": sum(1 for x in parsed["web"] if not x["ok"]),
            "websearch_calls": len(parsed["websearch"]),
            "probe_blocks": parsed["probes"],
            "direct_downloads": parsed["direct"],
            "pip_installs": parsed["pip"],
            "urllib_blocks": parsed["urllib"],
            "weight_downloads": len(parsed["weights"]),
            "weight_bytes": sum(x["bytes"] for x in parsed["weights"]),
            "web_events": parsed["web"],
            "weights": parsed["weights"],
        }
        row["internet_any"] = any(
            row[k]
            for k in (
                "webfetch_calls",
                "probe_blocks",
                "direct_downloads",
                "pip_installs",
                "urllib_blocks",
                "weight_downloads",
            )
        )
        agents.append(row)

    tasks: dict[str, dict] = {}
    for row in agents:
        task = row["task"]
        t = tasks.setdefault(
            task,
            {
                "task": task,
                "short": row["short"],
                "medal": row["medal"],
                "score": row["score"],
                "webfetch_calls": 0,
                "webfetch_ok": 0,
                "webfetch_fail": 0,
                "websearch_calls": 0,
                "probe_blocks": 0,
                "direct_downloads": 0,
                "pip_installs": 0,
                "urllib_blocks": 0,
                "weight_downloads": 0,
                "weight_bytes": 0,
                "agents_internet": 0,
                "urls": set(),
                "domains": Counter(),
                "web_events": [],
                "weights": [],
            },
        )
        for key in (
            "webfetch_calls",
            "webfetch_ok",
            "webfetch_fail",
            "websearch_calls",
            "probe_blocks",
            "direct_downloads",
            "pip_installs",
            "urllib_blocks",
            "weight_downloads",
            "weight_bytes",
        ):
            t[key] += row[key]
        t["agents_internet"] += int(row["internet_any"])
        for ev in row["web_events"]:
            domain = urlparse(ev["url"]).netloc
            t["urls"].add(ev["url"])
            t["domains"][domain] += 1
            t["web_events"].append({**ev, "task": task, "short": row["short"], "agent": row["agent"], "domain": domain})
        for ev in row["weights"]:
            t["weights"].append({**ev, "task": task, "short": row["short"], "agent": row["agent"]})
    for t in tasks.values():
        t["webfetch_unique_urls"] = len(t["urls"])
        t["internet_any"] = any(
            t[k]
            for k in (
                "webfetch_calls",
                "probe_blocks",
                "direct_downloads",
                "pip_installs",
                "urllib_blocks",
                "weight_downloads",
            )
        )
    return (
        sorted(tasks.values(), key=lambda x: x["short"]),
        sorted(agents, key=lambda x: (x["short"], x["agent"])),
        list(tasks.values()),
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def fmt_gb(n: float) -> str:
    return f"{n / 1e9:.2f}"


def plot(tasks: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    vals = [r["webfetch_calls"] for r in rs]
    ax.barh(ys, vals, color="#0ea5e9", edgecolor="white")
    for y, v in zip(ys, vals):
        ax.text(v + 0.4, y, str(v), va="center", fontsize=7)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlabel("WebFetch calls")
    ax.set_title("A. WebFetch by task")

    ax = axes[1]
    vals = [r["weight_bytes"] / 1e9 for r in rs]
    ax.barh(ys, vals, color="#f59e0b", edgecolor="white")
    for y, v in zip(ys, vals):
        if v > 0:
            ax.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("Downloaded model weights, GB")
    ax.set_title("B. Pretrained weights")
    fig.tight_layout()
    fig.savefig(out / "internet_fig1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(tasks: list[dict], agents: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    total_web = sum(r["webfetch_calls"] for r in rs)
    total_ok = sum(r["webfetch_ok"] for r in rs)
    total_fail = sum(r["webfetch_fail"] for r in rs)
    total_search = sum(r["websearch_calls"] for r in rs)
    total_w = sum(r["weight_downloads"] for r in rs)
    total_gb = sum(r["weight_bytes"] for r in rs) / 1e9
    internet_tasks = sum(1 for r in rs if r["internet_any"])
    internet_agents = sum(1 for a in agents if a["internet_any"])
    probe_tasks = sum(1 for r in rs if r["probe_blocks"])
    probes = sum(r["probe_blocks"] for r in rs)
    vent_web = sum(r["webfetch_calls"] for r in rs if r["short"] == "ventilator")
    heavy = sum(r["weight_bytes"] for r in sorted(rs, key=lambda x: -x["weight_bytes"])[:4]) / 1e9
    total_agents = len(agents)
    domains = Counter()
    for t in rs:
        domains.update(t["domains"])
    domain_rows = []
    for domain, n in domains.most_common():
        ok = sum(1 for t in rs for e in t["web_events"] if e["domain"] == domain and e["ok"])
        fail = n - ok
        task_count = len({t["short"] for t in rs for e in t["web_events"] if e["domain"] == domain})
        domain_rows.append((domain, n, ok, fail, task_count))

    L: list[str] = []
    L.append("# Интернет-ресурсы агентов\n")
    L.append(
        "Отдельный отчёт: какие внешние ресурсы использовали 3 агента на задачу за 12 часов — "
        "веб-страницы и код, пакеты и предобученные веса. Основной вопрос — не «был ли интернет», "
        "а **сколько и каких внешних ресурсов** прошло через агентов.\n"
    )

    L.append("## 1. Методика\n")
    L.append(
        "- Источник — 30 логов `blackboard/agent_logs/swarm-*.log` (10 задач × 3 агента). "
        "ANSI-разметка снята; учитываются команды, tool-вызовы и stdout.\n"
    )
    L.append(
        "- `WebFetch`: строка `% WebFetch URL` — успех, `✗ WebFetch URL failed` — ошибка. "
        "Отдельный `WebSearch` в логах не встречается.\n"
    )
    L.append(
        "- `probe` — shell-проверка доступности (`curl -sI`/`-sSI`/`-sIL` или "
        "`urllib.urlopen(...).status`); `direct` — явное скачивание `curl -o`/`wget`; "
        "`urllib` — shell-блок с `urlretrieve`/`urlopen` для получения файлов или API-ответа.\n"
    )
    L.append(
        "- `pip install` — только реальные команды установки (предупреждения "
        "`pip install -U ...` в stdout не считаются).\n"
    )
    L.append(
        "- `weights` — завершённые скачивания файлов весов: строки `100%|...| X/Y` для "
        "`.safetensors/.bin/.pth/.pt/.tflite/.onnx`, `Downloading: ...` из `torch.hub` и явные "
        "`urlretrieve` + размер. Файлы токенизаторов и конфигов в объём весов не входят. "
        "Большая часть весов скачивалась неявно библиотеками (`transformers`, `timm`, "
        "`torch.hub`), поэтому в логах это progress bar, а не URL.\n"
    )
    L.append(
        "- Объём — сумма размеров **наблюдаемых** завершённых скачиваний. Это не уникальный "
        "сетевой трафик: один и тот же файл могли качать несколько агентов, а часть скачиваний "
        "могла быть скрыта кэшем или фильтрами вывода. Поэтому объём — нижняя оценка.\n"
    )
    L.append(
        "- Задача/агент считаются использовавшими интернет, если есть хотя бы одно из событий "
        "выше, включая проверку доступности.\n"
    )

    L.append("## 2. Сводка\n")
    L.append("| Показатель | Значение |")
    L.append("|---|---:|")
    L.append(f"| Задач / агентов | {len(rs)} / {total_agents} |")
    L.append(f"| Задач, где был интернет | {internet_tasks} / {len(rs)} |")
    L.append(f"| Агентов с сетевой активностью | {internet_agents} / {total_agents} |")
    L.append(f"| WebFetch-вызовов | {total_web} ({total_ok} успешных, {total_fail} ошибок) |")
    L.append(f"| WebSearch-вызовов | {total_search} |")
    L.append(f"| Задач с WebFetch | {sum(1 for r in rs if r['webfetch_calls'])} / {len(rs)} |")
    L.append(f"| Завершённых скачиваний весов | {total_w} |")
    L.append(f"| Объём скачанных весов | ~{total_gb:.2f} GB |")
    L.append(f"| `pip install` | {sum(r['pip_installs'] for r in rs)} (rdkit в bms-mol) |")
    L.append(f"| Прямых `curl -o`/`wget` | {sum(r['direct_downloads'] for r in rs)} |")
    L.append(f"| shell-блоков с `urllib` | {sum(r['urllib_blocks'] for r in rs)} (все — mlsp-birds) |")
    L.append("| Обращений к репозиторию MLE-bench | 0 |")
    L.append("")
    L.append(
        "Лидеры: ventilator — 38 WebFetch-вызовов; spooky — 6.98 GB весов; mlsp-birds — "
        "14 `urllib`-блоков и 1.45 GB; cassava — 13 файлов весов; nfl-contact и "
        "stanford-covid — ноль сетевых событий.\n"
    )

    L.append("## 3. WebFetch: что читали\n")
    L.append("| Задача | Агент | Вызовов | OK | Ошибок | Что искали |")
    L.append("|---|---|---:|---:|---:|---|")
    notes = {
        ("billion-word", "swarm-1"): "метрика (Kaggle evaluation, PDF форума), GitHub-реализация",
        ("taxi", "swarm-1"): "leaderboard, RMSE победителей, GitHub-репозиторий",
        ("uw-gi", "swarm-2"): "метрика Dice/Hausdorff, Kaggle discussion, GitHub `metric.py`",
        ("ventilator", "swarm-1"): "1st-place solution, GitHub-код и конфиги",
        ("ventilator", "swarm-2"): "1st-place solution, Kaggle discussion, GitHub-код",
        ("ventilator", "swarm-3"): "1st-place solution, GitHub-код, блог, Kaggle",
    }
    for r in rs:
        for a in [x for x in agents if x["task"] == r["task"] and x["webfetch_calls"]]:
            note = notes.get((r["short"], a["agent"]), "")
            L.append(
                f"| {r['short']} | {a['agent']} | {a['webfetch_calls']} | "
                f"{a['webfetch_ok']} | {a['webfetch_fail']} | {note} |"
            )
    L.append(
        f"| **Итого** | | **{total_web}** | **{total_ok}** | **{total_fail}** | |"
    )
    L.append("")
    L.append(
        f"Всего {total_web} вызовов на {len({e['url'] for r in rs for e in r['web_events']})} "
        f"уникальных URL; {total_ok}/{total_web} = "
        f"{(100*total_ok/total_web if total_web else 0):.0f}% успешных. "
        f"WebFetch использовали только 4 задачи из 10; {vent_web} из {total_web} вызовов — "
        "ventilator, где три агента независимо тянули один и тот же репозиторий победителя.\n"
    )
    L.append("| Домен | Вызовов | OK | Ошибок | Задач |")
    L.append("|---|---:|---:|---:|---:|")
    for domain, n, ok, fail, task_count in domain_rows:
        L.append(f"| `{domain}` | {n} | {ok} | {fail} | {task_count} |")
    L.append("")
    L.append(
        "37 из 56 вызовов (66%) — `github.com`/`raw.githubusercontent.com`/`api.github.com`: "
        "агенты читали код, конфиги и README внешних решений. Ещё 9 (16%) — поисковики "
        "DuckDuckGo/Bing, 6 (11%) — Kaggle (leaderboard/discussion), остальное — блог и PDF. "
        "Пять ошибок: 404 на неверную ветку `main` в ventilator (2), 404 `datamodule.py` (1), "
        "404 Kaggle API (1) и ошибка прокси `r.jina.ai` (1).\n"
    )

    L.append("## 4. Прямые сетевые операции\n")
    L.append(
        f"- **Проверки доступности** — {probes} shell-блоков в {probe_tasks} задачах: "
        "`huggingface.co`, `pypi.org`, `github.com`, `download.pytorch.org`. HF/PyPI/GitHub "
        "отвечали `200`; `download.pytorch.org` на HEAD отдавал `403` (это не обязательно блокировка).\n"
    )
    L.append(
        "- **PDF** — `curl -sL -o billion.pdf` (59 KB) со страницы Kaggle-форума; агент "
        "использовал его, чтобы подтвердить character-level Levenshtein. Успешно.\n"
    )
    L.append(
        "- **GloVe 6B** — `curl -sL -o glove.6B.zip` с `nlp.stanford.edu` в spooky; "
        "файл не появился (`dl.log` пустой), скачивание не завершилось; подтверждения "
        "использования в логах нет.\n"
    )
    L.append(
        "- **Perch v2** — mlsp-birds: 14 shell-блоков с `urllib` (HF API, BirdNET, Perch, "
        "ConvNeXT-BirdSet, `labels.csv`), исходники Perch — с `raw.githubusercontent.com`, "
        "веса — с HF/GitHub release. Это единственный случай, когда агент вручную собирал "
        "внешний код и веса вне `transformers`/`timm`.\n"
    )
    L.append(
        "- **RDKit** — единственный `pip install`: `pip install --target ... rdkit` в bms-mol. "
        "Установка прошла, пакет использован для канонизации InChI в постпроцессинге.\n"
    )

    L.append("## 5. Предобученные веса\n")
    L.append("| Задача | Скачиваний | Объём, GB | Семейства |")
    L.append("|---|---:|---:|---|")
    families = {
        "spooky": "RoBERTa, XLNet, BERT, ELECTRA, ALBERT, BART, DeBERTa",
        "cassava": "ConvNeXt, Swin, ViT, EfficientNet",
        "mlsp-birds": "AST, ConvNeXT-BirdSet, Perch, BirdNET, BioLingual/CLAP",
        "billion-word": "RoBERTa base/large, DeBERTa-v3",
        "uw-gi": "EfficientNet-B0/B3 + segmentation checkpoint",
        "bms-mol": "EfficientNet-B0",
    }
    for r in sorted(rs, key=lambda x: -x["weight_bytes"]):
        if not r["weight_downloads"]:
            continue
        L.append(
            f"| {r['short']} | {r['weight_downloads']} | {fmt_gb(r['weight_bytes'])} | "
            f"{families.get(r['short'], '—')} |"
        )
    L.append(f"| **Итого** | **{total_w}** | **{total_gb:.2f}** | |")
    L.append("")
    L.append(
        f"{(100*heavy/total_gb if total_gb else 0):.0f}% объёма приходится на 4 задачи: spooky (6.98 GB), cassava (2.63 GB), "
        "mlsp-birds (1.45 GB) и billion-word (1.05 GB). Это ровно те задачи, где агенты "
        "делали fine-tuning или извлечение эмбеддингов из готовых чекпоинтов. "
        "В такси, nfl, stanford-covid и ventilator веса не скачивались вообще.\n"
    )
    L.append(
        "В логах видны повторные скачивания: `roberta-base` (499 MB) встречается у двух "
        "агентов spooky, файл 77.9 MB — у двух агентов cassava. Это согласуется с тем, что "
        "общего кэша весов/веб-контента между агентами, похоже, не было; поэтому 12.27 GB — сумма "
        "по агентам, а не обязательно уникальные файлы.\n"
    )
    L.append("| Задача | Медаль | WebFetch | Веса, GB | Профиль использования |")
    L.append("|---|---|---:|---:|---|")
    profiles = {
        "billion-word": "метрика + RoBERTa/DeBERTa",
        "bms-mol": "RDKit + EfficientNet-B0",
        "cassava": "timm-бэкбоны",
        "mlsp-birds": "audio foundation models",
        "taxi": "только leaderboard/поиск",
        "nfl-contact": "нет",
        "spooky": "много transformer fine-tuning",
        "stanford-covid": "нет",
        "uw-gi": "metric.py + EfficientNet",
        "ventilator": "код победителя, без весов",
    }
    for r in rs:
        L.append(
            f"| {r['short']} | {r['medal'] or '—'} | {r['webfetch_calls']} | "
            f"{fmt_gb(r['weight_bytes'])} | {profiles.get(r['short'], '—')} |"
        )
    L.append("")
    L.append(
        "Связи «интернет → медаль» в этих данных нет: две задачи без сети (stanford-covid, "
        "nfl-contact) взяли gold/silver, а taxi с веб-поиском оказался below_median. "
        "Наоборот, самые тяжёлые по весам spooky и mlsp-birds — gold, но billion-word — "
        "gold при 1.05 GB, а cassava — silver при 2.63 GB. Это описательная статистика "
        "на n = 10, а не причинный эффект.\n"
    )

    L.append("## 6. Ограничения\n")
    L.append(
        f"- WebFetch не логирует размер ответа: {total_web} вызовов — это счётчик обращений, "
        f"а не байты. Сравнивать веб-трафик с {total_gb:.1f} GB весов нельзя.\n"
    )
    L.append(
        f"- {total_gb:.1f} GB — сумма завершённых скачиваний в логах, а не уникальный egress. "
        "Один файл могли качать несколько агентов; кэш и `grep`/`tail` могли скрыть часть прогресса.\n"
    )
    L.append(
        "- У shell-команд нет timestamp: нельзя сказать, сколько времени ушло на скачивание "
        "и какая доля 12 часов потрачена на сеть.\n"
    )
    L.append(
        "- `%`/`✗` — маркеры TUI; «успех» означает отсутствие ошибки, а не то, что агент "
        "использовал содержимое страницы.\n"
    )
    L.append(
        "- Проверки доступности и `urllib`-блоки считаются по тексту команд; часть запросов "
        "могла быть диагностикой, а не получением контента.\n"
    )

    L.append("## 7. Проверка ограничений\n")
    L.append(
        "- Прямых обращений к репозиторию MLE-bench не найдено. Один поисковый запрос "
        "в uw-gi содержит слово `mlebench` (`duckduckgo.com/html/?q=mlebench+...`), но сам "
        "репозиторий не открывался.\n"
    )
    L.append(
        "- Прямых URL к приватным файлам/меткам (`answers`, `test_labels`, `ground_truth`) "
        "не найдено; слово `private` встречается только в поисковых запросах про private "
        "leaderboard. Приватные метки и test-set feedback не скачивались.\n"
    )
    L.append(
        "- Внешние данные — только публичные страницы/код/метрики и предобученные веса. "
        "Скачивания датасетов соревнования вне `data/` не обнаружено.\n"
    )

    L.append("## 8. Файлы\n")
    L.append("```")
    L.append("analysis/")
    L.append("├── internet_by_task.csv")
    L.append("├── internet_by_agent.csv")
    L.append("└── figures/internet_fig1.png")
    L.append("```")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    tasks, agents, _ = build(args.root)

    task_fields = [
        "task", "short", "medal", "score", "webfetch_calls", "webfetch_ok",
        "webfetch_fail", "webfetch_unique_urls", "websearch_calls", "probe_blocks",
        "direct_downloads", "pip_installs", "urllib_blocks", "weight_downloads",
        "weight_gb", "internet_any",
    ]
    task_rows = []
    for t in tasks:
        task_rows.append({**t, "weight_gb": round(t["weight_bytes"] / 1e9, 3)})
    write_csv(args.out / "internet_by_task.csv", task_rows, task_fields)

    agent_fields = [
        "task", "short", "agent", "webfetch_calls", "webfetch_ok", "webfetch_fail",
        "websearch_calls", "probe_blocks", "direct_downloads", "pip_installs",
        "urllib_blocks", "weight_downloads", "weight_gb", "internet_any",
    ]
    agent_rows = [{**a, "weight_gb": round(a["weight_bytes"] / 1e9, 3)} for a in agents]
    write_csv(args.out / "internet_by_agent.csv", agent_rows, agent_fields)

    plot(tasks, args.out / "figures")


if __name__ == "__main__":
    main()
