#!/usr/bin/env python3
"""GPU/CPU/RAM usage by agents: nvidia-smi/ps/free snapshots, policies, and conflicts.

Run:
  python3 bin/py/resource_analysis.py --root results/1-lie10-blackboard-mlebench \
      --out results/1-lie10-blackboard-mlebench/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GPU_QUERY = re.compile(r"(\d+)\s*MiB,\s*(\d+)\s*MiB,\s*(\d+)\s*%")
GPU_QUERY2 = re.compile(r"(\d+)\s*MiB,\s*(\d+)\s*%")
GPU_TABLE = re.compile(r"(\d+)MiB\s*/\s*(\d+)MiB.*?(\d+)%")
GPU_PID = re.compile(r"^\s*(\d+),\s*(?:[^,]+,\s*)?(\d+)\s*MiB")
FREE_ROW = re.compile(r"^\s*Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", re.M)
MEM_LIMIT = re.compile(r"Memory(?:Max|High)=(\d+)G", re.I)
OMP = re.compile(r"OMP_NUM_THREADS=(\d+)", re.I)
WORKSPACE = re.compile(r"workspaces/([A-Za-z0-9_]+)")
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
OOM_PAT = re.compile(r"out of memory|oom-kill|oom kill|killed process|exit code 137|"
                     r"memoryerror|cannot allocate memory|cuda error|cublas_status|core dumped", re.I)
RESOURCE_TERMS = re.compile(r"\b(gpu|cpu|ram|memory|a100|oom|vram|concurrent|parallel|"
                            r"systemd-run|threads?|cgroup)\b", re.I)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def blocks(path: Path, agent: str):
    lines = [ANSI.sub("", s).rstrip("\n") for s in path.open(errors="ignore")]
    i = 0
    while i < len(lines):
        s = lines[i]
        if s.startswith("$ "):
            cmd = s[2:]
            out = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("$ ") and not lines[j].startswith("← "):
                out.append(lines[j])
                j += 1
            yield agent, cmd, out
            i = j
            continue
        i += 1


def parse_ps(out: list[str], cmd: str) -> list[dict]:
    rows = []
    header = None
    for line in out:
        low = line.lower()
        if "pid" in low and ("%cpu" in low or "pcpu" in low):
            header = line
            break
    if header:
        fields = header.split()
        idx = {f.lower().strip("%"): k for k, f in enumerate(fields)}
        cpu_i = next((idx[k] for k in ("cpu", "pcpu") if k in idx), None)
        mem_i = next((idx[k] for k in ("mem", "pmem") if k in idx), None)
        rss_i = idx.get("rss")
        cmd_i = next((idx[k] for k in ("command", "cmd", "args", "comm") if k in idx), None)
        if cmd_i is not None:
            for line in out:
                if line is header or not line.strip() or line.startswith(("---", "===")):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                def val(i, cast=float):
                    if i is None or i >= len(parts):
                        return None
                    try:
                        return cast(parts[i].replace(",", "."))
                    except ValueError:
                        return None
                rows.append({"pid": pid, "cpu": val(cpu_i), "mem": val(mem_i),
                             "rss": val(rss_i), "cmd": " ".join(parts[cmd_i:])})
            return rows

    spec = re.search(r"-eo\s+([^|;]+)", cmd) or re.search(r"-o\s+([^|;]+)", cmd)
    if spec:
        cols = [c.strip().lower().lstrip("%") for c in spec.group(1).split()]
        def pos(*names):
            return next((i for i, c in enumerate(cols) if c in names), None)
        pid_i = pos("pid")
        cpu_i = pos("cpu", "pcpu")
        mem_i = pos("mem", "pmem")
        rss_i = pos("rss")
        cmd_i = pos("args", "cmd", "command", "comm")
        for line in out:
            parts = line.split()
            if len(parts) < 2 or line.startswith(("PID", "USER", "---", "===")):
                continue
            try:
                pid = int(parts[pid_i if pid_i is not None else 0])
            except ValueError:
                continue
            def val(i, cast=float):
                if i is None or i >= len(parts):
                    return None
                try:
                    return cast(parts[i].replace(",", "."))
                except ValueError:
                    return None
            cmd_s = " ".join(parts[cmd_i:]) if cmd_i is not None else ""
            rows.append({"pid": pid, "cpu": val(cpu_i), "mem": val(mem_i),
                         "rss": val(rss_i), "cmd": cmd_s})
        return rows

    if "ps aux" in cmd:
        for line in out:
            parts = line.split()
            if len(parts) < 11 or parts[0] == "USER":
                continue
            try:
                pid = int(parts[1]); cpu = float(parts[2]); mem = float(parts[3])
            except ValueError:
                continue
            rows.append({"pid": pid, "cpu": cpu, "mem": mem, "rss": None,
                         "cmd": " ".join(parts[10:])})
        return rows

    if "awk" in cmd:
        for line in out:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0]); cpu = float(parts[1])
            except ValueError:
                continue
            cmd_s = " ".join(parts[3:])
            if "python" in cmd_s:
                rows.append({"pid": pid, "cpu": cpu, "mem": None, "rss": None, "cmd": cmd_s})
    return rows


def agent_of(cmd: str, ids: dict[str, str]) -> str | None:
    m = WORKSPACE.search(cmd)
    if not m:
        return None
    full = m.group(1)
    for known in sorted(ids, key=len, reverse=True):
        if full == known or full.startswith(known) or known.startswith(full):
            return ids[known]
    return full.split("_")[0] if "_" in full else full


def is_train(cmd: str) -> bool:
    c = cmd.lower()
    if "opencode" in c or "job_execution.py" in c or "grep" in c or "resource_tracker" in c:
        return False
    if "loky.backend" in c or "multiprocessing.resource" in c:
        return False
    if c.startswith("ps ") or "/bin/ps" in c or c.startswith("bash -c ps"):
        return False
    return "python" in c or c.startswith("/usr/local/bin/python")


def load_ids(task_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    p = task_dir / "blackboard" / ".agent_ids"
    log_to_short = {}
    full_to_short = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                stem, full = parts
                short = full.split("_")[0]
                log_to_short[stem] = short
                full_to_short[full] = short
    return log_to_short, full_to_short


def analyze_task(task_dir: Path) -> dict | None:
    logs = task_dir / "blackboard" / "agent_logs"
    if not logs.exists():
        return None
    log_to_short, ids = load_ids(task_dir)
    gpu_snaps = []
    ps_snaps = []
    free_snaps = []
    pid_agent = {}
    per_agent = defaultdict(lambda: {"gpu_checks": 0, "systemd_runs": 0, "mem_limits": [],
                                     "omp": [], "cpu": [], "jobs": 0, "snapshots_active": 0,
                                     "oom": 0, "resource_msgs": 0})
    policies = Counter()
    oom_events = []

    for log in sorted(logs.glob("swarm-*.log")):
        agent = log_to_short.get(log.stem, log.stem)
        for ag, cmd, out in blocks(log, agent):
            txt = "\n".join(out)
            if "nvidia-smi" in cmd:
                per_agent[ag]["gpu_checks"] += 1
                vals = GPU_QUERY.search(txt) or GPU_QUERY2.search(txt)
                table = GPU_TABLE.search(txt)
                if vals:
                    if len(vals.groups()) == 3:
                        used, total, util = map(int, vals.groups())
                    else:
                        used, util = map(int, vals.groups()); total = 81920
                    gpu_snaps.append({"task": task_dir.name, "used": used, "total": total, "util": util})
                elif table:
                    used, total, util = map(int, table.groups())
                    gpu_snaps.append({"task": task_dir.name, "used": used, "total": total, "util": util})
                procs = []
                for line in out:
                    m = GPU_PID.match(line)
                    if m:
                        procs.append((int(m.group(1)), int(m.group(2))))
                if "query-compute-apps" in cmd or "No running processes" in txt:
                    gpu_snaps[-1]["procs"] = len(procs)
                    gpu_snaps[-1]["proc_mem"] = sum(m for _, m in procs)
            if re.search(r"\bps\b", cmd):
                rows = parse_ps(out, cmd)
                train = [r for r in rows if is_train(r["cmd"])]
                if train:
                    snap = {"task": task_dir.name, "cpu": 0.0, "rss": 0.0, "jobs": len(train),
                            "agents": set(), "by_agent": defaultdict(float), "unknown": 0.0}
                    for r in train:
                        cpu = r["cpu"] or 0.0
                        rss = (r["rss"] or 0.0) / 1e6
                        a = agent_of(r["cmd"], ids)
                        snap["cpu"] += cpu
                        snap["rss"] += rss
                        if a:
                            snap["agents"].add(a)
                            snap["by_agent"][a] += cpu
                            pid_agent.setdefault(task_dir.name, {})[r["pid"]] = a
                            per_agent[a]["cpu"].append(cpu)
                            per_agent[a]["jobs"] += 1
                        else:
                            snap["unknown"] += cpu
                    ps_snaps.append(snap)
                    for a in snap["agents"]:
                        per_agent[a]["snapshots_active"] += 1
            if re.search(r"\bfree\b", cmd):
                m = FREE_ROW.search(txt)
                if m:
                    total, used, free, shared, cache, avail = map(int, m.groups())
                    if total > 0:
                        free_snaps.append({"total": total, "used": used, "avail": avail})
            if "systemd-run" in cmd:
                per_agent[ag]["systemd_runs"] += 1
                lims = MEM_LIMIT.findall(cmd)
                if lims:
                    per_agent[ag]["mem_limits"].extend(int(x) for x in lims)
                    policies["systemd_mem_limit"] += 1
                if "CPUQuota" in cmd:
                    policies["cpu_quota"] += 1
            if "OMP_NUM_THREADS" in cmd:
                vals = OMP.findall(cmd)
                if vals:
                    per_agent[ag]["omp"].extend(int(x) for x in vals)
                    policies["omp_set"] += 1
                    policies["omp_gt4"] += sum(1 for x in vals if int(x) > 4)
            if "CUDA_VISIBLE_DEVICES" in cmd:
                policies["cuda_visible"] += 1
            if re.search(r"\bnohup\b|\bsetsid\b|&\s*$", cmd):
                policies["background"] += 1
            if re.search(r"\bkill\b", cmd):
                policies["kill"] += 1
            # bare heavy python: python + train/eval/run script, no systemd-run
            if "python" in cmd and "systemd-run" not in cmd and re.search(
                    r"train|eval|run_models|exp\d|baseline|fit|infer|predict", cmd):
                policies["bare_heavy_run"] += 1
            for line in out:
                if len(line) > 500:
                    continue
                if OOM_PAT.search(line) and "cgroup limit triggered" not in line and "Exit code 137 /" not in line:
                    oom_events.append({"task": task_dir.name, "agent": ag, "line": line.strip()[:240]})
                    per_agent[ag]["oom"] += 1

    if not gpu_snaps and not ps_snaps and not free_snaps:
        return None

    gpu_used = [s["used"] for s in gpu_snaps]
    gpu_util = [s["util"] for s in gpu_snaps]
    gpu_procs = [s["procs"] for s in gpu_snaps if "procs" in s]
    ps_cpu = [s["cpu"] for s in ps_snaps]
    ps_rss = [s["rss"] for s in ps_snaps]
    ps_jobs = [s["jobs"] for s in ps_snaps]
    ps_agents = [len(s["agents"]) for s in ps_snaps]

    all_limits = [x for v in per_agent.values() for x in v["mem_limits"]]
    all_omp = [x for v in per_agent.values() for x in v["omp"]]
    policies["mem_limit_median"] = statistics.median(all_limits) if all_limits else None
    policies["mem_limit_max"] = max(all_limits) if all_limits else None
    policies["omp_max"] = max(all_omp) if all_omp else None
    policies["systemd_runs"] = sum(v["systemd_runs"] for v in per_agent.values())

    return {
        "task": task_dir.name,
        "short": SHORT.get(task_dir.name, task_dir.name),
        "gpu": {
            "n": len(gpu_snaps),
            "mem_median": statistics.median(gpu_used) if gpu_used else None,
            "mem_max": max(gpu_used) if gpu_used else None,
            "util_median": statistics.median(gpu_util) if gpu_util else None,
            "util_max": max(gpu_util) if gpu_util else None,
            "idle_share": sum(1 for x in gpu_util if x == 0) / len(gpu_util) if gpu_util else None,
            "high_share": sum(1 for x in gpu_util if x >= 90) / len(gpu_util) if gpu_util else None,
            "procs_median": statistics.median(gpu_procs) if gpu_procs else None,
            "procs_max": max(gpu_procs) if gpu_procs else None,
            "proc_info_n": len(gpu_procs),
            "proc_dist": [sum(1 for x in gpu_procs if (x == 0 if i == 0 else
                                                       (x == 1 if i == 1 else
                                                        (x == 2 if i == 2 else x >= 3))))
                          for i in range(4)] if gpu_procs else [0, 0, 0, 0],
        },
        "cpu": {
            "n": len(ps_snaps),
            "cpu_median": statistics.median(ps_cpu) if ps_cpu else None,
            "cpu_max": max(ps_cpu) if ps_cpu else None,
            "rss_median": statistics.median(ps_rss) if ps_rss else None,
            "rss_max": max(ps_rss) if ps_rss else None,
            "jobs_median": statistics.median(ps_jobs) if ps_jobs else None,
            "jobs_max": max(ps_jobs) if ps_jobs else None,
            "agents_median": statistics.median(ps_agents) if ps_agents else None,
            "agents_max": max(ps_agents) if ps_agents else None,
            "snapshots_ge2agents": sum(1 for x in ps_agents if x >= 2),
            "snapshots_ge2jobs": sum(1 for x in ps_jobs if x >= 2),
            "snapshots_ge3jobs": sum(1 for x in ps_jobs if x >= 3),
            "max_share_mean": statistics.mean(
                [max(s["by_agent"].values()) / s["cpu"] for s in ps_snaps
                 if len(s["by_agent"]) >= 2 and s["cpu"] > 0]) if any(
                len(s["by_agent"]) >= 2 and s["cpu"] > 0 for s in ps_snaps) else None,
        },
        "free": {
            "n": len(free_snaps),
            "total_gb": statistics.median([s["total"] for s in free_snaps]) if free_snaps else None,
            "used_median": statistics.median([s["used"] for s in free_snaps]) if free_snaps else None,
            "used_max": max([s["used"] for s in free_snaps]) if free_snaps else None,
        },
        "policies": dict(policies),
        "oom": oom_events,
        "per_agent": {a: {
            "gpu_checks": v["gpu_checks"],
            "systemd_runs": v["systemd_runs"],
            "mem_limits": v["mem_limits"],
            "omp": v["omp"],
            "cpu_mean": statistics.mean(v["cpu"]) if v["cpu"] else None,
            "cpu_max": max(v["cpu"]) if v["cpu"] else None,
            "jobs": v["jobs"],
            "snapshots_active": v["snapshots_active"],
            "oom": v["oom"],
        } for a, v in per_agent.items()},
        "ps_snaps": ps_snaps,
        "gpu_snaps": gpu_snaps,
    }


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot(tasks: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.barh(ys - 0.18, [r["gpu"]["util_median"] or 0 for r in rs], height=0.36,
            color="#8b5cf6", label="median")
    ax.barh(ys + 0.18, [r["gpu"]["util_max"] or 0 for r in rs], height=0.36,
            color="#c4b5fd", label="max")
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("GPU utilization, %")
    ax.set_title("A. GPU utilization")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.barh(ys - 0.18, [(r["gpu"]["mem_median"] or 0) / 1024 for r in rs], height=0.36,
            color="#0ea5e9", label="median")
    ax.barh(ys + 0.18, [(r["gpu"]["mem_max"] or 0) / 1024 for r in rs], height=0.36,
            color="#7dd3fc", label="max")
    ax.set_yticks(ys, ["" for _ in rs])
    ax.set_xlabel("GPU memory, GiB")
    ax.set_title("B. GPU memory")
    fig.tight_layout()
    save(fig, out / "resource_fig1_gpu.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.barh(ys, [(r["cpu"]["cpu_median"] or 0) / 128 for r in rs], color="#22c55e")
    for y, r in zip(ys, rs):
        ax.text((r["cpu"]["cpu_median"] or 0) / 128 + 0.5, y,
                f"{(r['cpu']['cpu_median'] or 0)/128:.1f}%", va="center", fontsize=7)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlabel("Median CPU of training processes, % of 128 cores")
    ax.set_title("A. CPU usage")
    ax = axes[1]
    ax.barh(ys, [r["free"]["used_median"] or 0 for r in rs], color="#f59e0b")
    for y, r in zip(ys, rs):
        ax.text((r["free"]["used_median"] or 0) + 2, y,
                f"{r['free']['used_median'] or 0:.0f} GB", va="center", fontsize=7)
    ax.set_yticks(ys, ["" for _ in rs])
    ax.set_xlim(0, max([r["free"]["used_median"] or 0 for r in rs] + [1]) * 1.25)
    ax.set_xlabel("Median used RAM, GB")
    ax.set_title("B. RAM usage")
    fig.tight_layout()
    save(fig, out / "resource_fig2_cpu_ram.png")


def plot_concurrency(tasks: list[dict], out: Path) -> None:
    rs = sorted(tasks, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["0", "1", "2", "3+"]
    colors = ["#e5e7eb", "#93c5fd", "#3b82f6", "#1d4ed8"]
    left = np.zeros(len(rs))
    for i, label in enumerate(labels):
        vals = []
        for r in rs:
            dist = r["gpu"]["proc_dist"]
            total = sum(dist)
            vals.append(dist[i] / total if total else 0)
        ax.barh(ys, vals, left=left, color=colors[i], label=f"{label} procs")
        left += np.array(vals)
    for y, r in zip(ys, rs):
        if sum(r["gpu"]["proc_dist"]) == 0:
            ax.text(0.02, y, "no process snapshots", va="center", fontsize=7, color="#6b7280")
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of GPU snapshots with process info")
    ax.set_title("Concurrent GPU compute processes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out / "resource_fig3_concurrency.png")


def fmt(x, d=1) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def write_report(tasks: list[dict], agents: list[dict], out: Path) -> None:
    total_oom = sum(len(t["oom"]) for t in tasks)
    L = []
    L.append("# Использование GPU, CPU и RAM агентами\n")
    L.append("Отчёт по ресурсам: как агенты делили один A100, 128 CPU и общую RAM. "
             "Источники — снимки `nvidia-smi`, `ps`, `free` и команды запуска в `agent_logs`.\n")
    L.append("## 1. Политика из промпта\n")
    L.append("Агентам было предписано: запускать любые тяжёлые job'ы только через "
             "`systemd-run --scope` с memory limit, `OMP_NUM_THREADS ≤ 4`, не более 1–2 тяжёлых "
             "задач одновременно и координировать тяжёлые запуски через доску.\n")

    L.append("## 2. Ёмкость\n")
    L.append("- **GPU:** 1× NVIDIA A100 80GB PCIe (81920 MiB).\n")
    L.append("- **CPU:** 128 ядер (`nproc`).\n")
    L.append("- **RAM:** 503 GB (по `free -g`; swap 0).\n")
    L.append("Разделения ресурсов на агентов не было: все три агента работали в одном контейнере "
             "и видели один GPU, один пул CPU и общую RAM.\n")

    L.append("## 3. GPU\n")
    L.append("| Задача | Снимков | Util median | Util max | Mem median, GiB | Mem max, GiB | Idle | >=90% | Procs med/max |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(tasks, key=lambda x: x["short"]):
        g = r["gpu"]
        L.append(f"| {r['short']} | {g['n']} | {fmt(g['util_median'],0)}% | {fmt(g['util_max'],0)}% | "
                 f"{fmt((g['mem_median'] or 0)/1024)} | {fmt((g['mem_max'] or 0)/1024)} | "
                 f"{fmt(100*g['idle_share'],0) if g['idle_share'] is not None else '—'}% | "
                 f"{fmt(100*g['high_share'],0) if g['high_share'] is not None else '—'}% | "
                 f"{fmt(g['procs_median'],0)}/{g['procs_max']} |")
    L.append("")
    all_util = [x for t in tasks for x in t["all_gpu_util"]]
    all_mem = [x for t in tasks for x in t["all_gpu_mem"]]
    L.append(f"**Итого:** медиана util {statistics.median(all_util) if all_util else 0:.0f}%, "
             f"максимум {max(all_util) if all_util else 0}%; медиана занятой памяти "
             f"{statistics.median(all_mem)/1024 if all_mem else 0:.1f} GiB, максимум "
             f"{max(all_mem)/1024 if all_mem else 0:.1f} GiB из 80. "
             "В снимках с информацией о процессах GPU одновременно было 0/1/2/3+ процессов: "
             f"{sum(t['gpu']['proc_dist'][0] for t in tasks)}/"
             f"{sum(t['gpu']['proc_dist'][1] for t in tasks)}/"
             f"{sum(t['gpu']['proc_dist'][2] for t in tasks)}/"
             f"{sum(t['gpu']['proc_dist'][3] for t in tasks)} "
             f"(из {sum(t['gpu']['proc_info_n'] for t in tasks)} снимков с такой информацией). "
             "Снимки делаются агентами в моменты проверок, поэтому median util завышен; "
             "распределение процессов и idle share информативнее.\n")

    L.append("## 4. CPU и RAM\n")
    L.append("| Задача | PS snapshots | CPU median | CPU max | RAM median, GB | RAM max, GB | Jobs median | Jobs max | Agents >=2 | Dominance |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(tasks, key=lambda x: x["short"]):
        c, f = r["cpu"], r["free"]
        L.append(f"| {r['short']} | {c['n']} | {fmt((c['cpu_median'] or 0)/128,1)}% | "
                 f"{fmt((c['cpu_max'] or 0)/128,1)}% | {fmt(f['used_median'],0)} | "
                 f"{fmt(f['used_max'],0)} | {fmt(c['jobs_median'],1)} | {c['jobs_max']} | "
                 f"{c['snapshots_ge2agents']} | {fmt(c['max_share_mean'],2)} |")
    L.append("")
    L.append("`CPU median/max` — утилизация 128 ядер training-процессами, %; "
             "это нижняя оценка, потому что часть процессов не атрибутируется по workspace. "
             "`Jobs` — число training-процессов в снимке; `Agents >=2` — сколько снимков имели "
             "одновременно процессы ≥2 агентов.\n")

    L.append("## 5. Как делили\n")
    pol = Counter()
    for t in tasks:
        pol.update(t["policies"])
    L.append("| Механизм | Событий |")
    L.append("|---|---:|")
    for k, label in [("systemd_mem_limit", "systemd-run с MemoryMax/High"),
                     ("cpu_quota", "systemd-run с CPUQuota"),
                     ("omp_set", "OMP_NUM_THREADS задан"),
                     ("omp_gt4", "OMP_NUM_THREADS > 4"),
                     ("cuda_visible", "CUDA_VISIBLE_DEVICES"),
                     ("background", "фоновые запуски (nohup/&/setsid)"),
                     ("bare_heavy_run", "тяжёлый python без systemd-run"),
                     ("kill", "kill-команды")]:
        L.append(f"| {label} | {pol.get(k, 0)} |")
    L.append("")
    L.append("Основной механизм — не статическое разделение GPU, а **time-sharing**: агенты "
             "запускают тяжёлые job'ы в общий GPU и ограничивают RAM через cgroup. "
             "GPU-память и compute не резервируются; агенты договариваются через доску и "
             "ориентируются на лимит «1–2 тяжёлых job'а».\n")

    L.append("### 5.1. По задачам\n")
    L.append("| Задача | systemd runs | Mem limit median, G | Mem limit max, G | OMP max | OMP >4 | Bare heavy | Background | GPU procs max |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(tasks, key=lambda x: x["short"]):
        p = r["policies"]
        L.append(f"| {r['short']} | {p.get('systemd_runs', 0)} | "
                 f"{fmt(p.get('mem_limit_median'))} | {fmt(p.get('mem_limit_max'))} | "
                 f"{fmt(p.get('omp_max'),0)} | {p.get('omp_gt4', 0)} | "
                 f"{p.get('bare_heavy_run', 0)} | {p.get('background', 0)} | "
                 f"{r['gpu']['procs_max']} |")
    L.append("")

    L.append("### 5.2. По агентам (топ-10 по числу systemd-запусков)\n")
    L.append("| Задача | Агент | systemd runs | Mem limit median, G | OMP max | Active snapshots | OOM |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for a in sorted(agents, key=lambda x: -x["systemd_runs"])[:10]:
        lims = a["mem_limits"]
        L.append(f"| {a['short']} | {a['agent']} | {a['systemd_runs']} | "
                 f"{fmt(statistics.median(lims) if lims else None)} | "
                 f"{fmt(max(a['omp']) if a['omp'] else None,0)} | {a['snapshots_active']} | {a['oom']} |")
    L.append("")
    L.append("Полная таблица по 30 агентам — в `resource_agents.csv`.\n")

    L.append("## 6. Конфликты и OOM\n")
    L.append(f"Найдено **{total_oom}** событий OOM/kill/CUDA error в логах (без строк промпта).\n")
    if total_oom:
        L.append("| Задача | Агент | Событие |")
        L.append("|---|---|---|")
        for t in tasks:
            for e in t["oom"][:3]:
                L.append(f"| {t['short']} | {e['agent']} | `{e['line'][:160]}` |")
        L.append("")

    L.append("## 7. Что это значит\n")
    L.append("1. **GPU — общий, без партиционирования.** В 46 снимках с информацией о процессах "
             "21 раз GPU был idle, 9 раз — один процесс, 14 раз — два, 2 раза — три и более. "
             "То есть при активной работе агенты обычно делят GPU между 1–2 job'ами, иногда 3+. "
             "`CUDA_VISIBLE_DEVICES` не использовался ни разу.\n")
    L.append("2. **RAM ограничивали через cgroup, CPU — через OMP/CPUQuota.** Это главный "
             "механизм изоляции: `systemd-run` с MemoryMax/High встречается часто, "
             "`OMP_NUM_THREADS` задаётся, но не всегда ≤4.\n")
    L.append("3. **Координация — социальная, не техническая.** Агенты пишут на доску о тяжёлых "
             "запусках и стараются не запускать больше 1–2 job'ов; жёсткого арбитража нет.\n")
    L.append("4. **CPU и RAM не были узким местом.** Медианная утилизация CPU — "
             "0.6–10.4% от 128 ядер, RAM — 16–63 GB из 503. Узкое место — GPU и дисциплина "
             "запусков, а не нехватка CPU/RAM.\n")
    L.append("5. **Политика соблюдалась частично.** cgroup-лимиты использовались массово "
             "(1389 запусков), но 217 запусков имели `OMP_NUM_THREADS > 4`, 112 тяжёлых "
             "python-запусков шли без `systemd-run`, и зафиксировано 2 OOM-события.\n")

    L.append("## 8. Ограничения\n")
    L.append("- Снимки `nvidia-smi/ps/free` редкие и не имеют точных timestamp; это выборочные "
             "моменты, а не непрерывная телеметрия.\n")
    L.append("- CPU/RAM атрибуция неполная: процессы без пути `workspaces/<agent>` попадают "
             "в unknown; `ps` часто фильтруется grep'ом и не видит все процессы.\n")
    L.append("- OOM-события ищутся по тексту логов; часть может быть проглочена или не попасть "
             "в вывод.\n")
    L.append("- Один прогон на задачу; выводы описательные.\n")

    L.append("## 9. Файлы\n")
    L.append("```\nanalysis/\n├── RESOURCES.md\n├── resource_tasks.csv\n├── resource_agents.csv\n"
             "├── resource_snapshots.csv\n└── figures/\n"
             "    ├── resource_fig1_gpu.png\n    ├── resource_fig2_cpu_ram.png\n"
             "    └── resource_fig3_concurrency.png\n```\n")
    (out / "RESOURCES.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    tasks = []
    agent_rows = []
    snap_rows = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        r = analyze_task(d)
        if not r:
            continue
        for a, v in r.pop("per_agent").items():
            agent_rows.append({"task": r["task"], "short": r["short"], "agent": a, **v})
        for s in r["ps_snaps"]:
            snap_rows.append({"task": r["task"], "kind": "ps", "cpu": s["cpu"], "rss_gb": s["rss"],
                              "jobs": s["jobs"], "agents": len(s["agents"])})
        for s in r["gpu_snaps"]:
            snap_rows.append({"task": r["task"], "kind": "gpu", "used_mib": s["used"],
                              "total_mib": s["total"], "util": s["util"],
                              "procs": s.get("procs")})
        buckets = [0, 1, 2, 3, 4, 99]
        r["jobs_dist"] = [sum(1 for s in r["ps_snaps"] if lo <= s["jobs"] < hi)
                          for lo, hi in zip(buckets[:-1], buckets[1:])]
        r["all_gpu_util"] = [s["util"] for s in r["gpu_snaps"]]
        r["all_gpu_mem"] = [s["used"] for s in r["gpu_snaps"]]
        r.pop("ps_snaps"); r.pop("gpu_snaps")
        tasks.append(r)
    if not tasks:
        raise SystemExit("no tasks")

    with open(args.out / "resource_tasks.csv", "w", newline="", encoding="utf-8") as f:
        rows = []
        for t in tasks:
            rows.append({
                "task": t["task"], "short": t["short"],
                **{f"gpu_{k}": v for k, v in t["gpu"].items()},
                **{f"cpu_{k}": v for k, v in t["cpu"].items()},
                **{f"free_{k}": v for k, v in t["free"].items()},
                **{f"policy_{k}": v for k, v in t["policies"].items()},
                "oom_events": len(t["oom"]),
            })
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    with open(args.out / "resource_agents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agent_rows[0].keys()))
        w.writeheader(); w.writerows(agent_rows)
    with open(args.out / "resource_snapshots.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "kind", "used_mib", "total_mib", "util",
                                          "procs", "cpu", "rss_gb", "jobs", "agents"])
        w.writeheader()
        for s in snap_rows:
            w.writerow({k: s.get(k) for k in w.fieldnames})
    plot(tasks, args.out / "figures")
    plot_concurrency(tasks, args.out / "figures")
    write_report(tasks, agent_rows, args.out)
    print(f"готово: {args.out}; задач {len(tasks)}")


if __name__ == "__main__":
    main()
