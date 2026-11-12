#!/usr/bin/env python3
"""Statistics of board reading by agents from session logs.

Source: <task>/blackboard/agent_logs/swarm-*.log — transcripts of shell commands
and Write/Edit tool calls. Commands have no exact timestamps; read time is
estimated by linear interpolation between `date` calls in the logs and the
run boundaries.

Usage:
  python3 paper/analysis/code/board_reading.py --root results/1-lie10-blackboard-mlebench \
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DATE_FULL = re.compile(
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})\s+"
    r"(\d\d):(\d\d):(\d\d)\s+(\w+)\s+(20\d\d)"
)
DATE_ISO = re.compile(r"(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d)")
CONTENT_TOOL = re.compile(r"\b(cat|head|tail|grep|less|more|sed|awk)\b")
READERS = {"cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ack",
           "less", "more", "sed", "awk"}
LIST_TOOL = re.compile(r"\b(ls|find|wc|stat|du)\b")
MD_FILE = re.compile(r"blackboard/([A-Za-z0-9_.\-]+\.md)")
LOG_FILE = re.compile(r"agent_logs/(swarm-\d+\.log)")
DATE_CMD = re.compile(r"(^|[;&|]\s*)date(\s|$)")
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
TZ = {"UTC": 0, "GMT": 0, "MSK": 3, "MSD": 4}

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
    "gold": "#c9a227", "silver": "#9ca3af", "bronze": "#b45309",
    "above_median": "#3b82f6", "below_median": "#ef4444",
}


def args_parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def parse_date_block(lines: list[str], fallback: datetime) -> datetime | None:
    for s in lines:
        m = DATE_FULL.search(s)
        if m:
            _, mon, day, hh, mm, ss, tzname, year = m.groups()
            off = TZ.get(tzname.upper(), 0)
            return datetime(int(year), MONTHS[mon], int(day), int(hh), int(mm), int(ss),
                            tzinfo=timezone(timedelta(hours=off)))
        m = DATE_ISO.search(s)
        if m:
            try:
                return datetime.fromisoformat(m.group(1))
            except ValueError:
                pass
    return None


def _split_top(s: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    i = 0
    quote = None
    while i < len(s):
        c = s[i]
        if quote:
            cur.append(c)
            if c == quote and (i == 0 or s[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if c in "'\"":
            quote = c
            cur.append(c)
            i += 1
            continue
        if s.startswith("&&", i) or s.startswith("||", i):
            out.append("".join(cur))
            cur = []
            i += 2
            continue
        if c == ";":
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _first_word(seg: str) -> str:
    m = re.match(r"\s*(?:cd\s+\S+\s*&&\s*)?([A-Za-z0-9_.-]+)", seg)
    return m.group(1) if m else ""


def _head_n(expr: str) -> int | None:
    m = re.search(r"\bhead\s+(?:-n\s*)?-?(\d+)", expr)
    if m:
        return int(m.group(1))
    if re.search(r"\bhead\b", expr):
        return 10
    return None


def _latest_n(seg: str) -> int | None:
    for m in re.finditer(r"\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)", seg):
        expr = m.group(1)
        if "blackboard/*" in expr:
            n = _head_n(expr)
            if n:
                return n
    return None


def _is_write_to(cmd: str, f: str) -> bool:
    c = re.sub(r"/download_job_code/payload/sandbox/", "", cmd)
    pat = r"blackboard/" + re.escape(f)
    if re.search(r"(?<![<])(?:>>?)\s*[\"']?[^\s;\"']*" + pat, c):
        return True
    if re.search(r"\btee\s+(?:-a\s+)?[\"']?[^\s;\"']*" + pat, c):
        return True
    for seg in _split_top(c):
        if re.search(r"\bsed\b[^;]*\s-i", seg) and pat in seg:
            return True
    if re.search(r"open\s*\(\s*[\"'][^\"']*" + pat + r"[\"']\s*,\s*[\"'](?:w|a|x)", c):
        return True
    return False


def _bulk_kind(cmd: str) -> tuple[str | None, int | None]:
    c = re.sub(r"/download_job_code/payload/sandbox/", "", cmd)
    for m in re.finditer(r"for\s+\w+\s+in\s+(.+?)\s*;\s*do\s+(.+?)\s*done", c, re.S):
        lst, body = m.groups()
        if "blackboard/*" in lst and CONTENT_TOOL.search(body):
            n = _head_n(lst) if "ls" in lst else None
            return ("latest_n", n) if n else ("full_loop", None)
    for seg in _split_top(c):
        fw = _first_word(seg)
        if fw in ("ls", "find", "du", "stat", "wc", "for"):
            continue
        if "blackboard/*" not in seg:
            continue
        n = _latest_n(seg)
        if n:
            return ("latest_n", n)
        if fw in ("grep", "egrep", "fgrep", "rg", "ack"):
            return ("grep_scan", None)
        if re.search(r"\bgrep\b", seg):
            return ("grep_scan", None)
        if re.search(r"\b(?:head|tail)\b[^|]*blackboard/\*", seg):
            return ("glob", None)
        if re.search(r"\bcat\b[^;|&]*blackboard/\*", seg):
            return ("glob", None)
        if fw in ("sed", "awk", "less", "more"):
            return ("glob", None)
    for seg in _split_top(c):
        if re.search(r"\bgrep\s+-[a-zA-Z]*[rR]", seg) and "blackboard" in seg:
            return ("grep_scan", None)
    return (None, None)


def classify(cmd: str) -> dict:
    peer = False
    peer_files: set[str] = set()
    if "agent_logs" in cmd and CONTENT_TOOL.search(cmd):
        peer_files = set(LOG_FILE.findall(cmd))
        peer = bool(peer_files)

    board_view = re.sub(r"blackboard/agent_logs[^\s;|&]*", " ", cmd)
    segs = _split_top(board_view)
    kind, latest_n = _bulk_kind(board_view)
    bulk = kind is not None

    content_files: set[str] = set()
    for seg in segs:
        if "blackboard" not in seg:
            continue
        fw = _first_word(seg)
        if fw in READERS or re.search(r"\|\s*(?:%s)\b" % "|".join(READERS), seg):
            content_files |= set(MD_FILE.findall(seg))
    for m in re.finditer(r"for\s+\w+\s+in\s+(.+?)\s*;\s*do\s+(.+?)\s*done", board_view, re.S):
        lst, body = m.groups()
        if "blackboard/*" not in lst and CONTENT_TOOL.search(body):
            content_files |= set(MD_FILE.findall(lst))
    targeted_files = {f for f in content_files if not _is_write_to(board_view, f)}
    targeted = bool(targeted_files and not bulk)

    listing = any("blackboard" in seg and LIST_TOOL.search(seg) for seg in segs)
    deep = bulk or targeted
    return {
        "any": deep or listing or peer,
        "board": deep or listing,
        "peer_log": peer,
        "peer_files": peer_files,
        "deep": deep,
        "listing": listing,
        "bulk": bulk,
        "latest_n": latest_n if bulk else None,
        "files": [] if bulk else sorted(targeted_files),
    }


def parse_log(path: Path, own_log: str, run_start: datetime, run_end: datetime) -> dict:
    lines = [ANSI.sub("", s).rstrip("\n") for s in path.open(errors="ignore")]
    events = 0
    cmds: list[dict] = []
    writes: list[dict] = []
    reads: list[dict] = []
    anchors: list[tuple[int, datetime]] = []
    i = 0
    while i < len(lines):
        s = lines[i]
        if s.startswith("$ "):
            cmd = s[2:]
            out = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("$ ") and not lines[j].startswith("← Write blackboard/") and not lines[j].startswith("← Edit blackboard/"):
                out.append(lines[j])
                j += 1
            if DATE_CMD.search(cmd):
                t = parse_date_block(out, run_start)
                if t:
                    anchors.append((events, t))
            flags = classify(cmd)
            if flags.get("peer_log"):
                flags["peer_files"] = {f for f in flags["peer_files"] if f != own_log}
                flags["peer_log"] = bool(flags["peer_files"])
                flags["any"] = flags["board"] or flags["peer_log"]
            cmds.append({"idx": events, "cmd": cmd, **flags})
            if flags["any"]:
                reads.append({"idx": events, **flags})
            events += 1
            i = j
            continue
        if s.startswith("← Write blackboard/") or s.startswith("← Edit blackboard/"):
            m = MD_FILE.search(s)
            if m:
                writes.append({
                    "idx": events,
                    "file": m.group(1),
                    "edit": s.startswith("← Edit"),
                })
                events += 1
        i += 1

    raw = [(idx, t) for idx, t in anchors if run_start <= t <= run_end + timedelta(hours=1)]
    return {"cmds": cmds, "writes": writes, "reads": reads, "anchors_raw": raw,
            "events": max(events, 1), "own_log": own_log}


def build_est(raw: list[tuple[int, datetime]], run_start: datetime, run_end: datetime,
              last_idx: int) -> tuple:
    raw = sorted(raw, key=lambda x: x[0])
    clean: list[tuple[int, datetime]] = []
    for idx, t in raw:
        if not clean or t >= clean[-1][1]:
            clean.append((idx, t))
    anchors = [(0, run_start)] + clean + [(last_idx, run_end)]
    anchors.sort(key=lambda x: x[0])

    def est(idx: int) -> datetime:
        if idx <= anchors[0][0]:
            return anchors[0][1]
        if idx >= anchors[-1][0]:
            return anchors[-1][1]
        for (i0, t0), (i1, t1) in zip(anchors, anchors[1:]):
            if i0 <= idx <= i1:
                if i1 == i0:
                    return t0
                frac = (idx - i0) / (i1 - i0)
                return t0 + (t1 - t0) * frac
        return anchors[-1][1]

    return est, len(clean)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def describe(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "q1": float(np.percentile(xs, 25)),
        "q3": float(np.percentile(xs, 75)),
        "min": min(xs),
        "max": max(xs),
    }


def load_task(task_dir: Path) -> dict | None:
    bb = task_dir / "blackboard"
    logs_dir = bb / "agent_logs"
    if not logs_dir.exists():
        return None
    ids = {}
    ids_file = bb / ".agent_ids"
    if ids_file.exists():
        for line in ids_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                ids[parts[0]] = parts[1]

    msgs = []
    with open(task_dir / "analysis" / "swarm" / "messages.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "agent":
                msgs.append({"name": r["name"], "agent": r["agent"], "time": iso(r["time"]),
                             "tags": [t for t in r["tags"].split() if t]})
    if not msgs:
        return None
    by_name = {m["name"]: m for m in msgs}
    msg_names = set(by_name)
    run_start = min(m["time"] for m in msgs)
    run_end = max(m["time"] for m in msgs)
    duration_h = max((run_end - run_start).total_seconds() / 3600, 1e-6)

    known_agents = sorted({m["agent"] for m in msgs}, key=len, reverse=True)
    parsed = {}
    for log in sorted(logs_dir.glob("swarm-*.log")):
        own = log.name
        full = ids.get(log.stem, log.stem)
        agent = next((a for a in known_agents if full.startswith(a)), full)
        parsed[agent] = parse_log(log, own, run_start, run_end)

    for agent, info in parsed.items():
        n_writes = Counter()
        n_edits = Counter()
        for w in info["writes"]:
            if w["edit"]:
                n_edits[w["file"]] += 1
            else:
                n_writes[w["file"]] += 1
        extra = []
        for w in info["writes"]:
            f = w["file"]
            if not w["edit"] and n_writes[f] == 1 and n_edits[f] == 0 and f in by_name:
                extra.append((w["idx"], by_name[f]["time"]))
        est, n_anchors = build_est(info["anchors_raw"] + extra, run_start, run_end, info["events"])
        info["est"] = est
        info["anchors"] = n_anchors

    availability: dict[str, datetime] = {}
    for agent, info in parsed.items():
        first = {}
        for w in info["writes"]:
            if not w["edit"] and w["file"] not in first:
                first[w["file"]] = w["idx"]
        for f, idx in first.items():
            availability[f] = min(availability.get(f, run_end), info["est"](idx))
    for m in msgs:
        availability.setdefault(m["name"], m["time"])

    edges = []
    g = json.loads((task_dir / "analysis" / "graph.json").read_text(encoding="utf-8"))
    for e in g.get("edges", []):
        if e.get("kind") == "cite" and e.get("s") in by_name and e.get("t") in by_name:
            edges.append((e["s"], e["t"]))

    tta_rows = []
    tta_path = task_dir / "analysis" / "swarm" / "tta.csv"
    if tta_path.exists():
        with open(tta_path, encoding="utf-8") as f:
            tta_rows = [r for r in csv.DictReader(f) if r.get("direct_adopter")]

    def bulk_selected(agent: str, r: dict) -> list[str]:
        info = parsed[agent]
        t = info["est"](r["idx"]) + timedelta(minutes=2)
        avail = [m for m in msgs if availability[m["name"]] <= t]
        n = r.get("latest_n")
        if n:
            avail = sorted(avail, key=lambda m: availability[m["name"]], reverse=True)[:n]
        return [m["name"] for m in avail]

    def read_before(agent: str, target: str, limit_idx: int) -> bool:
        info = parsed[agent]
        t_avail = availability.get(target, run_end)
        for r in info["reads"]:
            if r["idx"] >= limit_idx:
                continue
            if target in r.get("files", []):
                return True
            if r.get("bulk") and info["est"](r["idx"]) + timedelta(minutes=2) >= t_avail:
                if target in bulk_selected(agent, r):
                    return True
        return False

    agent_rows = []
    for agent, info in parsed.items():
        cmds = info["cmds"]
        reads = info["reads"]
        board_reads = [r for r in reads if r.get("board")]
        n_cmds = len(cmds)
        n_reads = len(board_reads)
        deep = sum(1 for r in board_reads if r["deep"])
        listing = sum(1 for r in board_reads if r["listing"])
        peer_log = sum(1 for r in reads if r["peer_log"])

        targeted = Counter()
        for r in board_reads:
            for f in r.get("files", []):
                targeted[f] += 1
        targeted_total = sum(targeted.values())
        unique_targeted = len(targeted)

        coverage = set(targeted)
        bulk_reads = 0
        for r in board_reads:
            if r.get("bulk"):
                bulk_reads += 1
                coverage.update(bulk_selected(agent, r))
        coverage &= msg_names

        writes = info["writes"]
        read_before_write = 0
        prev_write = -1
        stale = []
        no_read = 0
        for w in writes:
            hist = [r["idx"] for r in board_reads if prev_write < r["idx"] < w["idx"]]
            if hist:
                read_before_write += 1
            else:
                no_read += 1
            last = max([r["idx"] for r in board_reads if r["idx"] < w["idx"]], default=None)
            if last is not None:
                t_read = info["est"](last)
                t_write = info["est"](w["idx"])
                if t_write >= t_read:
                    stale.append(sum(
                        1 for m in msgs
                        if m["agent"] != agent and t_read < availability[m["name"]] <= t_write
                    ))
            prev_write = w["idx"]

        last_write = {}
        for w in writes:
            last_write[w["file"]] = max(last_write.get(w["file"], -1), w["idx"])
        cited_targets = {tgt for src, tgt in edges if by_name[src]["agent"] == agent}
        read_cited = set()
        for f in set(targeted) & cited_targets:
            sources = [src for src, tgt in edges if tgt == f and by_name[src]["agent"] == agent]
            limit = max((last_write.get(src, -1) for src in sources), default=-1)
            if limit >= 0 and read_before(agent, f, limit):
                read_cited.add(f)
        cite_edges = [(s, t) for s, t in edges if by_name[s]["agent"] == agent]
        cite_read = sum(1 for s, t in cite_edges if read_before(agent, t, last_write.get(s, 10**9)))

        adoptions = 0
        adopt_read = 0
        for r in tta_rows:
            if by_name.get(r["direct_adopter"], {}).get("agent") != agent:
                continue
            adoptions += 1
            limit = last_write.get(r["direct_adopter"], 10**9)
            if read_before(agent, r["event"], limit):
                adopt_read += 1

        agent_rows.append({
            "task": task_dir.name,
            "agent": agent,
            "commands": n_cmds,
            "reads": n_reads,
            "reads_per_hour": n_reads / duration_h,
            "board_attention_share": n_reads / n_cmds if n_cmds else None,
            "deep_read_share": deep / n_reads if n_reads else None,
            "listing_share": listing / n_reads if n_reads else None,
            "peer_log_reads": peer_log,
            "peer_log_share": peer_log / (n_reads + peer_log) if n_reads + peer_log else None,
            "targeted_reads": targeted_total,
            "unique_targeted": unique_targeted,
            "bulk_reads": bulk_reads,
            "coverage": len(coverage),
            "content_coverage": len(coverage) / len(msg_names) if msg_names else None,
            "writes": len(writes),
            "writes_with_read": read_before_write,
            "no_read_writes": no_read,
            "read_before_write_share": read_before_write / len(writes) if writes else None,
            "posts_without_read_share": no_read / len(writes) if writes else None,
            "staleness_median": statistics.median(stale) if stale else None,
            "staleness_events": len(stale),
            "staleness_positive": sum(1 for x in stale if x > 0),
            "staleness_positive_share": (sum(1 for x in stale if x > 0) / len(stale)) if stale else None,
            "cross_agent_read_share": (targeted_total - sum(targeted[f] for f in targeted if f in by_name and by_name[f]["agent"] == agent)) / targeted_total if targeted_total else None,
            "cite_edges": len(cite_edges),
            "cite_read": cite_read,
            "read_before_cite_share": cite_read / len(cite_edges) if cite_edges else None,
            "read_to_cite_rate": len(read_cited) / unique_targeted if unique_targeted else None,
            "adoptions": adoptions,
            "adopt_with_read": adopt_read,
            "adopt_with_read_share": adopt_read / adoptions if adoptions else None,
            "coverage_set": coverage,
            "targeted_counts": {f: {"count": c, "author": by_name.get(f, {}).get("agent", "")}
                                for f, c in targeted.items()},
        })

    coverage_union = set().union(*[r["coverage_set"] for r in agent_rows]) if agent_rows else set()
    coverage_2plus = sum(1 for m in msg_names
                         if sum(1 for r in agent_rows if m in r["coverage_set"]) >= 2)
    total_reads = sum(r["reads"] for r in agent_rows)
    total_cmds = sum(r["commands"] for r in agent_rows)
    total_deep = sum(round(r["deep_read_share"] * r["reads"]) for r in agent_rows
                     if r["deep_read_share"] is not None)
    total_targeted = sum(r["targeted_reads"] for r in agent_rows)
    total_bulk_reads = sum(r["bulk_reads"] for r in agent_rows)
    total_writes = sum(r["writes"] for r in agent_rows)
    total_rbw = sum(r["writes_with_read"] for r in agent_rows)
    total_cite = sum(r["cite_edges"] for r in agent_rows)
    total_cite_read = sum(r["cite_read"] for r in agent_rows)
    total_adopt = sum(r["adoptions"] for r in agent_rows)
    total_adopt_read = sum(r["adopt_with_read"] for r in agent_rows)
    total_peer_log = sum(r["peer_log_reads"] for r in agent_rows)
    stale_vals = []
    for r in agent_rows:
        if r["staleness_median"] is not None:
            stale_vals.append(r["staleness_median"])
    stale_events = sum(r["staleness_events"] for r in agent_rows)
    stale_positive = sum(r["staleness_positive"] for r in agent_rows)
    read_cited_sum = sum(round((r["read_to_cite_rate"] or 0) * r["unique_targeted"]) for r in agent_rows)
    unique_targeted_sum = sum(r["unique_targeted"] for r in agent_rows)
    peer_targeted = sum(round((r["cross_agent_read_share"] or 0) * r["targeted_reads"]) for r in agent_rows)

    medal = ""
    grade_path = task_dir / "analysis" / "grade.json"
    if grade_path.exists():
        gf = json.loads(grade_path.read_text(encoding="utf-8"))["flags"]
        medal = ("gold" if gf.get("gold_medal") else "silver" if gf.get("silver_medal")
                 else "bronze" if gf.get("bronze_medal")
                 else "above_median" if gf.get("above_median") else "below_median")

    row = {
        "task": task_dir.name,
        "short": SHORT.get(task_dir.name, task_dir.name),
        "medal": medal,
        "messages": len(msg_names),
        "duration_h": duration_h,
        "reads": total_reads,
        "reads_per_hour": total_reads / (duration_h * max(len(agent_rows), 1)),
        "board_attention_share": total_reads / total_cmds if total_cmds else None,
        "deep_read_share": total_deep / total_reads if total_reads else None,
        "peer_log_reads": total_peer_log,
        "peer_log_read_share": total_peer_log / (total_reads + total_peer_log) if total_reads + total_peer_log else None,
        "content_coverage": len(coverage_union) / len(msg_names) if msg_names else None,
        "coverage_2plus": coverage_2plus / len(msg_names) if msg_names else None,
        "unread_content_share": 1 - len(coverage_union) / len(msg_names) if msg_names else None,
        "read_before_write_share": total_rbw / total_writes if total_writes else None,
        "staleness_median": statistics.median(stale_vals) if stale_vals else None,
        "staleness_positive_share": stale_positive / stale_events if stale_events else None,
        "cross_agent_read_share": peer_targeted / total_targeted if total_targeted else None,
        "read_to_cite_rate": read_cited_sum / unique_targeted_sum if unique_targeted_sum else None,
        "read_before_cite_share": total_cite_read / total_cite if total_cite else None,
        "adopt_with_read_share": total_adopt_read / total_adopt if total_adopt else None,
        "targeted_reads": total_targeted,
        "bulk_reads": total_bulk_reads,
        "anchors": sum(info["anchors"] for info in parsed.values()),
        "agent_rows": agent_rows,
    }
    return row


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_attention(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    vals = [r["reads_per_hour"] for r in rs]
    ax.barh(ys, vals, color="#0ea5e9", edgecolor="white")
    for y, v in zip(ys, vals):
        ax.text(v + 0.05, y, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlabel("Board reads per hour (mean per agent)")
    ax.set_title("A. Reading intensity")
    ax = axes[1]
    att = [r["board_attention_share"] for r in rs]
    deep = [r["deep_read_share"] for r in rs]
    ax.barh(ys - 0.18, att, height=0.36, color="#8b5cf6", label="board / all commands")
    ax.barh(ys + 0.18, deep, height=0.36, color="#22c55e", label="deep / board reads")
    ax.set_yticks(ys, ["" for _ in rs])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share")
    ax.set_title("B. Attention allocation")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, out / "reading_fig1_attention.png")
    plt.close("all")


def plot_coverage(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    unread = [r["unread_content_share"] for r in rs]
    one = [r["content_coverage"] - r["coverage_2plus"] for r in rs]
    two = [r["coverage_2plus"] for r in rs]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(ys, one, color="#93c5fd", label="read by 1 agent")
    ax.barh(ys, two, left=one, color="#2563eb", label="read by >= 2 agents")
    ax.barh(ys, unread, left=[a + b for a, b in zip(one, two)], color="#e5e7eb", label="unread")
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of board messages")
    ax.set_title("Content coverage of board messages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out / "reading_fig2_coverage.png")
    plt.close("all")


def plot_freshness(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    vals = [r["read_before_write_share"] for r in rs]
    axes[0].barh(ys, vals, color="#f59e0b", edgecolor="white")
    for y, v in zip(ys, vals):
        axes[0].text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=7)
    axes[0].set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    axes[0].set_xlim(0, 1.1)
    axes[0].set_xlabel("Share of posts with a prior board read")
    axes[0].set_title("A. Read before write")
    vals = [r["staleness_positive_share"] or 0 for r in rs]
    axes[1].barh(ys, vals, color="#ef4444", edgecolor="white")
    for y, v in zip(ys, vals):
        axes[1].text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=7)
    axes[1].set_yticks(ys, ["" for _ in rs])
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("Share of posts with unseen peer messages")
    axes[1].set_title("B. Stale posts")
    fig.tight_layout()
    save(fig, out / "reading_fig3_freshness.png")
    plt.close("all")


def plot_linkage(rows: list[dict], out: Path) -> None:
    rs = sorted(rows, key=lambda r: r["short"])
    ys = np.arange(len(rs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    series = [
        ("cross_agent_read_share", "cross-agent reads", "#0ea5e9"),
        ("read_to_cite_rate", "read -> cite", "#8b5cf6"),
        ("adopt_with_read_share", "adoption with read", "#22c55e"),
        ("peer_log_read_share", "peer-log reads", "#f97316"),
    ]
    h = 0.2
    for k, (key, label, color) in enumerate(series):
        vals = [r[key] or 0 for r in rs]
        ax.barh(ys + (k - 1.5) * h, vals, height=h, color=color, label=label)
    ax.set_yticks(ys, [r["short"] for r in rs], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Share")
    ax.set_title("A. Reading -> action linkage")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax = axes[1]
    xs = [r["peer_log_read_share"] or 0 for r in rs]
    ys2 = [r["board_attention_share"] or 0 for r in rs]
    cs = [COLORS.get(medal_for(r), "#6b7280") for r in rs]
    ax.scatter(xs, ys2, c=cs, s=55, edgecolor="white")
    for r, x, y in zip(rs, xs, ys2):
        ax.annotate(r["short"], (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Peer-log read share")
    ax.set_ylabel("Board attention share")
    ax.set_title("B. Bypass channel vs board attention")
    fig.tight_layout()
    save(fig, out / "reading_fig5_linkage.png")
    plt.close("all")


def medal_for(row: dict) -> str:
    return row.get("medal", "")


def write_report(rows: list[dict], med: dict, out: Path) -> None:
    def m(key: str, d: int = 2) -> str:
        v = med.get(key, {})
        return f"{v['median']:.{d}f} [{v['q1']:.{d}f}, {v['q3']:.{d}f}]" if v.get("n") else "—"

    rs = sorted(rows, key=lambda r: r["short"])
    L: list[str] = []
    L.append("# Статистика чтения доски\n")
    L.append("Отдельный отчёт по поведению агентов на доске: интенсивность чтения, покрытие сообщений, "
             "свежесть перед публикацией и связь чтения с цитированием.\n")
    L.append("## 1. Данные и парсинг\n")
    L.append("- Источник: `blackboard/agent_logs/swarm-{1,2,3}.log` — транскрипты shell-команд и "
             "tool-вызовов `Write`/`Edit`.\n")
    L.append("- **Listing** — `ls/find/wc` по доске: агент видит имена, но не содержание.\n")
    L.append("- **Targeted read** — `cat/head/tail/grep blackboard/<file>.md`; команды записи "
             "(`cat >`, `sed -i`) чтением не считаются.\n")
    L.append("- **Bulk read** — чтение содержимого нескольких сообщений: полный цикл по "
             "`blackboard/*.md`, `cat blackboard/*.md`, `grep` по доске или выборка последних N "
             "сообщений через `ls -t ... | head -N`.\n")
    L.append("- Для latest-N выборки засчитываются только N последних сообщений, доступных на "
             "момент команды, а не вся доска.\n")
    L.append("- **Peer-log read** — чтение `blackboard/agent_logs/swarm-*.log` другого агента (обход доски).\n")
    L.append("- Точных timestamp у команд нет. Время чтения оценивается линейной интерполяцией между "
             "выводами `date` и mtime сообщений с единственным `Write`; это приближение, особенно для "
             "bulk-чтений и staleness.\n")

    L.append("## 2. Интенсивность и покрытие\n")
    L.append("| Задача | Reads/h | Board attention | Deep read share | Bulk sweeps | Peer-log share |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rs:
        L.append(f"| {r['short']} | {r['reads_per_hour']:.1f} | {fmt(r['board_attention_share'])} | "
                 f"{fmt(r['deep_read_share'])} | {r['bulk_reads']} | {fmt(r['peer_log_read_share'],3)} |")
    L.append("")
    L.append(f"**Медиана [IQR]:** reads/h {m('reads_per_hour',1)}; board attention "
             f"{m('board_attention_share')}; deep share {m('deep_read_share')}; peer-log share "
             f"{m('peer_log_read_share',3)}.\n")

    L.append("## 3. Покрытие и свежесть\n")
    L.append("| Задача | Content coverage | Read by >=2 | Unread | Read before write | Staleness median | Stale posts share |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        L.append(f"| {r['short']} | {fmt(r['content_coverage'])} | {fmt(r['coverage_2plus'])} | "
                 f"{fmt(r['unread_content_share'])} | {fmt(r['read_before_write_share'])} | "
                 f"{fmt(r['staleness_median'],1)} | {fmt(r['staleness_positive_share'])} |")
    L.append("")
    L.append(f"**Медиана [IQR]:** coverage {m('content_coverage')}; read by >=2 "
             f"{m('coverage_2plus')}; unread {m('unread_content_share')}; read-before-write "
             f"{m('read_before_write_share')}; staleness median {m('staleness_median',1)}; "
             f"stale posts {m('staleness_positive_share')}.\n")
    L.append("Staleness — сколько **новых чужих** сообщений появилось между последним чтением доски "
             "и публикацией. 0 означает, что автор читал доску непосредственно перед постом; "
             "большие значения — публикация по устаревшему снимку.\n")

    L.append("## 4. Чтение → действие\n")
    L.append("| Задача | Cross-agent reads | Read → cite | Read before cite | Adoption with read |")
    L.append("|---|---:|---:|---:|---:|")
    for r in rs:
        L.append(f"| {r['short']} | {fmt(r['cross_agent_read_share'])} | {fmt(r['read_to_cite_rate'])} | "
                 f"{fmt(r['read_before_cite_share'])} | {fmt(r['adopt_with_read_share'])} |")
    L.append("")
    L.append(f"**Медиана [IQR]:** cross-agent reads {m('cross_agent_read_share')}; read→cite "
             f"{m('read_to_cite_rate')}; read-before-cite {m('read_before_cite_share')}; adoption with read "
             f"{m('adopt_with_read_share')}.\n")
    L.append("`Read before cite` — доля citation-рёбер, для которых в логах читателя было чтение цели "
             "до финальной версии источника. Значение ниже 100% не доказывает «ритуальное» цитирование: "
             "источник мог быть прочитан через peer-log, в другой сессии или до оцениваемого окна.\n")

    L.append("## 5. Что это значит для статьи\n")
    L.append(f"1. **Доска действительно читается.** Медиана {med['reads_per_hour']['median']:.1f} "
             "чтений/час на агента; board-read команды — "
             f"{100*med['board_attention_share']['median']:.0f}% всех shell-команд. Это не фоновое "
             "присутствие, а заметная доля рабочего времени.\n")
    L.append(f"2. **Покрытие умеренное и неравномерное.** Медиана content coverage "
             f"{100*med['content_coverage']['median']:.0f}%, прочитано ≥2 агентами "
             f"{100*med['coverage_2plus']['median']:.0f}%; unread — "
             f"{100*med['unread_content_share']['median']:.0f}%. Большая часть bulk-команд — "
             "это проверка последних N сообщений, а не полный перескан доски, поэтому покрытие "
             "заметно ниже, чем при подсчёте latest-N как чтения всей доски.\n")
    L.append(f"3. **Перед постом читают, но доска обновляется быстрее.** Read-before-write = "
             f"{100*med['read_before_write_share']['median']:.0f}%, медианный staleness = "
             f"{med['staleness_median']['median']:.1f} новых чужих сообщений. При этом у "
             f"{100*med['staleness_positive_share']['median']:.0f}% постов есть хотя бы одно новое "
             "чужое сообщение, которого автор не видел: агент смотрит на доску, но за время работы "
             "над сообщением она успевает измениться.\n")
    L.append(f"4. **Внимание направлено на коллег.** Cross-agent reads = "
             f"{100*med['cross_agent_read_share']['median']:.0f}% targeted-чтений; агенты читают "
             "в основном чужие сообщения.\n")
    L.append(f"5. **Чтение конвертируется в использование.** Read→cite = "
             f"{100*med['read_to_cite_rate']['median']:.0f}%: большинство явно прочитанных сообщений "
             "позже цитируются. При этом read-before-cite = "
             f"{100*med['read_before_cite_share']['median']:.0f}% — половина цитат не имеет "
             "наблюдаемого предшествующего чтения; это может быть peer-log bypass или пробел измерения, "
             "и требует осторожной формулировки.\n")
    L.append(f"6. **Peer-log bypass заметен.** Медиана {100*med['peer_log_read_share']['median']:.1f}% "
             "чтений — это чужие логи, а не доска; в отдельных задачах до 10.6%. Для статьи это важно: "
             "часть «координации» идёт вне основного канала.\n")

    L.append("## 6. Ограничения\n")
    L.append("- Нет точных timestamp команд; время чтения интерполируется между `date` и mtime "
             "единственных `Write`; coverage и staleness — приближённые.\n")
    L.append("- Bulk-чтение засчитывается как чтение только тех сообщений, которые команда реально "
             "выбирает: для latest-N — N последних, для полного цикла/grep — все доступные. "
             "Обрезанный вывод (`head`, `grep`) может видеть меньше содержимого.\n")
    L.append("- Listing (`ls`) не даёт содержания; он считается отдельным уровнем и не входит в "
             "content coverage.\n")
    L.append("- `read-before-cite` и `adoption with read` — нижние оценки: чтение через peer-log, "
             "старые сессии или пересказ другим агентом не видны.\n")
    L.append("- Логи могут быть неполными/обрезанными; парсер не разворачивает алиасы и сложные "
             "shell-конструкции.\n")
    L.append("- 3 агента и один прогон на задачу; выводы описательные, не причинные.\n")

    L.append("## 7. Файлы\n")
    L.append("```\nanalysis/\n├── reading_metrics.csv\n├── reading_agents.csv\n└── figures/\n"
             "    ├── reading_fig1_attention.png\n    ├── reading_fig2_coverage.png\n"
             "    ├── reading_fig3_freshness.png\n    └── reading_fig5_linkage.png\n```\n")


def fmt(x, d=2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def main() -> None:
    args = args_parse()
    args.out.mkdir(parents=True, exist_ok=True)
    figdir = args.out / "figures"
    figdir.mkdir(exist_ok=True)

    rows = []
    all_agent_rows = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        r = load_task(d)
        if not r:
            continue
        for ar in r["agent_rows"]:
            all_agent_rows.append({k: v for k, v in ar.items()
                                   if k not in ("coverage_set", "targeted_counts")})
        r.pop("agent_rows")
        rows.append(r)

    with open(args.out / "reading_metrics.csv", "w", newline="", encoding="utf-8") as f:
        keys = [k for k in rows[0] if k != "agent_rows"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    with open(args.out / "reading_agents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_agent_rows[0].keys()))
        w.writeheader()
        w.writerows(all_agent_rows)

    for stale in ("reading_rereads.csv", "reading_rereads_by_tag.csv",
                  "reading_rereads_by_tag_agg.csv", "figures/reading_fig4_rereads.png",
                  "figures/reading_fig6_rereads_by_tag.png"):
        (args.out / stale).unlink(missing_ok=True)

    plot_attention(rows, figdir)
    plot_coverage(rows, figdir)
    plot_freshness(rows, figdir)
    plot_linkage(rows, figdir)

    print(f"готово: {args.out}; задач {len(rows)}")


if __name__ == "__main__":
    main()
