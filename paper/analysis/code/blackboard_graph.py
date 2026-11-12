#!/usr/bin/env python3
"""Interactive swarm board graph: parses [[wiki links]] and #hashtags from
sandbox/blackboard/*.md and builds a self-contained HTML + JSON (no external
dependencies). The HTML template is embedded; if graph_template.html is present
next to it, that file is used.

Displays:
  - three views: "Graph" (citations + tag nodes), "Timeline" (swimlane by tags
    or agents), "Matrices" (tag co-occurrence and agent -> agent links);
  - citation direction: an arrow from a note to the note it relies on;
  - tags as nodes (diamonds) and doc -> #tag edges; tag filter (AND);
  - clicking a node highlights everything connected: green — ancestors, orange —
    descendants, blue — other connected nodes;
  - node color = agent, size = how many times the note is cited;
  - metrics only from <metric>2.23</metric> / <metric>2.23<\\metric> tags:
    values are not parsed from text, the "lower/higher is better" direction comes
    from task metadata (meta.json / data/README.md);
  - border = verdict (final / candidate / negative / verified);
  - time axis (from timestamps in headings) + interval filter;
  - artifacts (csv/npy) and broken links as separate nodes.

Examples:
  python3 paper/analysis/code/blackboard_graph.py --bb <board>
  python3 paper/analysis/code/blackboard_graph.py --bb <board> --out /tmp/blackboard.html
  python3 paper/analysis/code/blackboard_graph.py --bb <board> --no-json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Copy of bin/py/common.py: paper/analysis/code must not depend on the harness.
METRIC_OVERRIDES = {
    "stanford-covid-vaccine": "MCRMSE",
    "ventilator-pressure-prediction": "MAE",
}


def short_metric(name: str | None, comp_id: str | None = None) -> str:
    """Short metric name for labels: no repeated task id and no -score suffix."""
    s = (name or "").strip()
    if not s:
        return "metric"
    cid_any = str(comp_id or "").strip().lower()
    if cid_any in METRIC_OVERRIDES:
        return METRIC_OVERRIDES[cid_any]
    low = s.lower()
    if comp_id:
        cid = str(comp_id).strip().lower()
        if cid and (low == cid or low == f"{cid}-score"):
            return "score"
        if cid and low.startswith(cid + "-"):
            s = s[len(cid) + 1:]
            low = s.lower()
    if low == "score":
        return "score"
    if low.endswith("-score"):
        base = s[: -len("-score")].strip("-_ ")
        return base or "score"
    return s


DEFAULT_BB = ROOT / "sandbox" / "blackboard"
DEFAULT_OUT = DEFAULT_BB / "graph.html"
DEFAULT_JSON = DEFAULT_BB / "graph.json"

LINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]")
TITLE_TS_RE = re.compile(r"\((\d{1,2}):(\d{2})\s*(?:MSK|МСК)\)")
NAME_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_SYSTEM_")
INPUT_LINE_RE = re.compile(r"^\s*[*_>\s]*Ссылк[аи]?\b", re.IGNORECASE)
# Agents publish metrics only via tags (prompt rule 22):
# <metric>2.23</metric> (the closing tag is also written as <\metric>).
METRIC_TAG_RE = re.compile(
    r"<\s*\\?\s*metric\s*>"
    r"(?P<content>[^<]{1,80}?)"
    r"<\s*/?\s*\\?\s*metric\s*>",
    re.IGNORECASE | re.DOTALL,
)
METRIC_VALUE_RE = re.compile(r"-?\d{1,8}(?:[.,]\d{1,8})?")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
CODE_SPAN_RE = re.compile(r"`[^`\n]*`")
TAG_RE = re.compile(r"(?<![\w#])#([A-Za-zА-Яа-яЁё0-9_]{2,})")

TAG_ALIASES = {
    "результаты": "результат",
    "tта": "тта",
    "tta": "тта",
    "финальное": "финал",
    "проверки": "проверка",
}

PALETTE = [
    "#2563eb", "#e11d48", "#059669", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#4d7c0f",
    "#b45309", "#0f766e", "#9333ea", "#dc2626",
]
VERDICT_COLORS = {
    "final": "#111827",
    "candidate": "#f59e0b",
    "negative": "#ef4444",
    "verified": "#16a34a",
}
TAG_COLOR = "#0f766e"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bb", type=Path, default=DEFAULT_BB, help="каталог доски (default: sandbox/blackboard)")
    ap.add_argument("--competition", default=None,
                    help="id задачи MLE-bench для метрики/направления (default: активная из private/)")
    ap.add_argument("--out", type=Path, default=None, help="куда писать HTML (default: <bb>/graph.html)")
    ap.add_argument("--json-out", type=Path, default=None, help="куда писать JSON (default: <bb>/graph.json)")
    ap.add_argument("--no-json", action="store_true", help="не писать JSON рядом с HTML")
    ap.add_argument("--open", action="store_true", help="открыть HTML в браузере")
    return ap.parse_args()


def load_base_date(bb: Path, fallback: date) -> date:
    try:
        data = json.loads((bb / "deadline.json").read_text())
        return date.fromisoformat(str(data["start"])[:10])
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


def load_mtime_manifest(bb: Path) -> dict[str, float]:
    """Reads blackboard/.mtime.tsv (epoch<TAB>name) — timestamps that survived moving the board."""
    out: dict[str, float] = {}
    try:
        text = (bb / ".mtime.tsv").read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        epoch, _, name = line.partition("\t")
        name = name.strip()
        if epoch.strip().isdigit() and name and Path(name).name == name:
            out[name] = float(epoch.strip())
    return out


def parse_doc_time(path: Path, title: str, body: str, base: date,
                   manifest: dict[str, float] | None = None) -> tuple[datetime | None, int | None]:
    m = NAME_TS_RE.match(path.name)
    if m:
        d, hh, mm, _ss = m.groups()
        try:
            dt = datetime.fromisoformat(f"{d}T{hh}:{mm}")
            return dt, hh_int(dt, base)
        except ValueError:
            pass
    hay = title + " " + body[:400]
    m = TITLE_TS_RE.search(hay)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        return datetime(base.year, base.month, base.day, hh, mm), hh * 60 + mm
    if manifest and path.name in manifest:
        dt = datetime.fromtimestamp(manifest[path.name])
        return dt, hh_int(dt, base)
    if manifest:
        # The file is not in the manifest: it appeared after the run/export, so its mtime
        # does not reflect swarm runtime — use the end of the manifest window.
        dt = datetime.fromtimestamp(max(manifest.values()))
        return dt, hh_int(dt, base)
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None, None
    return mtime, (mtime.date() - base).days * 1440 + mtime.hour * 60 + mtime.minute


def hh_int(dt: datetime, base: date) -> int:
    return (dt.date() - base).days * 1440 + dt.hour * 60 + dt.minute


def clean_text(text: str) -> str:
    text = LINK_RE.sub(lambda m: m.group(1).strip(), text)
    text = re.sub(r"^#+.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[*`_>|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = LINK_RE.sub(lambda m: m.group(1).strip(), line[2:]).strip()
            return re.sub(r"[*`_]+", "", title).strip()
    return fallback


def verdict_of(name: str, title: str, body: str) -> str | None:
    blob = f"{name} {title} {body[:500]}".lower()
    if "negative" in blob:
        return "negative"
    if "кандидат" in blob or "candidate" in blob:
        return "candidate"
    if "final" in name.lower() or "итог" in title.lower() or "финал" in title.lower():
        return "final"
    if re.search(r"верифиц|подтвержд|confirmed|проверен|verification", blob):
        return "verified"
    return None


def load_task_metric(bb: Path, comp_id: str | None = None) -> tuple[str | None, str | None]:
    """Task metric name and direction (low/high) for the label and plot envelope.

    comp_id pins the task (for archived boards in results/); otherwise the active
    task is taken from private/competition.json. Then falls back to the sandbox
    README (<bb>/../data/README.md). Metric values are not touched — they come
    only from <metric> tags; this function handles task metadata only.
    """
    try:
        active = str(comp_id).strip() if comp_id else \
            json.loads((ROOT / "private" / "competition.json").read_text())["competition"]
        meta = json.loads((ROOT / "private" / "competitions" / active / "meta.json").read_text())
        name = str(meta.get("metric") or "").strip() or None
        if name:
            # for some tasks grader.name = "<id>-score" — show only the metric in the graph
            name = short_metric(name, active)
        lower = meta.get("is_lower_better")
        direction = "low" if lower else "high" if lower is not None else None
        if name or direction:
            return name, direction
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        readme = (bb.parent / "data" / "README.md").read_text(errors="replace")
    except OSError:
        return None, None
    m = re.search(r"метрика:\s*`([^`]+)`\s*\(([^)]+)\)", readme)
    if not m:
        return None, None
    direction = None
    if "меньше" in m.group(2):
        direction = "low"
    elif "больше" in m.group(2):
        direction = "high"
    name = m.group(1).strip() or None
    if name:
        # README may contain an outdated/long grader name — normalize it
        name = short_metric(name, comp_id)
    return name, direction


def extract_metrics(text: str, label: str = "metric", direction: str | None = None) -> list[dict]:
    """Metrics from <metric>2.23</metric> / <metric>2.23<\\metric> tags: [{label, v, dir}].

    No text heuristics: the value must be wrapped in a tag. Code in backticks
    and ``` blocks is ignored (the tag there may just be an example).
    label/direction are task metric metadata supplied by the caller (dir = low/high).
    """
    text = CODE_SPAN_RE.sub(" ", CODE_FENCE_RE.sub(" ", text))
    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for m in METRIC_TAG_RE.finditer(text):
        content = m.group("content")
        vm = METRIC_VALUE_RE.search(content)
        if not vm:
            continue
        try:
            v = float(vm.group(0).replace(",", "."))
        except ValueError:
            continue
        if "%" in content:
            v /= 100.0
        key = (label, round(v, 5))
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": label, "v": round(v, 5), "dir": direction})
    return out[:20]


def extract_metric(ms: list[dict], direction: str | None = None) -> str | None:
    """String for the tooltip/card: the best of the note's tagged metrics."""
    if not ms:
        return None
    if direction == "low":
        best = min(ms, key=lambda m: m["v"])
    else:
        best = max(ms, key=lambda m: m["v"])
    return f'{best["label"]} {best["v"]:.4f}'


def message_markdown(text: str, limit: int = 6000) -> str:
    """Raw message markdown without the first heading (the card displays it)."""
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return body[:limit]


def extract_tags(text: str) -> list[str]:
    stripped = LINK_RE.sub(" ", text)
    tags: list[str] = []
    seen: set[str] = set()
    for m in TAG_RE.finditer(stripped):
        t = m.group(1).lower()
        if not re.search(r"[A-Za-zА-Яа-яЁё]", t):
            continue
        t = TAG_ALIASES.get(t, t)
        if t not in seen:
            seen.add(t)
            tags.append(t)
    return tags


def input_link_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines()[:15]:
        if INPUT_LINE_RE.match(line.lstrip("*_> ")):
            names.update(m.group(1).strip() for m in LINK_RE.finditer(line))
        elif line.strip():
            continue
    return names


def build_graph(bb: Path, comp_id: str | None = None) -> dict:
    md_files = {p.name: p for p in sorted(bb.glob("*.md"))}
    other_files = {
        p.name: p
        for p in sorted(bb.iterdir())
        if p.is_file() and p.suffix.lower() != ".md" and not p.name.startswith(".")
    }
    lower_md = {name.lower(): name for name in md_files}

    agent_counter: Counter[str] = Counter()
    for name in md_files:
        if NAME_TS_RE.match(name):
            continue
        prefix = name.split("_", 1)[0]
        if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]{2,}", prefix):
            agent_counter[prefix] += 1
    agents = [a for a, c in agent_counter.most_common() if c >= 2]
    agent_colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(agents)}
    agent_colors.setdefault("system", "#64748b")
    agent_colors.setdefault("other", "#94a3b8")

    fallback_date = min(
        (datetime.fromtimestamp(p.stat().st_mtime).date() for p in md_files.values()),
        default=date.today(),
    )
    base = load_base_date(bb, fallback_date)
    manifest = load_mtime_manifest(bb)

    nodes: dict[str, dict] = {}
    docs: dict[str, dict] = {}
    metric_name, metric_dir = load_task_metric(bb, comp_id)

    def ensure_agent_node(name: str, kind: str) -> dict:
        if kind == "artifact":
            nid = f"artifact:{name}"
            node = {
                "id": nid, "label": name, "title": name, "author": "артефакт",
                "kind": "artifact", "verdict": None, "time": None, "minutes": None,
                "frac": None, "excerpt": "", "metric": None, "file": name,
            }
        else:
            nid = f"missing:{name}"
            node = {
                "id": nid, "label": name, "title": f"битая ссылка: {name}", "author": "missing",
                "kind": "missing", "verdict": None, "time": None, "minutes": None,
                "frac": None, "excerpt": "", "metric": None, "file": None,
            }
        nodes[nid] = node
        return node

    for name, path in md_files.items():
        text = path.read_text(errors="replace")
        is_system = bool(NAME_TS_RE.match(name))
        title = first_title(text, name)
        dt, minutes = parse_doc_time(path, title, text, base, manifest)
        prefix = "system" if is_system else name.split("_", 1)[0]
        if prefix not in agent_colors:
            prefix = "other"
        stem = name[:-3] if name.endswith(".md") else name
        # The SYSTEM grade.py score is tagged as score — drawn as a dashed line;
        # other points are the task metric (direction from its metadata)
        is_grade = is_system and "grade" in name.lower()
        tag_label = "score" if is_grade else (metric_name or "metric")
        tag_dir = None if is_grade else metric_dir
        node_metrics = extract_metrics(text, label=tag_label, direction=tag_dir)
        node = {
            "id": name,
            "label": stem,
            "title": title,
            "author": prefix,
            "kind": "system" if is_system else "doc",
            "verdict": "system" if is_system else verdict_of(name, title, text),
            "time": dt.isoformat(sep=" ") if dt else None,
            "minutes": minutes,
            "frac": None,
            "excerpt": clean_text(text)[:600],
            "metric": extract_metric(node_metrics, tag_dir),
            "file": name,
            "tags": extract_tags(text),
            "md": message_markdown(text),
            "metrics": node_metrics,
        }
        nodes[name] = node
        docs[name] = {"node": node, "inputs": input_link_names(text), "text": text}

    if not manifest:
        minute_counts = Counter(d["node"]["minutes"] for d in docs.values() if d["node"]["minutes"] is not None)
        total = sum(minute_counts.values())
        if total >= 5 and minute_counts.most_common(1)[0][1] >= 0.8 * total:
            print("WARN: mtime доски вырожден (>=80% файлов в одну минуту) — время, похоже, "
                  "потеряно при переносе из MLM. Запусти в контейнере "
                  "`bin/py/blackboard_mtime.py save`, а после pull — `restore`.", file=sys.stderr)

    edges: dict[tuple[str, str], dict] = {}
    unresolved: Counter[str] = Counter()
    for name, doc in docs.items():
        raw_names = [m.group(1).strip() for m in LINK_RE.finditer(doc["text"])]
        for raw in raw_names:
            target = raw[len("missing:"):] if raw.startswith("missing:") else raw
            if target in md_files:
                tid, tkind = target, "doc"
            elif target + ".md" in md_files:
                tid, tkind = target + ".md", "doc"
            elif target.lower() in lower_md:
                tid, tkind = lower_md[target.lower()], "doc"
            elif target in other_files:
                tid, tkind = f"artifact:{target}", "artifact"
            else:
                tid, tkind = f"missing:{target}", "missing"
            if tkind == "missing":
                unresolved[target] += 1
            node = nodes.get(tid) or ensure_agent_node(target, tkind)
            source = name
            key = (source, node["id"])
            edge = edges.get(key)
            if edge is None:
                edge = {"s": source, "t": node["id"], "input": False, "tag": False, "ref": 0, "weight": 0}
                edges[key] = edge
            edge["weight"] += 1
            if raw in doc["inputs"]:
                edge["input"] = True
            else:
                edge["ref"] += 1

    tag_docs: dict[str, list[str]] = defaultdict(list)
    for name, doc in docs.items():
        for t in doc["node"]["tags"]:
            tag_docs[t].append(name)

    tag_agents: dict[str, Counter[str]] = defaultdict(Counter)
    for t, ids in tag_docs.items():
        for nid in ids:
            tag_agents[t][nodes[nid]["author"]] += 1

    for t in sorted(tag_docs):
        ids = tag_docs[t]
        timed = sorted((nodes[n]["minutes"], nodes[n]["time"]) for n in ids if nodes[n]["minutes"] is not None)
        first_min, first_iso = timed[0] if timed else (None, None)
        nid = f"tag:{t}"
        nodes[nid] = {
            "id": nid, "label": f"#{t}", "title": f"тег #{t}", "author": "tag",
            "kind": "tag", "verdict": None, "time": first_iso, "minutes": first_min,
            "frac": None, "excerpt": "", "metric": None, "file": None,
            "tags": [], "tag": t, "count": len(ids),
        }
        for n in ids:
            edges[(n, nid)] = {"s": n, "t": nid, "input": False, "tag": True, "ref": 0, "weight": 1}

    all_minutes = [n["minutes"] for n in nodes.values() if n["minutes"] is not None]
    t0, t1 = (min(all_minutes), max(all_minutes)) if all_minutes else (0, 1)
    span = max(t1 - t0, 1)
    for n in nodes.values():
        if n["kind"] == "tag":
            continue
        if n["minutes"] is not None:
            n["frac"] = round((n["minutes"] - t0) / span, 4)
    for t, ids in tag_docs.items():
        fr = [nodes[n]["frac"] for n in ids if nodes[n]["frac"] is not None]
        if fr:
            nodes[f"tag:{t}"]["frac"] = round(sum(fr) / len(fr), 4)

    in_deg: Counter[str] = Counter()
    out_deg: Counter[str] = Counter()
    for (s, t) in edges:
        out_deg[s] += 1
        in_deg[t] += 1
    for nid, n in nodes.items():
        n["in"] = in_deg[nid]
        n["out"] = out_deg[nid]

    edge_list = []
    for edge in edges.values():
        target = nodes[edge["t"]]
        is_tag = bool(edge.get("tag"))
        kind = "tag" if is_tag else ("input" if edge["input"] else "cite")
        edge_list.append({
            "s": edge["s"], "t": edge["t"],
            "input": 1 if edge["input"] else 0,
            "artifact": 1 if target["kind"] == "artifact" else 0,
            "tag": 1 if is_tag else 0,
            "kind": kind,
            "weight": edge["weight"],
        })

    order = {name: i for i, name in enumerate(md_files)}
    node_list = sorted(
        nodes.values(),
        key=lambda n: (n["minutes"] is None, n["minutes"] if n["minutes"] is not None else 0,
                       order.get(n["file"] or "", 10**6)),
    )
    last = max(len(node_list) - 1, 1)
    for i, n in enumerate(node_list):
        n["order"] = i
        n["ofrac"] = round(i / last, 4)

    tag_list = []
    for t in sorted(tag_docs):
        ids = tag_docs[t]
        mins = [nodes[n]["minutes"] for n in ids if nodes[n]["minutes"] is not None]
        tag_list.append({
            "name": t,
            "count": len(ids),
            "first": min(mins) if mins else None,
            "last": max(mins) if mins else None,
            "agents": dict(tag_agents[t].most_common()),
            "docs": ids,
        })
    tag_list.sort(key=lambda x: (-x["count"], x["name"]))

    return {
        "generated_at": datetime.now().strftime("%F %H:%M:%S"),
        "source": str(bb),
        "base_date": base.isoformat(),
        "t0": t0,
        "t1": t1,
        "agents": agent_colors,
        "verdict_colors": VERDICT_COLORS,
        "tag_color": TAG_COLOR,
        "stats": {
            "docs": sum(1 for n in nodes.values() if n["kind"] in ("doc", "system")),
            "artifacts": sum(1 for n in nodes.values() if n["kind"] == "artifact"),
            "missing": sum(1 for n in nodes.values() if n["kind"] == "missing"),
            "tags": len(tag_docs),
            "edges": len(edge_list),
            "input_edges": sum(1 for e in edge_list if e["input"]),
            "tag_edges": sum(1 for e in edge_list if e["tag"]),
            "metric_points": sum(len(n.get("metrics", [])) for n in nodes.values()),
            "dangling": sum(unresolved.values()),
        },
        "dangling": unresolved.most_common(20),
        "tags": tag_list,
        "nodes": node_list,
        "edges": edge_list,
    }


TEMPLATE_PATH = Path(__file__).with_name("graph_template.html")


def render_html(graph: dict) -> str:
    stats = graph["stats"]
    title = (f"Граф доски роя: {stats['docs']} записей, {stats['edges']} связей, "
             f"{stats['tags']} тегов")
    data_js = json.dumps(graph, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    if TEMPLATE_PATH.is_file():
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = TEMPLATE
    return template.replace("__TITLE__", title).replace("__DATA__", data_js)


def main() -> None:
    a = parse_args()
    bb = a.bb if a.bb.is_absolute() else ROOT / a.bb
    if not bb.is_dir():
        sys.exit(f"нет каталога доски: {bb}")
    out = a.out or (bb / "graph.html")
    if not out.is_absolute():
        out = ROOT / out
    json_out = None if a.no_json else (a.json_out or (bb / "graph.json"))
    if json_out is not None and not json_out.is_absolute():
        json_out = ROOT / json_out

    graph = build_graph(bb, a.competition)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(graph), encoding="utf-8")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    s = graph["stats"]
    print(f"граф доски: {s['docs']} записей, {s['artifacts']} артефактов, "
          f"{s['missing']} битых ссылок, {s['tags']} тегов, "
          f"{s['edges']} связей ({s['input_edges']} — входные, {s['tag_edges']} — к тегам)")
    print(f"html: {out}")
    if json_out is not None:
        print(f"json: {json_out}")
    if a.open:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener, str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            print(f"не удалось открыть браузер: {e}", file=sys.stderr)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bar:#fff;--line:#e2e8f0;--ink:#0f172a;--mut:#64748b;--bg:#f8fafc}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font:13px/1.4 'DejaVu Sans','Segoe UI',Arial,sans-serif;color:var(--ink);background:var(--bg)}
#bar{position:fixed;top:0;left:0;right:0;height:46px;background:var(--bar);border-bottom:1px solid var(--line);
     display:flex;align-items:center;gap:10px;padding:0 12px;z-index:30;box-shadow:0 1px 4px rgba(0,0,0,.05)}
#bar .t{font-weight:700;font-size:15px;white-space:nowrap}
#bar .stat{font-size:12px;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#bar .spacer{flex:1}
button{border:1px solid #cbd5e1;background:#fff;border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer;color:var(--ink)}
button:hover{background:#f1f5f9}
button.on{background:#0f172a;color:#fff;border-color:#0f172a}
#side{position:fixed;top:46px;left:0;bottom:0;width:290px;background:#fff;border-right:1px solid var(--line);
      overflow-y:auto;padding:12px 12px 40px;z-index:20}
#side h3{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:#334155;margin:16px 0 7px}
#search{width:100%;padding:7px 9px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px}
.ck{display:flex;align-items:center;gap:7px;font-size:12.5px;padding:3px 0;cursor:pointer}
.ck input{margin:0}
.agent-row{display:flex;align-items:center;gap:7px;font-size:12.5px;padding:2px 0;cursor:pointer}
.sw{width:11px;height:11px;border-radius:3px;flex:none}
.trow{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mut);margin:4px 0}
.trow input{flex:1}
.trow .val{width:42px;text-align:right;font-variant-numeric:tabular-nums;color:#334155}
.legend{font-size:12px;color:#334155;display:flex;align-items:center;gap:7px;margin:3px 0}
.legend .ring{width:12px;height:12px;border-radius:50%;border:2.5px solid;flex:none}
.legend .sq{width:11px;height:11px;background:#64748b;border-radius:2px;flex:none}
#detail{margin-top:10px}
#detail .card{border:1px solid var(--line);border-radius:10px;padding:10px;background:#fff}
#detail .h{font-weight:700;font-size:13px;overflow-wrap:anywhere}
#detail .meta{font-size:11.5px;color:var(--mut);margin:4px 0 7px}
#detail .badge{display:inline-block;font-size:10.5px;border-radius:999px;padding:1px 7px;margin-right:4px;border:1px solid}
#detail .ex{font-size:11.5px;color:#475569;max-height:120px;overflow:auto;border-top:1px dashed var(--line);padding-top:6px;margin-top:6px;white-space:pre-wrap}
#detail .links{margin-top:7px;font-size:11.5px}
#detail .links b{display:block;margin:6px 0 3px;font-size:11px;color:#334155;text-transform:uppercase;letter-spacing:.4px}
.chip{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;padding:1px 6px;margin:2px 3px 0 0;
      cursor:pointer;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip:hover{background:#e2e8f0}
#stage{position:fixed;top:46px;left:290px;right:0;bottom:0}
#cv{display:block;width:100%;height:100%;cursor:grab}
#cv.drag{cursor:grabbing}
#tip{position:fixed;display:none;background:#0f172a;color:#f1f5f9;padding:8px 11px;border-radius:8px;font-size:12px;
     max-width:380px;z-index:50;box-shadow:0 8px 22px rgba(0,0,0,.28);pointer-events:none;line-height:1.45}
#tip b{font-size:12.5px}
#tip .m{color:#fbbf24}
#tip .g{color:#94a3b8}
</style></head>
<body>
<div id="bar">
  <span class="t">Граф доски роя</span>
  <span class="stat" id="stats"></span>
  <span class="spacer"></span>
  <button id="btn-time" class="on" title="раскладывать узлы слева-направо по времени публикации">Ось времени</button>
  <button id="btn-phys" class="on" title="включить/остановить симуляцию раскладки">Физика</button>
  <button id="btn-fit" title="показать весь граф">Уместить</button>
  <button id="btn-reset" title="снять выделение и фильтры">Сброс</button>
</div>
<aside id="side">
  <input id="search" placeholder="поиск: имя, заголовок, текст…">
  <h3>Агенты</h3><div id="agents"></div>
  <h3>Типы узлов</h3>
  <label class="ck"><input type="checkbox" id="ck-art" checked> артефакты (csv/npy)</label>
  <label class="ck"><input type="checkbox" id="ck-miss" checked> битые ссылки</label>
  <label class="ck"><input type="checkbox" id="ck-sys" checked> SYSTEM-сообщения</label>
  <h3>Время (<span id="t-span"></span>)</h3>
  <div class="trow"><span>с</span><input type="range" id="t0"><span class="val" id="t0v"></span></div>
  <div class="trow"><span>по</span><input type="range" id="t1"><span class="val" id="t1v"></span></div>
  <label class="ck" style="margin-top:4px"><input type="checkbox" id="ck-notime" checked> узлы без времени</label>
  <h3>Легенда</h3>
  <div id="legend"></div>
  <h3>Выбранный узел</h3>
  <div id="detail"><div class="legend" style="color:#64748b">клик по узлу — предшественники (зелёные), потомки (оранжевые)</div></div>
</aside>
<div id="stage"><canvas id="cv"></canvas></div>
<div id="tip"></div>
<script>
"use strict";
const DATA = __DATA__;
const VERD = DATA.verdict_colors;
const AGENTS = DATA.agents;
const AGENT_KEYS = Object.keys(AGENTS);
const AGENT_IDX = new Map(AGENT_KEYS.map((a, i) => [a, i]));
const WORLD = {w: 1680, h: 940};
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const stage = document.getElementById('stage');
const tip = document.getElementById('tip');

const nodes = DATA.nodes.map(n => Object.assign({}, n));
const byId = new Map(nodes.map(n => [n.id, n]));
const edges = DATA.edges.filter(e => byId.has(e.s) && byId.has(e.t))
  .map(e => Object.assign({}, e, {vis: 1}));
const adj = new Map(nodes.map(n => [n.id, {out: [], in: []}]));
for (const e of edges) { adj.get(e.s).out.push(e); adj.get(e.t).in.push(e); }

let alpha = 1, simOn = true, timeAxis = true, hovered = null, selected = null;
let ancSet = new Set(), desSet = new Set(), matchSet = null;
const view = {x: WORLD.w / 2, y: WORLD.h / 2, k: 0.55};
let W = 0, H = 0, DPR = 1;

nodes.forEach(n => {
  const ai = Math.max(0, AGENT_KEYS.indexOf(n.author));
  const lane = AGENT_KEYS.length > 1 ? ai / (AGENT_KEYS.length - 1) : 0.5;
  const xf = n.ofrac != null ? n.ofrac : (n.frac != null ? n.frac : 0.5);
  n.x = 80 + 1500 * xf + (Math.random() - 0.5) * 120;
  n.y = 70 + lane * 790 + (Math.random() - 0.5) * 200;
  n.vx = 0; n.vy = 0;
  n.deg = (n.in || 0) + (n.out || 0);
  n.r = 4.5 + Math.min(11, Math.sqrt(n.in || 0) * 2.1);
  n.vis = 1;
});

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtTime(min) {
  if (min == null) return '—';
  const h = Math.floor(min / 60) % 24, m = Math.round(min % 60);
  return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
}
function stem(n) { return n.label.replace(/\.md$/, ''); }
function nodeColor(n) { return n.kind === 'system' ? '#64748b' : (n.kind === 'artifact' ? '#64748b' : (n.kind === 'missing' ? '#b91c1c' : (AGENTS[n.author] || '#94a3b8'))); }

function bfs(id, dir) {
  const seen = new Set(), q = [id];
  while (q.length) {
    const u = q.pop();
    for (const e of adj.get(u)[dir]) {
      const v = dir === 'out' ? e.t : e.s;
      if (v !== id && !seen.has(v)) { seen.add(v); q.push(v); }
    }
  }
  return seen;
}
function setSelection(id) {
  selected = id;
  if (id) { ancSet = bfs(id, 'out'); desSet = bfs(id, 'in'); }
  else { ancSet = new Set(); desSet = new Set(); }
  renderDetail();
  draw();
}

function computeMatch() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  if (!q) { matchSet = null; return; }
  matchSet = new Set();
  for (const n of nodes) {
    if ((n.label + ' ' + n.title + ' ' + (n.excerpt || '')).toLowerCase().includes(q)) matchSet.add(n.id);
  }
}

const filterState = {agents: new Set(AGENT_KEYS), art: true, miss: true, sys: true, notime: true, t0: DATA.t0, t1: DATA.t1};

function applyFilters() {
  for (const n of nodes) {
    let vis = 1;
    if (n.author === 'system' && !filterState.sys) vis = 0;
    else if (n.kind === 'artifact' && !filterState.art) vis = 0;
    else if (n.kind === 'missing' && !filterState.miss) vis = 0;
    else if (n.minutes == null) vis = filterState.notime ? 1 : 0;
    else if (n.minutes < filterState.t0 || n.minutes > filterState.t1) vis = 0;
    if (vis && AGENT_KEYS.includes(n.author) && !filterState.agents.has(n.author)) vis = 0;
    n.vis = vis;
  }
  for (const e of edges) e.vis = (byId.get(e.s).vis && byId.get(e.t).vis) ? 1 : 0;
  const nv = nodes.filter(n => n.vis).length, ev = edges.filter(e => e.vis).length;
  document.getElementById('stats').textContent =
    `видно ${nv}/${nodes.length} узлов, ${ev}/${edges.length} связей · всего ${DATA.stats.docs} записей` +
    (DATA.stats.missing ? ` · битых ссылок ${DATA.stats.missing}` : '') +
    ` · ${DATA.generated_at}`;
  updateToggleButtons();
}

function buildLegend() {
  const lg = document.getElementById('legend');
  let html = '';
  for (const a of AGENT_KEYS) {
    html += `<div class="legend"><span class="sw" style="background:${AGENTS[a]}"></span>${esc(a)}</div>`;
  }
  html += `<div class="legend"><span class="sq"></span>артефакт (данные/код)</div>`;
  html += `<div class="legend"><span class="ring" style="border-color:#b91c1c;border-style:dashed"></span>битая ссылка</div>`;
  html += `<div class="legend" style="margin-top:7px">рамка вердикта:</div>`;
  for (const [v, c] of Object.entries(VERD)) {
    if (v === 'system') continue;
    html += `<div class="legend"><span class="ring" style="border-color:${c}"></span>${esc(v)}</div>`;
  }
  lg.innerHTML = html;
  const ag = document.getElementById('agents');
  ag.innerHTML = AGENT_KEYS.map(a =>
    `<label class="agent-row"><input type="checkbox" data-agent="${esc(a)}" checked>
     <span class="sw" style="background:${AGENTS[a]}"></span>${esc(a)}</label>`).join('');
  ag.querySelectorAll('input').forEach(cb => cb.addEventListener('change', () => {
    if (cb.checked) filterState.agents.add(cb.dataset.agent); else filterState.agents.delete(cb.dataset.agent);
    applyFilters(); draw();
  }));
}

function renderDetail() {
  const box = document.getElementById('detail');
  if (!selected) {
    box.innerHTML = '<div class="legend" style="color:#64748b">клик по узлу — предшественники (зелёные), потомки (оранжевые)</div>';
    return;
  }
  const n = byId.get(selected);
  const outs = adj.get(selected).out.map(e => byId.get(e.t));
  const ins = adj.get(selected).in.map(e => byId.get(e.s));
  let badges = '';
  if (n.verdict && n.verdict !== 'system') badges += `<span class="badge" style="border-color:${VERD[n.verdict] || '#cbd5e1'};color:${VERD[n.verdict] || '#475569'}">${esc(n.verdict)}</span>`;
  if (n.kind === 'artifact') badges += `<span class="badge" style="border-color:#94a3b8;color:#475569">артефакт</span>`;
  if (n.kind === 'missing') badges += `<span class="badge" style="border-color:#b91c1c;color:#b91c1c">битая ссылка</span>`;
  const chip = (x, dir) => `<span class="chip" data-goto="${esc(x.id)}" title="${esc(stem(x))}">${esc(stem(x))}</span>`;
  let html = `<div class="card">
    <div class="h">${esc(n.title || stem(n))}</div>
    <div class="meta">${esc(stem(n))} · автор: <b>${esc(n.author)}</b> · ${esc(fmtTime(n.minutes))} MSK
      ${n.metric ? ' · метрика <b class="m">' + esc(n.metric) + '</b>' : ''}
      · ссылается: ${outs.length} · цитируют: ${ins.length}<br>${badges}</div>`;
  if (n.excerpt) html += `<div class="ex">${esc(n.excerpt.slice(0, 420))}</div>`;
  if (outs.length) html += `<div class="links"><b>опирается на (предшественники)</b>${outs.map(x => chip(x, 'out')).join('')}</div>`;
  if (ins.length) html += `<div class="links"><b>использовали / цитируют (потомки)</b>${ins.map(x => chip(x, 'in')).join('')}</div>`;
  html += '</div>';
  box.innerHTML = html;
  box.querySelectorAll('.chip').forEach(el => el.addEventListener('click', () => setSelection(el.dataset.goto)));
  if (selected) box.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

function tick() {
  if (!simOn || alpha < 0.01) return;
  alpha *= 0.99;
  const vs = nodes.filter(n => n.vis);
  for (let i = 0; i < vs.length; i++) {
    const a = vs[i];
    for (let j = i + 1; j < vs.length; j++) {
      const b = vs[j];
      let dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy;
      if (d2 > 360000) continue;
      let d = Math.sqrt(d2);
      if (d2 < 0.25) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d = Math.hypot(dx, dy) || 0.5; }
      let f = Math.min(2000 / d2, 3.4) * alpha;
      const minD = a.r + b.r + 22;
      if (d < minD) f += (minD - d) * 0.35 * alpha;
      dx /= d; dy /= d;
      a.vx -= dx * f; a.vy -= dy * f; b.vx += dx * f; b.vy += dy * f;
    }
  }
  for (const e of edges) {
    if (!e.vis) continue;
    const a = byId.get(e.s), b = byId.get(e.t);
    let dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
    const L = e.artifact ? 80 : 46 + 70 / (1 + Math.min(a.deg, 10));
    const f = (d - L) * 0.0025 * alpha;
    dx /= d; dy /= d;
    a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
  }
  for (const n of vs) {
    if (n.fixed) continue;
    const xf = n.ofrac != null ? n.ofrac : n.frac;
    if (timeAxis) {
      if (xf != null) {
        const tx = 80 + 1520 * xf;
        n.vx += (tx - n.x) * 0.013 * alpha;
      }
      const lanes = Math.max(AGENT_KEYS.length - 1, 1);
      const ai = AGENT_IDX.has(n.author) ? AGENT_IDX.get(n.author) : lanes;
      const ly = 70 + (ai / lanes) * 790;
      n.vy += (ly - n.y) * 0.03 * alpha;
    }
    n.vx += (WORLD.w / 2 - n.x) * 0.00035 * alpha;
    n.vy += (WORLD.h / 2 - n.y) * (timeAxis ? 0.00005 : 0.0006) * alpha;
    n.vx *= 0.87; n.vy *= 0.87;
    n.x += n.vx; n.y += n.vy;
    n.x = Math.max(-600, Math.min(WORLD.w + 600, n.x));
    n.y = Math.max(-300, Math.min(WORLD.h + 300, n.y));
  }
}

function toWorld(mx, my) {
  return {x: view.x + (mx - W / 2) / view.k, y: view.y + (my - H / 2) / view.k};
}
function hitNode(mx, my) {
  const p = toWorld(mx, my);
  let best = null, bestD = 1e9;
  for (const n of nodes) {
    if (!n.vis) continue;
    const rr = n.r + 6;
    const d = Math.hypot(n.x - p.x, n.y - p.y);
    if (d < rr && d < bestD) { best = n; bestD = d; }
  }
  return best;
}

function draw() {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(W / 2, H / 2);
  ctx.scale(view.k, view.k);
  ctx.translate(-view.x, -view.y);
  const lw = 1.05 / view.k;

  for (const e of edges) {
    if (!e.vis) continue;
    const a = byId.get(e.s), b = byId.get(e.t);
    let state = 'norm';
    if (selected) {
      if (ancSet.has(e.t) && ancSet.has(e.s)) state = 'anc';
      else if (desSet.has(e.t) && desSet.has(e.s)) state = 'des';
      else state = 'dim';
    } else if (matchSet) {
      state = (matchSet.has(e.s) || matchSet.has(e.t)) ? 'norm' : 'dim';
    }
    let col = e.artifact ? '#94a3b8' : '#cbd5e1';
    if (state === 'anc') col = '#16a34a';
    else if (state === 'des') col = '#f59e0b';
    ctx.globalAlpha = state === 'dim' ? 0.07 : (e.artifact ? 0.75 : 0.55);
    ctx.strokeStyle = col;
    ctx.lineWidth = (state === 'dim' ? 1 : (e.input ? 1.35 : 1)) * lw;
    ctx.setLineDash(e.input && !e.artifact ? [] : (e.artifact ? [3 / view.k, 3 / view.k] : [5 / view.k, 4 / view.k]));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
    const d = Math.hypot(b.x - a.x, b.y - a.y) || 1;
    const ux = (b.x - a.x) / d, uy = (b.y - a.y) / d;
    const ex = b.x - ux * (b.r + 5), ey = b.y - uy * (b.r + 5);
    const ah = 4.6 / view.k;
    ctx.globalAlpha = state === 'dim' ? 0.07 : 0.9;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ux * ah - uy * ah * 0.5, ey - uy * ah + ux * ah * 0.5);
    ctx.lineTo(ex - ux * ah + uy * ah * 0.5, ey - uy * ah - ux * ah * 0.5);
    ctx.closePath();
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  const labelCand = [];

  for (const n of nodes) {
    if (!n.vis) continue;
    let state = 'norm';
    if (selected) {
      if (n.id === selected) state = 'sel';
      else if (ancSet.has(n.id)) state = 'anc';
      else if (desSet.has(n.id)) state = 'des';
      else state = 'dim';
    } else if (matchSet) {
      state = matchSet.has(n.id) ? 'sel' : 'dim';
    }
    const dimmed = state === 'dim';
    ctx.globalAlpha = dimmed ? 0.09 : 1;
    const col = nodeColor(n);
    const r = n.r;
    if (n.kind === 'artifact') {
      ctx.fillStyle = col;
      ctx.fillRect(n.x - r * 0.85, n.y - r * 0.85, r * 1.7, r * 1.7);
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = lw;
      ctx.strokeRect(n.x - r * 0.85, n.y - r * 0.85, r * 1.7, r * 1.7);
    } else {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, 6.2832);
      ctx.fillStyle = col;
      ctx.fill();
      if (n.kind === 'missing') {
        ctx.setLineDash([3 / view.k, 2.5 / view.k]);
        ctx.strokeStyle = '#b91c1c';
        ctx.lineWidth = 1.6 * lw;
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (n.verdict && n.verdict !== 'system' && VERD[n.verdict]) {
        ctx.strokeStyle = VERD[n.verdict];
        ctx.lineWidth = 2.4 * lw;
        ctx.stroke();
      } else {
        ctx.strokeStyle = 'rgba(15,23,42,.35)';
        ctx.lineWidth = lw;
        ctx.stroke();
      }
    }
    let ring = null;
    if (state === 'sel') ring = '#111827';
    else if (state === 'anc') ring = '#16a34a';
    else if (state === 'des') ring = '#f59e0b';
    if (ring) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 3.5, 0, 6.2832);
      ctx.strokeStyle = ring;
      ctx.lineWidth = 2.2 * lw;
      ctx.stroke();
    }
    const showLabel = view.k > 1.25 || n.deg >= 10 || n === hovered || n.id === selected || (matchSet && matchSet.has(n.id));
    if (showLabel) labelCand.push(n);
  }

  labelCand.sort((a, b) => labelPrio(b) - labelPrio(a));
  ctx.font = (10.5 / view.k) + 'px "DejaVu Sans",Arial,sans-serif';
  ctx.textAlign = 'center';
  const placed = [];
  for (const n of labelCand) {
    const txt = stem(n).slice(0, 30);
    const tw = ctx.measureText(txt).width;
    const h = 11.5 / view.k;
    const lx = n.x - tw / 2 - 2, ly = n.y + n.r + 2 / view.k;
    let clash = false;
    for (const q of placed) {
      if (lx < q.x + q.w && lx + tw + 4 > q.x && ly < q.y + q.h && ly + h > q.y) { clash = true; break; }
    }
    if (clash) continue;
    placed.push({x: lx, y: ly, w: tw + 4, h: h});
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = 'rgba(255,255,255,.78)';
    ctx.fillRect(lx, ly, tw + 4, h);
    ctx.fillStyle = '#0f172a';
    ctx.fillText(txt, n.x, ly + h - 1.5 / view.k);
  }
  ctx.restore();
  ctx.globalAlpha = 1;
}

function labelPrio(n) {
  return (n.id === selected ? 100 : 0) + (n === hovered ? 90 : 0) +
    (matchSet && matchSet.has(n.id) ? 80 : 0) + Math.min(n.deg, 50);
}

function fit() {
  const vs = nodes.filter(n => n.vis);
  if (!vs.length) return;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const n of vs) { x0 = Math.min(x0, n.x - n.r); y0 = Math.min(y0, n.y - n.r); x1 = Math.max(x1, n.x + n.r); y1 = Math.max(y1, n.y + n.r); }
  const bw = Math.max(x1 - x0 + 120, 200), bh = Math.max(y1 - y0 + 120, 200);
  view.k = Math.max(0.12, Math.min(2.5, Math.min(W / bw, H / bh) * 0.92));
  view.x = (x0 + x1) / 2; view.y = (y0 + y1) / 2;
  draw();
}

let dragNode = null, panning = null, moved = 0, downPos = null, interacted = false;
cv.addEventListener('mousedown', ev => {
  interacted = true;
  const rect = cv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  downPos = {x: mx, y: my}; moved = 0;
  const n = hitNode(mx, my);
  if (n) { dragNode = n; n.fixed = true; alpha = Math.max(alpha, 0.25); }
  else { panning = {x: mx, y: my, vx: view.x, vy: view.y}; cv.classList.add('drag'); }
  ev.preventDefault();
});
window.addEventListener('mousemove', ev => {
  const rect = cv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  if (dragNode) {
    const p = toWorld(mx, my);
    dragNode.x = p.x; dragNode.y = p.y; moved = 99;
  } else if (panning) {
    view.x = panning.vx - (mx - panning.x) / view.k;
    view.y = panning.vy - (my - panning.y) / view.k;
    moved = 99;
  } else {
    const n = hitNode(mx, my);
    if (n !== hovered) { hovered = n; if (!n) tip.style.display = 'none'; else showTip(n, ev.clientX, ev.clientY); }
    else if (n) showTip(n, ev.clientX, ev.clientY);
  }
});
window.addEventListener('mouseup', ev => {
  if (dragNode) { dragNode.fixed = false; dragNode = null; }
  if (panning) { panning = null; cv.classList.remove('drag'); }
  if (downPos && moved < 5) {
    const rect = cv.getBoundingClientRect();
    const n = hitNode(ev.clientX - rect.left, ev.clientY - rect.top);
    setSelection(n ? n.id : null);
  }
  downPos = null;
});
cv.addEventListener('click', ev => ev.preventDefault());
cv.addEventListener('mouseleave', () => { tip.style.display = 'none'; hovered = null; draw(); });
cv.addEventListener('wheel', ev => {
  interacted = true;
  const rect = cv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const p = toWorld(mx, my);
  const factor = Math.exp(-ev.deltaY * 0.0011);
  view.k = Math.max(0.1, Math.min(4, view.k * factor));
  view.x = p.x - (mx - W / 2) / view.k;
  view.y = p.y - (my - H / 2) / view.k;
  draw();
  ev.preventDefault();
}, {passive: false});

function showTip(n, cx, cy) {
  let out = '';
  out += `<b>${esc(stem(n))}</b><br>`;
  if (n.title && n.title !== n.label) out += `<span class="g">${esc(n.title.slice(0, 160))}</span><br>`;
  out += `${esc(n.author)} · ${esc(fmtTime(n.minutes))} MSK · →${n.out} ←${n.in}`;
  if (n.metric) out += ` · <span class="m">${esc(n.metric)}</span>`;
  if (n.kind === 'artifact') out += ' · <span class="g">артефакт</span>';
  if (n.kind === 'missing') out += ' · <span class="g">битая ссылка</span>';
  tip.innerHTML = out;
  tip.style.display = 'block';
  const pad = 14, tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = cx + pad, y = cy + pad;
  if (x + tw > window.innerWidth - 6) x = cx - tw - pad;
  if (y + th > window.innerHeight - 6) y = cy - th - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}

function resize() {
  const rect = stage.getBoundingClientRect();
  DPR = window.devicePixelRatio || 1;
  W = rect.width; H = rect.height;
  cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  applyFilters();
  if (interacted) draw(); else fit();
}
window.addEventListener('resize', resize);

function updateToggleButtons() {
  document.getElementById('btn-time').className = timeAxis ? 'on' : '';
  document.getElementById('btn-phys').className = simOn ? 'on' : '';
}
document.getElementById('btn-time').addEventListener('click', () => { timeAxis = !timeAxis; alpha = Math.max(alpha, 0.5); updateToggleButtons(); });
document.getElementById('btn-phys').addEventListener('click', () => { simOn = !simOn; if (simOn) alpha = Math.max(alpha, 0.4); updateToggleButtons(); });
document.getElementById('btn-fit').addEventListener('click', () => { interacted = false; fit(); });
document.getElementById('btn-reset').addEventListener('click', () => {
  document.getElementById('search').value = '';
  matchSet = null;
  filterState.agents = new Set(AGENT_KEYS);
  filterState.art = filterState.miss = filterState.sys = filterState.notime = true;
  filterState.t0 = DATA.t0; filterState.t1 = DATA.t1;
  document.querySelectorAll('#agents input').forEach(cb => { cb.checked = true; });
  for (const id of ['ck-art', 'ck-miss', 'ck-sys', 'ck-notime']) document.getElementById(id).checked = true;
  document.getElementById('t0').value = DATA.t0; document.getElementById('t1').value = DATA.t1;
  syncTimeLabels();
  setSelection(null); applyFilters(); fit();
});
document.getElementById('search').addEventListener('input', () => { computeMatch(); draw(); });
for (const id of ['ck-art', 'ck-miss', 'ck-sys', 'ck-notime']) {
  document.getElementById(id).addEventListener('change', ev => {
    if (id === 'ck-art') filterState.art = ev.target.checked;
    if (id === 'ck-miss') filterState.miss = ev.target.checked;
    if (id === 'ck-sys') filterState.sys = ev.target.checked;
    if (id === 'ck-notime') filterState.notime = ev.target.checked;
    applyFilters(); draw();
  });
}
function syncTimeLabels() {
  document.getElementById('t0v').textContent = fmtTime(+document.getElementById('t0').value);
  document.getElementById('t1v').textContent = fmtTime(+document.getElementById('t1').value);
  document.getElementById('t-span').textContent = fmtTime(DATA.t0) + '–' + fmtTime(DATA.t1);
}
for (const id of ['t0', 't1']) {
  const el = document.getElementById(id);
  el.min = DATA.t0; el.max = DATA.t1; el.step = 5; el.value = id === 't0' ? DATA.t0 : DATA.t1;
  el.addEventListener('input', () => {
    let a = +document.getElementById('t0').value, b = +document.getElementById('t1').value;
    if (a > b) { if (id === 't0') a = b; else b = a; document.getElementById('t0').value = a; document.getElementById('t1').value = b; }
    filterState.t0 = a; filterState.t1 = b;
    syncTimeLabels(); applyFilters(); draw();
  });
}

buildLegend();
syncTimeLabels();
for (let i = 0; i < 520; i++) tick();
resize();
fit();
setSelection(null);
(function frame() { tick(); draw(); requestAnimationFrame(frame); })();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
