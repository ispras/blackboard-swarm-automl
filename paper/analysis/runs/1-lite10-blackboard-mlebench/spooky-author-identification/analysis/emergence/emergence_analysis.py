#!/usr/bin/env python3
"""Emergence analysis for a Sisyphus blackboard run.

The script implements four families of metrics on the archived blackboard:

1. Specialization Index (SI) from normalized Shannon entropy of tag profiles.
2. Behavioral divergence from Jensen-Shannon divergence (JSD) between agents.
3. Temporal convergence from rolling tag windows and permutation tests.
4. Reactivity and influence from explicit [[wiki-link]] citations.

A second, expert-coded workstream taxonomy is analyzed as a robustness check,
because generic hashtags can collapse distinct technical roles into the same
few labels. All figures and tables are written next to this script.

Usage:
    python emergence_analysis.py
    python emergence_analysis.py --experiment /path/to/experiment
    python emergence_analysis.py --bootstrap 5000 --permutations 5000
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from scipy.optimize import curve_fit
from scipy.stats import linregress

AGENTS = ["rigel", "deneb", "arcturus"]
DISPLAY = {
    "rigel": "rigel_xi",
    "deneb": "deneb_upsilon",
    "arcturus": "arcturus_alpha",
}
COLORS = {
    "rigel": "#e11d48",
    "deneb": "#2563eb",
    "arcturus": "#059669",
}
WORKSTREAMS = [
    "ensemble",
    "transformer",
    "training_tech",
    "classical",
    "coordination",
    "submission",
    "negative",
]
WORKSTREAM_LABELS = {
    "ensemble": "Ensembling",
    "transformer": "Transformer training",
    "training_tech": "Training technique",
    "classical": "Classical models",
    "coordination": "Coordination",
    "submission": "Submission / release",
    "negative": "Negative result",
}
WORKSTREAM_COLORS = {
    "ensemble": "#7c3aed",
    "transformer": "#2563eb",
    "training_tech": "#ea580c",
    "classical": "#059669",
    "coordination": "#64748b",
    "submission": "#111827",
    "negative": "#dc2626",
}
TAG_SOURCE_LABEL = "hashtag"
def keyword_workstream(title: str) -> str:
    text = title.lower()
    if re.search(r"blend|stack|combine|meta-?combine", text):
        return "ensemble"
    if re.search(r"final|submission|verify|closing|code added", text):
        return "submission"
    if re.search(r"\b(fgm|seed|pseudo|adversarial)", text):
        return "training_tech"
    if re.search(r"deberta|roberta|xlnet|\bbert\b|electra|ernie|mpnet|bart|distil", text):
        return "transformer"
    if re.search(r"tf-?idf|logistic|classical|lgbm|lightgbm|baseline|linear", text):
        return "classical"
    if re.search(r"negative|weak|bug|blocker", text):
        return "negative"
    return "coordination"


def cohen_kappa(true_labels: list[str], predicted_labels: list[str]) -> float:
    total = len(true_labels)
    if total == 0:
        return float("nan")
    observed = sum(a == b for a, b in zip(true_labels, predicted_labels)) / total
    true_counts = Counter(true_labels)
    predicted_counts = Counter(predicted_labels)
    expected = sum(true_counts[key] * predicted_counts.get(key, 0) for key in true_counts) / (total * total)
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")


@dataclass
class Message:
    id: str
    author: str
    time: datetime
    title: str
    tags: list[str]
    text: str
    workstream: str = ""
    workstream_note: str = ""
    order: int = 0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_experiment = here.parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=Path, default=default_experiment)
    ap.add_argument("--out", type=Path, default=here.parent)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tags-json", type=Path, default=None)
    ap.add_argument("--tag-label", default="hashtag")
    ap.add_argument("--tag-window", type=int, default=16)
    ap.add_argument("--tag-stride", type=int, default=4)
    ap.add_argument("--jsd-window", type=int, default=24)
    return ap.parse_args()


def load_tag_overrides(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "annotations" in data:
        items = data["annotations"]
    elif isinstance(data, dict):
        items = [{"id": key, "tags": value} for key, value in data.items()]
    else:
        raise SystemExit(f"unsupported tags JSON: {path}")
    overrides = {}
    for item in items:
        message_id = str(item.get("id", ""))
        tags = item.get("tags")
        if not message_id or not isinstance(tags, list) or not tags:
            raise SystemExit(f"invalid tag override for {message_id!r} in {path}")
        cleaned = []
        for tag in tags:
            tag = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
            if tag and tag not in cleaned:
                cleaned.append(tag)
        if not cleaned:
            raise SystemExit(f"empty tag override for {message_id!r}")
        overrides[message_id] = cleaned
    return overrides


def load_messages(experiment: Path, labels_path: Path) -> tuple[list[Message], dict]:
    graph_path = experiment / "blackboard" / "graph.json"
    if not graph_path.is_file():
        raise SystemExit(
            f"missing {graph_path}; regenerate it with "
            f"bin/blackboard_graph.py --bb {experiment / 'blackboard'}"
        )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    labels: dict[str, tuple[str, str]] = {}
    if labels_path.is_file():
        frame = pd.read_csv(labels_path)
        labels = {
            str(row["file"]): (str(row["workstream"]), str(row.get("note", "")))
            for _, row in frame.iterrows()
        }
    messages: list[Message] = []
    for node in graph["nodes"]:
        if node.get("kind") != "doc" or node.get("author") not in AGENTS:
            continue
        stamp = node.get("time")
        if not stamp:
            continue
        tags = [str(tag).lower() for tag in (node.get("tags") or []) if str(tag).strip()]
        tags = [tag for tag in tags if any(ch.isalpha() for ch in tag)]
        workstream = ""
        workstream_note = ""
        if node["id"] in labels:
            workstream, workstream_note = labels[node["id"]]
        messages.append(
            Message(
                id=node["id"],
                author=node["author"],
                time=datetime.fromisoformat(stamp),
                title=str(node.get("title") or node["id"]),
                tags=tags,
                text=str(node.get("md") or ""),
                workstream=workstream,
                workstream_note=workstream_note,
                order=int(node.get("order") or 0),
            )
        )
    messages.sort(key=lambda msg: (msg.time, msg.order))
    missing = [msg.id for msg in messages if not msg.workstream]
    if missing:
        raise SystemExit(f"missing workstream labels for {len(missing)} messages: {missing[:5]}")
    return messages, graph


def tag_universe(messages: Iterable[Message]) -> list[str]:
    return sorted({tag for msg in messages for tag in msg.tags})


def tag_counts(messages: Iterable[Message]) -> Counter:
    return Counter(tag for msg in messages for tag in msg.tags)


def profile_from_counts(counts: Counter, universe: list[str]) -> np.ndarray:
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(universe), dtype=float)
    return np.array([counts.get(tag, 0) / total for tag in universe], dtype=float)


def tag_profile(messages: Iterable[Message], universe: list[str]) -> np.ndarray:
    return profile_from_counts(tag_counts(messages), universe)


def shannon_entropy(profile: np.ndarray) -> float:
    positive = profile[profile > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def normalized_entropy(profile: np.ndarray, universe_size: int) -> float:
    if universe_size <= 1:
        return 0.0
    return shannon_entropy(profile) / math.log(universe_size)


def specialization_index(profile: np.ndarray, universe_size: int) -> float:
    return float(1.0 - normalized_entropy(profile, universe_size))


def jsd(profile_a: np.ndarray, profile_b: np.ndarray, base: float = 2.0) -> float:
    midpoint = 0.5 * (profile_a + profile_b)
    mask_a = profile_a > 0
    mask_b = profile_b > 0
    kl_a = float(np.sum(profile_a[mask_a] * np.log(profile_a[mask_a] / midpoint[mask_a])))
    kl_b = float(np.sum(profile_b[mask_b] * np.log(profile_b[mask_b] / midpoint[mask_b])))
    return 0.5 * (kl_a + kl_b) / math.log(base)


def jsd_matrix(profiles: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(AGENTS), len(AGENTS)), dtype=float)
    for i, a in enumerate(AGENTS):
        for j, b in enumerate(AGENTS):
            matrix[i, j] = jsd(profiles[a], profiles[b])
    return matrix


def bootstrap_si(
    messages: list[Message],
    universe: list[str],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sample = [messages[i] for i in rng.integers(0, len(messages), len(messages))]
        values.append(specialization_index(tag_profile(sample, universe), len(universe)))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(np.mean(values)), float(high)


def bootstrap_workstream_si(
    messages: list[Message],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sample = [messages[i] for i in rng.integers(0, len(messages), len(messages))]
        counts = Counter(msg.workstream for msg in sample)
        profile = profile_from_counts(counts, WORKSTREAMS)
        values.append(specialization_index(profile, len(WORKSTREAMS)))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(np.mean(values)), float(high)


def tag_windows_for_agent(
    messages: list[Message],
    agent: str,
    universe: list[str],
    k: int,
    stride: int,
    start: datetime,
) -> pd.DataFrame:
    tokens = [
        (msg.time, tag)
        for msg in messages
        if msg.author == agent
        for tag in msg.tags
    ]
    records = []
    for begin in range(0, max(len(tokens) - k + 1, 0), stride):
        window = tokens[begin : begin + k]
        counts = Counter(tag for _, tag in window)
        profile = profile_from_counts(counts, universe)
        end = window[-1][0]
        records.append(
            {
                "agent": agent,
                "t_end": end,
                "hours": (end - start).total_seconds() / 3600.0,
                "si": specialization_index(profile, len(universe)),
                "n_tokens": len(window),
                "n_observed_tags": int(np.count_nonzero(profile)),
            }
        )
    return pd.DataFrame(records)


def window_si_with_ci(
    messages: list[Message],
    agent: str,
    universe: list[str],
    k: int,
    which: str,
    n_boot: int,
    seed: int,
) -> dict:
    tokens = [
        (msg.time, tag)
        for msg in messages
        if msg.author == agent
        for tag in msg.tags
    ]
    if len(tokens) < k:
        k = len(tokens)
    window = tokens[:k] if which == "first" else tokens[-k:]
    counts = Counter(tag for _, tag in window)
    profile = profile_from_counts(counts, universe)
    observed = specialization_index(profile, len(universe))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = [window[i] for i in rng.integers(0, len(window), len(window))]
        counts_b = Counter(tag for _, tag in sample)
        values.append(specialization_index(profile_from_counts(counts_b, universe), len(universe)))
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "si": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_tokens": len(window),
        "start": window[0][0].isoformat(sep=" "),
        "end": window[-1][0].isoformat(sep=" "),
    }


def permutation_jsd(
    messages: list[Message],
    universe: list[str],
    n_perm: int,
    seed: int,
) -> dict:
    observed_profiles = {
        agent: tag_profile([msg for msg in messages if msg.author == agent], universe)
        for agent in AGENTS
    }
    observed_matrix = jsd_matrix(observed_profiles)
    observed_pairs = {
        f"{a}-{b}": float(observed_matrix[i, j])
        for i, a in enumerate(AGENTS)
        for j, b in enumerate(AGENTS)
        if i < j
    }
    observed_mean = float(np.mean(list(observed_pairs.values())))
    rng = np.random.default_rng(seed)
    counts = {agent: sum(1 for msg in messages if msg.author == agent) for agent in AGENTS}
    null_mean: list[float] = []
    null_pairs: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(n_perm):
        shuffled = [messages[i] for i in rng.permutation(len(messages))]
        profiles = {}
        cursor = 0
        for agent in AGENTS:
            profiles[agent] = tag_profile(shuffled[cursor : cursor + counts[agent]], universe)
            cursor += counts[agent]
        matrix = jsd_matrix(profiles)
        pairs = [
            matrix[i, j]
            for i, a in enumerate(AGENTS)
            for j, b in enumerate(AGENTS)
            if i < j
        ]
        null_mean.append(float(np.mean(pairs)))
        for i, a in enumerate(AGENTS):
            for j, b in enumerate(AGENTS):
                if i < j:
                    null_pairs[f"{a}-{b}"].append(float(matrix[i, j]))
    p_mean = (1 + sum(value >= observed_mean for value in null_mean)) / (n_perm + 1)
    p_pairs = {
        key: (1 + sum(value >= observed_pairs[key] for value in values)) / (n_perm + 1)
        for key, values in null_pairs.items()
    }
    return {
        "observed_matrix": observed_matrix.tolist(),
        "observed_pairs": observed_pairs,
        "observed_mean": observed_mean,
        "null_mean": null_mean,
        "null_pairs": {key: values for key, values in null_pairs.items()},
        "p_mean": float(p_mean),
        "p_pairs": {key: float(value) for key, value in p_pairs.items()},
    }


def rolling_tag_si(messages: list[Message], universe: list[str], k: int, stride: int) -> pd.DataFrame:
    start = min(msg.time for msg in messages)
    frames = [
        tag_windows_for_agent(messages, agent, universe, k, stride, start)
        for agent in AGENTS
    ]
    return pd.concat(frames, ignore_index=True)


def tag_trend_tests(
    messages: list[Message],
    universe: list[str],
    k: int,
    stride: int,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict:
    windows = rolling_tag_si(messages, universe, k, stride)
    slopes = {}
    intercepts = {}
    r2 = {}
    for agent in AGENTS:
        subset = windows[windows["agent"] == agent]
        fit = linregress(subset["hours"], subset["si"])
        slopes[agent] = float(fit.slope)
        intercepts[agent] = float(fit.intercept)
        r2[agent] = float(fit.rvalue ** 2)
    observed_pooled = float(np.nanmean(list(slopes.values())))

    rng = np.random.default_rng(seed)
    by_agent = {agent: [msg for msg in messages if msg.author == agent] for agent in AGENTS}
    null_slopes: defaultdict[str, list[float]] = defaultdict(list)
    null_pooled: list[float] = []
    for _ in range(n_perm):
        shuffled_messages = []
        for agent in AGENTS:
            agent_messages = by_agent[agent]
            pool = [tag for msg in agent_messages for tag in msg.tags]
            rng.shuffle(pool)
            cursor = 0
            for msg in agent_messages:
                fresh_tags = pool[cursor : cursor + len(msg.tags)]
                cursor += len(msg.tags)
                shuffled_messages.append(replace(msg, tags=list(fresh_tags)))
        shuffled_windows = rolling_tag_si(shuffled_messages, universe, k, stride)
        values = {}
        for agent in AGENTS:
            subset = shuffled_windows[shuffled_windows["agent"] == agent]
            values[agent] = float(linregress(subset["hours"], subset["si"]).slope)
        for agent, value in values.items():
            null_slopes[agent].append(value)
        null_pooled.append(float(np.nanmean(list(values.values()))))
    p_agents = {
        agent: (1 + sum(value >= slopes[agent] for value in null_slopes[agent])) / (n_perm + 1)
        for agent in AGENTS
    }
    valid_null = [value for value in null_pooled if not math.isnan(value)]
    p_pooled = (1 + sum(value >= observed_pooled for value in valid_null)) / (len(valid_null) + 1)

    return {
        "windows": windows,
        "slopes": slopes,
        "intercepts": intercepts,
        "r2": r2,
        "p_agents": {agent: float(value) for agent, value in p_agents.items()},
        "pooled_slope": observed_pooled,
        "p_pooled": float(p_pooled),
        "null_slopes": {agent: values for agent, values in null_slopes.items()},
        "null_pooled": null_pooled,
    }


def fit_entropy_decay(windows: pd.DataFrame) -> dict:
    fits = {}
    for agent in AGENTS:
        subset = windows[windows["agent"] == agent].sort_values("hours")
        x = subset["hours"].to_numpy(dtype=float)
        entropy = 1.0 - subset["si"].to_numpy(dtype=float)
        if len(x) < 4:
            fits[agent] = {"status": "too_few_windows", "n": int(len(x))}
            continue

        def model(time: np.ndarray, h_inf: float, h_zero: float, rate: float) -> np.ndarray:
            return h_inf + (h_zero - h_inf) * np.exp(-rate * time)

        p0 = [float(np.min(entropy)), float(np.max(entropy)), 0.1]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 5.0])
        try:
            params, _ = curve_fit(model, x, entropy, p0=p0, bounds=bounds, maxfev=50000)
            predicted = model(x, *params)
            denominator = float(np.sum((entropy - np.mean(entropy)) ** 2))
            r2 = 1.0 - float(np.sum((entropy - predicted) ** 2)) / denominator if denominator > 0 else float("nan")
            fits[agent] = {
                "status": "ok",
                "n": int(len(x)),
                "h_inf": float(params[0]),
                "h_zero": float(params[1]),
                "lambda_per_hour": float(params[2]),
                "r2": r2,
            }
        except (RuntimeError, ValueError) as exc:
            fits[agent] = {"status": f"failed: {exc}", "n": int(len(x))}
    return fits


def workstream_profile(messages: Iterable[Message]) -> np.ndarray:
    counts = Counter(msg.workstream for msg in messages)
    return profile_from_counts(counts, WORKSTREAMS)


def workstream_jsd(messages: list[Message]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    profiles = {
        agent: workstream_profile([msg for msg in messages if msg.author == agent])
        for agent in AGENTS
    }
    matrix = np.zeros((len(AGENTS), len(AGENTS)), dtype=float)
    for i, a in enumerate(AGENTS):
        for j, b in enumerate(AGENTS):
            matrix[i, j] = jsd(profiles[a], profiles[b])
    return matrix, profiles


def rolling_workstream_si(messages: list[Message], k: int, stride: int) -> pd.DataFrame:
    records = []
    for agent in AGENTS:
        subset = [msg for msg in messages if msg.author == agent]
        for begin in range(0, max(len(subset) - k + 1, 0), stride):
            window = subset[begin : begin + k]
            profile = workstream_profile(window)
            records.append(
                {
                    "agent": agent,
                    "end_index": begin + k - 1,
                    "end_time": window[-1].time,
                    "si": specialization_index(profile, len(WORKSTREAMS)),
                    "n_messages": len(window),
                }
            )
    return pd.DataFrame(records)


def workstream_first_last(
    messages: list[Message],
    n_boot: int,
    seed: int,
) -> dict:
    result = {}
    for agent in AGENTS:
        subset = [msg for msg in messages if msg.author == agent]
        n = min(8, len(subset) // 2)
        early = subset[:n]
        late = subset[-n:]
        early_si = specialization_index(workstream_profile(early), len(WORKSTREAMS))
        late_si = specialization_index(workstream_profile(late), len(WORKSTREAMS))
        rng = np.random.default_rng(seed)
        early_values = []
        late_values = []
        for _ in range(n_boot):
            sample = [early[i] for i in rng.integers(0, len(early), len(early))]
            early_values.append(specialization_index(workstream_profile(sample), len(WORKSTREAMS)))
            sample = [late[i] for i in rng.integers(0, len(late), len(late))]
            late_values.append(specialization_index(workstream_profile(sample), len(WORKSTREAMS)))
        result[agent] = {
            "n_early": len(early),
            "n_late": len(late),
            "early_si": float(early_si),
            "early_ci_low": float(np.percentile(early_values, 2.5)),
            "early_ci_high": float(np.percentile(early_values, 97.5)),
            "late_si": float(late_si),
            "late_ci_low": float(np.percentile(late_values, 2.5)),
            "late_ci_high": float(np.percentile(late_values, 97.5)),
        }
    return result


def workstream_trend_test(
    messages: list[Message],
    n_perm: int,
    seed: int,
) -> dict:
    windows = rolling_workstream_si(messages, k=8, stride=1)
    slopes = {}
    for agent in AGENTS:
        subset = windows[windows["agent"] == agent]
        if len(subset) < 3:
            slopes[agent] = float("nan")
        else:
            x = np.arange(len(subset), dtype=float)
            slopes[agent] = float(linregress(x, subset["si"].to_numpy()).slope)
    observed_pooled = float(np.nanmean(list(slopes.values())))
    rng = np.random.default_rng(seed)
    by_agent = {agent: [msg for msg in messages if msg.author == agent] for agent in AGENTS}
    null_slopes: defaultdict[str, list[float]] = defaultdict(list)
    null_pooled: list[float] = []
    for _ in range(n_perm):
        shuffled = []
        for agent in AGENTS:
            labels = [msg.workstream for msg in by_agent[agent]]
            rng.shuffle(labels)
            for msg, label in zip(by_agent[agent], labels):
                shuffled.append(replace(msg, workstream=label))
        shuffled_windows = rolling_workstream_si(shuffled, k=8, stride=1)
        values = {}
        for agent in AGENTS:
            subset = shuffled_windows[shuffled_windows["agent"] == agent]
            if len(subset) < 3:
                values[agent] = float("nan")
            else:
                x = np.arange(len(subset), dtype=float)
                values[agent] = float(linregress(x, subset["si"].to_numpy()).slope)
        for agent, value in values.items():
            if not math.isnan(value):
                null_slopes[agent].append(value)
        valid = [value for value in values.values() if not math.isnan(value)]
        if valid:
            null_pooled.append(float(np.mean(valid)))
    p_agents = {}
    for agent in AGENTS:
        if math.isnan(slopes[agent]) or not null_slopes[agent]:
            p_agents[agent] = float("nan")
        else:
            p_agents[agent] = (1 + sum(value >= slopes[agent] for value in null_slopes[agent])) / (
                len(null_slopes[agent]) + 1
            )
    p_pooled = float("nan")
    if null_pooled and not math.isnan(observed_pooled):
        p_pooled = (1 + sum(value >= observed_pooled for value in null_pooled)) / (len(null_pooled) + 1)
    return {
        "windows": windows,
        "slopes": slopes,
        "p_agents": p_agents,
        "pooled_slope": observed_pooled,
        "p_pooled": p_pooled,
        "null_pooled": null_pooled,
    }


def citation_edges(messages: list[Message], graph: dict) -> tuple[list[tuple[str, str, int]], list[tuple[str, str]]]:
    known = {msg.id for msg in messages}
    weighted = []
    expanded = []
    for edge in graph["edges"]:
        if edge.get("kind") != "cite":
            continue
        source = edge["s"]
        target = edge["t"]
        if source not in known or target not in known:
            continue
        weight = int(edge.get("weight") or 1)
        weighted.append((source, target, weight))
        expanded.extend([(source, target)] * weight)
    return weighted, expanded


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def reactivity_stats(messages: list[Message], expanded: list[tuple[str, str]]) -> dict:
    by_id = {msg.id: msg for msg in messages}
    result = {}
    for agent in AGENTS:
        agent_messages = [msg for msg in messages if msg.author == agent]
        external_messages = 0
        any_messages = 0
        for msg in agent_messages:
            targets = [target for source, target in expanded if source == msg.id]
            if targets:
                any_messages += 1
            if any(target != agent for target in (by_id[target].author for target in targets)):
                external_messages += 1
        out_links = [(source, target) for source, target in expanded if by_id[source].author == agent]
        external_links = sum(1 for source, target in out_links if by_id[target].author != agent)
        rho = external_messages / len(agent_messages) if agent_messages else float("nan")
        rho_low, rho_high = wilson_interval(external_messages, len(agent_messages))
        link_share = external_links / len(out_links) if out_links else float("nan")
        link_low, link_high = wilson_interval(external_links, len(out_links))
        result[agent] = {
            "messages": len(agent_messages),
            "messages_with_any_citation": any_messages,
            "messages_with_external_citation": external_messages,
            "rho_message": float(rho),
            "rho_message_ci": [float(rho_low), float(rho_high)],
            "out_links": len(out_links),
            "external_links": external_links,
            "external_link_share": float(link_share),
            "external_link_ci": [float(link_low), float(link_high)],
        }
    return result


def influence_matrix(messages: list[Message], expanded: list[tuple[str, str]]) -> dict:
    by_id = {msg.id: msg for msg in messages}
    index = {agent: i for i, agent in enumerate(AGENTS)}
    weights = np.zeros((len(AGENTS), len(AGENTS)), dtype=int)
    for source, target in expanded:
        weights[index[by_id[source].author], index[by_id[target].author]] += 1
    row_sums = weights.sum(axis=1, keepdims=True)
    row_normalized = np.divide(weights, row_sums, out=np.zeros_like(weights, dtype=float), where=row_sums > 0)
    net_flow = weights.sum(axis=1) - weights.sum(axis=0)
    asymmetry = 0.0
    denominator = 0.0
    for i, a in enumerate(AGENTS):
        for j, b in enumerate(AGENTS):
            if i < j:
                asymmetry += abs(float(weights[i, j] - weights[j, i]))
                denominator += float(weights[i, j] + weights[j, i])
    asymmetry = asymmetry / denominator if denominator > 0 else float("nan")
    return {
        "agents": AGENTS,
        "weights": weights.tolist(),
        "row_normalized": row_normalized.tolist(),
        "row_sums": weights.sum(axis=1).tolist(),
        "column_sums": weights.sum(axis=0).tolist(),
        "net_flow": net_flow.tolist(),
        "asymmetry": float(asymmetry),
    }


def citation_lag_tests(
    messages: list[Message],
    expanded: list[tuple[str, str]],
    n_perm: int,
    seed: int,
) -> dict:
    by_id = {msg.id: msg for msg in messages}
    sources = np.array([by_id[source].time.timestamp() for source, _ in expanded], dtype=float)
    observed_lags = np.array(
        [(by_id[source].time - by_id[target].time).total_seconds() / 60.0 for source, target in expanded],
        dtype=float,
    )
    observed_lags = np.clip(observed_lags, 0.0, None)
    author = np.array([by_id[msg.id].author for msg in messages])
    index_of = {msg.id: i for i, msg in enumerate(messages)}
    times = np.array([msg.time.timestamp() for msg in messages], dtype=float)
    available: dict[int, np.ndarray] = {}
    for source_id in {source for source, _ in expanded}:
        source_index = index_of[source_id]
        mask = (times <= times[source_index]) & (np.arange(len(messages)) != source_index)
        choices = np.flatnonzero(mask)
        if choices.size == 0:
            choices = np.flatnonzero(np.arange(len(messages)) != source_index)
        available[source_index] = choices
    edge_source_indices = np.array([index_of[source] for source, _ in expanded], dtype=int)
    edge_source_authors = np.array([by_id[source].author for source, _ in expanded])
    observed_external = np.array(
        [by_id[target].author != by_id[source].author for source, target in expanded], dtype=bool
    )
    observed_external_share = float(np.mean(observed_external))
    observed_median = float(np.median(observed_lags))
    observed_mean = float(np.mean(observed_lags))

    rng = np.random.default_rng(seed)
    null_median = []
    null_mean = []
    null_external_share = []
    null_agent_median: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(n_perm):
        lags = np.empty(len(expanded), dtype=float)
        external = np.empty(len(expanded), dtype=bool)
        for edge_index, source_index in enumerate(edge_source_indices):
            choices = available[int(source_index)]
            target_index = int(choices[rng.integers(0, len(choices))])
            lags[edge_index] = (sources[edge_index] - times[target_index]) / 60.0
            external[edge_index] = author[target_index] != edge_source_authors[edge_index]
        null_median.append(float(np.median(lags)))
        null_mean.append(float(np.mean(lags)))
        null_external_share.append(float(np.mean(external)))
        for agent in AGENTS:
            mask = edge_source_authors == agent
            if mask.any():
                null_agent_median[agent].append(float(np.median(lags[mask])))
    p_median = (1 + sum(value <= observed_median for value in null_median)) / (n_perm + 1)
    p_mean = (1 + sum(value <= observed_mean for value in null_mean)) / (n_perm + 1)
    p_external = (1 + sum(value >= observed_external_share for value in null_external_share)) / (n_perm + 1)
    per_agent = {}
    for agent in AGENTS:
        mask = edge_source_authors == agent
        agent_lags = observed_lags[mask]
        agent_median = float(np.median(agent_lags)) if len(agent_lags) else float("nan")
        p_agent = float("nan")
        if null_agent_median[agent] and not math.isnan(agent_median):
            p_agent = (1 + sum(value <= agent_median for value in null_agent_median[agent])) / (
                len(null_agent_median[agent]) + 1
            )
        per_agent[agent] = {
            "n_links": int(mask.sum()),
            "median_lag_minutes": agent_median,
            "mean_lag_minutes": float(np.mean(agent_lags)) if len(agent_lags) else float("nan"),
            "null_median_mean": float(np.mean(null_agent_median[agent])) if null_agent_median[agent] else float("nan"),
            "p_shorter_than_null": p_agent,
        }
    return {
        "n_links": len(expanded),
        "observed_median_lag_minutes": observed_median,
        "observed_mean_lag_minutes": observed_mean,
        "null_median_lag_mean": float(np.mean(null_median)),
        "null_median_lag_ci": [float(np.percentile(null_median, 2.5)), float(np.percentile(null_median, 97.5))],
        "p_median_shorter_than_null": float(p_median),
        "p_mean_shorter_than_null": float(p_mean),
        "observed_external_share": observed_external_share,
        "null_external_share_mean": float(np.mean(null_external_share)),
        "null_external_share_ci": [
            float(np.percentile(null_external_share, 2.5)),
            float(np.percentile(null_external_share, 97.5)),
        ],
        "p_external_share_above_null": float(p_external),
        "per_agent": per_agent,
        "null_median": null_median,
        "null_mean": null_mean,
        "null_external_share": null_external_share,
        "observed_lags": observed_lags.tolist(),
    }


def graph_statistics(
    messages: list[Message],
    weighted: list[tuple[str, str, int]],
    n_perm: int,
    seed: int,
) -> dict:
    by_id = {msg.id: msg for msg in messages}
    graph = nx.DiGraph()
    for msg in messages:
        graph.add_node(msg.id, author=msg.author, title=msg.title, time=msg.time)
    for source, target, weight in weighted:
        graph.add_edge(source, target, weight=weight)
    undirected = graph.to_undirected()
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    modularity = nx.algorithms.community.modularity(undirected, communities, weight="weight")
    observed_assortativity = float(nx.attribute_assortativity_coefficient(graph, "author"))
    nodes = list(graph.nodes)
    original_authors = [graph.nodes[node]["author"] for node in nodes]
    rng = np.random.default_rng(seed)
    null_assortativity = []
    for _ in range(n_perm):
        shuffled = rng.permutation(original_authors)
        for node, author in zip(nodes, shuffled):
            graph.nodes[node]["author"] = author
        null_assortativity.append(float(nx.attribute_assortativity_coefficient(graph, "author")))
    for node, author in zip(nodes, original_authors):
        graph.nodes[node]["author"] = author
    p_assortativity = (1 + sum(value <= observed_assortativity for value in null_assortativity)) / (
        n_perm + 1
    )
    community_records = []
    for community_index, community in enumerate(communities, start=1):
        composition = Counter(by_id[node].author for node in community)
        community_records.append(
            {
                "community": community_index,
                "size": len(community),
                "composition": dict(composition),
            }
        )
    in_degree = dict(graph.in_degree(weight="weight"))
    out_degree = dict(graph.out_degree(weight="weight"))
    external_in_degree = defaultdict(int, {msg.id: 0 for msg in messages})
    for source, target, weight in weighted:
        if by_id[source].author != by_id[target].author:
            external_in_degree[target] += weight
    per_agent_degree = {}
    for agent in AGENTS:
        agent_messages = [msg for msg in messages if msg.author == agent]
        per_agent_degree[agent] = {
            "messages": len(agent_messages),
            "in_degree": int(sum(in_degree.get(msg.id, 0) for msg in agent_messages)),
            "external_in_degree": int(sum(external_in_degree.get(msg.id, 0) for msg in agent_messages)),
            "external_per_message": float(
                sum(external_in_degree.get(msg.id, 0) for msg in agent_messages) / len(agent_messages)
            )
            if agent_messages
            else 0.0,
        }
    top_cited = sorted(
        messages,
        key=lambda msg: (-in_degree.get(msg.id, 0), -external_in_degree.get(msg.id, 0), msg.time),
    )[:15]
    top_records = []
    for rank, msg in enumerate(top_cited, start=1):
        top_records.append(
            {
                "rank": rank,
                "file": msg.id,
                "agent": msg.author,
                "time": msg.time.isoformat(sep=" "),
                "title": msg.title,
                "in_degree": int(in_degree.get(msg.id, 0)),
                "external_in_degree": int(external_in_degree.get(msg.id, 0)),
                "out_degree": int(out_degree.get(msg.id, 0)),
            }
        )
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": float(nx.density(graph)),
        "reciprocity": float(nx.reciprocity(graph)),
        "assortativity_author": observed_assortativity,
        "assortativity_null_mean": float(np.mean(null_assortativity)),
        "assortativity_null_sd": float(np.std(null_assortativity)),
        "assortativity_p_lower": float(p_assortativity),
        "assortativity_null": null_assortativity,
        "weak_components": [len(component) for component in nx.weakly_connected_components(graph)],
        "communities": community_records,
        "modularity": float(modularity),
        "top_cited": top_records,
        "mean_in_degree": float(np.mean([in_degree.get(msg.id, 0) for msg in messages])),
        "mean_external_in_degree": float(np.mean([external_in_degree.get(msg.id, 0) for msg in messages])),
        "per_agent_degree": per_agent_degree,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.6)


def add_time_top_axis(ax: plt.Axes, start: datetime, end_hours: float) -> None:
    ticks = np.arange(0, end_hours + 0.5, 3.0)
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks)
    top.set_xticklabels([(start + timedelta(hours=float(t))).strftime("%H:%M") for t in ticks], fontsize=7)
    top.set_xlabel("clock time (MSK)", fontsize=7, labelpad=4)
    top.tick_params(length=2, pad=2)


def plot_tag_heatmap(
    messages: list[Message],
    universe: list[str],
    panels: list[str],
    out: Path,
    start: datetime,
) -> None:
    bins = 4
    edges = [messages[int(len(messages) * i / bins)].time for i in range(bins)] + [messages[-1].time + timedelta(seconds=1)]
    labels = [f"Q{i+1}\n{edges[i].strftime('%H:%M')}" for i in range(bins)]
    shares = np.zeros((len(panels), len(AGENTS), bins), dtype=float)
    samples = np.zeros_like(shares, dtype=int)
    for panel_index, tag in enumerate(panels):
        for agent_index, agent in enumerate(AGENTS):
            for bin_index in range(bins):
                subset = [
                    msg
                    for msg in messages
                    if msg.author == agent and edges[bin_index] <= msg.time < edges[bin_index + 1]
                ]
                total = sum(len(msg.tags) for msg in subset)
                count = sum(1 for msg in subset if tag in msg.tags)
                shares[panel_index, agent_index, bin_index] = count / total if total else 0.0
                samples[panel_index, agent_index, bin_index] = total
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.8), sharey=True)
    fig.subplots_adjust(top=0.88, bottom=0.16, hspace=0.45, wspace=0.18)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e5e7eb")
    vmax = max(0.25, float(np.nanmax(shares)) * 1.05)
    images = []
    for panel_index, (ax, tag) in enumerate(zip(axes.ravel(), panels)):
        matrix = np.ma.masked_where(samples[panel_index] < 5, shares[panel_index])
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        images.append(image)
        ax.set_title(tag.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xticks(range(bins))
        if panel_index >= 3:
            ax.set_xticklabels(labels, fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(range(len(AGENTS)))
        ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
        ax.tick_params(length=0)
        for i in range(len(AGENTS)):
            for j in range(bins):
                value = shares[panel_index, i, j]
                n = samples[panel_index, i, j]
                if n < 5:
                    text = f"n={n}"
                    color = "#475569"
                else:
                    text = f"{value:.0%}"
                    color = "white" if value < vmax * 0.55 else "#111827"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("share of tag assignments", fontsize=7)
    fig.suptitle(f"{TAG_SOURCE_LABEL} prevalence by agent and time quartile", fontsize=11, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Quartiles: Q1 18:53–19:42 · Q2 19:43–20:55 · Q3 20:57–00:17 · Q4 00:22–06:10",
        ha="center",
        fontsize=7,
        color="#475569",
    )
    save_figure(fig, out / "fig01_tag_evolution")


def plot_specialization_dynamics(
    messages: list[Message],
    universe: list[str],
    trend: dict,
    first_last: dict[str, dict],
    out: Path,
    start: datetime,
) -> None:
    windows = trend["windows"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.34)
    ax = axes[0]
    for agent in AGENTS:
        subset = windows[windows["agent"] == agent].sort_values("hours")
        ax.step(subset["hours"], subset["si"], where="post", color=COLORS[agent], linewidth=1.2, alpha=0.45)
        ax.scatter(subset["hours"], subset["si"], color=COLORS[agent], s=10, alpha=0.8, label=DISPLAY[agent])
        x = subset["hours"].to_numpy(dtype=float)
        y = subset["si"].to_numpy(dtype=float)
        slope = trend["slopes"][agent]
        intercept = trend["intercepts"][agent]
        ax.plot(x, intercept + slope * x, color=COLORS[agent], linewidth=1.4, linestyle="--", alpha=0.9)
        ax.text(
            0.02,
            0.96 - 0.09 * AGENTS.index(agent),
            f"{DISPLAY[agent]}: {slope:+.3f}/h (p={trend['p_agents'][agent]:.3f})",
            transform=ax.transAxes,
            fontsize=7,
            color=COLORS[agent],
            va="top",
        )
    ax.set_xlim(0, max(windows["hours"]) * 1.02)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("hours since first message")
    ax.set_ylabel("Specialization Index (SI)")
    ax.set_title(f"A. Rolling {TAG_SOURCE_LABEL} specialization", fontsize=9, fontweight="bold", loc="left")
    add_time_top_axis(ax, start, float(max(windows["hours"])))
    style_axis(ax)

    ax = axes[1]
    width = 0.34
    positions = np.arange(len(AGENTS))
    early = [first_last[agent]["si"] for agent in AGENTS]
    early_err = [
        [max(0.0, first_last[agent]["si"] - first_last[agent]["ci_low"]) for agent in AGENTS],
        [max(0.0, first_last[agent]["ci_high"] - first_last[agent]["si"]) for agent in AGENTS],
    ]
    late = [first_last[f"{agent}_late"]["si"] for agent in AGENTS]
    late_err = [
        [max(0.0, first_last[f"{agent}_late"]["si"] - first_last[f"{agent}_late"]["ci_low"]) for agent in AGENTS],
        [max(0.0, first_last[f"{agent}_late"]["ci_high"] - first_last[f"{agent}_late"]["si"]) for agent in AGENTS],
    ]
    ax.bar(
        positions - width / 2,
        early,
        width,
        yerr=early_err,
        capsize=3,
        color=[COLORS[agent] for agent in AGENTS],
        alpha=0.45,
        edgecolor="white",
        label="first window",
    )
    ax.bar(
        positions + width / 2,
        late,
        width,
        yerr=late_err,
        capsize=3,
        color=[COLORS[agent] for agent in AGENTS],
        alpha=1.0,
        edgecolor="white",
        label="last window",
    )
    for i, agent in enumerate(AGENTS):
        ax.text(
            i - width / 2,
            early[i] / 2,
            f"{early[i]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        )
        ax.text(
            i + width / 2,
            late[i] / 2,
            f"{late[i]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox=dict(facecolor=COLORS[agent], edgecolor="none", alpha=1.0, pad=1.2),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([agent for agent in AGENTS], fontsize=8)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("SI")
    ax.set_title("B. First (light) vs last (dark) window", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = axes[2]
    null = np.array(trend["null_pooled"])
    null = null[np.isfinite(null)]
    ax.hist(null, bins=35, color="#94a3b8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(trend["pooled_slope"], color="#111827", linewidth=1.8)
    ax.text(
        0.97,
        0.94,
        f"observed = {trend['pooled_slope']:.4f}/h\np = {trend['p_pooled']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("pooled SI slope under tag shuffling (per hour)")
    ax.set_ylabel("count")
    ax.set_title("C. Null test for temporal trend", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)
    fig.suptitle(f"Temporal consolidation of {TAG_SOURCE_LABEL} profiles", fontsize=11, fontweight="bold", y=0.99)
    save_figure(fig, out / "fig02_specialization_dynamics")


def plot_jsd(
    messages: list[Message],
    universe: list[str],
    permutation: dict,
    out: Path,
    start: datetime,
    k: int,
) -> None:
    fig = plt.figure(figsize=(13.0, 4.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.0])
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.42)
    ax = fig.add_subplot(grid[0, 0])
    matrix = np.array(permutation["observed_matrix"])
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(0.2, float(matrix.max()) * 1.1))
    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels([DISPLAY[agent] for agent in AGENTS], rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=7)
    for i in range(len(AGENTS)):
        for j in range(len(AGENTS)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] < 0.12 else "#111827")
    ax.set_title("A. JSD matrix (bits)", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02).set_label("JSD (bits)", fontsize=7)

    ax = fig.add_subplot(grid[0, 1])
    tokens = {
        agent: [(msg.time, tag) for msg in messages if msg.author == agent for tag in msg.tags]
        for agent in AGENTS
    }
    max_time = max(token[0] for values in tokens.values() for token in values)
    grid_times = np.arange(1.0, (max_time - start).total_seconds() / 3600.0, 0.25)
    pairs = list(itertools.combinations(AGENTS, 2))
    pair_colors = {
        ("rigel", "deneb"): "#7c3aed",
        ("rigel", "arcturus"): "#ea580c",
        ("deneb", "arcturus"): "#0891b2",
    }
    all_curve_values = []
    for a, b in pairs:
        values = []
        for hours in grid_times:
            cutoff = start + timedelta(hours=float(hours))
            profile_a = None
            profile_b = None
            values_a = [token for token in tokens[a] if token[0] <= cutoff]
            values_b = [token for token in tokens[b] if token[0] <= cutoff]
            if len(values_a) >= k:
                counts = Counter(tag for _, tag in values_a[-k:])
                profile_a = profile_from_counts(counts, universe)
            if len(values_b) >= k:
                counts = Counter(tag for _, tag in values_b[-k:])
                profile_b = profile_from_counts(counts, universe)
            values.append(jsd(profile_a, profile_b) if profile_a is not None and profile_b is not None else np.nan)
        values = np.array(values, dtype=float)
        all_curve_values.extend(values[np.isfinite(values)].tolist())
        safe = np.nan_to_num(values, nan=np.nanmean(values))
        smooth = np.convolve(safe, np.ones(5) / 5.0, mode="same")
        ax.plot(grid_times, smooth, color=pair_colors[(a, b)], linewidth=1.5, label=f"{DISPLAY[a]}–{DISPLAY[b]}")
    ax.set_xlabel("hours since first message")
    ax.set_ylabel("rolling JSD (bits)")
    if all_curve_values:
        ymax = max(0.45, min(0.9, float(np.nanmax(all_curve_values)) * 1.08))
    else:
        ymax = 0.45
    ax.set_ylim(0, ymax)
    ax.set_title(f"B. Rolling divergence (k={k})", fontsize=9, fontweight="bold", loc="left")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=7, loc="upper right")
    add_time_top_axis(ax, start, float(max(grid_times)))
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 2])
    null = np.array(permutation["null_mean"])
    ax.hist(null, bins=35, color="#94a3b8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(permutation["observed_mean"], color="#111827", linewidth=1.8)
    ax.text(
        0.97,
        0.94,
        f"observed = {permutation['observed_mean']:.3f}\np = {permutation['p_mean']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("mean pairwise JSD under author shuffling")
    ax.set_ylabel("count")
    ax.set_title("C. Null test for divergence", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)
    fig.suptitle(f"Behavioral divergence of {TAG_SOURCE_LABEL} profiles", fontsize=11, fontweight="bold", y=0.99)
    save_figure(fig, out / "fig03_behavioral_divergence")


def plot_workstream_analysis(
    messages: list[Message],
    matrix: np.ndarray,
    profiles: dict[str, np.ndarray],
    first_last: dict[str, dict],
    trend: dict,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(12.0, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.52, wspace=0.30)
    fig.subplots_adjust(top=0.88, bottom=0.14)

    ax = fig.add_subplot(grid[0, 0])
    left = np.zeros(len(AGENTS))
    for workstream in WORKSTREAMS:
        values = np.array([profiles[agent][WORKSTREAMS.index(workstream)] for agent in AGENTS])
        ax.barh(
            np.arange(len(AGENTS)),
            values,
            left=left,
            color=WORKSTREAM_COLORS[workstream],
            edgecolor="white",
            linewidth=0.8,
            label=WORKSTREAM_LABELS[workstream],
        )
        for index, value in enumerate(values):
            if value >= 0.10:
                ax.text(left[index] + value / 2, index, f"{value:.0%}", ha="center", va="center", fontsize=7, color="white")
        left += values
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of messages")
    ax.set_title("A. Workstream profile (all messages)", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(0.7, float(matrix.max()) * 1.1))
    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels([DISPLAY[agent] for agent in AGENTS], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    for i in range(len(AGENTS)):
        for j in range(len(AGENTS)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] < 0.3 else "#111827")
    ax.set_title("B. Workstream JSD (bits)", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, 0])
    bins = 4
    edges = [messages[int(len(messages) * i / bins)].time for i in range(bins)] + [messages[-1].time + timedelta(seconds=1)]
    for agent_index, agent in enumerate(AGENTS):
        for bin_index in range(bins):
            subset = [
                msg
                for msg in messages
                if msg.author == agent and edges[bin_index] <= msg.time < edges[bin_index + 1]
            ]
            counts = Counter(msg.workstream for msg in subset)
            dominant = max(WORKSTREAMS, key=lambda value: counts.get(value, 0))
            share = counts.get(dominant, 0) / len(subset) if subset else 0.0
            color = WORKSTREAM_COLORS[dominant]
            ax.add_patch(Rectangle((bin_index - 0.48, agent_index - 0.42), 0.96, 0.84, facecolor=color, edgecolor="white", linewidth=1.2))
            text_color = "white" if dominant not in ("training_tech",) else "#111827"
            ax.text(
                bin_index,
                agent_index,
                f"{WORKSTREAM_LABELS[dominant]}\n{share:.0%} ({counts.get(dominant, 0)}/{len(subset)})",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                linespacing=1.1,
            )
    ax.set_xlim(-0.5, bins - 0.5)
    ax.set_ylim(len(AGENTS) - 0.5, -0.5)
    ax.set_xticks(range(bins))
    ax.set_xticklabels([f"Q{i+1}\n{edges[i].strftime('%H:%M')}–{edges[i+1].strftime('%H:%M')}" for i in range(bins)], fontsize=7)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    ax.set_title("C. Dominant workstream by time quartile", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    width = 0.34
    positions = np.arange(len(AGENTS))
    early = [first_last[agent]["early_si"] for agent in AGENTS]
    late = [first_last[agent]["late_si"] for agent in AGENTS]
    early_err = [
        [max(0.0, first_last[agent]["early_si"] - first_last[agent]["early_ci_low"]) for agent in AGENTS],
        [max(0.0, first_last[agent]["early_ci_high"] - first_last[agent]["early_si"]) for agent in AGENTS],
    ]
    late_err = [
        [max(0.0, first_last[agent]["late_si"] - first_last[agent]["late_ci_low"]) for agent in AGENTS],
        [max(0.0, first_last[agent]["late_ci_high"] - first_last[agent]["late_si"]) for agent in AGENTS],
    ]
    ax.bar(positions - width / 2, early, width, yerr=early_err, capsize=3, color=[COLORS[agent] for agent in AGENTS], alpha=0.45, edgecolor="white", label="first 8 messages")
    ax.bar(positions + width / 2, late, width, yerr=late_err, capsize=3, color=[COLORS[agent] for agent in AGENTS], alpha=1.0, edgecolor="white", label="last 8 messages")
    for i, agent in enumerate(AGENTS):
        ax.text(
            i - width / 2,
            early[i] / 2,
            f"{early[i]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        )
        ax.text(
            i + width / 2,
            late[i] / 2,
            f"{late[i]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox=dict(facecolor=COLORS[agent], edgecolor="none", alpha=1.0, pad=1.2),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{agent}\n(n={first_last[agent]['n_early']}/{first_last[agent]['n_late']})"
            for agent in AGENTS
        ],
        fontsize=8,
    )
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Workstream SI")
    ax.set_title("D. Early (light) vs late (dark) specialization", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)
    handles = [Patch(facecolor=WORKSTREAM_COLORS[value], edgecolor="white", label=WORKSTREAM_LABELS[value]) for value in WORKSTREAMS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.suptitle("Expert-coded workstreams reveal functional differentiation", fontsize=11, fontweight="bold", y=0.98)
    save_figure(fig, out / "fig04_workstream_analysis")


def plot_reactivity_influence(
    reactivity: dict,
    influence: dict,
    graph_stats: dict,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(10.5, 6.6))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    positions = np.arange(len(AGENTS))
    values = [reactivity[agent]["rho_message"] for agent in AGENTS]
    errors = [
        [reactivity[agent]["rho_message"] - reactivity[agent]["rho_message_ci"][0] for agent in AGENTS],
        [reactivity[agent]["rho_message_ci"][1] - reactivity[agent]["rho_message"] for agent in AGENTS],
    ]
    ax.bar(positions, values, color=[COLORS[agent] for agent in AGENTS], yerr=errors, capsize=4, edgecolor="white", linewidth=1.0)
    for index, agent in enumerate(AGENTS):
        record = reactivity[agent]
        top = record["rho_message_ci"][1]
        ax.text(index, min(top + 0.025, 1.06), f"{values[index]:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(
            index,
            0.06,
            f"{record['messages_with_external_citation']}/{record['messages']}",
            ha="center",
            fontsize=7,
            color="white",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"$\rho_i$ (messages citing another agent)")
    ax.set_title("A. Message-level reactivity quotient", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    weights = np.array(influence["weights"])
    image = ax.imshow(weights, cmap="magma")
    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels([DISPLAY[agent] for agent in AGENTS], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    for i in range(len(AGENTS)):
        for j in range(len(AGENTS)):
            ax.text(j, i, str(weights[i, j]), ha="center", va="center", fontsize=9, color="white" if weights[i, j] < weights.max() * 0.6 else "#111827")
    ax.set_xlabel("cited agent (target)")
    ax.set_ylabel("citing agent (source)")
    ax.set_title("B. Influence matrix $W$ (citation links)", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, 0])
    net = np.array(influence["net_flow"])
    colors = [COLORS[agent] if value >= 0 else "#cbd5e1" for agent, value in zip(AGENTS, net)]
    ax.barh(positions, net, color=colors, edgecolor="#94a3b8", linewidth=0.7)
    for index, value in enumerate(net):
        ax.text(
            value + (0.5 if value >= 0 else -0.5),
            index,
            f"{value:+d}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    ax.set_xlim(float(np.min(net)) - 4.0, float(np.max(net)) + 3.0)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    ax.set_xlabel("net citation flow: outgoing - incoming")
    ax.set_title("C. Who consumes whose artifacts", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = fig.add_subplot(grid[1, 1])
    external = [graph_stats["per_agent_degree"][agent]["external_per_message"] for agent in AGENTS]
    ax.bar(positions, external, color=[COLORS[agent] for agent in AGENTS], edgecolor="white")
    for index, value in enumerate(external):
        ax.text(index, value + 0.08, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=8)
    ax.set_ylabel("external in-degree per message")
    ax.set_title("D. External citation leverage", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)
    fig.suptitle("Reactivity and knowledge-flow structure", fontsize=11, fontweight="bold", y=0.98)
    save_figure(fig, out / "fig05_reactivity_influence")


def plot_citation_null_tests(lag: dict, graph_stats: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    fig.subplots_adjust(top=0.80, bottom=0.18, wspace=0.32)
    ax = axes[0]
    null = np.array(lag["null_median"])
    ax.hist(null, bins=35, color="#94a3b8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(lag["observed_median_lag_minutes"], color="#111827", linewidth=1.8)
    ax.text(
        0.97,
        0.94,
        f"observed = {lag['observed_median_lag_minutes']:.1f} min\np = {lag['p_median_shorter_than_null']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("median citation lag under availability-preserving null (min)")
    ax.set_ylabel("count")
    ax.set_title("A. Observed citations are more recent", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = axes[1]
    null = np.array(lag["null_external_share"])
    ax.hist(null, bins=35, color="#94a3b8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(lag["observed_external_share"], color="#111827", linewidth=1.8)
    ax.text(
        0.97,
        0.94,
        f"observed = {lag['observed_external_share']:.3f}\np = {lag['p_external_share_above_null']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("share of cross-agent citations under random rewiring")
    ax.set_ylabel("count")
    ax.set_title("B. Cross-agent citation exceeds chance", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = axes[2]
    null = np.array(graph_stats["assortativity_null"])
    ax.hist(null, bins=35, color="#94a3b8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(graph_stats["assortativity_author"], color="#111827", linewidth=1.8)
    ax.text(
        0.03,
        0.94,
        f"observed = {graph_stats['assortativity_author']:+.3f}\np = {graph_stats['assortativity_p_lower']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("author assortativity under label shuffling")
    ax.set_ylabel("count")
    ax.set_title("C. Cross-agent mixing exceeds chance", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)
    fig.suptitle("Citations are reactive and cross-agent", fontsize=11, fontweight="bold", y=0.99)
    save_figure(fig, out / "fig06_citation_null_tests")


def short_title(title: str, limit: int = 38) -> str:
    title = re.sub(r"<metric>.*?</metric>", "", title, flags=re.DOTALL)
    title = re.sub(r"^[A-Za-z0-9_]+\s*[—–-]\s*", "", title)
    title = title.replace("⚡", "").strip()
    return textwrap.shorten(title, width=limit, placeholder="…")


def plot_knowledge_graph(messages: list[Message], weighted: list[tuple[str, str, int]], stats: dict, out: Path) -> None:
    by_id = {msg.id: msg for msg in messages}
    graph = nx.DiGraph()
    for msg in messages:
        graph.add_node(msg.id, author=msg.author, title=msg.title, time=msg.time)
    for source, target, weight in weighted:
        graph.add_edge(source, target, weight=weight)
    position = nx.spring_layout(graph, seed=7, k=0.48, iterations=300, weight="weight")
    in_degree = dict(graph.in_degree(weight="weight"))
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    fig.subplots_adjust(left=0.02, right=0.70, top=0.90, bottom=0.03)
    for source, target, data in graph.edges(data=True):
        x0, y0 = position[source]
        x1, y1 = position[target]
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=5,
            linewidth=0.55,
            color=COLORS[by_id[source].author],
            alpha=0.22,
            connectionstyle="arc3,rad=0.08",
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
        ax.add_patch(arrow)
    for agent in AGENTS:
        nodes = [node for node in graph.nodes if by_id[node].author == agent]
        sizes = [18 + 32 * math.sqrt(max(in_degree.get(node, 0), 0)) for node in nodes]
        ax.scatter(
            [position[node][0] for node in nodes],
            [position[node][1] for node in nodes],
            s=sizes,
            color=COLORS[agent],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
            label=DISPLAY[agent],
            zorder=2,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
        handle.set_sizes([60])
    table_lines = ["MOST CITED MESSAGES", "in  agent      message", "─" * 44]
    for record in stats["top_cited"][:7]:
        table_lines.append(
            f"{record['in_degree']:>2}  {record['agent']:<9}  {short_title(record['title'], 29)}"
        )
    fig.text(
        0.715,
        0.86,
        "\n".join(table_lines),
        family="monospace",
        fontsize=6.3,
        va="top",
        color="#111827",
        linespacing=1.45,
    )
    fig.text(
        0.715,
        0.38,
        "node size = citations received\nedge color = citing agent",
        fontsize=7,
        va="top",
        color="#475569",
    )
    info = (
        f"nodes = {stats['nodes']}\n"
        f"citation edges = {stats['edges']}\n"
        f"modularity = {stats['modularity']:.2f}\n"
        f"author assortativity = {stats['assortativity_author']:+.2f}\n"
        f"communities = {len(stats['communities'])} (all multi-agent)"
    )
    fig.text(
        0.715,
        0.28,
        info,
        fontsize=7,
        va="top",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#e2e8f0"),
    )
    ax.set_title("Knowledge graph of the final blackboard state", fontsize=11, fontweight="bold", y=0.99)
    save_figure(fig, out / "fig07_knowledge_graph")


def plot_temporal_network(messages: list[Message], weighted: list[tuple[str, str, int]], out: Path, start: datetime) -> None:
    by_id = {msg.id: msg for msg in messages}
    lanes = {"rigel": 2.0, "deneb": 1.0, "arcturus": 0.0}
    in_degree = Counter()
    for source, target, weight in weighted:
        in_degree[target] += weight
    position = {}
    lane_counts = Counter()
    for msg in messages:
        lane = lanes[msg.author]
        lane_counts[msg.author] += 1
        jitter = ((lane_counts[msg.author] % 7) - 3) * 0.045
        position[msg.id] = ((msg.time - start).total_seconds() / 3600.0, lane + jitter)
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    fig.subplots_adjust(top=0.80, right=0.86)
    for lane in lanes.values():
        ax.axhspan(lane - 0.28, lane + 0.28, color="#f1f5f9", zorder=0)
    for source, target, weight in weighted:
        x0, y0 = position[source]
        x1, y1 = position[target]
        dy = y1 - y0
        rad = 0.16 if abs(dy) > 0.6 else 0.06
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=4.5,
            linewidth=0.45,
            color=COLORS[by_id[source].author],
            alpha=0.18,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
        ax.add_patch(arrow)
    for agent in AGENTS:
        nodes = [msg for msg in messages if msg.author == agent]
        sizes = [12 + 22 * math.sqrt(max(in_degree.get(msg.id, 0), 0)) for msg in nodes]
        ax.scatter(
            [position[msg.id][0] for msg in nodes],
            [position[msg.id][1] for msg in nodes],
            s=sizes,
            color=COLORS[agent],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            zorder=2,
            label=DISPLAY[agent],
        )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([DISPLAY["arcturus"], DISPLAY["deneb"], DISPLAY["rigel"]], fontsize=8)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xlim(0, max(x for x, _ in position.values()) * 1.02)
    ax.set_xlabel("hours since first message")
    ax.set_title("Temporal citation network across agent lanes", fontsize=11, fontweight="bold", pad=36)
    add_time_top_axis(ax, start, float(max(x for x, _ in position.values())))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    style_axis(ax)
    save_figure(fig, out / "fig08_temporal_network")


def plot_dashboard(
    messages: list[Message],
    trend: dict,
    permutation: dict,
    workstream_profiles: dict[str, np.ndarray],
    influence: dict,
    lag: dict,
    out: Path,
    start: datetime,
) -> None:
    fig = plt.figure(figsize=(11.0, 7.8))
    grid = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)
    fig.subplots_adjust(top=0.88, bottom=0.10)

    ax = fig.add_subplot(grid[0, 0])
    for agent in AGENTS:
        subset = trend["windows"][trend["windows"]["agent"] == agent].sort_values("hours")
        ax.plot(subset["hours"], subset["si"], color=COLORS[agent], linewidth=1.4, marker="o", markersize=2.5, label=DISPLAY[agent])
    ax.set_xlabel("hours since first message")
    ax.set_ylabel(f"{TAG_SOURCE_LABEL} SI")
    ax.set_title("A. Specialization over time", fontsize=9, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7)
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    left = np.zeros(len(AGENTS))
    for workstream in WORKSTREAMS:
        values = np.array([workstream_profiles[agent][WORKSTREAMS.index(workstream)] for agent in AGENTS])
        ax.barh(np.arange(len(AGENTS)), values, left=left, color=WORKSTREAM_COLORS[workstream], edgecolor="white", linewidth=0.7)
        left += values
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of messages")
    ax.set_title("B. Workstream profiles", fontsize=9, fontweight="bold", loc="left")
    style_axis(ax)

    ax = fig.add_subplot(grid[1, 0])
    weights = np.array(influence["weights"])
    image = ax.imshow(weights, cmap="magma")
    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels([DISPLAY[agent] for agent in AGENTS], rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([DISPLAY[agent] for agent in AGENTS], fontsize=7)
    for i in range(len(AGENTS)):
        for j in range(len(AGENTS)):
            ax.text(j, i, str(weights[i, j]), ha="center", va="center", fontsize=8, color="white" if weights[i, j] < weights.max() * 0.6 else "#111827")
    ax.set_title("C. Influence matrix $W$", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, 1])
    null = np.array(lag["null_median"])
    ax.hist(null, bins=30, color="#94a3b8", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(lag["observed_median_lag_minutes"], color="#111827", linewidth=1.8)
    ax.set_xlabel("median citation lag (min)")
    ax.set_ylabel("count")
    ax.set_title("D. Citations are more recent than chance", fontsize=9, fontweight="bold", loc="left")
    ax.text(
        0.97,
        0.94,
        f"observed={lag['observed_median_lag_minutes']:.1f}\np={lag['p_median_shorter_than_null']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    style_axis(ax)
    handles = [Patch(facecolor=WORKSTREAM_COLORS[value], edgecolor="white", label=WORKSTREAM_LABELS[value]) for value in WORKSTREAMS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.suptitle("Emergence dashboard — spooky-author-identification", fontsize=12, fontweight="bold", y=0.98)
    save_figure(fig, out / "fig00_dashboard")


def md_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                if math.isnan(float(value)):
                    values.append("—")
                else:
                    values.append(float_format.format(float(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_tables(
    messages: list[Message],
    universe: list[str],
    tag_profiles: dict[str, np.ndarray],
    tag_si: dict,
    tag_ci: dict,
    tag_first_last: dict[str, dict],
    tag_trend: dict,
    tag_permutation: dict,
    tag_distinctiveness: dict,
    workstream_profiles: dict[str, np.ndarray],
    workstream_si: dict,
    workstream_ci: dict,
    workstream_early_late: dict,
    workstream_matrix: np.ndarray,
    workstream_trend: dict,
    workstream_distinctiveness: dict,
    coding_validation: dict,
    reactivity: dict,
    influence: dict,
    lag: dict,
    graph_stats: dict,
) -> str:
    sections = []
    rows = []
    for agent in AGENTS:
        rows.append(
            {
                "agent": DISPLAY[agent],
                "messages": sum(1 for msg in messages if msg.author == agent),
                "tag tokens": sum(len(msg.tags) for msg in messages if msg.author == agent),
                "unique tags": int(np.count_nonzero(tag_profiles[agent])),
                "H (normalized)": normalized_entropy(tag_profiles[agent], len(universe)),
                "SI": tag_si[agent],
                "JSD to pooled": tag_distinctiveness[agent],
                "SI 95% CI": f"[{tag_ci[agent][0]:.3f}, {tag_ci[agent][2]:.3f}]",
                "first window SI": tag_first_last[agent]["si"],
                "last window SI": tag_first_last[f"{agent}_late"]["si"],
                "slope/h": tag_trend["slopes"][agent],
                "p (trend)": tag_trend["p_agents"][agent],
            }
        )
    sections.append(f"### Table 1. {TAG_SOURCE_LABEL} specialization per agent\n\n" + md_table(pd.DataFrame(rows)))

    rows = []
    for a, b in itertools.combinations(range(len(AGENTS)), 2):
        key = f"{AGENTS[a]}-{AGENTS[b]}"
        rows.append(
            {
                "pair": f"{DISPLAY[AGENTS[a]]}–{DISPLAY[AGENTS[b]]}",
                "JSD (bits)": tag_permutation["observed_pairs"][key],
                "permutation p": tag_permutation["p_pairs"][key],
            }
        )
    rows.append(
        {
            "pair": "mean",
            "JSD (bits)": tag_permutation["observed_mean"],
            "permutation p": tag_permutation["p_mean"],
        }
    )
    sections.append(f"### Table 2. {TAG_SOURCE_LABEL} behavioral divergence\n\n" + md_table(pd.DataFrame(rows)))

    rows = []
    for agent in AGENTS:
        rows.append(
            {
                "agent": DISPLAY[agent],
                "messages": sum(1 for msg in messages if msg.author == agent),
                "SI": workstream_si[agent],
                "JSD to pooled": workstream_distinctiveness[agent],
                "SI 95% CI": f"[{workstream_ci[agent][0]:.3f}, {workstream_ci[agent][2]:.3f}]",
                "early SI": workstream_early_late[agent]["early_si"],
                "late SI": workstream_early_late[agent]["late_si"],
                "slope/window": workstream_trend["slopes"][agent],
                "p (trend)": workstream_trend["p_agents"][agent],
            }
        )
    sections.append("### Table 3. Workstream specialization per agent\n\n" + md_table(pd.DataFrame(rows)))
    sections.append(
        "### Table 3b. Coding validation\n\n"
        + md_table(
            pd.DataFrame(
                [
                    {
                        "keyword classifier accuracy": coding_validation["keyword_classifier_accuracy"],
                        "Cohen's kappa": coding_validation["cohen_kappa"],
                    }
                ]
            )
        )
    )

    rows = []
    for a, b in itertools.combinations(range(len(AGENTS)), 2):
        rows.append(
            {
                "pair": f"{DISPLAY[AGENTS[a]]}–{DISPLAY[AGENTS[b]]}",
                "workstream JSD (bits)": float(workstream_matrix[a, b]),
            }
        )
    sections.append("### Table 4. Workstream behavioral divergence\n\n" + md_table(pd.DataFrame(rows)))

    rows = []
    for agent in AGENTS:
        record = reactivity[agent]
        rows.append(
            {
                "agent": DISPLAY[agent],
                "messages": record["messages"],
                "messages citing peers": record["messages_with_external_citation"],
                "rho message": record["rho_message"],
                "rho 95% CI": f"[{record['rho_message_ci'][0]:.3f}, {record['rho_message_ci'][1]:.3f}]",
                "external link share": record["external_link_share"],
                "external link 95% CI": f"[{record['external_link_ci'][0]:.3f}, {record['external_link_ci'][1]:.3f}]",
            }
        )
    sections.append("### Table 5. Reactivity quotient\n\n" + md_table(pd.DataFrame(rows)))

    weights = np.array(influence["weights"])
    rows = []
    for i, agent in enumerate(AGENTS):
        rows.append(
            {
                "source \\ target": DISPLAY[agent],
                **{DISPLAY[target]: int(weights[i, j]) for j, target in enumerate(AGENTS)},
                "row sum": int(influence["row_sums"][i]),
                "net flow": int(influence["net_flow"][i]),
            }
        )
    sections.append("### Table 6. Influence matrix W (citation links)\n\n" + md_table(pd.DataFrame(rows)))

    rows = [
        {
            "statistic": "observed median citation lag",
            "value": lag["observed_median_lag_minutes"],
            "null reference": lag["null_median_lag_mean"],
            "p": lag["p_median_shorter_than_null"],
        },
        {
            "statistic": "observed mean citation lag",
            "value": lag["observed_mean_lag_minutes"],
            "null reference": float(np.mean(lag["null_mean"])),
            "p": lag["p_mean_shorter_than_null"],
        },
        {
            "statistic": "observed external citation share",
            "value": lag["observed_external_share"],
            "null reference": lag["null_external_share_mean"],
            "p": lag["p_external_share_above_null"],
        },
    ]
    sections.append("### Table 7. Citation null tests\n\n" + md_table(pd.DataFrame(rows)))

    rows = []
    for agent in AGENTS:
        record = lag["per_agent"][agent]
        rows.append(
            {
                "agent": DISPLAY[agent],
                "links": record["n_links"],
                "median lag (min)": record["median_lag_minutes"],
                "null median lag (min)": record["null_median_mean"],
                "p (shorter)": record["p_shorter_than_null"],
            }
        )
    sections.append("### Table 8. Per-agent external citation lags\n\n" + md_table(pd.DataFrame(rows)))

    rows = [
        {"statistic": "nodes", "value": graph_stats["nodes"]},
        {"statistic": "citation edges", "value": graph_stats["edges"]},
        {"statistic": "density", "value": round(graph_stats["density"], 4)},
        {"statistic": "reciprocity", "value": round(graph_stats["reciprocity"], 4)},
        {"statistic": "author assortativity", "value": round(graph_stats["assortativity_author"], 4)},
        {"statistic": "assortativity null mean", "value": round(graph_stats["assortativity_null_mean"], 4)},
        {"statistic": "assortativity permutation p", "value": round(graph_stats["assortativity_p_lower"], 4)},
        {"statistic": "modularity", "value": round(graph_stats["modularity"], 4)},
        {"statistic": "communities", "value": len(graph_stats["communities"])},
    ]
    sections.append("### Table 9. Citation graph statistics\n\n" + md_table(pd.DataFrame(rows)))

    sections.append(
        "### Table 10. Top cited messages\n\n"
        + md_table(
            pd.DataFrame(graph_stats["top_cited"])[
                ["rank", "agent", "time", "title", "in_degree", "external_in_degree"]
            ]
        )
    )
    return "\n\n".join(sections)


def main() -> None:
    global TAG_SOURCE_LABEL
    configure_style()
    args = parse_args()
    TAG_SOURCE_LABEL = args.tag_label
    experiment = args.experiment.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    labels_path = out / "workstream_labels.csv"
    if not labels_path.is_file():
        labels_path = Path(__file__).resolve().parent / "workstream_labels.csv"
    messages, graph = load_messages(experiment, labels_path)
    if args.tags_json:
        overrides = load_tag_overrides(args.tags_json.resolve())
        missing_tags = [msg.id for msg in messages if msg.id not in overrides]
        if missing_tags:
            raise SystemExit(f"no tag override for {len(missing_tags)} messages: {missing_tags[:5]}")
        for msg in messages:
            msg.tags = overrides[msg.id]
    keyword_labels = {msg.id: keyword_workstream(msg.title) for msg in messages}
    coding_accuracy = sum(keyword_labels[msg.id] == msg.workstream for msg in messages) / len(messages)
    coding_kappa = cohen_kappa(
        [msg.workstream for msg in messages],
        [keyword_labels[msg.id] for msg in messages],
    )
    universe = tag_universe(messages)
    if args.tags_json:
        tag_panels = [tag for tag, _ in tag_counts(messages).most_common(6)]
    else:
        tag_panels = [
            "ensembling",
            "status",
            "submission",
            "architecture",
            "postprocessing",
            "regularization",
        ]
    start = messages[0].time
    end = messages[-1].time

    tag_profiles = {
        agent: tag_profile([msg for msg in messages if msg.author == agent], universe)
        for agent in AGENTS
    }
    tag_si = {agent: specialization_index(tag_profiles[agent], len(universe)) for agent in AGENTS}
    pooled_tag_profile = profile_from_counts(tag_counts(messages), universe)
    tag_distinctiveness = {agent: jsd(tag_profiles[agent], pooled_tag_profile) for agent in AGENTS}
    tag_ci = {
        agent: bootstrap_si(
            [msg for msg in messages if msg.author == agent],
            universe,
            args.bootstrap,
            args.seed + index,
        )
        for index, agent in enumerate(AGENTS)
    }
    tag_first_last = {}
    for index, agent in enumerate(AGENTS):
        tag_first_last[agent] = window_si_with_ci(
            messages, agent, universe, args.tag_window, "first", args.bootstrap, args.seed + 10 + index
        )
        tag_first_last[f"{agent}_late"] = window_si_with_ci(
            messages, agent, universe, args.tag_window, "last", args.bootstrap, args.seed + 20 + index
        )
    tag_permutation = permutation_jsd(messages, universe, args.permutations, args.seed + 30)
    tag_trend = tag_trend_tests(
        messages,
        universe,
        k=args.tag_window,
        stride=args.tag_stride,
        n_perm=args.permutations,
        n_boot=min(args.bootstrap, 1000),
        seed=args.seed + 40,
    )
    tag_decay = fit_entropy_decay(
        rolling_tag_si(messages, universe, k=args.tag_window, stride=args.tag_stride)
    )

    workstream_matrix, workstream_profiles = workstream_jsd(messages)
    workstream_si = {
        agent: specialization_index(workstream_profiles[agent], len(WORKSTREAMS))
        for agent in AGENTS
    }
    pooled_workstream_profile = workstream_profile(messages)
    workstream_distinctiveness = {
        agent: jsd(workstream_profiles[agent], pooled_workstream_profile) for agent in AGENTS
    }
    workstream_ci = {
        agent: bootstrap_workstream_si(
            [msg for msg in messages if msg.author == agent],
            args.bootstrap,
            args.seed + 50 + index,
        )
        for index, agent in enumerate(AGENTS)
    }
    workstream_early_late = workstream_first_last(messages, args.bootstrap, args.seed + 60)
    workstream_trend = workstream_trend_test(messages, args.permutations, args.seed + 70)

    weighted, expanded = citation_edges(messages, graph)
    reactivity = reactivity_stats(messages, expanded)
    influence = influence_matrix(messages, expanded)
    lag = citation_lag_tests(messages, expanded, args.permutations, args.seed + 80)
    graph_stats = graph_statistics(messages, weighted, args.permutations, args.seed + 90)

    metrics = {
        "experiment": {
            "name": experiment.name,
            "task": json.loads((experiment / "info.json").read_text(encoding="utf-8")).get("task")
            if (experiment / "info.json").is_file()
            else experiment.name,
            "start": start.isoformat(sep=" "),
            "end": end.isoformat(sep=" "),
            "duration_hours": (end - start).total_seconds() / 3600.0,
            "messages": len(messages),
            "agents": {agent: sum(1 for msg in messages if msg.author == agent) for agent in AGENTS},
            "tag_universe": universe,
            "tag_source_label": TAG_SOURCE_LABEL,
            "tags_json": str(args.tags_json.resolve()) if args.tags_json else None,
            "tag_token_count": sum(len(msg.tags) for msg in messages),
        },
        ("tags" if args.tags_json else "hashtags"): {
            "profiles": {
                agent: {
                    "counts": {tag: int(tag_counts([msg for msg in messages if msg.author == agent]).get(tag, 0)) for tag in universe},
                    "profile": tag_profiles[agent].tolist(),
                    "entropy": normalized_entropy(tag_profiles[agent], len(universe)),
                    "si": tag_si[agent],
                    "jsd_to_pooled": tag_distinctiveness[agent],
                    "si_ci": list(tag_ci[agent]),
                    "first_window": tag_first_last[agent],
                    "last_window": tag_first_last[f"{agent}_late"],
                }
                for agent in AGENTS
            },
            "jsd": {
                "matrix": tag_permutation["observed_matrix"],
                "pairs": tag_permutation["observed_pairs"],
                "mean": tag_permutation["observed_mean"],
                "p_mean": tag_permutation["p_mean"],
                "p_pairs": tag_permutation["p_pairs"],
            },
            "trend": {
                "slopes": tag_trend["slopes"],
                "intercepts": tag_trend["intercepts"],
                "r2": tag_trend["r2"],
                "p_agents": tag_trend["p_agents"],
                "pooled_slope": tag_trend["pooled_slope"],
                "p_pooled": tag_trend["p_pooled"],
                "window_k": args.tag_window,
                "window_stride": args.tag_stride,
            },
            "entropy_decay_fit_k16": tag_decay,
        },
        "workstreams": {
            "categories": WORKSTREAMS,
            "coding_validation": {
                "keyword_classifier_accuracy": coding_accuracy,
                "cohen_kappa": coding_kappa,
            },
            "profiles": {
                agent: {
                    "counts": {workstream: int(Counter(msg.workstream for msg in messages if msg.author == agent).get(workstream, 0)) for workstream in WORKSTREAMS},
                    "profile": workstream_profiles[agent].tolist(),
                    "si": workstream_si[agent],
                    "jsd_to_pooled": workstream_distinctiveness[agent],
                    "si_ci": list(workstream_ci[agent]),
                    "first_last": workstream_early_late[agent],
                }
                for agent in AGENTS
            },
            "jsd_matrix": workstream_matrix.tolist(),
            "trend": {
                "slopes": workstream_trend["slopes"],
                "p_agents": workstream_trend["p_agents"],
                "pooled_slope": workstream_trend["pooled_slope"],
                "p_pooled": workstream_trend["p_pooled"],
                "window_k": 8,
                "window_stride": 1,
            },
        },
        "reactivity": {
            "by_agent": reactivity,
            "overall_external_message_share": sum(
                record["messages_with_external_citation"] for record in reactivity.values()
            )
            / len(messages),
        },
        "influence": influence,
        "citation_lag": {key: value for key, value in lag.items() if key not in ("null_median", "null_external_share", "observed_lags")},
        "graph": graph_stats,
        "citation_lag_null": {
            "null_median": lag["null_median"],
            "null_external_share": lag["null_external_share"],
        },
        "tag_trend_null_pooled": tag_trend["null_pooled"],
    }

    (out / "emergence_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    tables = build_tables(
        messages,
        universe,
        tag_profiles,
        tag_si,
        tag_ci,
        tag_first_last,
        tag_trend,
        tag_permutation,
        tag_distinctiveness,
        workstream_profiles,
        workstream_si,
        workstream_ci,
        workstream_early_late,
        workstream_matrix,
        workstream_trend,
        workstream_distinctiveness,
        {
            "keyword_classifier_accuracy": coding_accuracy,
            "cohen_kappa": coding_kappa,
        },
        reactivity,
        influence,
        lag,
        graph_stats,
    )
    (out / "emergence_tables.md").write_text(tables + "\n", encoding="utf-8")

    profile_rows = []
    for agent in AGENTS:
        for tag_index, tag in enumerate(universe):
            profile_rows.append(
                {
                    "agent": DISPLAY[agent],
                    "tag": tag,
                    "count": int(tag_counts([msg for msg in messages if msg.author == agent]).get(tag, 0)),
                    "share": float(tag_profiles[agent][tag_index]),
                }
            )
    pd.DataFrame(profile_rows).to_csv(out / "tag_profiles.csv", index=False)

    workstream_rows = []
    for agent in AGENTS:
        counts = Counter(msg.workstream for msg in messages if msg.author == agent)
        for workstream in WORKSTREAMS:
            workstream_rows.append(
                {
                    "agent": DISPLAY[agent],
                    "workstream": workstream,
                    "count": int(counts.get(workstream, 0)),
                    "share": float(workstream_profiles[agent][WORKSTREAMS.index(workstream)]),
                }
            )
    pd.DataFrame(workstream_rows).to_csv(out / "workstream_profiles.csv", index=False)

    labels_full = pd.DataFrame(
        [
            {
                "file": msg.id,
                "agent": DISPLAY[msg.author],
                "time": msg.time.isoformat(sep=" "),
                "title": msg.title,
                "workstream": msg.workstream,
                "keyword_label": keyword_labels[msg.id],
                "matches": keyword_labels[msg.id] == msg.workstream,
                "note": msg.workstream_note,
            }
            for msg in messages
        ]
    )
    labels_full.to_csv(out / "workstream_labels_full.csv", index=False)
    by_id = {msg.id: msg for msg in messages}
    pd.DataFrame(
        [
            {
                "source": DISPLAY[by_id[source].author],
                "target": DISPLAY[by_id[target].author],
                "weight": weight,
            }
            for source, target, weight in weighted
        ]
    ).to_csv(out / "influence_edges.csv", index=False)
    pd.DataFrame(graph_stats["top_cited"]).to_csv(out / "top_cited_messages.csv", index=False)

    plot_tag_heatmap(messages, universe, tag_panels, out / "figures", start)
    plot_specialization_dynamics(
        messages,
        universe,
        tag_trend,
        tag_first_last,
        out / "figures",
        start,
    )
    plot_jsd(messages, universe, tag_permutation, out / "figures", start, args.jsd_window)
    plot_workstream_analysis(
        messages,
        workstream_matrix,
        workstream_profiles,
        workstream_early_late,
        workstream_trend,
        out / "figures",
    )
    plot_reactivity_influence(reactivity, influence, graph_stats, out / "figures")
    plot_citation_null_tests(lag, graph_stats, out / "figures")
    plot_knowledge_graph(messages, weighted, graph_stats, out / "figures")
    plot_temporal_network(messages, weighted, out / "figures", start)
    plot_dashboard(
        messages,
        tag_trend,
        tag_permutation,
        workstream_profiles,
        influence,
        lag,
        out / "figures",
        start,
    )

    print(f"messages: {len(messages)}")
    print(f"tag universe: {len(universe)}")
    print(f"tag SI: " + ", ".join(f"{agent}={tag_si[agent]:.3f}" for agent in AGENTS))
    print(f"tag JSD mean: {tag_permutation['observed_mean']:.4f} (p={tag_permutation['p_mean']:.4f})")
    print(f"tag trend pooled: {tag_trend['pooled_slope']:+.5f}/h (p={tag_trend['p_pooled']:.4f})")
    print(f"workstream SI: " + ", ".join(f"{agent}={workstream_si[agent]:.3f}" for agent in AGENTS))
    print(f"reactivity: " + ", ".join(f"{agent}={reactivity[agent]['rho_message']:.3f}" for agent in AGENTS))
    print(f"citation median lag: {lag['observed_median_lag_minutes']:.2f} min (null {lag['null_median_lag_mean']:.2f}, p={lag['p_median_shorter_than_null']:.4f})")
    print(f"external citation share: {lag['observed_external_share']:.3f} (p={lag['p_external_share_above_null']:.4f})")
    print(f"graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges, modularity={graph_stats['modularity']:.3f}, assortativity={graph_stats['assortativity_author']:+.3f}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
