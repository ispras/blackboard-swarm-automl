#!/usr/bin/env python3
"""Tag blackboard messages with the prompt's fixed hashtag vocabulary.

Every message is sent in a fresh context after removing all hashtags from its
title and body. The annotator must choose 1-4 tags from the exact vocabulary
that the swarm agents were originally instructed to use. The raw prompt and
response are retained for auditability.

Usage:
    python llm_tagging.py
    python llm_tagging.py --resume
    python llm_tagging.py --model deepseek-flash --max-tags 4
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

AGENTS = ["rigel", "deneb", "arcturus"]
DISPLAY = {
    "rigel": "rigel_xi",
    "deneb": "deneb_upsilon",
    "arcturus": "arcturus_alpha",
}
PROMPT_TAGS = [
    "data_preprocessing",
    "data_pipeline",
    "feature_engineering",
    "augmentation",
    "architecture",
    "loss",
    "regularization",
    "optimization",
    "training",
    "ensembling",
    "model_selection",
    "postprocessing",
    "inference",
    "plan",
    "communication",
    "submission",
    "status",
    "conflict",
    "problem",
    "bug",
    "insight",
    "other",
]
TAG_DEFINITIONS = {
    "data_preprocessing": "Cleaning, parsing, tokenization, or construction of the modelling dataset.",
    "data_pipeline": "Data loading, caching, fold generation, file formats, or artifact I/O plumbing.",
    "feature_engineering": "Handcrafted features, TF-IDF/n-gram features, stylometric features, or feature selection.",
    "augmentation": "Synthetic examples, augmentation, mixup, or other artificial data expansion.",
    "architecture": "Choice or design of a model architecture, backbone, or model family.",
    "loss": "Loss function, objective, label smoothing, or class weighting.",
    "regularization": "Dropout, weight decay, early stopping, or adversarial regularization such as FGM/AWP.",
    "optimization": "Optimizer, learning-rate schedule, batch size, or training-loop optimization.",
    "training": "Running, fine-tuning, or training a specific model, including epochs and seeds.",
    "ensembling": "Blending, stacking, combining, or weighting predictions from multiple models.",
    "model_selection": "Comparing candidate models or choosing the best model or configuration.",
    "postprocessing": "Calibration, thresholding, probability normalization, or submission formatting.",
    "inference": "Generating predictions on validation/test data or test-time inference.",
    "plan": "Planning, role division, strategy, or lane allocation.",
    "communication": "Coordination, acknowledgement, requests, or status updates to other agents.",
    "submission": "Creating, checking, or publishing the final submission or final solution.",
    "status": "A progress report whose main purpose is to say what has been done or is running.",
    "conflict": "Disagreement, resource contention, ownership conflict, or warning about a race.",
    "problem": "A blocker, failure, weak result, or unresolved issue.",
    "bug": "A code bug, crash, shape mismatch, or reproducibility error.",
    "insight": "An empirical finding, lesson, or unexpectedly strong/weak result.",
    "other": "Use only when none of the above tags applies.",
}
HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-zА-Яа-яЁё0-9_]+")
API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_AUTH = Path.home() / ".local/share/opencode/auth.json"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=Path, default=here.parents[2])
    ap.add_argument("--out", type=Path, default=here.parent)
    ap.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tags", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def read_api_key(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        key = data["deepseek"]["key"]
    except KeyError as exc:
        raise SystemExit(f"no deepseek key in {path}") from exc
    if not key:
        raise SystemExit(f"empty deepseek key in {path}")
    return str(key)


def strip_hashtags(text: str) -> str:
    cleaned = HASHTAG_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_messages(experiment: Path) -> list[dict]:
    graph_path = experiment / "blackboard" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    messages = []
    for node in graph["nodes"]:
        if node.get("kind") != "doc" or node.get("author") not in AGENTS:
            continue
        title = strip_hashtags(str(node.get("title") or ""))
        body = strip_hashtags(str(node.get("md") or ""))
        messages.append(
            {
                "id": node["id"],
                "agent": node["author"],
                "time": node.get("time", ""),
                "title": title,
                "text": body,
                "order": int(node.get("order") or 0),
            }
        )
    messages.sort(key=lambda item: (item["time"], item["order"]))
    for message in messages:
        leftover = HASHTAG_RE.findall(message["title"] + "\n" + message["text"])
        if leftover:
            raise SystemExit(f"hashtags remain in {message['id']}: {leftover[:5]}")
    return messages


def chat(
    key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    attempts: int = 5,
) -> tuple[dict, dict]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content), data
        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt, 20)
            print(f"  retry {attempt + 1}/{attempts} after {wait}s: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {attempts} attempts: {last_error}")


def build_prompt(message: dict, max_tags: int) -> list[dict]:
    vocabulary = ", ".join(PROMPT_TAGS)
    definitions = "\n".join(f"- {tag}: {TAG_DEFINITIONS[tag]}" for tag in PROMPT_TAGS)
    system = (
        "You are a careful annotator of multi-agent machine-learning engineering messages. "
        "Assign only the tags needed to describe the message's primary workstream(s). "
        "Use 1-3 tags for most messages; use 4 only if the message genuinely covers four distinct workstreams. "
        "Do not pad the tag list. Do not add generic tags (status, communication, training, plan) merely "
        "because the message is a report, acknowledgement, or mentions training. Add a generic tag only "
        "when it is the primary content. Prefer the most specific applicable tags. "
        "Use only exact tag strings from the fixed vocabulary; never invent a tag. "
        "Do not infer tags from the agent's identity; use only the message content. "
        "Return JSON only with this schema: "
        '{"tags":["..."],"confidence":0.0,"evidence":"short verbatim phrase"}.'
    )
    user = (
        f"Allowed tags: {vocabulary}\n\n"
        f"Tag definitions:\n{definitions}\n\n"
        f"Agent: {DISPLAY[message['agent']]}\n"
        f"Title: {message['title']}\n\n"
        f"Message:\n{message['text']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate(payload: dict, max_tags: int) -> dict:
    tags = payload.get("tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= max_tags:
        raise ValueError(f"expected 1-{max_tags} tags")
    normalized = []
    for tag in tags:
        tag = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
        if tag not in PROMPT_TAGS:
            raise ValueError(f"tag not in fixed vocabulary: {tag}")
        if tag not in normalized:
            normalized.append(tag)
    confidence = float(payload.get("confidence", 0.0))
    confidence = min(1.0, max(0.0, confidence))
    evidence = str(payload.get("evidence", "")).strip()[:300]
    return {"tags": normalized, "confidence": confidence, "evidence": evidence}


def write_report(messages: list[dict], annotations: list[dict], out: Path) -> None:
    tag_counts = Counter(tag for item in annotations for tag in item["tags"])
    agent_tag = {agent: Counter() for agent in AGENTS}
    for item in annotations:
        agent_tag[item["agent"]].update(item["tags"])
    lines = [
        "# LLM content tagging — fixed prompt vocabulary",
        "",
        f"- Messages: {len(messages)}",
        f"- Tags per message: {sum(len(item['tags']) for item in annotations) / len(annotations):.2f} on average",
        f"- Mean confidence: {sum(item['confidence'] for item in annotations) / len(annotations):.2f}",
        f"- Low-confidence messages (<0.6): {sum(1 for item in annotations if item['confidence'] < 0.6)}",
        f"- Messages with `other`: {sum(1 for item in annotations if 'other' in item['tags'])}",
        "",
        "## Tag counts",
        "",
        "| tag | messages | rigel_xi | deneb_upsilon | arcturus_alpha |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in PROMPT_TAGS:
        lines.append(
            f"| `{tag}` | {tag_counts.get(tag, 0)} | {agent_tag['rigel'][tag]} | "
            f"{agent_tag['deneb'][tag]} | {agent_tag['arcturus'][tag]} |"
        )
    lines += ["", "## Lowest-confidence messages", ""]
    for item in sorted(annotations, key=lambda value: value["confidence"])[:12]:
        lines.append(
            f"- `{item['id']}` ({item['confidence']:.2f}) — {', '.join(item['tags'])} — {item['evidence'][:140]}"
        )
    (out / "llm_tagging_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    messages = load_messages(args.experiment.resolve())
    with (out / "messages_no_tags.jsonl").open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")
    (out / "prompt_tag_vocabulary.json").write_text(
        json.dumps({"tags": PROMPT_TAGS, "definitions": TAG_DEFINITIONS}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    key = read_api_key(args.auth)
    by_id = {message["id"]: message for message in messages}
    previous = {}
    if args.resume and (out / "llm_tags.json").is_file():
        old = json.loads((out / "llm_tags.json").read_text(encoding="utf-8"))
        previous = {item["id"]: item for item in old.get("annotations", [])}
        print(f"resume: {len(previous)} annotations loaded", flush=True)
    annotations = dict(previous)
    raw_calls = []
    pending = [message for message in messages if message["id"] not in annotations]
    print(f"messages: {len(messages)}; pending: {len(pending)}", flush=True)
    for index, message in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {message['time'][11:16]} {DISPLAY[message['agent']]} {message['id']}",
            flush=True,
        )
        last_error = None
        result = None
        responses = []
        for attempt in range(3):
            prompt = build_prompt(message, args.max_tags)
            if attempt and last_error is not None:
                prompt.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous answer was invalid: {last_error}. "
                            f"Return JSON with 1-{args.max_tags} tags chosen only from the allowed vocabulary."
                        ),
                    }
                )
            try:
                parsed, raw = chat(
                    key,
                    args.model,
                    prompt,
                    temperature=args.temperature,
                    max_tokens=800,
                )
                validated = validate(parsed, args.max_tags)
                result = {"id": message["id"], "agent": message["agent"], **validated}
                responses.append({"prompt": prompt, "response": parsed, "usage": raw.get("usage", {})})
                break
            except Exception as exc:
                last_error = exc
                print(f"  invalid: {exc}", flush=True)
                time.sleep(1.0 * (attempt + 1))
        if result is None:
            result = {
                "id": message["id"],
                "agent": message["agent"],
                "tags": ["other"],
                "confidence": 0.0,
                "evidence": f"annotation failed: {last_error}",
            }
        annotations[message["id"]] = result
        raw_calls.append({"id": message["id"], "attempts": responses, "fallback": result["tags"] == ["other"] and result["confidence"] == 0.0})
        ordered = [annotations[msg["id"]] for msg in messages if msg["id"] in annotations]
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "vocabulary": PROMPT_TAGS,
            "annotations": ordered,
        }
        (out / "llm_tags.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "llm_tagging_raw.json").write_text(
            json.dumps(raw_calls, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        time.sleep(0.2)
    ordered = [annotations[message["id"]] for message in messages]
    for item in ordered:
        item["title"] = by_id[item["id"]]["title"]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "vocabulary": PROMPT_TAGS,
        "definitions": TAG_DEFINITIONS,
        "annotations": ordered,
    }
    (out / "llm_tags.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(messages, ordered, out)
    print(f"done: {len(ordered)} annotations -> {out / 'llm_tags.json'}", flush=True)
    print(f"report: {out / 'llm_tagging_report.md'}", flush=True)


if __name__ == "__main__":
    main()
