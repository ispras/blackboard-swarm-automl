#!/usr/bin/env python3
"""Tag blackboard messages with the expanded hierarchical vocabulary.

Each message is sent in a fresh context after removing all hashtags. The model
assigns 1-4 content tags (specific tags preferred, base tags only as fallback)
and exactly one primary role tag. The script writes three tag tables for the
downstream analysis:

  tags_specific.json  specific tags only (other fallback)
  tags_rollup.json    specific tags rolled up to their base parents
  tags_roles.json     the single primary role per message

Usage:
    python llm_tagging_expanded.py
    python llm_tagging_expanded.py --resume
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
HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-zА-Яа-яЁё0-9_]+")
API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_AUTH = Path.home() / ".local/share/opencode/auth.json"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=Path, default=here.parents[2])
    ap.add_argument("--out", type=Path, default=here.parent)
    ap.add_argument("--vocabulary", type=Path, default=here.parent / "expanded_tag_vocabulary.json")
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
    graph = json.loads((experiment / "blackboard" / "graph.json").read_text(encoding="utf-8"))
    messages = []
    for node in graph["nodes"]:
        if node.get("kind") != "doc" or node.get("author") not in AGENTS:
            continue
        messages.append(
            {
                "id": node["id"],
                "agent": node["author"],
                "time": node.get("time", ""),
                "title": strip_hashtags(str(node.get("title") or "")),
                "text": strip_hashtags(str(node.get("md") or "")),
                "order": int(node.get("order") or 0),
            }
        )
    messages.sort(key=lambda item: (item["time"], item["order"]))
    for message in messages:
        leftover = HASHTAG_RE.findall(message["title"] + "\n" + message["text"])
        if leftover:
            raise SystemExit(f"hashtags remain in {message['id']}: {leftover[:5]}")
    return messages


def load_vocabulary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    specific = data["specific_tags"]
    base = data["base_tags"]
    roles = data["role_layer"]
    return {
        "base": base,
        "specific": specific,
        "specific_by_tag": {item["tag"]: item for item in specific},
        "base_set": set(base),
        "specific_set": {item["tag"] for item in specific},
        "roles": roles,
        "role_set": {item["tag"] for item in roles},
        "raw": data,
    }


def chat(key: str, model: str, messages: list[dict], temperature: float, max_tokens: int, attempts: int = 5):
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
            return json.loads(data["choices"][0]["message"]["content"]), data
        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt, 20)
            print(f"  retry {attempt + 1}/{attempts} after {wait}s: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {attempts} attempts: {last_error}")


def build_prompt(message: dict, vocab: dict, max_tags: int) -> list[dict]:
    specific_lines = "\n".join(f"- {item['tag']} (parent {item['parent']}): {item['definition']}" for item in vocab["specific"])
    base_lines = "\n".join(f"- {tag}" for tag in vocab["base"])
    role_lines = "\n".join(f"- {item['tag']}: {item['definition']}" for item in vocab["roles"])
    system = (
        "You are a careful annotator of multi-agent machine-learning engineering messages. "
        "Assign 1-4 content tags and exactly one primary role to the message. "
        "Prefer the most specific applicable tag. Use a base tag only when no specific tag applies "
        "(for example pure status, pure communication, pure planning, or other). "
        "Never use a specific tag and its parent together on the same message. "
        "Use 'other' alone only when nothing else fits. "
        "The role must be exactly one role_* tag describing the message's primary functional contribution, "
        "not the author's overall identity. Do not infer tags from the agent's identity; use only the message content. "
        "Return JSON only with this schema: "
        '{"tags":["..."],"role":"role_...","confidence":0.0,"evidence":"short verbatim phrase"}.'
    )
    user = (
        "Specific tags (preferred):\n"
        f"{specific_lines}\n\n"
        "Base fallback tags (use only if no specific tag applies):\n"
        f"{base_lines}\n\n"
        "Role tags (choose exactly one):\n"
        f"{role_lines}\n\n"
        f"Agent: {DISPLAY[message['agent']]}\n"
        f"Title: {message['title']}\n\n"
        f"Message:\n{message['text']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate(payload: dict, vocab: dict, max_tags: int) -> dict:
    tags = payload.get("tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= max_tags:
        raise ValueError(f"expected 1-{max_tags} tags")
    normalized = []
    for tag in tags:
        tag = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
        if tag not in vocab["base_set"] and tag not in vocab["specific_set"]:
            raise ValueError(f"tag not in vocabulary: {tag}")
        if tag not in normalized:
            normalized.append(tag)
    if "other" in normalized and len(normalized) > 1:
        raise ValueError("'other' must be used alone")
    for tag in normalized:
        if tag in vocab["specific_by_tag"]:
            parent = vocab["specific_by_tag"][tag]["parent"]
            if parent in normalized:
                raise ValueError(f"specific tag {tag} and parent {parent} used together")
    role = str(payload.get("role", "")).strip().lower()
    if role not in vocab["role_set"]:
        raise ValueError(f"invalid role: {role}")
    confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
    evidence = str(payload.get("evidence", "")).strip()[:300]
    return {"tags": normalized, "role": role, "confidence": confidence, "evidence": evidence}


def rollup(tags: list[str], vocab: dict) -> list[str]:
    result = []
    for tag in tags:
        if tag in vocab["specific_by_tag"]:
            parent = vocab["specific_by_tag"][tag]["parent"]
        else:
            parent = tag
        if parent not in result:
            result.append(parent)
    return result or ["other"]


def specific_only(tags: list[str], vocab: dict) -> list[str]:
    result = [tag for tag in tags if tag in vocab["specific_set"]]
    return result or ["other"]


def write_report(messages: list[dict], annotations: list[dict], vocab: dict, out: Path) -> None:
    tag_counts = Counter(tag for item in annotations for tag in item["tags"])
    role_counts = Counter(item["role"] for item in annotations)
    agent_role = {agent: Counter() for agent in AGENTS}
    for item in annotations:
        agent_role[item["agent"]][item["role"]] += 1
    lines = [
        "# Expanded-vocabulary LLM tagging — audit report",
        "",
        f"- Messages: {len(messages)}",
        f"- Tags per message: {sum(len(item['tags']) for item in annotations) / len(annotations):.2f} on average",
        f"- Mean confidence: {sum(item['confidence'] for item in annotations) / len(annotations):.2f}",
        f"- Low-confidence messages (<0.6): {sum(1 for item in annotations if item['confidence'] < 0.6)}",
        f"- Messages using `other`: {sum(1 for item in annotations if item['tags'] == ['other'])}",
        "",
        "## Role counts",
        "",
        "| role | messages | rigel_xi | deneb_upsilon | arcturus_alpha |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in vocab["roles"]:
        role = item["tag"]
        lines.append(
            f"| `{role}` | {role_counts.get(role, 0)} | {agent_role['rigel'][role]} | "
            f"{agent_role['deneb'][role]} | {agent_role['arcturus'][role]} |"
        )
    lines += ["", "## Content tag counts", "", "| tag | messages |", "|---|---:|"]
    for tag, count in tag_counts.most_common():
        lines.append(f"| `{tag}` | {count} |")
    lines += ["", "## Lowest-confidence messages", ""]
    for item in sorted(annotations, key=lambda value: value["confidence"])[:12]:
        lines.append(
            f"- `{item['id']}` ({item['confidence']:.2f}) — {', '.join(item['tags'])} / {item['role']} — {item['evidence'][:140]}"
        )
    (out / "llm_tagging_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    messages = load_messages(args.experiment.resolve())
    vocab = load_vocabulary(args.vocabulary.resolve())
    (out / "prompt_vocabulary_expanded.json").write_text(
        json.dumps(vocab["raw"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out / "messages_no_tags.jsonl").open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")
    key = read_api_key(args.auth)
    annotations = {}
    if args.resume and (out / "llm_tags_expanded.json").is_file():
        old = json.loads((out / "llm_tags_expanded.json").read_text(encoding="utf-8"))
        annotations = {item["id"]: item for item in old.get("annotations", [])}
        print(f"resume: {len(annotations)} annotations loaded", flush=True)
    pending = [message for message in messages if message["id"] not in annotations]
    print(f"messages: {len(messages)}; pending: {len(pending)}", flush=True)
    raw_calls = []
    for index, message in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {message['time'][11:16]} {DISPLAY[message['agent']]} {message['id']}",
            flush=True,
        )
        last_error = None
        result = None
        attempts_log = []
        for attempt in range(3):
            prompt = build_prompt(message, vocab, args.max_tags)
            if attempt and last_error is not None:
                prompt.append(
                    {
                        "role": "user",
                        "content": f"Your previous answer was invalid: {last_error}. "
                        f"Return JSON with 1-{args.max_tags} tags from the vocabulary and exactly one role.",
                    }
                )
            try:
                parsed, raw = chat(key, args.model, prompt, args.temperature, 900)
                validated = validate(parsed, vocab, args.max_tags)
                result = {"id": message["id"], "agent": message["agent"], **validated}
                attempts_log.append({"prompt": prompt, "response": parsed, "usage": raw.get("usage", {})})
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
                "role": "role_coordinator",
                "confidence": 0.0,
                "evidence": f"annotation failed: {last_error}",
            }
        annotations[message["id"]] = result
        raw_calls.append({"id": message["id"], "attempts": attempts_log})
        ordered = [annotations[msg["id"]] for msg in messages if msg["id"] in annotations]
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "annotations": ordered,
        }
        (out / "llm_tags_expanded.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "llm_tagging_raw.json").write_text(json.dumps(raw_calls, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.2)
    ordered = [annotations[message["id"]] for message in messages]
    for item in ordered:
        item["title"] = next(message["title"] for message in messages if message["id"] == item["id"])
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "vocabulary": vocab["raw"],
        "annotations": ordered,
    }
    (out / "llm_tags_expanded.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    specific = {"annotations": [{"id": item["id"], "tags": specific_only(item["tags"], vocab)} for item in ordered]}
    rollup_payload = {"annotations": [{"id": item["id"], "tags": rollup(item["tags"], vocab)} for item in ordered]}
    roles = {"annotations": [{"id": item["id"], "tags": [item["role"]]} for item in ordered]}
    (out / "tags_specific.json").write_text(json.dumps(specific, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tags_rollup.json").write_text(json.dumps(rollup_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tags_roles.json").write_text(json.dumps(roles, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(messages, ordered, vocab, out)
    print(f"done: {len(ordered)} annotations -> {out / 'llm_tags_expanded.json'}", flush=True)
    print(f"report: {out / 'llm_tagging_report.md'}", flush=True)


if __name__ == "__main__":
    main()
