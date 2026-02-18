#!/usr/bin/env python3
"""Convert Hugging Face datasets to OpenAI-style chat JSONL.

Outputs JSONL where each line is:
{
  "messages": [
    {"role": "system|user|assistant|tool", "content": "...", ...},
    ...
  ],
  "tools": [... optional ...]
}

This script favors robust heuristics over dataset-specific code. It supports:
- datasets already containing `messages`
- ShareGPT-style `conversations`
- instruction/response pairs (instruction+input -> response)
- preference datasets with `chosen`/`rejected` (uses chosen)
- HH-RLHF style "Human:/Assistant:" transcripts
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import VerificationMode, load_dataset

logger = logging.getLogger("hf_to_openai_chat")

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _stringify_content(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    normalized: List[Dict[str, Any]] = []
    for m in messages:
        role_raw = m.get("role") or m.get("from") or m.get("speaker")
        if role_raw is None:
            continue
        role = ROLE_MAP.get(str(role_raw).lower())
        if role is None:
            continue
        content = _stringify_content(m.get("content") or m.get("value") or m.get("text"))
        if not content and role != "tool":
            continue
        out = {"role": role, "content": content}
        # Preserve tool calls if present
        if "tool_calls" in m:
            out["tool_calls"] = m["tool_calls"]
        if "name" in m and role == "tool":
            out["name"] = m["name"]
        normalized.append(out)
    return normalized or None


def _parse_hh_transcript(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    parts = re.split(r"\n\n(Human|Assistant):", text)
    if len(parts) < 3:
        return None
    messages: List[Dict[str, Any]] = []
    # parts = [prefix, role1, content1, role2, content2, ...]
    for i in range(1, len(parts), 2):
        role = ROLE_MAP.get(parts[i].lower())
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if role and content:
            messages.append({"role": role, "content": content})
    return messages or None


def _first(row: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _instruction_to_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    instruction = _first(row, ("instruction", "prompt", "question", "query"))
    input_text = _first(row, ("input", "context"))
    response = _first(row, ("response", "output", "answer", "completion"))
    if response is None:
        response = _first(row, ("chosen",))

    if instruction is None and response is None:
        return None

    user_content = str(instruction or "").strip()
    if input_text:
        user_content = f"{user_content}\n{str(input_text).strip()}".strip()

    if not user_content or not response:
        return None

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": str(response).strip()},
    ]


def _row_to_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if isinstance(row.get("messages"), list):
        return _normalize_messages(row["messages"])

    if isinstance(row.get("conversations"), list):
        return _normalize_messages(row["conversations"])

    chosen = row.get("chosen")
    if isinstance(chosen, list):
        messages = _normalize_messages(chosen)
        if messages:
            return messages
    if isinstance(chosen, str):
        hh_messages = _parse_hh_transcript(chosen)
        if hh_messages:
            return hh_messages

    return _instruction_to_messages(row)


def _row_tools(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    tools = row.get("tools")
    if isinstance(tools, list):
        return tools
    return None


def convert_dataset(dataset_id: str, split: str, name: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    logger.info("Loading %s (split=%s)", dataset_id, split)
    ds = load_dataset(
        dataset_id,
        name=name,
        split=split,
        streaming=False,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    if limit:
        ds = ds.select(range(limit))

    converted: List[Dict[str, Any]] = []
    skipped = 0
    for row in ds:
        messages = _row_to_messages(row)
        if not messages:
            skipped += 1
            continue
        item: Dict[str, Any] = {"messages": messages}
        tools = _row_tools(row)
        if tools:
            item["tools"] = tools
        converted.append(item)

    logger.info("Converted %d rows, skipped %d rows", len(converted), skipped)
    return converted


def write_jsonl(items: List[Dict[str, Any]], output_path: Path) -> None:
    _ensure_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF datasets to OpenAI-format chat JSONL")
    parser.add_argument("--dataset", action="append", required=True, help="HF dataset id (repeatable)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--name", default=None, help="Dataset config/subset name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (per dataset)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    all_items: List[Dict[str, Any]] = []
    for dataset_id in args.dataset:
        items = convert_dataset(dataset_id, args.split, args.name, args.limit)
        all_items.extend(items)

    output_path = Path(args.output)
    write_jsonl(all_items, output_path)
    logger.info("Wrote %d samples to %s", len(all_items), output_path)


if __name__ == "__main__":
    main()
