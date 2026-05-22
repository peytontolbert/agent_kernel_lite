#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any


STRUCTURED = "<AK_STRUCTURED>"
ACTION_RESPOND = "<AK_ACTION_RESPOND>"
ACTION_ASK_USER = "<AK_ACTION_ASK_USER>"
ACTION_EXTENSION_REQUEST = "<AK_ACTION_EXTENSION_REQUEST>"
ACTION_SAVE_MEMORY = "<AK_ACTION_SAVE_MEMORY>"
CONTENT = "<AK_CONTENT>"
CONTENT_END = "</AK_CONTENT>"
TASK_TYPE = "<AK_TASK_TYPE>"
INTENT = "<AK_INTENT>"
FIELD = "<AK_FIELD>"
FIELD_NAME = "<AK_FIELD_NAME>"
FIELD_VALUE = "<AK_FIELD_VALUE>"
FIELDS = "<AK_FIELDS>"
FRESHNESS = "<AK_FRESHNESS>"
END = "<AK_END>"
COPY_USER_SOURCE_1 = "<AK_COPY_USER_SOURCE_1>"

ACTION_TOKENS = {
    "respond": ACTION_RESPOND,
    "ask_user": ACTION_ASK_USER,
    "extension_request": ACTION_EXTENSION_REQUEST,
    "save_memory": ACTION_SAVE_MEMORY,
}

CORE_TOKENS = [
    STRUCTURED,
    ACTION_RESPOND,
    ACTION_ASK_USER,
    ACTION_EXTENSION_REQUEST,
    ACTION_SAVE_MEMORY,
    CONTENT,
    CONTENT_END,
    TASK_TYPE,
    INTENT,
    FIELD,
    FIELD_NAME,
    FIELD_VALUE,
    FIELDS,
    FRESHNESS,
    END,
    *[f"<AK_COPY_USER_SOURCE_{index}>" for index in range(1, 25)],
]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_content(payload: dict[str, Any]) -> str:
    content = payload.get("content", "")
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(content or "")


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def json_to_structured_tokens(text: str, *, source_text: str = "", use_copy_source: bool = False) -> str:
    payload = _parse_json_object(text)
    if payload is None:
        return f"{STRUCTURED} {ACTION_RESPOND} {CONTENT} {str(text or '').strip()} {CONTENT_END} {END}".strip()
    action = str(payload.get("action") or "respond").strip()
    action_token = ACTION_TOKENS.get(action, ACTION_RESPOND)
    metadata = payload.get("proposal_metadata") if isinstance(payload.get("proposal_metadata"), dict) else {}
    task_type = str(metadata.get("task_type") or "").strip()
    content = _extract_json_content(payload)
    parts = [STRUCTURED, action_token]
    if task_type:
        parts.extend([TASK_TYPE, task_type])
    content_payload = _parse_json_object(content)
    if content_payload is not None:
        intent = str(content_payload.get("intent") or "").strip()
        if intent:
            parts.extend([INTENT, intent])
        freshness = str(content_payload.get("freshness") or "").strip()
        if freshness:
            parts.extend([FRESHNESS, freshness])
        fields = content_payload.get("fields")
        if isinstance(fields, list) and fields:
            parts.append(FIELDS)
            parts.extend(str(item) for item in fields)
        for key, value in sorted(content_payload.items()):
            if key in {"intent", "freshness", "fields"}:
                continue
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            else:
                rendered = str(value)
            parts.extend([FIELD, FIELD_NAME, str(key), FIELD_VALUE, rendered])
    else:
        if bool(use_copy_source) and source_text and _norm_text(content) == _norm_text(source_text):
            parts.extend([CONTENT, COPY_USER_SOURCE_1, CONTENT_END])
        else:
            parts.extend([CONTENT, content, CONTENT_END])
    parts.append(END)
    return " ".join(part for part in parts if str(part).strip())


def json_to_content_tokens(text: str, *, source_text: str = "", use_copy_source: bool = False) -> str:
    payload = _parse_json_object(text)
    content = _extract_json_content(payload) if payload is not None else str(text or "").strip()
    if bool(use_copy_source) and source_text and _norm_text(content) == _norm_text(source_text):
        content = COPY_USER_SOURCE_1
    return f"{CONTENT} {content} {CONTENT_END} {END}".strip()


def _read_until_token(text: str, start_token: str, end_token: str) -> str:
    match = re.search(
        rf"{re.escape(start_token)}\s*(.*?)\s*{re.escape(end_token)}",
        str(text or ""),
        flags=re.S,
    )
    return match.group(1).strip() if match else ""


def _read_value_after(text: str, token: str) -> str:
    match = re.search(rf"{re.escape(token)}\s+(.+?)(?=\s+<AK_[A-Z0-9_/]+>|$)", str(text or ""), flags=re.S)
    return match.group(1).strip() if match else ""


def _read_field_values(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    pattern = re.compile(
        rf"{re.escape(FIELD)}\s+{re.escape(FIELD_NAME)}\s+(.+?)\s+{re.escape(FIELD_VALUE)}\s+(.+?)(?=\s+{re.escape(FIELD)}|\s+{re.escape(CONTENT)}|\s+{re.escape(END)}|$)",
        flags=re.S,
    )
    for match in pattern.finditer(str(text or "")):
        name = match.group(1).strip()
        value = match.group(2).strip()
        if name:
            fields[name] = value
    return fields


def structured_tokens_to_json(text: str, *, source_text: str = "") -> str:
    raw = str(text or "").strip()
    if STRUCTURED not in raw:
        if CONTENT in raw:
            content = _read_until_token(raw, CONTENT, CONTENT_END)
            if content == COPY_USER_SOURCE_1 and source_text:
                content = str(source_text).strip()
            return json.dumps(
                {"action": "respond", "content": content, "proposal_metadata": {"task_type": "active_agent_content"}},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        payload = _parse_json_object(raw)
        if payload is not None:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.dumps(
            {"action": "respond", "content": raw, "proposal_metadata": {"task_type": "active_agent_unknown"}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if ACTION_EXTENSION_REQUEST in raw:
        action = "extension_request"
    elif ACTION_ASK_USER in raw:
        action = "ask_user"
    elif ACTION_SAVE_MEMORY in raw:
        action = "save_memory"
    else:
        action = "respond"
    task_type = _read_value_after(raw, TASK_TYPE) or "active_agent_unknown"
    content = _read_until_token(raw, CONTENT, CONTENT_END)
    if content == COPY_USER_SOURCE_1 and source_text:
        content = str(source_text).strip()
    intent = _read_value_after(raw, INTENT)
    freshness = _read_value_after(raw, FRESHNESS)
    fields_raw = _read_value_after(raw, FIELDS)
    field_values = _read_field_values(raw)
    if not content and (intent or freshness or fields_raw or field_values):
        content_payload: dict[str, Any] = {}
        if intent:
            content_payload["intent"] = intent
        if freshness:
            content_payload["freshness"] = freshness
        if fields_raw:
            content_payload["fields"] = [item for item in re.split(r"[\s,]+", fields_raw) if item and not item.startswith("<AK_")]
        content_payload.update(field_values)
        content = json.dumps(content_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    metadata: dict[str, Any] = {"task_type": task_type}
    if action == "extension_request":
        metadata.update(
            {
                "extension_id": "web_search",
                "capability": "web.search",
                "max_sources": 5,
                "requires_user_approval": True,
            }
        )
        if source_text:
            metadata["query"] = str(source_text).strip()
    payload = {"action": action, "content": content, "proposal_metadata": metadata}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", choices=("structured", "json"), default="structured")
    parser.add_argument("--list-tokens", action="store_true")
    parser.add_argument("--source-text", default="")
    parser.add_argument("--use-copy-source", type=int, choices=(0, 1), default=0)
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()
    if args.list_tokens:
        print(json.dumps({"protocol": "pocketpal_ak_tokens_v1", "tokens": CORE_TOKENS}, indent=2, sort_keys=True))
        return
    text = " ".join(args.text) if args.text else sys.stdin.read()
    if args.to == "structured":
        print(json_to_structured_tokens(text, source_text=str(args.source_text), use_copy_source=bool(args.use_copy_source)))
    else:
        print(structured_tokens_to_json(text, source_text=str(args.source_text)))


if __name__ == "__main__":
    main()
