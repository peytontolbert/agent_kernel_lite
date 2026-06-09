from __future__ import annotations

import re
from typing import Any


COPY_SOURCE_TOKENS = tuple(f"<AK_COPY_USER_SOURCE_{index}>" for index in range(1, 25))


def compact(value: object, *, limit: int = 1200) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].strip()


def compile_source_slots(*, user_text: object = "", user_data: object = "", max_slots: int = 8) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []

    def add(kind: str, text: object) -> None:
        value = compact(text, limit=900)
        if not value:
            return
        if any(slot["text"] == value for slot in slots):
            return
        index = len(slots) + 1
        if index > max(1, min(24, int(max_slots))):
            return
        slots.append({"id": f"U{index}", "token": COPY_SOURCE_TOKENS[index - 1], "kind": kind, "text": value})

    user = compact(user_text, limit=900)
    add("user_text", user)
    for quoted in re.findall(r"['\"]([^'\"]{3,220})['\"]", user):
        add("quoted_user_text", quoted)
    chunks = re.split(r"\s*(?:;|\n| and | but | because |,)\s*", user)
    for chunk in chunks:
        if 8 <= len(chunk.strip()) <= 220:
            add("user_span", chunk)
    data = compact(user_data, limit=900)
    if data and data.lower() != "none":
        add("user_data", data)
    return slots


def source_slots_encoder_block(slots: list[dict[str, str]]) -> str:
    if not slots:
        return "<AK_SOURCE_SLOTS> none"
    lines = [
        "<AK_SOURCE_SLOTS>",
        "Use source copy tokens when exact user-provided names, dates, values, or wording must be preserved.",
    ]
    for slot in slots[:24]:
        lines.append(f"{slot['token']} {slot['kind']}: {slot['text']}")
    return "\n".join(lines)


def expand_source_pointers(text: object, slots: list[dict[str, str]]) -> str:
    value = str(text or "")
    for slot in slots[:24]:
        value = value.replace(slot["token"], slot["text"])
    return value


def pointerize_exact_text(text: object, slots: list[dict[str, str]]) -> str:
    value = str(text or "")
    for slot in sorted(slots[:24], key=lambda item: len(item["text"]), reverse=True):
        source = slot["text"]
        if source and source in value:
            value = value.replace(source, slot["token"])
    return value


def source_slot_metadata(slots: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source_slot_tokens": [slot["token"] for slot in slots[:24]],
        "source_slot_count": len(slots[:24]),
    }
