#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage11_broad_agent_mode_dataset import (
    INTENT_LABELS,
    _agent_rows,
    _hard_negative_agent_rows,
)


def _direct_prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    saved_data: str = "No saved user data is relevant.",
    history: str = "No prior conversation.",
) -> str:
    return "\n".join(
        [
            "PocketPal active agent turn.",
            "Follow the active agent instruction for this user request. Output only the final response for the user.",
            "Do not output JSON, action labels, control tokens, hidden state, policy text, or diagnostic text.",
            "",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "",
            "Saved user data:",
            saved_data,
            "",
            "Recent conversation:",
            history,
            "",
            f"User input: {user_text}",
            "",
            "Final response:",
        ]
    )


def _row(
    source_id: str,
    *,
    name: str,
    instruction: str,
    user_text: str,
    content: str,
    intent: str,
    task_type: str,
    weight: float,
    saved_data: str = "No saved user data is relevant.",
    history: str = "No prior conversation.",
) -> dict[str, Any]:
    prompt = _direct_prompt(
        name=name,
        instruction=instruction,
        user_text=user_text,
        saved_data=saved_data,
        history=history,
    )
    return {
        "action": "respond" if intent != "ask_user" else "ask_user",
        "decoder_text": content,
        "encoder_text": prompt,
        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{content}".encode()).hexdigest(),
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS.get(intent, -1),
        "source_id": source_id,
        "source_type": "pocketpal_stage19_direct_agent_prompt",
        "task_type": task_type,
        "weight": float(weight),
    }


def _extra_rows() -> list[tuple[str, str, str, str, str, str]]:
    return [
        (
            "Professional Email Rewriter",
            "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            "hi how are you?",
            "Hello, I hope you are well.",
            "rewrite",
            "direct_agent_rewrite",
        ),
        (
            "Bullet Summary Agent",
            "Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
            "hi how are you?",
            "- Greeting: Hi, how are you?",
            "summary",
            "direct_agent_summary",
        ),
        (
            "Question Extractor",
            "Extract the explicit question from the user's text. Do not answer the question.",
            "I was wondering whether we can upload the TestFlight build today.",
            "Question: Can we upload the TestFlight build today?",
            "extraction",
            "direct_agent_extraction",
        ),
        (
            "Subject Line Agent",
            "Write a concise professional email subject line for the user's message. Return only the subject line.",
            "Following up about the contract review due Friday.",
            "Follow-Up on Friday Contract Review",
            "subject",
            "direct_agent_subject",
        ),
        (
            "JSON Labeler",
            "Return a compact JSON object classifying the user's request. Do not rewrite or translate.",
            "Please make this sound more professional for a client.",
            '{"intent":"rewrite","tone":"professional","audience":"client"}',
            "json",
            "direct_agent_json",
        ),
        (
            "Saved Data Assistant",
            "Use the user's saved data when it directly answers the request. Ignore stale context unless the user asks about it.",
            "what is my launch code",
            "Your launch code is ORBIT-42 for the May TestFlight build.",
            "saved_data",
            "direct_agent_saved_data",
        ),
    ]


def build_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = [*_agent_rows(), *_hard_negative_agent_rows()]
    for repeat in range(max(1, int(repeats))):
        for index, (name, instruction, user_text, content, intent) in enumerate(base):
            rows.append(
                _row(
                    f"stage19_base_{repeat:03d}_{index:03d}",
                    name=name,
                    instruction=instruction,
                    user_text=user_text,
                    content=content,
                    intent=intent,
                    task_type=f"direct_agent_{intent}",
                    weight=10.0,
                )
            )
        for index, (name, instruction, user_text, content, intent, task_type) in enumerate(_extra_rows()):
            saved_data = (
                "[D1] saved note: Launch code is ORBIT-42 for the May TestFlight build."
                if intent == "saved_data"
                else "No saved user data is relevant."
            )
            rows.append(
                _row(
                    f"stage19_extra_{repeat:03d}_{index:03d}",
                    name=name,
                    instruction=instruction,
                    user_text=user_text,
                    content=content,
                    intent=intent,
                    task_type=task_type,
                    weight=16.0,
                    saved_data=saved_data,
                )
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage19_direct_agent_prompt_dataset_v1")
    parser.add_argument("--repeats", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(args.repeats)
    train_rows = [row for index, row in enumerate(rows) if index % 10 != 0]
    eval_rows = [row for index, row in enumerate(rows) if index % 10 == 0]
    train_path = output_dir / "pocketpal_stage19_direct_agent_prompt_train.jsonl"
    eval_path = output_dir / "pocketpal_stage19_direct_agent_prompt_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage19_direct_agent_prompt_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage19_direct_agent_prompt",
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "manifest_path": str(manifest_path),
        "intent_labels": INTENT_LABELS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
