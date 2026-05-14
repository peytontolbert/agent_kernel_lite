#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage11_broad_agent_mode_dataset import INTENT_LABELS, _agent_rows, _hard_negative_agent_rows


def _task_for(intent: str) -> str:
    return "active_agent_rewrite" if intent == "rewrite" else f"active_agent_{intent}"


def _prompt(name: str, instruction: str, user_text: str, intent: str) -> str:
    task = _task_for(intent)
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            f"<AK_TASK_HINT> intent={intent} task={task} source_text_required={intent != 'casual'}",
            "<AK_CONTEXT> Saved user data: none",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _row(source_id: str, name: str, instruction: str, user_text: str, content: str, intent: str, weight: float) -> dict[str, Any]:
    action = "ask_user" if intent == "ask_user" else "respond"
    task = _task_for(intent)
    prompt = _prompt(name, instruction, user_text, intent)
    decoder = {"action": action, "content": content, "proposal_metadata": {"task_type": task}}
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": prompt,
        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{json.dumps(decoder, sort_keys=True)}".encode()).hexdigest(),
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS.get(intent, -1),
        "source_id": source_id,
        "source_type": "pocketpal_stage20_hinted_agent_json",
        "task_type": task,
        "weight": float(weight),
    }


EXTRA: list[tuple[str, str, str, str, str]] = [
    ("Reword Agent", "Reword the user input for a professional email.", "hi how are you?", "Hello, I hope you are well.", "rewrite"),
    ("Reword Agent", "Reword what the user wrote so it sounds professional.", "hey john i need the report by friday", "Hi John, could you please send the report by Friday?", "rewrite"),
    ("Polish Agent", "Make the user's text sound polished and work appropriate.", "send me that thing today", "Could you please send that to me today?", "rewrite"),
    ("Email Agent", "Turn the user's rough note into a professional email.", "client wants the docs today can you send them", "Hello, could you please send the documents today? The client is asking for them.", "rewrite"),
    ("Bullet Agent", "Convert the user input into bullet points.", "Maria owns launch slides and Devin fixes login by Thursday.", "- Maria: launch slides\n- Devin: fix login by Thursday", "action_items"),
    ("List Agent", "Turn the user's text into a clean bullet list.", "Review the proposal, confirm budget, and send invoice tomorrow.", "- Review the proposal\n- Confirm the budget\n- Send the invoice tomorrow", "checklist"),
    ("Task Agent", "Pull out owners and deadlines as bullets.", "Sam books the room today and Priya sends notes Friday.", "- Sam: book the room today\n- Priya: send notes Friday", "action_items"),
    ("Summary Agent", "Make the user's text shorter without changing facts.", "The team fixed payment bugs but still needs legal approval before launch.", "The team fixed payment bugs, but launch still needs legal approval.", "summary"),
    ("Title Agent", "Create a short title for the user text.", "Notes about fixing web search approval and clickable result links.", "Web Search Approval Fixes", "title"),
    ("Question Agent", "Extract the question from the user's text.", "I wonder if we can upload the build today.", "Question: Can we upload the build today?", "extraction"),
    ("Spanish Agent", "Put the user's English text into Spanish.", "Please send the invoice tomorrow morning.", "Por favor, envia la factura manana por la manana.", "translation"),
    ("JSON Agent", "Classify the user request as compact JSON.", "Please make this more professional.", "{\"intent\":\"rewrite\",\"tone\":\"professional\"}", "json"),
]


def build_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = [*_agent_rows(), *_hard_negative_agent_rows(), *EXTRA]
    for repeat in range(max(1, int(repeats))):
        for index, (name, instruction, user_text, content, intent) in enumerate(base):
            rows.append(_row(f"stage20_{repeat:03d}_{index:03d}", name, instruction, user_text, content, intent, 14.0))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage20_hinted_agent_json_dataset_v1")
    parser.add_argument("--repeats", type=int, default=160)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(args.repeats)
    train_rows = [row for index, row in enumerate(rows) if index % 10 != 0]
    eval_rows = [row for index, row in enumerate(rows) if index % 10 == 0]
    train_path = output_dir / "pocketpal_stage20_hinted_agent_json_train.jsonl"
    eval_path = output_dir / "pocketpal_stage20_hinted_agent_json_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage20_hinted_agent_json_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage20_hinted_agent_json",
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
