#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage11_broad_agent_mode_dataset import INTENT_LABELS
from build_pocketpal_stage22_json_agent_repair_dataset import JSON_CASES


def _prompt(name: str, instruction: str, user_text: str) -> str:
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
            "<AK_TASK_HINT> intent=json task=active_agent_json source_text_required=true",
            "<AK_CONTEXT> Saved user data: none",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return only the active agent result as raw compact JSON. Do not wrap it in action/content metadata.",
        ]
    )


def build_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    instructions = [
        "Classify the user request as compact JSON.",
        "Return only a compact JSON object describing the user's intent.",
        "Map the user's request to a JSON object with intent and useful fields.",
    ]
    for repeat in range(max(1, int(repeats))):
        for instruction_index, instruction in enumerate(instructions):
            for case_index, (user_text, content) in enumerate(JSON_CASES):
                prompt = _prompt("JSON Classifier", instruction, user_text)
                source_id = f"stage23_raw_json_{repeat:03d}_{instruction_index:03d}_{case_index:03d}"
                rows.append(
                    {
                        "action": "respond",
                        "decoder_prefix": "",
                        "decoder_text": content,
                        "encoder_text": prompt,
                        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{content}".encode()).hexdigest(),
                        "expected_content": content,
                        "intent_label": "json",
                        "intent_label_id": INTENT_LABELS["json"],
                        "source_id": source_id,
                        "source_type": "pocketpal_stage23_raw_json_agent",
                        "task_type": "active_agent_json_raw",
                        "weight": 28.0,
                    }
                )
    return rows


def _is_eval_row(row: dict[str, Any]) -> bool:
    digest = hashlib.sha256(str(row["source_id"]).encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage23_raw_json_agent_v1")
    parser.add_argument("--repeats", type=int, default=260)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(args.repeats)
    train_rows = [row for row in rows if not _is_eval_row(row)]
    eval_rows = [row for row in rows if _is_eval_row(row)]
    train_path = output_dir / "pocketpal_stage23_raw_json_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_stage23_raw_json_agent_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage23_raw_json_agent_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "decoder_prefix": "",
        "objective": "pocketpal_stage23_raw_json_agent",
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
