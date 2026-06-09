#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage21_active_agent_eval_curriculum import _row


JSON_CASES: list[tuple[str, str]] = [
    ("Please make this more professional.", '{"intent":"rewrite","tone":"professional"}'),
    ("Find current news about Apple TestFlight.", '{"intent":"web_search","freshness":"current"}'),
    ("Translate this into Spanish.", '{"intent":"translation","target_language":"spanish"}'),
    ("Summarize these meeting notes.", '{"intent":"summary","format":"concise"}'),
    ("Extract the owner and deadline.", '{"intent":"extraction","fields":["owner","deadline"]}'),
    ("Make a checklist from this note.", '{"intent":"checklist","format":"bullets"}'),
    ("Rank these tasks by urgency.", '{"intent":"ranking","criterion":"urgency"}'),
    ("Write a subject line for this email.", '{"intent":"subject","format":"email_subject"}'),
    ("Use my saved launch code.", '{"intent":"saved_data","data_needed":"launch_code"}'),
    ("Search online for current pricing.", '{"intent":"web_search","freshness":"current"}'),
    ("Create action items from this paragraph.", '{"intent":"action_items","format":"bullets"}'),
    ("Brainstorm names for this feature.", '{"intent":"brainstorm","format":"ideas"}'),
    ("Review this plan for risks.", '{"intent":"risks","format":"concise"}'),
    ("Return the same text exactly.", '{"intent":"source_echo","preserve":"exact"}'),
]


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
                row = _row(
                    f"stage22_json_{repeat:03d}_{instruction_index:03d}_{case_index:03d}",
                    name="JSON Classifier",
                    instruction=instruction,
                    user_text=user_text,
                    content=content,
                    intent="json",
                    weight=24.0,
                )
                rows.append(row)
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
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage22_json_agent_repair_v1")
    parser.add_argument("--repeats", type=int, default=240)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(args.repeats)
    train_rows = [row for row in rows if not _is_eval_row(row)]
    eval_rows = [row for row in rows if _is_eval_row(row)]
    train_path = output_dir / "pocketpal_stage22_json_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_stage22_json_agent_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage22_json_agent_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "decoder_prefix": '{"action": "respond", "content": "',
        "objective": "pocketpal_stage22_json_agent_repair",
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
