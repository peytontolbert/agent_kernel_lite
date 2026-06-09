#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pocketpal_agent_gates import GATES
from scripts.build_pocketpal_stage11_broad_agent_mode_dataset import INTENT_LABELS


TARGETS: dict[str, dict[str, Any]] = {
    "runtime_plain_greeting": {
        "action": "respond",
        "content": "I'm doing well. What would you like help with?",
        "metadata": {"task_type": "runtime_plain_chat"},
    },
    "active_agent_rewrite_greeting": {
        "action": "respond",
        "content": "Hello, I hope you are well.",
        "metadata": {"task_type": "agent_instruction_following"},
    },
    "active_agent_bullet_summary_same_input": {
        "action": "respond",
        "content": "- Greeting: Hi, how are you?",
        "metadata": {"task_type": "agent_instruction_following"},
    },
    "professional_email_rewrite": {
        "action": "respond",
        "content": "Hi John,\n\nCould you please send the report by Friday? The client is asking, and we are behind.\n\nThank you.",
        "metadata": {"task_type": "agent_instruction_following"},
    },
    "casual_retention": {
        "action": "respond",
        "content": "It's going well. What would you like to work on?",
        "metadata": {"task_type": "agent_instruction_following"},
    },
    "source_echo_no_slots": {
        "action": "respond",
        "content": "Source text: vendor invoice INV-2048 is blocked until finance approves $1,200",
        "metadata": {"task_type": "source_slot_copy"},
    },
    "ask_missing_text": {
        "action": "ask_user",
        "content": "What text should I rewrite?",
        "metadata": {"task_type": "ask_missing_text"},
    },
    "saved_data_use": {
        "action": "respond",
        "content": "Your launch code is ORBIT-42 for the May TestFlight build.",
        "metadata": {"task_type": "agent_instruction_following"},
    },
    "missing_user_data": {
        "action": "ask_user",
        "content": "I do not have saved data for your hotel confirmation code. Can you add or paste the relevant saved data?",
        "metadata": {"task_type": "ask_missing_saved_data"},
    },
    "web_search_request": {
        "action": "extension_request",
        "content": "Requesting approval to search the web.",
        "metadata": {
            "task_type": "web_search_request",
            "extension_id": "web_search",
            "capability": "web.search",
            "query": "search the web for current TestFlight upload limits",
            "max_sources": 5,
            "requires_user_approval": True,
        },
    },
}

GATE_INTENTS: dict[str, str] = {
    "runtime_plain_greeting": "casual",
    "active_agent_rewrite_greeting": "rewrite",
    "active_agent_bullet_summary_same_input": "summary",
    "professional_email_rewrite": "rewrite",
    "casual_retention": "casual",
    "source_echo_no_slots": "source_echo",
    "ask_missing_text": "ask_user",
    "saved_data_use": "saved_data",
    "missing_user_data": "ask_user",
    "web_search_request": "web_search",
}


def _split(source_id: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_rows(repeats: int, *, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for gate in GATES:
            gate_id = str(gate.get("id") or "")
            if bool(gate.get("experimental")) or gate_id not in TARGETS:
                continue
            target = TARGETS[gate_id]
            payload = {
                "action": target["action"],
                "content": target["content"],
                "proposal_metadata": target["metadata"],
            }
            intent = GATE_INTENTS[gate_id]
            source_id = f"pocketpal_agent_gate_repair_{repeat:04d}_{gate_id}"
            rows.append(
                {
                    "source_id": source_id,
                    "source_type": "pocketpal_agent_gate_repair",
                    "task_type": str(target["metadata"].get("task_type") or "agent_gate_repair"),
                    "action": str(target["action"]),
                    "intent_label": intent,
                    "intent_label_id": INTENT_LABELS.get(intent, -1),
                    "encoder_text": str(gate["prompt"]),
                    "decoder_text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    "weight": float(weight),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_agent_gate_repair_v1")
    parser.add_argument("--repeats", type=int, default=240)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--weight", type=float, default=28.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = build_rows(int(args.repeats), weight=float(args.weight))
    train_rows = [row for row in rows if _split(str(row["source_id"]), float(args.eval_fraction)) == "train"]
    eval_rows = [row for row in rows if _split(str(row["source_id"]), float(args.eval_fraction)) == "eval"]
    if not eval_rows and train_rows:
        eval_rows.append(train_rows.pop())

    train_path = output_dir / "pocketpal_agent_gate_repair_train.jsonl"
    eval_path = output_dir / "pocketpal_agent_gate_repair_eval.jsonl"
    manifest_path = output_dir / "pocketpal_agent_gate_repair_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    action_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
        intent_counts[str(row["intent_label"])] = intent_counts.get(str(row["intent_label"]), 0) + 1
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_agent_gate_repair",
        "dataset_format": "jsonl",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"pocketpal_agent_gate_repair": len(rows)},
        "intent_labels": INTENT_LABELS,
        "intent_label_counts": dict(sorted(intent_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
