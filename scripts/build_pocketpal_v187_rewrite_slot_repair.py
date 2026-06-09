#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {"rewrite": 2, "casual": 5, "source_echo": 6, "saved_data": 7, "ask_user": 8, "summary": 9, "web_search": 4}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_gates():
    path = _repo_root() / "scripts" / "evaluate_pocketpal_agent_gates.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_agent_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load gates module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


def _add(rows: list[dict[str, Any]], split: str, idx: int, prompt: str, action: str, content: str, task_type: str, intent: str, weight: float, negative: str = "") -> int:
    rows.append(
        {
            "action": action,
            "decoder_text": _payload(action, content, task_type),
            "encoder_text": prompt,
            "example_id": f"v187_{task_type}_{split}_{idx:06d}",
            "expected_content": content,
            "intent_label": intent,
            "intent_label_id": LABELS.get(intent, -1),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "source_id": f"v187_{task_type}",
            "source_type": "pocketpal_v187_rewrite_slot_repair",
            "split": split,
            "task_type": task_type,
            "weight": float(weight),
        }
    )
    return idx + 1


def _rows(split: str, repeats: int) -> list[dict[str, Any]]:
    gates = _load_gates()
    rows: list[dict[str, Any]] = []
    idx = 0
    slot_template = "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you."
    rewrite_cases = [
        ("John", "report", "friday", "The client is asking", "hey john i need the report by friday because the client is asking and we are behind"),
        ("Lena", "budget draft", "June 3", "Finance is waiting", "yo lena send the budget draft by june 3 because finance is waiting"),
        ("Ava", "invoice", "tomorrow morning", "Finance needs it", "ava please send the invoice tomorrow morning because finance needs it"),
        ("Devon", "roadmap update", "next Tuesday", "Planning needs it", "devon get the roadmap update ready next tuesday because planning needs it"),
    ]
    source_text = "vendor invoice INV-2048 is blocked until finance approves $1,200"
    for _ in range(repeats):
        for name, item, deadline, reason, user_text in rewrite_cases:
            idx = _add(
                rows,
                split,
                idx,
                gates._agent_prompt(
                    name="Professional Email Rewriter",
                    instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                    user_text=user_text,
                    text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
                ),
                "respond",
                slot_template,
                "active_agent_rewrite_slots",
                "rewrite",
                44.0,
                _payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REAS", "negative_truncated_slot"),
            )
        idx = _add(
            rows,
            split,
            idx,
            gates._agent_prompt(
                name="Source Echo Agent",
                instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
                user_text=source_text,
                text_slots={"SOURCE_TEXT": source_text},
            ),
            "respond",
            "Source text: [[SOURCE_TEXT]]",
            "active_agent_source_echo",
            "source_echo",
            32.0,
        )
        idx = _add(
            rows,
            split,
            idx,
            gates._agent_prompt(
                name="Casual Assistant",
                instruction="Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it.",
                user_text="How's it going?",
                stale_context="Selected paper [P1]: unrelated optimization notes from a previous research turn.",
            ),
            "respond",
            "It's going well. What would you like help with?",
            "active_agent_casual",
            "casual",
            24.0,
        )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-repeats", type=int, default=100)
    parser.add_argument("--eval-repeats", type=int, default=12)
    args = parser.parse_args()
    train_rows = _rows("train", int(args.train_repeats))
    eval_rows = _rows("eval", int(args.eval_repeats))
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, train_rows)
    _write(eval_path, eval_rows)
    all_rows = train_rows + eval_rows
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v187_rewrite_slot_repair",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
