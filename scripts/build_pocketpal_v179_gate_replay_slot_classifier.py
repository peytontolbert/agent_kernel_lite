#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


INTENT_LABELS = {
    "plan": 0,
    "action_items": 1,
    "rewrite": 2,
    "translation": 3,
    "web_search": 4,
    "casual": 5,
    "source_echo": 6,
    "saved_data": 7,
    "ask_user": 8,
    "summary": 9,
    "title": 10,
    "checklist": 11,
    "risks": 12,
    "json": 13,
    "ranking": 14,
    "extraction": 15,
    "subject": 16,
    "brainstorm": 17,
}


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


def _payload(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    proposal_metadata = {"task_type": task_type}
    if metadata:
        proposal_metadata.update(metadata)
    return json.dumps({"action": action, "content": content, "proposal_metadata": proposal_metadata}, ensure_ascii=False, sort_keys=True)


def _add(rows: list[dict[str, Any]], split: str, idx: int, prompt: str, action: str, content: str, task_type: str, intent: str, weight: float, negative: str = "", metadata: dict[str, Any] | None = None) -> int:
    rows.append(
        {
            "action": action,
            "decoder_text": _payload(action, content, task_type, metadata),
            "encoder_text": prompt,
            "example_id": f"v179_{task_type}_{split}_{idx:06d}",
            "expected_content": content,
            "intent_label": intent,
            "intent_label_id": INTENT_LABELS.get(intent, -1),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "source_id": f"v179_{task_type}",
            "source_type": "pocketpal_v179_gate_replay_slot_classifier",
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
    exact_gate_replay = [
        (gates._runtime_plain_chat_prompt("Hi how are you?"), "respond", "I'm doing well. What would you like help with?", "runtime_plain_chat", "casual", 30.0, ""),
        (
            gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "Hello, I hope you are well.",
            "active_agent_rewrite_greeting",
            "rewrite",
            36.0,
            _payload("respond", slot_template, "negative_slot_when_only_source_text"),
        ),
        (
            gates._agent_prompt(
                name="Bullet Summary Agent",
                instruction="Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "- Greeting: Hi, how are you?",
            "active_agent_summary",
            "summary",
            30.0,
            "",
        ),
        (
            gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="rewrite this",
            ),
            "ask_user",
            "What text should I rewrite?",
            "active_agent_missing_text",
            "ask_user",
            36.0,
            _payload("respond", slot_template, "negative_slot_when_missing_text"),
        ),
    ]
    for _ in range(repeats):
        for prompt, action, content, task_type, intent, weight, negative in exact_gate_replay:
            idx = _add(rows, split, idx, prompt, action, content, task_type, intent, weight, negative)
        for name, item, deadline, reason, user_text in [
            ("John", "report", "friday", "The client is asking", "hey john i need the report by friday because the client is asking and we are behind"),
            ("Devon", "roadmap update", "next Tuesday", "Planning needs it", "devon get the roadmap update ready next tuesday because planning needs it"),
            ("Lena", "budget draft", "June 3", "Finance is waiting", "yo lena send the budget draft by june 3 because finance is waiting"),
            ("Ava", "invoice", "tomorrow morning", "Finance needs it", "ava please send the invoice tomorrow morning because finance needs it"),
        ]:
            prompt = gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=user_text,
                text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
            )
            idx = _add(
                rows,
                split,
                idx,
                prompt,
                "respond",
                slot_template,
                "active_agent_rewrite_slots",
                "rewrite",
                28.0,
                _payload("respond", "Maria, Devin, and Sam each have launch tasks with near-term deadlines.", "negative_summary_attractor"),
            )
        for text, label, wrong in [
            ("Please approve invoice INV-2048 for $1,200.", "finance", "web_search"),
            ("Can you rewrite this note professionally?", "writing", "web_search"),
            ("Find current Swift WKWebView examples online.", "web_search", "finance"),
            ("Move the launch meeting to Thursday at 2 PM.", "schedule", "finance"),
            ("The hotel code is HCN-9921 for July 14.", "travel", "finance"),
        ]:
            prompt = gates._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text=text,
                text_slots={"SOURCE_TEXT": text},
            )
            idx = _add(rows, split, idx, prompt, "respond", label, "active_agent_classify", "web_search" if label == "web_search" else "casual", 24.0, _payload("respond", wrong, "negative_classifier_attractor"))
        idx = _add(
            rows,
            split,
            idx,
            gates._web_agent_prompt(user_text="search the web for current Swift WKWebView navigation policy examples"),
            "extension_request",
            "Requesting approval to search the web.",
            "runtime_web_search_request",
            "web_search",
            22.0,
            "",
            {
                "capability": "web.search",
                "extension_id": "web_search",
                "max_sources": 5,
                "query": "search the web for current Swift WKWebView navigation policy examples",
                "requires_user_approval": True,
            },
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
    parser.add_argument("--train-repeats", type=int, default=260)
    parser.add_argument("--eval-repeats", type=int, default=24)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_rows = _rows("train", int(args.train_repeats))
    eval_rows = _rows("eval", int(args.eval_repeats))
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
        "intent_labels": INTENT_LABELS,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v179_gate_replay_slot_classifier",
        "target_action_counts": dict(sorted(Counter(str(row.get("action") or "") for row in all_rows).items())),
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
