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


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        sort_keys=True,
    )


def _add(
    rows: list[dict[str, Any]],
    *,
    split: str,
    task_type: str,
    prompt: str,
    action: str,
    content: str,
    intent: str,
    idx: int,
    weight: float,
    negative: str = "",
) -> None:
    rows.append(
        {
            "action": action,
            "decoder_text": _payload(action, content, task_type),
            "encoder_text": prompt,
            "example_id": f"{task_type}_{split}_{idx:06d}",
            "expected_content": content,
            "intent_label": intent,
            "intent_label_id": INTENT_LABELS.get(intent, -1),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "source_id": f"v174_cert_two_failure_{task_type}",
            "source_type": "pocketpal_v174_cert_two_failure_repair",
            "split": split,
            "task_type": task_type,
            "weight": float(weight),
        }
    )


def _build_rows(split: str, repeats: int) -> list[dict[str, Any]]:
    gates = _load_gates()
    rows: list[dict[str, Any]] = []
    idx = 0

    rewrite_cases = [
        ("Devon", "roadmap update", "next Tuesday", "Planning needs it", "devon get the roadmap update ready next tuesday because planning needs it"),
        ("Devon", "roadmap update", "next Tuesday", "Planning needs it", "devon prepare the roadmap update by next tuesday because planning needs it"),
        ("Nora", "badge list", "Friday", "Security needs it", "nora share the badge list by friday because security needs it"),
        ("Eli", "release checklist", "Wednesday", "QA is blocked", "eli update the release checklist by wednesday because qa is blocked"),
        ("Sam", "slide deck", "Thursday at 2 PM", "The client review moved up", "sam finish the slide deck by thursday at 2 pm because the client review moved up"),
    ]
    classify_cases = [
        ("Please approve invoice INV-2048 for $1,200.", "finance"),
        ("Approve invoice INV-2048 for $1,200.", "finance"),
        ("The budget draft is waiting on finance approval.", "finance"),
        ("Finance needs approval for invoice INV-2048.", "finance"),
        ("Can you rewrite this note professionally?", "writing"),
        ("Find current Swift WKWebView examples online.", "web_search"),
        ("Move the launch meeting to Thursday at 2 PM.", "schedule"),
        ("The hotel code is HCN-9921 for July 14.", "travel"),
    ]
    replay = [
        (
            gates._runtime_plain_chat_prompt("Hi how are you?"),
            "respond",
            "I'm doing well. What would you like help with?",
            "runtime_plain_chat_replay",
            "casual",
        ),
        (
            gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "Hello, I hope you are well.",
            "rewrite_greeting_replay",
            "rewrite",
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
            "summary_greeting_replay",
            "summary",
        ),
        (
            gates._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text="what is my hotel confirmation code",
                text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
            ),
            "ask_user",
            "I do not have that in saved data. Which saved data source should I use?",
            "missing_saved_data_replay",
            "ask_user",
        ),
    ]

    for _ in range(int(repeats)):
        for name, item, deadline, reason, user_text in rewrite_cases:
            slots = {"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason}
            prompt = gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=user_text,
                text_slots=slots,
            )
            content = f"Hi {name}, could you please send the {item} by {deadline}? {reason}. Thank you."
            if "get the" in user_text or "prepare the" in user_text:
                content = f"Hi {name}, could you please prepare the {item} by {deadline}? {reason}. Thank you."
            if "share the" in user_text:
                content = f"Hi {name}, could you please share the {item} by {deadline}? {reason}. Thank you."
            if "update the" in user_text:
                content = f"Hi {name}, could you please update the {item} by {deadline}? {reason}. Thank you."
            if "finish the" in user_text:
                content = f"Hi {name}, could you please finish the {item} by {deadline}? {reason}. Thank you."
            _add(
                rows,
                split=split,
                task_type="v174_cert_rewrite_value_copy",
                prompt=prompt,
                action="respond",
                content=content,
                intent="rewrite",
                idx=idx,
                weight=16.0,
                negative=_payload("respond", "Maria, Devin, and Sam each have launch tasks with near-term deadlines.", "negative_summary_attractor"),
            )
            idx += 1
        for text, label in classify_cases:
            prompt = gates._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text=text,
                text_slots={"SOURCE_TEXT": text},
            )
            negative_label = "web_search" if label == "finance" else "finance"
            _add(
                rows,
                split=split,
                task_type="v174_cert_classifier_label",
                prompt=prompt,
                action="respond",
                content=label,
                intent="extraction",
                idx=idx,
                weight=14.0,
                negative=_payload("respond", negative_label, "negative_classifier_attractor"),
            )
            idx += 1
        for prompt, action, content, task_type, intent in replay:
            _add(
                rows,
                split=split,
                task_type=task_type,
                prompt=prompt,
                action=action,
                content=content,
                intent=intent,
                idx=idx,
                weight=6.0,
            )
            idx += 1
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-repeats", type=int, default=220)
    parser.add_argument("--eval-repeats", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_rows = _build_rows("train", int(args.train_repeats))
    eval_rows = _build_rows("eval", int(args.eval_repeats))
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    counts = Counter(str(row.get("action") or "") for row in train_rows + eval_rows)
    tasks = Counter(str(row.get("task_type") or "") for row in train_rows + eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": INTENT_LABELS,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v174_cert_two_failure_repair",
        "target_action_counts": dict(sorted(counts.items())),
        "task_type_counts": dict(sorted(tasks.items())),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
