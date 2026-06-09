#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

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


def _load_eval_module():
    path = _repo_root() / "scripts" / "evaluate_pocketpal_agent_gates.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_agent_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(action: str, content: str) -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": {"task_type": "slot_rewrite_classify_repair"}},
        ensure_ascii=False,
    )


def _add(
    rows: list[dict],
    split: str,
    task_type: str,
    prompt: str,
    action: str,
    content: str,
    idx: int,
    weight: float,
    negative: str = "",
    intent: str = "",
) -> None:
    intent_label = intent or {
        "extension_request": "web_search",
        "ask_user": "ask_user",
    }.get(action, "")
    rows.append(
        {
            "example_id": f"{task_type}_{split}_{idx:05d}",
            "split": split,
            "source_type": "pocketpal_slot_rewrite_classify_repair",
            "source_id": task_type,
            "task_type": task_type,
            "encoder_text": prompt,
            "decoder_text": _payload(action, content),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "action": action,
            "intent_label": intent_label,
            "intent_label_id": INTENT_LABELS.get(intent_label, -1),
            "weight": float(weight),
        }
    )


def _build_rows(split: str, repeats: int) -> list[dict]:
    ev = _load_eval_module()
    rows: list[dict] = []
    idx = 0
    rewrites = [
        ("Lena", "budget draft", "June 3", "Finance is waiting", "yo lena send the budget draft by june 3 because finance is waiting"),
        ("Ava", "invoice", "tomorrow morning", "Finance needs it", "ava send the invoice tomorrow morning because finance needs it"),
        ("Priya", "launch notes", "Monday", "The team is waiting", "priya please send launch notes monday because the team is waiting"),
        ("Marco", "contract", "end of day", "Legal needs it", "marco send contract by end of day because legal needs it"),
        ("Sam", "slide deck", "Thursday at 2 PM", "The client review moved up", "sam get the slide deck done by thursday at 2 pm because the client review moved up"),
        ("Nora", "badge list", "Friday", "Security needs it", "nora send the badge list friday because security needs it"),
        ("Devon", "roadmap update", "next Tuesday", "The planning meeting moved up", "devon send roadmap update next tuesday because the planning meeting moved up"),
    ]
    classify = [
        ("Can you rewrite this note professionally?", "writing"),
        ("Please approve invoice INV-2048 for $1,200.", "finance"),
        ("Move the launch meeting to Thursday at 2 PM.", "schedule"),
        ("The hotel code is HCN-9921 for July 14.", "travel"),
        ("Search the web for current TestFlight limits.", "web_search"),
        ("Could you make this email sound more polished?", "writing"),
        ("Find current Swift WKWebView examples online.", "web_search"),
        ("Reserve a room for July 14 under Alex.", "travel"),
        ("The budget draft is waiting on finance approval.", "finance"),
        ("Put the roadmap review on Monday morning.", "schedule"),
    ]
    fixed_replay = [
        (ev._runtime_plain_chat_prompt("Hi how are you?"), "respond", "I'm doing well. How can I help?"),
        (
            ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "Hello, I hope you are well.",
        ),
        (
            ev._agent_prompt(
                name="Bullet Summary Agent",
                instruction="Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "- Greeting: Hi, how are you?",
        ),
        (
            ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="hey john i need the report by friday because the client is asking and we are behind",
                text_slots={"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
            ),
            "respond",
            "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]], and we are behind. Thank you.",
        ),
        (
            ev._agent_prompt(
                name="Source Echo Agent",
                instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
                user_text="vendor invoice INV-2048 is blocked until finance approves $1,200",
                text_slots={"SOURCE_TEXT": "vendor invoice INV-2048 is blocked until finance approves $1,200"},
            ),
            "respond",
            "Source text: [[SOURCE_TEXT]]",
        ),
        (
            ev._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text="what is my launch code",
                text_slots={"SOURCE_TEXT": "what is my launch code", "DATA_CONTEXT": "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."},
            ),
            "respond",
            "I found this in your saved data: [[DATA_CONTEXT]]",
        ),
        (
            ev._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text="what is my hotel confirmation code",
                text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
            ),
            "ask_user",
            "I do not have that in saved data. Which saved data source should I use?",
        ),
        (ev._web_agent_prompt(user_text="search the web for current TestFlight upload limits"), "extension_request", "Requesting approval to search the web."),
    ]
    for _ in range(int(repeats)):
        for name, item, deadline, reason, user_text in rewrites:
            prompt = ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=user_text,
                text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
            )
            _add(
                rows,
                split,
                "slot_rewrite_generalization",
                prompt,
                "respond",
                "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
                idx,
                6.0,
                negative=_payload("respond", "Hi John, could you please send the report by Friday? The client is asking, and we are behind. Thank you."),
                intent="rewrite",
            )
            idx += 1
        for text, label in classify:
            prompt = ev._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text=text,
                text_slots={"SOURCE_TEXT": text},
            )
            _add(
                rows,
                split,
                "classifier_label_generalization",
                prompt,
                "respond",
                label,
                idx,
                6.0,
                negative=_payload("respond", "Source text: [[SOURCE_TEXT]]"),
                intent="extraction",
            )
            idx += 1
        for prompt, action, target in fixed_replay:
            replay_intent = "rewrite" if "Professional Email Rewriter" in prompt else "source_echo" if "Source Echo Agent" in prompt else "saved_data" if "Saved Data Assistant" in prompt else "summary" if "Bullet Summary Agent" in prompt else ""
            _add(rows, split, "fixed_gate_replay", prompt, action, target, idx, 2.5, intent=replay_intent)
            idx += 1
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-repeats", type=int, default=120)
    parser.add_argument("--eval-repeats", type=int, default=10)
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
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_slot_rewrite_classify_repair",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(train_rows) + len(eval_rows),
        "target_action_counts": dict(sorted(counts.items())),
        "intent_labels": INTENT_LABELS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
