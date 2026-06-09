#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _payload(action: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": metadata or {"task_type": "balanced_agent_repair"}},
        ensure_ascii=False,
    )


def _row(
    *,
    rows: list[dict[str, Any]],
    split: str,
    task_type: str,
    encoder_text: str,
    action: str,
    content: str,
    index: int,
    negative: str = "",
    weight: float = 2.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "example_id": f"{task_type}_{split}_{index:05d}",
            "split": split,
            "source_type": "pocketpal_balanced_agent_repair",
            "source_id": task_type,
            "task_type": task_type,
            "encoder_text": encoder_text,
            "decoder_text": _payload(action, content, metadata),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "action": action,
            "weight": float(weight),
        }
    )


def _web_metadata(query: str) -> dict[str, Any]:
    return {
        "task_type": "balanced_web_search_request",
        "extension_id": "web_search",
        "capability": "web.search",
        "query": query,
        "max_sources": 5,
        "requires_user_approval": True,
    }


def _build_rows(split: str, repeats: int) -> list[dict[str, Any]]:
    ev = _load_eval_module()
    rows: list[dict[str, Any]] = []
    idx = 0
    greetings = [
        ("Hi how are you?", "I'm doing well. How can I help?"),
        ("hello how's your day", "I'm doing well. How can I help?"),
        ("hey, how are things?", "I'm doing well. How can I help?"),
        ("good morning", "Good morning. How can I help?"),
    ]
    casual = [
        ("How's it going?", "It's going well. How's it going for you?"),
        ("what's up?", "Not much. What can I help with?"),
        ("how are things?", "Things are going well. How can I help?"),
        ("how is your day going?", "It's going well. How can I help?"),
    ]
    rewrite_greetings = [
        ("Hi how are you?", "Hello, I hope you are well."),
        ("hey how are you doing", "Hello, I hope you are doing well."),
        ("hi hope you're good", "Hello, I hope you are well."),
    ]
    emails = [
        ("John", "report", "friday", "The client is asking"),
        ("Ava", "invoice", "tomorrow", "Finance needs it"),
        ("Priya", "launch notes", "Monday", "The team is waiting"),
        ("Marco", "contract", "end of day", "Legal needs it"),
    ]
    source_texts = [
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "launch build ORBIT-42 is waiting for May TestFlight review",
        "Casey owes ACME receipt R-778 for $54.20 by June 3",
        "hotel confirmation HCN-9921 is saved under Alex for July 14",
    ]
    missing = [
        ("what is my hotel confirmation code", "I do not have that in saved data. Which saved data source should I use?"),
        ("what is my passport number", "I do not have that in saved data. Which saved data source should I use?"),
        ("find my reservation id", "I do not have that in saved data. Which saved data source should I use?"),
    ]
    saved = [
        ("what is my launch code", "[D1] saved note: Launch code is ORBIT-42 for the May TestFlight build.", "Your launch code is ORBIT-42 for the May TestFlight build."),
        ("what is the beta date", "[D1] saved note: Beta date is June 3 with web search enabled.", "Your beta date is June 3 with web search enabled."),
    ]
    searches = [
        "search the web for current TestFlight upload limits",
        "look up today's App Store Connect status",
        "find the latest iPhone beta release notes",
    ]

    for repeat in range(int(repeats)):
        for text, content in greetings:
            _row(rows=rows, split=split, task_type="runtime_plain_chat_repair", encoder_text=ev._runtime_plain_chat_prompt(text), action="respond", content=content, index=idx, weight=2.0)
            idx += 1
        for text, content in casual:
            prompt = ev._agent_prompt(name="Casual Assistant", instruction="Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it.", user_text=text)
            _row(rows=rows, split=split, task_type="casual_retention_repair", encoder_text=prompt, action="respond", content=content, index=idx, negative=_payload("respond", "I found this in your saved data: Client Review"), weight=3.0)
            idx += 1
        for text, content in rewrite_greetings:
            prompt = ev._agent_prompt(name="Professional Email Rewriter", instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.", user_text=text, text_slots={"SOURCE_TEXT": text})
            _row(rows=rows, split=split, task_type="rewrite_greeting_repair", encoder_text=prompt, action="respond", content=content, index=idx, negative=_payload("respond", "Hi John, Could you please send the report by Friday? The client is asking. Thank you."), weight=3.0)
            idx += 1
        for name, item, deadline, reason in emails:
            user_text = f"hey {name.lower()} i need the {item} by {deadline} because {reason.lower()} and we are behind"
            prompt = ev._agent_prompt(name="Professional Email Rewriter", instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.", user_text=user_text, text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason})
            _row(rows=rows, split=split, task_type="professional_email_repair", encoder_text=prompt, action="respond", content="Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]], and we are behind. Thank you.", index=idx, weight=2.0)
            idx += 1
        for text in source_texts:
            prompt = ev._agent_prompt(name="Source Echo Agent", instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.", user_text=text, text_slots={"SOURCE_TEXT": text})
            _row(rows=rows, split=split, task_type="source_echo_repair", encoder_text=prompt, action="respond", content="Source text: [[SOURCE_TEXT]]", index=idx, negative=_payload("respond", "Source text: vendor invoice INV-20: Deview"), weight=5.0)
            idx += 1
        for text, content in missing:
            prompt = ev._agent_prompt(name="Saved Data Assistant", instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.", user_text=text, text_slots={"SOURCE_TEXT": text})
            _row(rows=rows, split=split, task_type="missing_saved_data_repair", encoder_text=prompt, action="ask_user", content=content, index=idx, negative=_payload("ask_user", "I do not have that reservation code in the provided data. Where should I look for it?"), weight=3.0)
            idx += 1
        for user_text, data_context, content in saved:
            prompt = ev._agent_prompt(name="Saved Data Assistant", instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.", user_text=user_text, text_slots={"SOURCE_TEXT": user_text, "DATA_CONTEXT": data_context})
            _row(rows=rows, split=split, task_type="saved_data_use_repair", encoder_text=prompt, action="respond", content="I found this in your saved data: [[DATA_CONTEXT]]", index=idx, weight=2.0)
            idx += 1
        for query in searches:
            prompt = ev._web_agent_prompt(user_text=query)
            _row(rows=rows, split=split, task_type="web_search_request_repair", encoder_text=prompt, action="extension_request", content="Requesting approval to search the web.", index=idx, weight=2.0, metadata=_web_metadata(query))
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
    parser.add_argument("--train-repeats", type=int, default=30)
    parser.add_argument("--eval-repeats", type=int, default=4)
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
    task_counts = Counter(str(row.get("task_type") or "") for row in train_rows + eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_balanced_agent_repair",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(train_rows) + len(eval_rows),
        "target_action_counts": dict(sorted(counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
