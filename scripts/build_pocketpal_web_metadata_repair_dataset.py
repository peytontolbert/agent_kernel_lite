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
    proposal = {"task_type": "web_metadata_repair"}
    proposal.update(metadata or {})
    return json.dumps({"action": action, "content": content, "proposal_metadata": proposal}, ensure_ascii=False)


def _add(rows: list[dict[str, Any]], split: str, task_type: str, prompt: str, action: str, content: str, idx: int, weight: float, metadata: dict[str, Any] | None = None, negative: str = "") -> None:
    rows.append(
        {
            "example_id": f"{task_type}_{split}_{idx:05d}",
            "split": split,
            "source_type": "pocketpal_web_metadata_repair",
            "source_id": task_type,
            "task_type": task_type,
            "encoder_text": prompt,
            "decoder_text": _payload(action, content, metadata),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "action": action,
            "weight": float(weight),
        }
    )


def _build_rows(split: str, repeats: int) -> list[dict[str, Any]]:
    ev = _load_eval_module()
    rows: list[dict[str, Any]] = []
    idx = 0
    web_queries = [
        "search the web for current TestFlight upload limits",
        "search the web for current Swift WKWebView navigation policy examples",
        "find current Apple TestFlight beta review requirements online",
        "look up the latest App Store Connect TestFlight rules",
    ]
    metadata_base = {
        "extension_id": "web_search",
        "capability": "web.search",
        "max_sources": 5,
        "requires_user_approval": True,
    }
    replay = [
        (ev._runtime_plain_chat_prompt("Hi how are you?"), "respond", "I'm doing well. How can I help?", None),
        (
            ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "Hello, I hope you are well.",
            None,
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
            None,
        ),
        (
            ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="hey john i need the report by friday because the client is asking and we are behind",
                text_slots={"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
            ),
            "respond",
            "Hi John, could you please send the report by Friday? The client is asking, and we are behind. Thank you.",
            None,
        ),
        (
            ev._agent_prompt(
                name="Casual Assistant",
                instruction="Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it.",
                user_text="How's it going?",
                stale_context="Selected paper [P1]: unrelated optimization notes from a previous research turn.",
            ),
            "respond",
            "It's going well. How's the build today?",
            None,
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
            None,
        ),
        (
            ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="rewrite this",
            ),
            "ask_user",
            "What text should I rewrite?",
            None,
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
            None,
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
            None,
        ),
    ]
    for _ in range(int(repeats)):
        for query in web_queries:
            metadata = dict(metadata_base)
            metadata["query"] = query
            _add(
                rows,
                split,
                "web_metadata",
                ev._web_agent_prompt(user_text=query),
                "extension_request",
                "Requesting approval to search the web.",
                idx,
                8.0,
                metadata,
                negative=_payload("extension_request", "Requesting approval to search the web.", {"extension_id": "weanced"}),
            )
            idx += 1
        for prompt, action, content, metadata in replay:
            _add(rows, split, "fixed_gate_replay", prompt, action, content, idx, 4.0, metadata)
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
    parser.add_argument("--train-repeats", type=int, default=100)
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
    task_counts = Counter(str(row.get("task_type") or "") for row in train_rows + eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_web_metadata_repair",
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
