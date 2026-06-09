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
        {"action": action, "content": content, "proposal_metadata": metadata or {"task_type": "saved_data_slot_repair"}},
        ensure_ascii=False,
    )


def _add(rows: list[dict[str, Any]], split: str, task_type: str, prompt: str, action: str, content: str, idx: int, weight: float, negative: str = "") -> None:
    rows.append(
        {
            "example_id": f"{task_type}_{split}_{idx:05d}",
            "split": split,
            "source_type": "pocketpal_saved_data_slot_repair",
            "source_id": task_type,
            "task_type": task_type,
            "encoder_text": prompt,
            "decoder_text": _payload(action, content),
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
    saved_cases = [
        (
            "what is my launch code",
            "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build.",
        ),
        (
            "what badge code do I use",
            "[D1] saved note: Nora's badge code is GATE-17 for the south entrance.",
        ),
        (
            "what is the hotel code",
            "[D1] saved note: Hotel confirmation code is HCN-9921 under Alex for July 14.",
        ),
    ]
    replay_cases = [
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
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="hey john i need the report by friday because the client is asking and we are behind",
                text_slots={"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
            ),
            "respond",
            "Hi John, could you please send the report by Friday? The client is asking, and we are behind. Thank you.",
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
                user_text="what is my hotel confirmation code",
                text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
            ),
            "ask_user",
            "I do not have that in saved data. Which saved data source should I use?",
        ),
        (ev._web_agent_prompt(user_text="search the web for current TestFlight upload limits"), "extension_request", "Requesting approval to search the web."),
    ]
    for _ in range(int(repeats)):
        for user_text, data_context in saved_cases:
            prompt = ev._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text=user_text,
                stale_context="Selected paper [P1]: unrelated transformer attention paper.",
                text_slots={"SOURCE_TEXT": user_text, "DATA_CONTEXT": data_context},
            )
            _add(
                rows,
                split,
                "saved_data_slot_copy_repair",
                prompt,
                "respond",
                "I found this in your saved data: [[DATA_CONTEXT]]",
                idx,
                6.0,
                negative=_payload("respond", "Your launch code is ORBIT- Am- Am1,200"),
            )
            idx += 1
        for prompt, action, content in replay_cases:
            _add(rows, split, "saved_data_slot_replay", prompt, action, content, idx, 2.0)
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
    parser.add_argument("--train-repeats", type=int, default=80)
    parser.add_argument("--eval-repeats", type=int, default=8)
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
        "objective": "pocketpal_saved_data_slot_repair",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(train_rows) + len(eval_rows),
        "target_action_counts": dict(sorted(counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
