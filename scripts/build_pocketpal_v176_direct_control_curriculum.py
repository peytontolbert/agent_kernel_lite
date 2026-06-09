#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FOCUS_TASKS = {
    "active_agent_action_items",
    "active_agent_brainstorm",
    "active_agent_checklist",
    "active_agent_extraction",
    "active_agent_json",
    "active_agent_plan",
    "active_agent_rewrite",
    "active_agent_risks",
    "active_agent_subject",
    "active_agent_summary",
    "active_agent_translation",
}


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


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


def _decision_content(text: str) -> tuple[str, str]:
    try:
        payload = json.loads(str(text or ""))
    except json.JSONDecodeError:
        return "respond", str(text or "").strip()
    if not isinstance(payload, dict):
        return "respond", str(text or "").strip()
    return str(payload.get("action") or "respond"), str(payload.get("content") or "").strip()


def _hash_split(key: str, eval_fraction: float) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if value < eval_fraction else "train"


def _add_replay(rows: list[dict[str, Any]], split: str, repeats: int) -> None:
    gates = _load_gates()
    replay = [
        (gates._runtime_plain_chat_prompt("Hi how are you?"), "respond", "I'm doing well. What would you like help with?", "runtime_plain_chat", "casual", 10.0),
        (
            gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            "Hello, I hope you are well.",
            "active_agent_rewrite",
            "rewrite",
            24.0,
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
            12.0,
        ),
    ]
    for repeat in range(repeats):
        for index, (prompt, action, content, task_type, intent, weight) in enumerate(replay):
            rows.append(
                {
                    "action": action,
                    "decoder_text": _payload(action, content, task_type),
                    "encoder_text": prompt,
                    "example_id": f"v176_replay_{split}_{repeat:04d}_{index:02d}",
                    "expected_content": content,
                    "intent_label": intent,
                    "intent_label_id": INTENT_LABELS.get(intent, -1),
                    "negative_decoder_text": "",
                    "negative_loss_weight": 0.0,
                    "source_id": "v176_replay",
                    "source_type": "pocketpal_v176_direct_control_curriculum",
                    "split": split,
                    "task_type": task_type,
                    "weight": weight,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()

    source_manifest = json.loads(Path(args.input_manifest).read_text(encoding="utf-8"))
    source_paths = [Path(source_manifest["train_dataset_path"]), Path(source_manifest["eval_dataset_path"])]
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        for row in _iter_jsonl(path):
            task_type = str(row.get("task_type") or "")
            if task_type not in FOCUS_TASKS:
                continue
            action, content = _decision_content(str(row.get("decoder_text") or ""))
            if not content:
                continue
            for repeat in range(int(args.repeat)):
                converted = dict(row)
                converted["action"] = action
                converted["decoder_text"] = _payload(action, content, task_type)
                converted["expected_content"] = content
                converted["example_id"] = f"{row.get('example_id') or row.get('source_id')}_v176_{repeat:02d}"
                converted["negative_decoder_text"] = ""
                converted["negative_loss_weight"] = 0.0
                converted["source_type"] = "pocketpal_v176_direct_control_curriculum"
                converted["split"] = _hash_split(str(converted["example_id"]), float(args.eval_fraction))
                converted["weight"] = max(float(row.get("weight") or 1.0), 20.0)
                rows.append(converted)

    _add_replay(rows, "train", 260)
    _add_replay(rows, "eval", 24)

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    for path, split_rows in [(train_path, train_rows), (eval_path, eval_rows)]:
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    actions = Counter(str(row.get("action") or "") for row in rows)
    tasks = Counter(str(row.get("task_type") or "") for row in rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v176_direct_control_curriculum",
        "source_manifest_path": str(Path(args.input_manifest).resolve()),
        "target_action_counts": dict(sorted(actions.items())),
        "task_type_counts": dict(sorted(tasks.items())),
        "total_examples": len(rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
