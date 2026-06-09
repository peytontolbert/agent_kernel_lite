#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        sort_keys=True,
    )


def _row(
    *,
    split: str,
    index: int,
    prompt: str,
    action: str,
    content: str,
    task_type: str,
    intent: str,
    labels: dict[str, int],
    weight: float,
    negative: str = "",
) -> dict[str, Any]:
    return {
        "example_id": f"v172_{split}_{index:07d}",
        "split": split,
        "source_type": "pocketpal_v172_greeting_rewrite_curriculum",
        "source_id": f"v172_{task_type}_{index:07d}",
        "task_type": task_type,
        "encoder_text": prompt,
        "decoder_text": _payload(action, content, task_type),
        "negative_decoder_text": negative,
        "negative_loss_weight": 1.0 if negative else 0.0,
        "action": action,
        "weight": float(weight),
        "intent_label": intent,
        "intent_label_id": int(labels.get(intent, -1)),
    }


GREETINGS = [
    ("Hi how are you?", "Hello, I hope you are well."),
    ("hi how are you", "Hello, I hope you are well."),
    ("hey how are you doing", "Hello, I hope you are doing well."),
    ("hope you're doing ok", "Hello, I hope you are doing well."),
    ("good morning how are you", "Good morning, I hope you are well."),
    ("hello just checking in", "Hello, I am just checking in."),
    ("hey wanted to see how you are", "Hello, I wanted to see how you are doing."),
    ("hi hope all is good", "Hello, I hope all is well."),
    ("checking in to see how things are", "Hello, I am checking in to see how things are going."),
    ("hope your week is going well", "Hello, I hope your week is going well."),
]

SHORT_MESSAGES = [
    ("thanks for getting back to me", "Hello, thank you for getting back to me."),
    ("can you review this when you can", "Hello, could you please review this when you can?"),
    ("please send me the notes today", "Hello, could you please send me the notes today?"),
    ("following up on the draft", "Hello, I am following up on the draft."),
    ("can we move the meeting to monday", "Hello, could we move the meeting to Monday?"),
]


def build_rows(repeats: int, eval_repeats: int, labels: dict[str, int]) -> list[dict[str, Any]]:
    ev = _load_module(ROOT / "scripts" / "evaluate_pocketpal_agent_gates.py", "evaluate_pocketpal_agent_gates")
    gate_repair = _load_module(ROOT / "scripts" / "build_pocketpal_agent_gate_repair_dataset.py", "gate_repair")
    rows: list[dict[str, Any]] = []
    rng = random.Random(1720)
    index = 0
    negative_slot = _payload(
        "respond",
        "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
        "v172_negative_named_slots_unavailable",
    )
    instructions = [
        "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
        "Rewrite the user's provided text as a polished professional message. Preserve the intent and do not invent unavailable placeholders.",
        "Rewrite the provided message in a polite workplace tone. Use only placeholders that are available for this turn.",
    ]
    for split, count in (("train", repeats), ("eval", eval_repeats)):
        for _ in range(int(count)):
            cases = list(GREETINGS + SHORT_MESSAGES)
            rng.shuffle(cases)
            for source, target in cases:
                instruction = rng.choice(instructions)
                prompt = ev._agent_prompt(
                    name=rng.choice(["Professional Email Rewriter", "Professional Rewrite Agent", "Workplace Message Rewriter"]),
                    instruction=instruction,
                    user_text=source,
                    text_slots={"SOURCE_TEXT": source},
                )
                rows.append(
                    _row(
                        split=split,
                        index=index,
                        prompt=prompt,
                        action="respond",
                        content=target,
                        task_type="v172_greeting_or_short_source_rewrite",
                        intent="rewrite",
                        labels=labels,
                        weight=12.0,
                        negative=negative_slot,
                    )
                )
                index += 1
        # Gate replay keeps the focused pass from forgetting non-rewrite contracts.
        for replay_index, replay in enumerate(gate_repair.build_rows(max(1, int(count // 10)), weight=4.0)):
            replay = dict(replay)
            replay["split"] = split
            replay["example_id"] = f"v172_{split}_gate_replay_{index:07d}_{replay_index:04d}"
            replay["source_type"] = "pocketpal_v172_gate_replay"
            replay["weight"] = 4.0
            rows.append(replay)
            index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_v172_greeting_rewrite_curriculum")
    parser.add_argument("--intent-labels-json", default="tmp/pocketpal_v171_broad_teacher_source_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--train-repeats", type=int, default=1000)
    parser.add_argument("--eval-repeats", type=int, default=40)
    args = parser.parse_args()

    labels: dict[str, int] = {}
    path = Path(args.intent_labels_json)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        labels = {str(k): int(v) for k, v in manifest.get("intent_labels", {}).items()}

    rows = build_rows(int(args.train_repeats), int(args.eval_repeats), labels)
    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "agentkernel_lite_encdec_train.jsonl"
    eval_path = out / "agentkernel_lite_encdec_eval.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train_rows), encoding="utf-8")
    eval_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in eval_rows), encoding="utf-8")
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_v172_greeting_rewrite_curriculum",
        "dataset_format": "jsonl",
        "manifest_path": str(out / "agentkernel_lite_encdec_dataset_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "intent_labels": labels,
        "source_counts": {
            "pocketpal_v172_greeting_rewrite_curriculum": len(rows),
        },
    }
    (out / "agentkernel_lite_encdec_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
