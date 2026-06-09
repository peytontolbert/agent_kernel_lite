#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
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


def _payload(action: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {
            "action": action,
            "content": content,
            "proposal_metadata": metadata or {"task_type": "v168e_slot_availability_microrepair"},
        },
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
    intent_labels: dict[str, int],
    weight: float,
    negative: str = "",
) -> dict[str, Any]:
    return {
        "example_id": f"v168e_{split}_{task_type}_{index:06d}",
        "split": split,
        "source_type": "pocketpal_v168e_slot_availability_microrepair",
        "source_id": f"v168e_{task_type}_{index:06d}",
        "task_type": task_type,
        "encoder_text": prompt,
        "decoder_text": _payload(action, content, {"task_type": task_type}),
        "negative_decoder_text": negative,
        "negative_loss_weight": 1.0 if negative else 0.0,
        "action": action,
        "weight": float(weight),
        "intent_label": intent,
        "intent_label_id": int(intent_labels.get(intent, -1)),
    }


def build_rows(split: str, repeats: int, intent_labels: dict[str, int]) -> list[dict[str, Any]]:
    ev = _load_module(ROOT / "scripts" / "evaluate_pocketpal_agent_gates.py", "evaluate_pocketpal_agent_gates")
    rows: list[dict[str, Any]] = []
    idx = 0

    for _ in range(int(repeats)):
        prompt = ev._agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text="Hi how are you?",
            text_slots={"SOURCE_TEXT": "Hi how are you?"},
        )
        rows.append(
            _row(
                split=split,
                index=idx,
                prompt=prompt,
                action="respond",
                content="Hello, I hope you are well.",
                task_type="v168e_rewrite_source_text_only",
                intent="rewrite",
                intent_labels=intent_labels,
                weight=5.0,
                negative=_payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you."),
            )
        )
        idx += 1

        prompt = ev._agent_prompt(
            name="Bullet Summary Agent",
            instruction="Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
            user_text="Hi how are you?",
            text_slots={"SOURCE_TEXT": "Hi how are you?"},
        )
        rows.append(
            _row(
                split=split,
                index=idx,
                prompt=prompt,
                action="respond",
                content="- Greeting: Hi, how are you?",
                task_type="v168e_summary_source_text_only",
                intent="summary",
                intent_labels=intent_labels,
                weight=5.0,
                negative=_payload("respond", "- Summary: [[SOURCE_TEXT]]"),
            )
        )
        idx += 1

        prompt = ev._agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text="",
            text_slots={},
        )
        rows.append(
            _row(
                split=split,
                index=idx,
                prompt=prompt,
                action="ask_user",
                content="What text should I rewrite?",
                task_type="v168e_ask_missing_editable_text",
                intent="ask_user",
                intent_labels=intent_labels,
                weight=6.0,
                negative=_payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you."),
            )
        )
        idx += 1

        for name, item, deadline, reason in [
            ("Lena", "budget draft", "June 3", "finance is waiting"),
            ("John", "report", "friday", "The client is asking"),
            ("Priya", "slide deck", "Monday", "the team is waiting"),
        ]:
            user_text = f"yo {name.lower()} send the {item} by {deadline.lower()} because {reason}"
            prompt = ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=user_text,
                text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
            )
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="respond",
                    content="Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
                    task_type="v168e_rewrite_named_slots_available",
                    intent="rewrite",
                    intent_labels=intent_labels,
                    weight=4.0,
                )
            )
            idx += 1

        prompt = ev._agent_prompt(
            name="Saved Data Assistant",
            instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
            user_text="what is my launch code",
            text_slots={
                "SOURCE_TEXT": "what is my launch code",
                "DATA_CONTEXT": "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build.",
            },
        )
        rows.append(
            _row(
                split=split,
                index=idx,
                prompt=prompt,
                action="respond",
                content="I found this in your saved data: [[DATA_CONTEXT]]",
                task_type="v168e_saved_data_slot_available",
                intent="saved_data",
                intent_labels=intent_labels,
                weight=4.0,
            )
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
    parser.add_argument("--train-repeats", type=int, default=1200)
    parser.add_argument("--eval-repeats", type=int, default=120)
    args = parser.parse_args()

    labels_module = _load_module(
        ROOT / "scripts" / "build_pocketpal_stage11_broad_agent_mode_dataset.py",
        "build_pocketpal_stage11_broad_agent_mode_dataset",
    )
    intent_labels = dict(labels_module.INTENT_LABELS)
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_rows = build_rows("train", int(args.train_repeats), intent_labels)
    eval_rows = build_rows("eval", int(args.eval_repeats), intent_labels)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_v168e_slot_availability_microrepair",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(train_rows) + len(eval_rows),
        "source_counts": {"pocketpal_v168e_slot_availability_microrepair": len(train_rows) + len(eval_rows)},
        "intent_labels": intent_labels,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
