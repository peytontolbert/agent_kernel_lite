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
            "proposal_metadata": metadata or {"task_type": "v168f_slot_contract_curriculum"},
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
        "example_id": f"v168f_{split}_{task_type}_{index:06d}",
        "split": split,
        "source_type": "pocketpal_v168f_slot_contract_curriculum",
        "source_id": f"v168f_{task_type}_{index:06d}",
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

    source_rewrites = [
        ("Hi how are you?", "Hello, I hope you are well."),
        ("hey can you check this today", "Hello, could you please check this today?"),
        ("please send over the notes when you can", "Hello, could you please send over the notes when you can?"),
        ("thanks for helping me earlier", "Hello, thank you for helping me earlier."),
        ("i need to move our meeting to monday", "Hello, I need to move our meeting to Monday."),
    ]
    summaries = [
        ("Hi how are you?", "- Greeting: Hi, how are you?"),
        ("the invoice is blocked until finance approves $1,200", "- Invoice blocked until finance approves $1,200."),
        ("move the launch review to Tuesday because QA needs more time", "- Launch review should move to Tuesday because QA needs more time."),
        ("Nora has the south entrance badge code GATE-17", "- Nora has the south entrance badge code GATE-17."),
        ("ask Priya to review the draft before noon", "- Priya should review the draft before noon."),
    ]
    extracts = [
        ("Nora has code GATE-17 for Thursday at 2 PM", "Nora; GATE-17; Thursday at 2 PM"),
        ("invoice INV-2048 is blocked until finance approves $1,200", "INV-2048; finance approval; $1,200"),
        ("call Lena at 4 PM about the budget draft", "Lena; 4 PM; budget draft"),
    ]
    classes = [
        ("rewrite this professionally: hey john send report", "writing"),
        ("summarize this note into bullets", "summary"),
        ("search the web for current TestFlight limits", "web_search"),
        ("save my parking code as BLUE-19", "memory"),
    ]
    translations = [
        ("good morning", "Buenos días."),
        ("thank you for your help", "Gracias por tu ayuda."),
        ("see you tomorrow", "Nos vemos mañana."),
    ]

    for _ in range(int(repeats)):
        for source, target in source_rewrites:
            prompt = ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=source,
                text_slots={"SOURCE_TEXT": source},
            )
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="respond",
                    content=target,
                    task_type="v168f_rewrite_source_text_directly",
                    intent="rewrite",
                    intent_labels=intent_labels,
                    weight=6.0,
                    negative=_payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you."),
                )
            )
            idx += 1

        for source, target in summaries:
            prompt = ev._agent_prompt(
                name="Bullet Summary Agent",
                instruction="Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
                user_text=source,
                text_slots={"SOURCE_TEXT": source},
            )
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="respond",
                    content=target,
                    task_type="v168f_summarize_source_text_semantically",
                    intent="summary",
                    intent_labels=intent_labels,
                    weight=6.0,
                    negative=_payload("respond", "- Summary: [[SOURCE_TEXT]]"),
                )
            )
            idx += 1

        for source, target in extracts:
            prompt = ev._agent_prompt(
                name="Extraction Agent",
                instruction="Extract the important names, dates, codes, amounts, or requested facts from the user's provided text. If there is no source text, ask for it.",
                user_text=source,
                text_slots={"SOURCE_TEXT": source},
            )
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="respond",
                    content=target,
                    task_type="v168f_extract_from_source_text",
                    intent="extract",
                    intent_labels=intent_labels,
                    weight=3.0,
                    negative=_payload("respond", "[[NAME]]; [[ITEM]]; [[DEADLINE]]"),
                )
            )
            idx += 1

        for source, target in classes:
            prompt = ev._agent_prompt(
                name="Intent Classifier",
                instruction="Classify the user's request into one short label. Return only the label.",
                user_text=source,
                text_slots={"SOURCE_TEXT": source},
            )
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="respond",
                    content=target,
                    task_type="v168f_classify_source_text",
                    intent="classify",
                    intent_labels=intent_labels,
                    weight=2.5,
                )
            )
            idx += 1

        for source, target in translations:
            prompt = ev._agent_prompt(
                name="Spanish Translator",
                instruction="Translate the user's provided text into Spanish. Preserve meaning and do not answer as a chatbot.",
                user_text=source,
                text_slots={"SOURCE_TEXT": source},
            )
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="respond",
                    content=target,
                    task_type="v168f_translate_source_text",
                    intent="translate",
                    intent_labels=intent_labels,
                    weight=2.0,
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
                    task_type="v168f_named_slots_available",
                    intent="rewrite",
                    intent_labels=intent_labels,
                    weight=3.5,
                )
            )
            idx += 1

        for agent_name, instruction, ask in [
            (
                "Professional Email Rewriter",
                "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                "What text should I rewrite?",
            ),
            (
                "Bullet Summary Agent",
                "Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
                "What text should I summarize?",
            ),
            (
                "Spanish Translator",
                "Translate the user's provided text into Spanish. Preserve meaning and do not answer as a chatbot.",
                "What text should I translate?",
            ),
        ]:
            prompt = ev._agent_prompt(name=agent_name, instruction=instruction, user_text="", text_slots={})
            rows.append(
                _row(
                    split=split,
                    index=idx,
                    prompt=prompt,
                    action="ask_user",
                    content=ask,
                    task_type="v168f_missing_source_text",
                    intent="ask_user",
                    intent_labels=intent_labels,
                    weight=4.0,
                    negative=_payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you."),
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
    parser.add_argument("--train-repeats", type=int, default=800)
    parser.add_argument("--eval-repeats", type=int, default=80)
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
        "objective": "pocketpal_v168f_slot_contract_curriculum",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(train_rows) + len(eval_rows),
        "source_counts": {"pocketpal_v168f_slot_contract_curriculum": len(train_rows) + len(eval_rows)},
        "intent_labels": intent_labels,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
