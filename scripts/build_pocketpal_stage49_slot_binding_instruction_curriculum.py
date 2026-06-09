#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_eval_module():
    path = ROOT / "scripts" / "evaluate_pocketpal_agent_gates.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_agent_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_intent_labels() -> dict[str, int]:
    path = ROOT / "scripts" / "build_pocketpal_stage11_broad_agent_mode_dataset.py"
    spec = importlib.util.spec_from_file_location("build_pocketpal_stage11_broad_agent_mode_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load intent labels module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.INTENT_LABELS)


def _payload(action: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {
            "action": action,
            "content": content,
            "proposal_metadata": metadata or {"task_type": "stage49_slot_binding_instruction"},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _web_metadata(query: str) -> dict[str, Any]:
    return {
        "task_type": "stage49_web_search_request",
        "extension_id": "web_search",
        "capability": "web.search",
        "query": query,
        "max_sources": 5,
        "requires_user_approval": True,
    }


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _add(
    rows: list[dict[str, Any]],
    *,
    split: str,
    task_type: str,
    encoder_text: str,
    action: str,
    content: str,
    intent: str,
    intent_labels: dict[str, int],
    index: int,
    weight: float,
    negative: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "example_id": _stable_id("stage49", split, task_type, str(index), encoder_text, content),
            "split": split,
            "source_type": "pocketpal_stage49_slot_binding_instruction_curriculum",
            "source_id": f"stage49_{task_type}_{index:06d}",
            "task_type": task_type,
            "encoder_text": encoder_text,
            "decoder_text": _payload(action, content, metadata),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "action": action,
            "weight": float(weight),
            "intent_label": intent,
            "intent_label_id": int(intent_labels.get(intent, -1)),
        }
    )


def _build_rows(split: str, repeats: int, seed: int, intent_labels: dict[str, int]) -> list[dict[str, Any]]:
    ev = _load_eval_module()
    rng = random.Random(seed + (0 if split == "train" else 1_000_000))
    rows: list[dict[str, Any]] = []
    idx = 0

    people = ["Ava", "Priya", "Marco", "Lena", "Sam", "Nora", "Devon", "Iris", "Omar", "Mina", "Tao", "Riley"]
    items = [
        "budget draft",
        "slide deck",
        "receipt",
        "timeline",
        "launch notes",
        "invoice",
        "contract",
        "vendor form",
        "security brief",
        "test plan",
    ]
    deadlines = ["June 3", "Friday", "Monday", "tomorrow morning", "July 14", "end of day", "Thursday at 2 PM"]
    reasons = [
        "finance is waiting",
        "the client is asking",
        "legal needs it",
        "the launch meeting moved up",
        "the TestFlight review is blocked",
        "the team is waiting",
    ]
    source_texts = [
        "Nora's access code is GATE-17 for Thursday at 2 PM",
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "Casey owes ACME receipt R-778 for $54.20 by June 3",
        "hotel confirmation HCN-9921 is saved under Alex for July 14",
        "Sam moved the roadmap review to Thursday at 2 PM",
        "launch build ORBIT-42 is waiting for May TestFlight review",
        "Iris needs the security brief by end of day because audit starts Monday",
    ]
    classify_examples = [
        ("Can you rewrite this note professionally?", "writing"),
        ("Please approve invoice INV-2048 for $1,200.", "finance"),
        ("Move the launch meeting to Thursday at 2 PM.", "schedule"),
        ("The hotel code is HCN-9921 for July 14.", "travel"),
        ("Search the web for current TestFlight limits.", "web_search"),
    ]
    translations = [
        ("Please send the invoice tomorrow morning.", "Spanish", "Por favor, envia la factura manana por la manana."),
        ("The launch meeting moved to Thursday.", "Spanish", "La reunion de lanzamiento se movio al jueves."),
        ("Thank you for the update.", "French", "Merci pour la mise a jour."),
        ("Please review the contract by Friday.", "German", "Bitte prufen Sie den Vertrag bis Freitag."),
    ]
    web_queries = [
        "search the web for current Swift WKWebView navigation policy examples",
        "find current App Store Connect TestFlight processing status",
        "look up today's iPhone beta release notes",
        "search current DuckDuckGo result page format",
        "find recent Apple developer WKWebView documentation updates",
    ]

    for _ in range(int(repeats)):
        name = rng.choice(people)
        item = rng.choice(items)
        deadline = rng.choice(deadlines)
        reason = rng.choice(reasons)
        user_text = f"yo {name.lower()} send the {item} by {deadline.lower()} because {reason}"
        prompt = ev._agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text=user_text,
            text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
        )
        _add(
            rows,
            split=split,
            task_type="stage49_rewrite_slot_binding",
            encoder_text=prompt,
            action="respond",
            content="Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
            intent="rewrite",
            intent_labels=intent_labels,
            index=idx,
            weight=5.0,
            negative=_payload("respond", "Hi John, could you please send the report by Friday? The client is asking. Thank you."),
        )
        idx += 1

        text = rng.choice(source_texts)
        prompt = ev._agent_prompt(
            name="Source Echo Agent",
            instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
            user_text=text,
            text_slots={"SOURCE_TEXT": text},
        )
        _add(
            rows,
            split=split,
            task_type="stage49_source_slot_binding",
            encoder_text=prompt,
            action="respond",
            content="Source text: [[SOURCE_TEXT]]",
            intent="source_echo",
            intent_labels=intent_labels,
            index=idx,
            weight=4.0,
            negative=_payload("respond", "Source text: vendor invoice INV-20: inviewown-searchsearch"),
        )
        idx += 1

        classify_text, label = rng.choice(classify_examples)
        prompt = ev._agent_prompt(
            name="Classifier Agent",
            instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
            user_text=classify_text,
            text_slots={"SOURCE_TEXT": classify_text},
        )
        wrong = "finance" if label != "finance" else "writing"
        _add(
            rows,
            split=split,
            task_type="stage49_classify_instruction_binding",
            encoder_text=prompt,
            action="respond",
            content=label,
            intent="json",
            intent_labels=intent_labels,
            index=idx,
            weight=4.0,
            negative=_payload("respond", wrong),
        )
        idx += 1

        prompt = ev._agent_prompt(
            name="Bullet Summary Agent",
            instruction="Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
            user_text=text,
            text_slots={"SOURCE_TEXT": text},
        )
        _add(
            rows,
            split=split,
            task_type="stage49_summary_instruction_binding",
            encoder_text=prompt,
            action="respond",
            content="- Summary: [[SOURCE_TEXT]]",
            intent="summary",
            intent_labels=intent_labels,
            index=idx,
            weight=2.0,
            negative=_payload("respond", "I'm doing well. How can I help?"),
        )
        idx += 1

        prompt = ev._agent_prompt(
            name="Extractor Agent",
            instruction="Extract the key names, dates, codes, money values, and blockers from the user text. If a field is missing, say not provided.",
            user_text=text,
            text_slots={"SOURCE_TEXT": text},
        )
        _add(
            rows,
            split=split,
            task_type="stage49_extract_instruction_binding",
            encoder_text=prompt,
            action="respond",
            content="Extracted details: [[SOURCE_TEXT]]",
            intent="extraction",
            intent_labels=intent_labels,
            index=idx,
            weight=2.0,
        )
        idx += 1

        trans_text, lang, translated = rng.choice(translations)
        prompt = ev._agent_prompt(
            name="Translator Agent",
            instruction=f"Translate the user's provided text into {lang}. Preserve names, dates, and values. Return only the translation.",
            user_text=trans_text,
            text_slots={"SOURCE_TEXT": trans_text},
        )
        _add(
            rows,
            split=split,
            task_type="stage49_translate_instruction_binding",
            encoder_text=prompt,
            action="respond",
            content=translated,
            intent="translation",
            intent_labels=intent_labels,
            index=idx,
            weight=2.0,
        )
        idx += 1

        data_text = rng.choice(source_texts)
        prompt = ev._agent_prompt(
            name="Saved Data Assistant",
            instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
            user_text="what saved detail should I use",
            stale_context="Selected paper [P1]: unrelated optimization notes from a previous research turn.",
            text_slots={
                "SOURCE_TEXT": "what saved detail should I use",
                "DATA_CONTEXT": f"[D1] saved note: {data_text}",
            },
        )
        _add(
            rows,
            split=split,
            task_type="stage49_memory_instruction_binding",
            encoder_text=prompt,
            action="respond",
            content="I found this in your saved data: [[DATA_CONTEXT]]",
            intent="saved_data",
            intent_labels=intent_labels,
            index=idx,
            weight=2.5,
            negative=_payload("respond", "Selected paper [P1] focuses on unrelated optimization notes."),
        )
        idx += 1

        query = rng.choice(web_queries)
        prompt = ev._web_agent_prompt(user_text=query)
        _add(
            rows,
            split=split,
            task_type="stage49_web_search_request",
            encoder_text=prompt,
            action="extension_request",
            content="Requesting approval to search the web.",
            intent="web_search",
            intent_labels=intent_labels,
            index=idx,
            weight=2.5,
            metadata=_web_metadata(query),
            negative=_payload("respond", "I cannot browse the web."),
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
    parser.add_argument("--train-repeats", type=int, default=6000)
    parser.add_argument("--eval-repeats", type=int, default=400)
    parser.add_argument("--seed", type=int, default=491)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    intent_labels = _load_intent_labels()
    train_rows = _build_rows("train", int(args.train_repeats), int(args.seed), intent_labels)
    eval_rows = _build_rows("eval", int(args.eval_repeats), int(args.seed), intent_labels)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    all_rows = train_rows + eval_rows
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage49_slot_binding_instruction_curriculum",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(all_rows),
        "source_counts": {"pocketpal_stage49_slot_binding_instruction_curriculum": len(all_rows)},
        "target_action_counts": dict(sorted(Counter(str(row["action"]) for row in all_rows).items())),
        "task_type_counts": dict(sorted(Counter(str(row["task_type"]) for row in all_rows).items())),
        "intent_labels": intent_labels,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
