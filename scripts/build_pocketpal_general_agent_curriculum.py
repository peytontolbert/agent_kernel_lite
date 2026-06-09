#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
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
        {
            "action": action,
            "content": content,
            "proposal_metadata": metadata or {"task_type": "general_agent_curriculum"},
        },
        ensure_ascii=False,
    )


def _web_metadata(query: str) -> dict[str, Any]:
    return {
        "task_type": "general_web_search_request",
        "extension_id": "web_search",
        "capability": "web.search",
        "query": query,
        "max_sources": 5,
        "requires_user_approval": True,
    }


def _add(
    rows: list[dict[str, Any]],
    *,
    split: str,
    task_type: str,
    encoder_text: str,
    action: str,
    content: str,
    index: int,
    weight: float = 1.0,
    negative: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "example_id": f"{task_type}_{split}_{index:06d}",
            "split": split,
            "source_type": "pocketpal_general_agent_curriculum",
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


def _build_rows(split: str, repeats: int, seed: int) -> list[dict[str, Any]]:
    ev = _load_eval_module()
    rng = random.Random(seed + (0 if split == "train" else 100_000))
    rows: list[dict[str, Any]] = []
    idx = 0

    people = ["John", "Ava", "Priya", "Marco", "Lena", "Sam", "Nora", "Devon"]
    items = ["report", "invoice", "launch notes", "contract", "budget draft", "slide deck", "receipt", "timeline"]
    deadlines = ["Friday", "tomorrow morning", "Monday", "end of day", "June 3", "July 14"]
    reasons = [
        "the client is asking",
        "finance needs it",
        "the team is waiting",
        "legal needs it",
        "the TestFlight review is blocked",
        "the launch meeting moved up",
    ]
    source_texts = [
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "launch build ORBIT-42 is waiting for May TestFlight review",
        "Casey owes ACME receipt R-778 for $54.20 by June 3",
        "hotel confirmation HCN-9921 is saved under Alex for July 14",
        "Sam moved the roadmap review to Thursday at 2 PM",
        "Nora's badge code is GATE-17 for the south entrance",
    ]
    labels = [
        ("The hotel code is HCN-9921 for July 14.", "travel"),
        ("Please approve invoice INV-2048 for $1,200.", "finance"),
        ("Move the launch meeting to Thursday at 2 PM.", "schedule"),
        ("Can you rewrite this note professionally?", "writing"),
        ("Search the web for current TestFlight limits.", "web_search"),
    ]
    translations = [
        ("Please send the invoice tomorrow morning.", "Spanish", "Por favor, envia la factura manana por la manana."),
        ("The launch meeting moved to Thursday.", "Spanish", "La reunion de lanzamiento se movio al jueves."),
        ("Thank you for the update.", "French", "Merci pour la mise a jour."),
    ]
    plans = [
        (
            "prepare for TestFlight launch",
            ["Confirm upload requirements", "Check clickable links", "Run active-agent evals", "Submit the build"],
        ),
        (
            "organize the client report",
            ["Collect open numbers", "Draft the summary", "Review blockers", "Send the final report"],
        ),
        (
            "clean up saved user data",
            ["Identify duplicate notes", "Archive stale files", "Keep current records", "Verify retrieval works"],
        ),
    ]
    searches = [
        "search the web for current TestFlight upload limits",
        "look up today's App Store Connect status",
        "find the latest iPhone beta release notes",
        "search for Swift WKWebView navigation policy examples",
        "find current DuckDuckGo search result page format",
    ]

    for _ in range(int(repeats)):
        name = rng.choice(people)
        item = rng.choice(items)
        deadline = rng.choice(deadlines)
        reason = rng.choice(reasons)
        user_text = f"hey {name.lower()} i need the {item} by {deadline.lower()} because {reason} and we are behind"
        prompt = ev._agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text=user_text,
            text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
        )
        _add(
            rows,
            split=split,
            task_type="rewrite_professional_general",
            encoder_text=prompt,
            action="respond",
            content="Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]], and we are behind. Thank you.",
            index=idx,
            weight=2.0,
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
            task_type="source_exact_copy_general",
            encoder_text=prompt,
            action="respond",
            content="Source text: [[SOURCE_TEXT]]",
            index=idx,
            weight=4.0,
            negative=_payload("respond", "Source text: vendor invoice INV-20: Deview"),
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
            task_type="summarize_general",
            encoder_text=prompt,
            action="respond",
            content="- Summary: [[SOURCE_TEXT]]",
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
            task_type="extract_general",
            encoder_text=prompt,
            action="respond",
            content="Extracted details from [[SOURCE_TEXT]]",
            index=idx,
            weight=1.5,
        )
        idx += 1

        classify_text, label = rng.choice(labels)
        prompt = ev._agent_prompt(
            name="Classifier Agent",
            instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
            user_text=classify_text,
            text_slots={"SOURCE_TEXT": classify_text},
        )
        _add(
            rows,
            split=split,
            task_type="classify_general",
            encoder_text=prompt,
            action="respond",
            content=label,
            index=idx,
            weight=1.5,
            negative=_payload("respond", "finance" if label != "finance" else "travel"),
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
            task_type="translate_general",
            encoder_text=prompt,
            action="respond",
            content=translated,
            index=idx,
            weight=1.5,
        )
        idx += 1

        plan_goal, steps = rng.choice(plans)
        prompt = ev._agent_prompt(
            name="Planner Agent",
            instruction="Turn the user's goal into a short numbered plan. Keep it practical and do not invent private data.",
            user_text=plan_goal,
            text_slots={"SOURCE_TEXT": plan_goal},
        )
        _add(
            rows,
            split=split,
            task_type="plan_general",
            encoder_text=prompt,
            action="respond",
            content="\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps)),
            index=idx,
            weight=1.5,
        )
        idx += 1

        data_text = rng.choice(source_texts)
        question = "what is saved about this"
        prompt = ev._agent_prompt(
            name="Saved Data Assistant",
            instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
            user_text=question,
            stale_context="Selected paper [P1]: unrelated optimization notes from a previous research turn.",
            text_slots={"SOURCE_TEXT": question, "DATA_CONTEXT": f"[D1] saved note: {data_text}"},
        )
        _add(
            rows,
            split=split,
            task_type="memory_use_general",
            encoder_text=prompt,
            action="respond",
            content="I found this in your saved data: [[DATA_CONTEXT]]",
            index=idx,
            weight=2.0,
            negative=_payload("respond", "Selected paper [P1] focuses on unrelated optimization notes."),
        )
        idx += 1

        missing_query = rng.choice(["what is my passport number", "find my reservation id", "what is my tax pin"])
        prompt = ev._agent_prompt(
            name="Saved Data Assistant",
            instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
            user_text=missing_query,
            text_slots={"SOURCE_TEXT": missing_query},
        )
        _add(
            rows,
            split=split,
            task_type="missing_data_general",
            encoder_text=prompt,
            action="ask_user",
            content="I do not have that in saved data. Which saved data source should I use?",
            index=idx,
            weight=2.0,
            negative=_payload("ask_user", "I do not have that in the provided data. Where should I look for it?"),
        )
        idx += 1

        query = rng.choice(searches)
        prompt = ev._web_agent_prompt(user_text=query)
        _add(
            rows,
            split=split,
            task_type="web_search_general",
            encoder_text=prompt,
            action="extension_request",
            content="Requesting approval to search the web.",
            metadata=_web_metadata(query),
            index=idx,
            weight=2.0,
            negative=_payload("respond", "I cannot browse the web."),
        )
        idx += 1

        casual_text = rng.choice(["Hi how are you?", "How's it going?", "what's up?", "good morning"])
        prompt = ev._agent_prompt(
            name="Casual Assistant",
            instruction="Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it.",
            user_text=casual_text,
            stale_context="Selected paper [P1]: unrelated transformer paper.",
        )
        _add(
            rows,
            split=split,
            task_type="casual_general",
            encoder_text=prompt,
            action="respond",
            content="It's going well. How can I help?",
            index=idx,
            weight=2.0,
            negative=_payload("respond", "The selected paper focuses on transformer optimization."),
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
    parser.add_argument("--train-repeats", type=int, default=300)
    parser.add_argument("--eval-repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_rows = _build_rows("train", int(args.train_repeats), int(args.seed))
    eval_rows = _build_rows("eval", int(args.eval_repeats), int(args.seed))
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
        "objective": "pocketpal_general_agent_curriculum",
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
