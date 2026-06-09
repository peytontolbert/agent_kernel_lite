#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_pocketpal_stage25_diverse_instruction_curriculum import INTENT_LABELS, _row, _stable_split


NAMES = ["Nora", "Iris", "Maya", "Owen", "Leo", "Rina", "Tessa", "Caleb", "Mina", "Eli"]
REVIEWERS = ["Quinn", "Riley", "Sofia", "Mateo", "Noah", "Ava", "Lena", "Theo", "Zara", "Miles"]
OBJECTS = ["vendor form", "release checklist", "travel memo", "invoice draft", "search report", "access list"]
DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow"]
BLOCKERS = ["legal review", "finance approval", "QA signoff", "design feedback", "security approval", "manager approval"]

JSON_REQUESTS = [
    ("Please rewrite this professionally.", '{"intent":"rewrite","tone":"professional"}'),
    ("Summarize this note in one sentence.", '{"intent":"summary","format":"one_sentence"}'),
    ("Extract the owner and blocker.", '{"intent":"extraction","fields":["owner","blocker"]}'),
    ("Make a checklist from this.", '{"intent":"checklist","format":"bullets"}'),
    ("Translate this into French.", '{"intent":"translation","target_language":"french"}'),
    ("Search current Apple developer status.", '{"intent":"web_search","freshness":"current"}'),
    ("Rank these tasks by urgency.", '{"intent":"ranking","criterion":"urgency"}'),
]

TRANSLATIONS = [
    ("Please send the report before lunch.", "Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Por favor, envia el informe antes del almuerzo."),
    ("Please send the report before lunch.", "French Translator", "Translate the user's English text into French. Return only the translation.", "Veuillez envoyer le rapport avant le dejeuner."),
    ("The meeting moved to Thursday morning.", "Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "La reunion se cambio al jueves por la manana."),
    ("The meeting moved to Thursday morning.", "French Translator", "Translate the user's English text into French. Return only the translation.", "La reunion a ete deplacee a jeudi matin."),
    ("Can you call me after the client review?", "Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Puedes llamarme despues de la revision del cliente?"),
    ("Can you call me after the client review?", "French Translator", "Translate the user's English text into French. Return only the translation.", "Pouvez-vous m'appeler apres la revue client?"),
]

PLANS = [
    ("Prepare the weekly launch update.", "1. Gather open launch items.\n2. Confirm owners and deadlines.\n3. Write the update.\n4. Send it to the team."),
    ("Set up local retrieval for my notes.", "1. Choose the note folders.\n2. Remove private files that should be skipped.\n3. Import the folders locally.\n4. Test retrieval with sample questions."),
    ("Verify the active agent before release.", "1. Run rewrite, summary, and extraction checks.\n2. Test web search approval.\n3. Review malformed outputs.\n4. Package only after gates pass."),
    ("Organize a client review tomorrow.", "1. Confirm the attendee list.\n2. Prepare the agenda.\n3. Collect supporting notes.\n4. Send the reminder."),
]

BRAINSTORMS = [
    ("Ideas for making a private assistant more useful.", "1. Add per-agent memory collections\n2. Let users choose tool permissions\n3. Show clear source links for retrieved answers"),
    ("Ways to make search results easier to trust.", "1. Show source cards with titles and links\n2. Include dates and snippets\n3. Let users open each result"),
    ("Ideas for improving a rewrite agent.", "1. Preserve source facts exactly\n2. Offer tone presets\n3. Show the rewritten text without extra commentary"),
    ("Ways to organize local user data.", "1. Create topic folders\n2. Add tags for projects and people\n3. Let users exclude sensitive files"),
]


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _control_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for index in range(180):
            owner = NAMES[(index + repeat) % len(NAMES)]
            reviewer = REVIEWERS[(index * 2 + repeat) % len(REVIEWERS)]
            item = OBJECTS[(index * 3 + repeat) % len(OBJECTS)]
            date = DATES[(index + repeat * 2) % len(DATES)]
            blocker = BLOCKERS[(index * 5 + repeat) % len(BLOCKERS)]
            stale = (
                f"Selected paper [P1]: stale note says Blake owns the launch memo by Monday, "
                f"but this is unrelated to the current user text."
            )
            user_text = f"{owner} will send the {item} by {date}. {reviewer} will review it, and launch is blocked by {blocker}."
            common = {
                "user_text": user_text,
                "saved_data": "none",
                "stale_context": stale,
            }
            rows.extend(
                [
                    _row(
                        _id("stage50_summary", str(repeat), str(index)),
                        name="Summary Agent",
                        instruction="Summarize the current source text in one sentence. Ignore stale context.",
                        intent="summary",
                        content=f"{owner} will send the {item} by {date}, {reviewer} will review it, and {blocker} is blocking launch.",
                        weight=18.0,
                        **common,
                    ),
                    _row(
                        _id("stage50_extract", str(repeat), str(index)),
                        name="Exact Extractor",
                        instruction="Extract owner, reviewer, object, date, and blocker from the current source text only.",
                        intent="extraction",
                        content=f"- Owner: {owner}\n- Reviewer: {reviewer}\n- Object: {item}\n- Date: {date}\n- Blocker: {blocker}",
                        weight=22.0,
                        **common,
                    ),
                    _row(
                        _id("stage50_rewrite", str(repeat), str(index)),
                        name="Professional Rewriter",
                        instruction="Rewrite the current source text as a polished workplace update. Do not extract bullets.",
                        intent="rewrite",
                        content=f"{owner} will send the {item} by {date}. {reviewer} will review it, and launch is currently blocked by {blocker}.",
                        weight=20.0,
                        **common,
                    ),
                    _row(
                        _id("stage50_checklist", str(repeat), str(index)),
                        name="Checklist Agent",
                        instruction="Turn the current source text into a checklist. Preserve the current object and blocker.",
                        intent="checklist",
                        content=f"- Send the {item} by {date}\n- Review the {item}\n- Resolve {blocker}",
                        weight=18.0,
                        **common,
                    ),
                    _row(
                        _id("stage50_actions", str(repeat), str(index)),
                        name="Action Item Agent",
                        instruction="Extract concrete action items with owners from the current source text only.",
                        intent="action_items",
                        content=f"- {owner}: send the {item} by {date}\n- {reviewer}: review the {item}",
                        weight=18.0,
                        **common,
                    ),
                    _row(
                        _id("stage50_risks", str(repeat), str(index)),
                        name="Risk Agent",
                        instruction="List launch risks from the current source text. Do not copy stale paper context.",
                        intent="risks",
                        content=f"- {item.title()} may miss the {date} deadline\n- {reviewer}'s review could delay launch\n- {blocker.title()} is still unresolved",
                        weight=16.0,
                        **common,
                    ),
                ]
            )
    return rows


def _weak_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for index, (user_text, content) in enumerate(JSON_REQUESTS):
            rows.append(
                _row(
                    _id("stage50_json", str(repeat), str(index)),
                    name="JSON Classifier",
                    instruction="Return only compact JSON for the user's intent. Do not translate or rewrite.",
                    user_text=user_text,
                    intent="json",
                    content=content,
                    stale_context="Selected paper [P1]: unrelated research paper context.",
                    weight=24.0,
                )
            )
        for index, (user_text, name, instruction, content) in enumerate(TRANSLATIONS):
            rows.append(
                _row(
                    _id("stage50_translate", str(repeat), str(index), name),
                    name=name,
                    instruction=instruction,
                    user_text=user_text,
                    intent="translation",
                    content=content,
                    stale_context="Selected paper [P1]: stale Spanish sentence La reunion se ha cambiado al viernes.",
                    weight=24.0,
                )
            )
        for index, (user_text, content) in enumerate(PLANS):
            rows.append(
                _row(
                    _id("stage50_plan", str(repeat), str(index)),
                    name="Planner",
                    instruction="Create a concise practical plan for the current user goal. Do not use stale context.",
                    user_text=user_text,
                    intent="plan",
                    content=content,
                    weight=22.0,
                )
            )
        for index, (user_text, content) in enumerate(BRAINSTORMS):
            rows.append(
                _row(
                    _id("stage50_brainstorm", str(repeat), str(index)),
                    name="Brainstorm Agent",
                    instruction="Generate three concise ideas that fit the current request. Do not answer as a chat assistant.",
                    user_text=user_text,
                    intent="brainstorm",
                    content=content,
                    weight=22.0,
                )
            )
    return rows


def build_rows(repeats: int) -> list[dict[str, Any]]:
    return [*_control_rows(repeats), *_weak_rows(repeats)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage50_source_binding_curriculum_v1")
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()

    rows = build_rows(args.repeats)
    train_rows = [row for row in rows if not _stable_split(str(row["source_id"]))]
    eval_rows = [row for row in rows if _stable_split(str(row["source_id"]))]
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest: dict[str, Any] = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": INTENT_LABELS,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_stage50_source_binding_curriculum",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in rows).items())),
        "total_examples": len(rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
