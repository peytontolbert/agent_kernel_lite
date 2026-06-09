#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


AK_CHAT = "<AK_CHAT>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_PROFILE = "<AK_PROFILE>"
AK_RESPOND = "<AK_RESPOND>"
AK_SLOT = "<AK_SLOT>"
AK_SLOT_NAME = "<AK_SLOT_NAME>"
AK_SLOT_VALUE = "<AK_SLOT_VALUE>"
AK_TASK_HINT = "<AK_TASK_HINT>"
AK_USER = "<AK_USER>"

NAMES = [
    "Ari",
    "Blair",
    "Cam",
    "Drew",
    "Eden",
    "Fern",
    "Gray",
    "Hana",
    "Iris",
    "Jules",
    "Kai",
    "Lena",
    "Mika",
    "Nora",
    "Owen",
    "Priya",
    "Quinn",
    "Riley",
    "Sage",
    "Tara",
    "Uma",
    "Vera",
    "Wren",
    "Yara",
]
OBJECTS = [
    "budget sheet",
    "client deck",
    "invoice packet",
    "launch memo",
    "release notes",
    "security review",
    "search flow",
    "test plan",
    "vendor checklist",
    "roadmap update",
    "support report",
    "privacy brief",
]
DATES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "tomorrow",
    "next Tuesday",
    "May 18",
    "June 3",
    "2 PM",
]
BLOCKERS = [
    "design feedback",
    "finance review",
    "legal approval",
    "manager approval",
    "QA signoff",
    "vendor confirmation",
    "Apple processing",
    "missing screenshots",
    "security approval",
    "data export",
]
PLACES = ["Austin", "Boston", "Chicago", "Denver", "Miami", "Seattle", "Toronto"]
AMOUNTS = ["$85", "$120", "$420", "$1,200", "$2,400", "17 units", "42 files"]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _decision(content: str, task_type: str) -> str:
    return json.dumps(
        {"action": "respond", "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        sort_keys=True,
    )


def _example(
    *,
    source_id: str,
    task_type: str,
    intent_label: str,
    agent_name: str,
    agent_instruction: str,
    source_text: str,
    user_text: str,
    expected: str,
    weight: float,
    negative: str,
) -> dict[str, Any]:
    encoder_text = (
        f"{AK_CHAT} {AK_RESPOND} PocketPal counterfactual source-binding example.\n"
        "<AK_AGENT_ACTIVE>\n"
        f"Agent name: {agent_name}\n"
        f"Agent instruction: {agent_instruction}\n"
        "Retrieval policy: current_context_only\n"
        "Tool policy: no_tools\n"
        "Action policy: respond_or_ask\n"
        "The active agent instruction is the primary task contract for this turn.\n"
        "</AK_AGENT_ACTIVE>\n"
        f"{AK_TASK_HINT} intent={intent_label} task={task_type} source_text_required=true\n"
        f"{AK_CONTEXT} Stale selected paper context: unrelated research context with Blake, launch memo, and finance review.\n"
        f"{AK_PROFILE} User text slots:\n"
        f"{AK_SLOT} {AK_SLOT_NAME}=SOURCE_TEXT {AK_SLOT_VALUE}={source_text}\n"
        "Use only SOURCE_TEXT for this task. Do not copy stale names, objects, dates, or blockers.\n"
        f"{AK_USER} {user_text}\n"
        "Return compact JSON with action=respond and content that follows the active agent instruction."
    )
    decoder_text = _decision(expected, task_type)
    example_id = hashlib.sha256(f"{source_id}\n{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
    return {
        "example_id": example_id,
        "source_type": "pocketpal_stage52_counterfactual_slot_transform_curriculum",
        "source_id": source_id,
        "task_type": task_type,
        "encoder_text": encoder_text,
        "decoder_text": decoder_text,
        "action": "respond",
        "source_action": "respond",
        "intent_label": intent_label,
        "retrieval_query_text": "",
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "negative_decoder_text": negative,
        "negative_loss_weight": 1.0,
        "weight": float(weight),
    }


def _make_rows(count: int, *, seed: int, weight: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    stale_negative = _decision("- Owner: Blake\n- Object: launch memo\n- Date: Monday\n- Blocker: finance review", "stale_context_copy")
    for index in range(count):
        owner, reviewer = rng.sample(NAMES, 2)
        obj = rng.choice(OBJECTS)
        date = rng.choice(DATES)
        blocker = rng.choice(BLOCKERS)
        source = f"{owner} will send the {obj} by {date}. {reviewer} will review it, and {blocker} is blocking launch."
        variants = [
            (
                "active_agent_extraction",
                "extract",
                "Extractor",
                "Extract owner, reviewer, object, date, and blocker from SOURCE_TEXT. Keep labels exact.",
                "extract the fields",
                f"- Owner: {owner}\n- Reviewer: {reviewer}\n- Object: {obj}\n- Date: {date}\n- Blocker: {blocker}",
            ),
            (
                "active_agent_action_items",
                "action_items",
                "Action Item Maker",
                "Turn SOURCE_TEXT into concrete action items with the named people.",
                "make action items",
                f"- {owner}: send the {obj} by {date}\n- {reviewer}: review the {obj}",
            ),
            (
                "active_agent_summary",
                "summary",
                "Summarizer",
                "Summarize SOURCE_TEXT in one sentence without adding facts.",
                "summarize this",
                f"{owner} will send the {obj} by {date}, {reviewer} will review it, and {blocker} is blocking launch.",
            ),
            (
                "active_agent_risks",
                "risks",
                "Risk Finder",
                "List launch risks from SOURCE_TEXT only.",
                "list risks",
                f"- {obj.title()} may miss the {date} deadline\n- {reviewer}'s review could delay launch\n- {blocker.title()} is still unresolved",
            ),
            (
                "active_agent_subject",
                "subject",
                "Subject Writer",
                "Write a short subject line from SOURCE_TEXT.",
                "write a subject",
                f"{obj.title()} Review and Launch Blocker",
            ),
            (
                "active_agent_json",
                "json",
                "JSON Classifier",
                "Return only a compact JSON object describing intent and key fields from SOURCE_TEXT.",
                "return json",
                json.dumps({"intent": "project_update", "owner": owner, "object": obj, "deadline": date}, separators=(",", ":")),
            ),
            (
                "active_agent_plan",
                "plan",
                "Planner",
                "Make a short plan from SOURCE_TEXT. Use only the stated facts.",
                "make a plan",
                f"1. Ask {owner} for the {obj} by {date}.\n2. Have {reviewer} review the {obj}.\n3. Resolve {blocker} before launch.",
            ),
        ]
        for task_type, intent, agent_name, instruction, user_text, expected in variants:
            rows.append(
                _example(
                    source_id=f"stage52_task_{index:05d}_{task_type}",
                    task_type=task_type,
                    intent_label=intent,
                    agent_name=agent_name,
                    agent_instruction=instruction,
                    source_text=source,
                    user_text=user_text,
                    expected=expected,
                    weight=weight,
                    negative=stale_negative,
                )
            )
        place = rng.choice(PLACES)
        amount = rng.choice(AMOUNTS)
        fact_source = f"{owner} has invoice {amount} for {place} on {date}."
        rows.append(
            _example(
                source_id=f"stage52_invoice_{index:05d}",
                task_type="active_agent_extraction",
                intent_label="extract",
                agent_name="Field Extractor",
                agent_instruction="Extract name, place, date, and amount from SOURCE_TEXT.",
                source_text=fact_source,
                user_text="extract fields",
                expected=f"- Name: {owner}\n- Place: {place}\n- Date: {date}\n- Amount: {amount}",
                weight=weight,
                negative=stale_negative,
            )
        )
        phrase = rng.choice(
            [
                ("Please review the proposal before Friday.", "Veuillez examiner la proposition avant vendredi."),
                ("Could you call me after lunch?", "Pouvez-vous m'appeler apres le dejeuner?"),
                ("The meeting moved to Friday.", "La reunion a ete deplacee a vendredi."),
            ]
        )
        rows.append(
            _example(
                source_id=f"stage52_translate_{index:05d}",
                task_type="active_agent_translation",
                intent_label="translation",
                agent_name="Translator",
                agent_instruction="Translate SOURCE_TEXT into French. Return only the translation.",
                source_text=phrase[0],
                user_text="translate to french",
                expected=phrase[1],
                weight=weight,
                negative=_decision("La reunion se ha cambiado al viernes.", "wrong_language_memory"),
            )
        )
    return rows


def _hash_split(example_id: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="tmp/pocketpal_stage51_local_agentkernel_trace_curriculum/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage52_counterfactual_slot_transform_curriculum")
    parser.add_argument("--examples", type=int, default=3500)
    parser.add_argument("--weight", type=float, default=18.0)
    parser.add_argument("--eval-fraction", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    base_manifest = json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    all_rows = list(_iter_jsonl(Path(base_manifest["train_dataset_path"]))) + list(
        _iter_jsonl(Path(base_manifest["eval_dataset_path"]))
    )
    local_rows = _make_rows(int(args.examples), seed=int(args.seed), weight=float(args.weight))
    dedup: dict[str, dict[str, Any]] = {}
    for row in [*all_rows, *local_rows]:
        key = str(row.get("example_id") or "")
        if not key:
            key = hashlib.sha256(f"{row.get('encoder_text','')}-->{row.get('decoder_text','')}".encode("utf-8")).hexdigest()
            row["example_id"] = key
        dedup[key] = row
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in sorted(dedup.values(), key=lambda item: str(item.get("example_id", ""))):
        if row.get("source_type") == "pocketpal_stage52_counterfactual_slot_transform_curriculum":
            row["split"] = _hash_split(str(row["example_id"]), float(args.eval_fraction))
        split = str(row.get("split") or "train")
        if split == "eval":
            eval_rows.append(row)
        else:
            train_rows.append(row)
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    task_counts: dict[str, int] = {}
    for row in [*train_rows, *eval_rows]:
        task = str(row.get("task_type") or "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage52_counterfactual_slot_transform_curriculum",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "local_examples": len(local_rows),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
