#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

TASK_TO_INTENT = {
    "active_agent_action_items": "action_items",
    "active_agent_brainstorm": "brainstorm",
    "active_agent_casual": "casual",
    "active_agent_checklist": "checklist",
    "active_agent_classify": "json",
    "active_agent_extraction": "extraction",
    "active_agent_json": "json",
    "active_agent_missing_text": "ask_user",
    "active_agent_missing_user_data": "ask_user",
    "active_agent_plan": "plan",
    "active_agent_rewrite": "rewrite",
    "active_agent_rewrite_greeting": "rewrite",
    "active_agent_rewrite_slots": "rewrite",
    "active_agent_risks": "risks",
    "active_agent_saved_data": "saved_data",
    "active_agent_source_echo": "source_echo",
    "active_agent_subject": "subject",
    "active_agent_summary": "summary",
    "active_agent_translation": "translation",
    "runtime_plain_chat": "casual",
    "runtime_web_search_request": "web_search",
    "runtime_web_search_result": "summary",
}

NAMES = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Lena", "Nora"]
OBJECTS = ["launch memo", "budget sheet", "client deck", "release notes", "invoice packet", "security review"]
DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next Tuesday"]
BLOCKERS = ["legal approval", "finance review", "design feedback", "vendor confirmation", "manager approval", "QA signoff"]


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


def _intent_for_task(task_type: str, fallback: str = "") -> str:
    if fallback in INTENT_LABELS:
        return fallback
    return TASK_TO_INTENT.get(task_type, fallback if fallback in INTENT_LABELS else "")


def _normalize_row(row: dict[str, Any], *, source_suffix: str, weight: float | None = None) -> dict[str, Any]:
    out = dict(row)
    task_type = str(out.get("task_type") or "")
    intent = _intent_for_task(task_type, str(out.get("intent_label") or ""))
    if intent:
        out["intent_label"] = intent
        out["intent_label_id"] = INTENT_LABELS[intent]
    if weight is not None:
        out["weight"] = float(weight)
    out["example_id"] = f"{out.get('example_id')}_{source_suffix}"
    out["source_id"] = f"{out.get('source_id')}_{source_suffix}"
    out["source_type"] = f"{out.get('source_type', 'unknown')}_{source_suffix}"
    return out


def _row(example_id: str, encoder_text: str, decoder_text: str, task_type: str, *, weight: float, negative: str = "") -> dict[str, Any]:
    intent = _intent_for_task(task_type)
    return {
        "action": json.loads(decoder_text)["action"],
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": example_id,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS[intent],
        "negative_decoder_text": negative or None,
        "negative_loss_weight": 0.9 if negative else None,
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "retrieval_query_text": "",
        "source_id": example_id,
        "source_type": "pocketpal_v199_task_discrimination",
        "split": "train",
        "task_type": task_type,
        "weight": weight,
    }


def _agent_prompt(*, name: str, instruction: str, user_text: str, task: str) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            f"<AK_TASK_HINT> intent={task} task=active_agent_{task}",
            "<AK_CONTEXT> Saved user data: none",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    bad_translation = _payload("respond", "{a reunion se ha cambiado al viernes.", "active_agent_translation")
    bad_rewrite = _payload("respond", "Hi John, could you please send the report by Friday?", "active_agent_rewrite")
    bad_extract = _payload("respond", "- Owner: Jordan\n- Reviewer: Casey\n- Object: budget sheet\n- Date: Tuesday\n- Blocker: finance review", "active_agent_extraction")
    for owner in NAMES:
        for reviewer in NAMES:
            if owner == reviewer:
                continue
            obj = OBJECTS[index % len(OBJECTS)]
            date = DATES[index % len(DATES)]
            blocker = BLOCKERS[index % len(BLOCKERS)]
            text = f"{owner} will send the {obj} by {date}. {reviewer} will review it, and {blocker} is blocking launch."
            rows.append(
                _row(
                    f"v199_extract_{index:05d}",
                    _agent_prompt(
                        name="Extractor",
                        instruction="Extract owner, reviewer, object, date, and blocker as compact bullets. Preserve exact words from the user's text.",
                        user_text=text,
                        task="extraction",
                    ),
                    _payload("respond", f"- Owner: {owner}\n- Reviewer: {reviewer}\n- Object: {obj}\n- Date: {date}\n- Blocker: {blocker}", "active_agent_extraction"),
                    "active_agent_extraction",
                    weight=30.0,
                    negative=bad_translation,
                )
            )
            rows.append(
                _row(
                    f"v199_actions_{index:05d}",
                    _agent_prompt(
                        name="Action Extractor",
                        instruction="Convert the user's text into action items. Preserve names and objects from the user's text.",
                        user_text=text,
                        task="action_items",
                    ),
                    _payload("respond", f"- {owner}: send the {obj} by {date}\n- {reviewer}: review the {obj}", "active_agent_action_items"),
                    "active_agent_action_items",
                    weight=18.0,
                    negative=bad_extract,
                )
            )
            index += 1

    json_specs = [
        ("Please make this more professional.", "{\"intent\":\"rewrite\",\"tone\":\"professional\"}"),
        ("search the web for current TestFlight upload limits", "{\"intent\":\"web_search\",\"freshness\":\"current\"}"),
        ("Can you check if web search is active?", "{\"intent\":\"web_search\",\"freshness\":\"current\"}"),
        ("Schedule a review for next Tuesday.", "{\"intent\":\"schedule\",\"date\":\"next Tuesday\"}"),
    ]
    for repeat in range(32):
        for idx, (text, content) in enumerate(json_specs):
            rows.append(
                _row(
                    f"v199_json_{repeat:03d}_{idx:02d}",
                    _agent_prompt(name="JSON Classifier", instruction="Classify the user request as compact JSON. Return only the JSON object string in content.", user_text=text, task="json"),
                    _payload("respond", content, "active_agent_json"),
                    "active_agent_json",
                    weight=24.0,
                    negative=bad_rewrite,
                )
            )

    translation_specs = [
        ("Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Please send the invoice tomorrow morning.", "Por favor, envia la factura manana por la manana.", "active_agent_translation"),
        ("French Translator", "Translate the user's English text into French. Return only the translation.", "Please review the proposal before Friday.", "Veuillez examiner la proposition avant vendredi.", "active_agent_translation"),
        ("French Translator", "Translate the user's English text into French. Return only the translation.", "Can you call me after lunch?", "Pouvez-vous m'appeler apres le dejeuner?", "active_agent_translation"),
    ]
    for repeat in range(24):
        for idx, (agent, instruction, text, content, task_type) in enumerate(translation_specs):
            rows.append(
                _row(
                    f"v199_translation_{repeat:03d}_{idx:02d}",
                    _agent_prompt(name=agent, instruction=instruction, user_text=text, task="translation"),
                    _payload("respond", content, task_type),
                    task_type,
                    weight=16.0,
                    negative=_payload("respond", "{\"intent\":\"rewrite\",\"tone\":\"professional\"}", "active_agent_json"),
                )
            )
    return rows


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--protocol-manifest", default="tmp/pocketpal_v193_protocol_cleanup/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=199)
    parser.add_argument("--per-task-broad", type=int, default=900)
    parser.add_argument("--retrieval-protect", type=int, default=2200)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    rows = _synthetic_rows()
    eval_rows = rows[: min(800, len(rows))]

    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(Path(broad_manifest["train_dataset_path"])):
        task_type = str(row.get("task_type") or "")
        if task_type in TASK_TO_INTENT or task_type.startswith("broad_agent_") or task_type.startswith("confusion_repair_"):
            normalized_task = task_type
            if task_type.startswith("broad_agent_"):
                normalized_task = "active_agent_" + task_type.removeprefix("broad_agent_")
            if task_type.startswith("confusion_repair_"):
                normalized_task = "active_agent_" + task_type.removeprefix("confusion_repair_")
            out = dict(row)
            out["task_type"] = normalized_task
            buckets[normalized_task].append(out)
    for task_type, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        limit = int(args.per_task_broad)
        for idx, row in enumerate(bucket[:limit]):
            rows.append(_normalize_row(row, source_suffix=f"v199_broad_{idx:04d}", weight=min(max(float(row.get("weight") or 1.0), 4.0), 20.0)))

    for manifest_arg, label, repeat_count, cap_weight in [
        (args.protocol_manifest, "protocol", 1, 26.0),
        (args.rewrite_slot_manifest, "rewrite", 3, 22.0),
    ]:
        manifest = json.loads(Path(manifest_arg).read_text(encoding="utf-8"))
        for repeat in range(repeat_count):
            for path_key, dest in [("train_dataset_path", rows), ("eval_dataset_path", eval_rows)]:
                for row in _iter_jsonl(Path(manifest[path_key])):
                    dest.append(_normalize_row(row, source_suffix=f"v199_{label}_{repeat:02d}", weight=cap_weight))

    retrieval = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
    added = 0
    for row in _iter_jsonl(Path(retrieval["train_dataset_path"])):
        if added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            rows.append(_normalize_row(row, source_suffix="v199_retrieval", weight=0.0))
            added += 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, rows)
    _write(eval_path, eval_rows)
    all_rows = rows + eval_rows
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": INTENT_LABELS,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v199_task_discrimination_drill",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
