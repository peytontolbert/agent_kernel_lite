#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


NAMES = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Lena", "Nora"]
OBJECTS = ["launch memo", "budget sheet", "client deck", "release notes", "invoice packet", "security review"]
DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next Tuesday"]
BLOCKERS = ["legal approval", "finance review", "design feedback", "vendor confirmation", "manager approval", "QA signoff"]


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


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
            f"<AK_TASK_HINT> intent={task} task=active_agent_{task} source_text_required=true",
            "<AK_CONTEXT> Saved user data: none",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _row(example_id: str, encoder_text: str, decoder_text: str, task_type: str, *, weight: float, negative: str = "") -> dict[str, Any]:
    return {
        "action": json.loads(decoder_text)["action"],
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": example_id,
        "intent_label": task_type.replace("active_agent_", ""),
        "intent_label_id": 0,
        "negative_decoder_text": negative or None,
        "negative_loss_weight": 0.8 if negative else None,
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "retrieval_query_text": "",
        "source_id": example_id,
        "source_type": "pocketpal_v198_structured_exactness",
        "split": "train",
        "task_type": task_type,
        "weight": weight,
    }


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


def _extraction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for owner in NAMES:
        for reviewer in NAMES:
            if owner == reviewer:
                continue
            for obj in OBJECTS:
                date = DATES[index % len(DATES)]
                blocker = BLOCKERS[index % len(BLOCKERS)]
                text = f"{owner} will send the {obj} by {date}. {reviewer} will review it, and {blocker} is blocking launch."
                content = f"- Owner: {owner}\n- Reviewer: {reviewer}\n- Object: {obj}\n- Date: {date}\n- Blocker: {blocker}"
                prompt = _agent_prompt(
                    name="Extractor",
                    instruction="Extract owner, reviewer, object, date, and blocker as compact bullets. Preserve exact names and wording from the user's text.",
                    user_text=text,
                    task="extraction",
                )
                negative = _payload("respond", "- Owner: Jordan\n- Reviewer: Casey\n- Object: budget sheet\n- Date: Tuesday\n- Blocker: finance review", "active_agent_extraction")
                rows.append(_row(f"v198_extract_{index:05d}", prompt, _payload("respond", content, "active_agent_extraction"), "active_agent_extraction", weight=72.0, negative=negative))
                index += 1
    simple = [
        ("Hi John, please send the invoice for $1,200 by Friday.", "- Name: John\n- Object: invoice\n- Amount: $1,200\n- Date: Friday"),
        ("Question: Can you check if web search is active?", "Question: Can you check if web search is active?"),
    ]
    for offset, (text, content) in enumerate(simple):
        prompt = _agent_prompt(name="Extractor", instruction="Extract names, dates, amounts, questions, and objects as compact bullets.", user_text=text, task="extraction")
        rows.append(_row(f"v198_extract_simple_{offset:02d}", prompt, _payload("respond", content, "active_agent_extraction"), "active_agent_extraction", weight=90.0))
    return rows


def _json_rows() -> list[dict[str, Any]]:
    specs = [
        ("Please make this more professional.", "{\"intent\":\"rewrite\",\"tone\":\"professional\"}"),
        ("search the web for current TestFlight upload limits", "{\"intent\":\"web_search\",\"freshness\":\"current\"}"),
        ("Can you check if web search is active?", "{\"intent\":\"web_search\",\"freshness\":\"current\"}"),
        ("Schedule a review for next Tuesday.", "{\"intent\":\"schedule\",\"date\":\"next Tuesday\"}"),
        ("Approve invoice INV-2048 for $1,200.", "{\"intent\":\"finance\",\"object\":\"invoice\",\"amount\":\"$1,200\"}"),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(80):
        for index, (text, content) in enumerate(specs):
            prompt = _agent_prompt(name="JSON Classifier", instruction="Classify the user request as compact JSON. Return only the JSON object string in content.", user_text=text, task="json")
            negative = _payload("respond", "{reeting", "active_agent_classify")
            rows.append(_row(f"v198_json_{repeat:03d}_{index:02d}", prompt, _payload("respond", content, "active_agent_json"), "active_agent_json", weight=84.0, negative=negative))
    return rows


def _translation_rows() -> list[dict[str, Any]]:
    specs = [
        ("Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Please send the invoice tomorrow morning.", "Por favor, envia la factura manana por la manana."),
        ("French Translator", "Translate the user's English text into French. Return only the translation.", "Please review the proposal before Friday.", "Veuillez examiner la proposition avant vendredi."),
        ("French Translator", "Translate the user's English text into French. Return only the translation.", "Can you call me after lunch?", "Pouvez-vous m'appeler apres le dejeuner?"),
        ("Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Can you call me after lunch?", "Puedes llamarme despues del almuerzo?"),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(80):
        for index, (agent, instruction, text, content) in enumerate(specs):
            prompt = _agent_prompt(name=agent, instruction=instruction, user_text=text, task="translation")
            wrong = "Puedes llamarme despues del almuerzo?" if "French" in agent else "Pouvez-vous m'appeler apres le dejeuner?"
            rows.append(_row(f"v198_translation_{repeat:03d}_{index:02d}", prompt, _payload("respond", content, "active_agent_translation"), "active_agent_translation", weight=84.0, negative=_payload("respond", wrong, "active_agent_translation")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", default="tmp/pocketpal_v193_protocol_cleanup/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieval-protect", type=int, default=2500)
    args = parser.parse_args()

    rows = _extraction_rows() + _json_rows() + _translation_rows()
    eval_rows = rows[: min(1000, len(rows))]

    for manifest_path, label, repeat_count in [
        (Path(args.protocol_manifest), "protocol", 1),
        (Path(args.rewrite_slot_manifest), "rewrite", 4),
    ]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for repeat in range(repeat_count):
            for path_key, dest in [("train_dataset_path", rows), ("eval_dataset_path", eval_rows)]:
                for row in _iter_jsonl(Path(manifest[path_key])):
                    out = dict(row)
                    out["example_id"] = f"{out.get('example_id')}_v198_{label}_{repeat:02d}"
                    out["source_type"] = f"{out.get('source_type', 'unknown')}_v198_{label}_protect"
                    out["weight"] = min(max(float(out.get("weight") or 1.0), 18.0), 48.0)
                    dest.append(out)

    retrieval = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
    added = 0
    for row in _iter_jsonl(Path(retrieval["train_dataset_path"])):
        if added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            out = dict(row)
            out["example_id"] = f"{out.get('example_id')}_v198_retrieval"
            out["source_type"] = "v182_retrieval_protection_v198"
            out["weight"] = 0.0
            rows.append(out)
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
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v198_structured_exactness_drill",
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
