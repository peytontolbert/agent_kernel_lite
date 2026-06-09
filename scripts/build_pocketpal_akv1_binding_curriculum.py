#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from pocketpal_structured_decode import json_to_structured_tokens


PEOPLE = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Priya", "Maria"]
OBJECTS = ["launch memo", "budget sheet", "client deck", "invoice packet", "security review", "release notes"]
DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow morning"]
BLOCKERS = ["legal approval", "finance review", "design feedback", "vendor confirmation", "QA signoff", "Apple processing"]

TRANSLATIONS = [
    ("Please send the invoice tomorrow morning.", "Spanish", "Por favor envia la factura manana por la manana."),
    ("Can you call me after lunch?", "French", "Pouvez-vous m'appeler apres le dejeuner?"),
    ("Please review the proposal before Friday.", "French", "Veuillez examiner la proposition avant vendredi."),
    ("The meeting moved to Friday afternoon.", "Spanish", "La reunion se movio al viernes por la tarde."),
]

JSON_INTENTS = [
    ("Classify this as a professional rewrite request.", {"intent": "rewrite", "tone": "professional"}),
    ("Classify this as an extraction request for owner and deadline.", {"intent": "extraction", "fields": ["owner", "deadline"]}),
    ("Classify this as a Spanish translation request.", {"intent": "translation", "target_language": "spanish"}),
    ("Classify this as a summary request.", {"intent": "summary", "format": "concise"}),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _agent_prompt(*, name: str, instruction: str, task_hint: str, user_text: str, source_text: str = "") -> str:
    source = source_text or user_text
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
            f"<AK_TASK_HINT> {task_hint}",
            "<AK_CONTEXT> Saved user data: none",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={source}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _payload(content: str, task_type: str) -> str:
    return json.dumps(
        {"action": "respond", "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _row(
    *,
    prompt: str,
    content: str,
    task_type: str,
    intent_label: str,
    intent_labels: dict[str, int],
    source_type: str,
    weight: float,
    suffix: str,
) -> dict[str, Any]:
    json_text = _payload(content, task_type)
    return {
        "action": "respond",
        "answer_confidence_target": None,
        "decoder_loss_weight": float(weight),
        "decoder_text": json_to_structured_tokens(json_text),
        "encoder_text": prompt,
        "example_id": _stable_id(source_type, task_type, prompt, content, suffix),
        "intent_label": intent_label,
        "intent_label_id": int(intent_labels.get(intent_label, 0)),
        "json_decoder_text": json_text,
        "needs_verification_target": None,
        "negative_decoder_text": None,
        "negative_loss_weight": None,
        "ood_evidence_target": None,
        "ood_query_target": None,
        "paper_action_validity_target": None,
        "query_confidence_target": None,
        "retrieval_coverage_target": None,
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "retrieval_query_text": "",
        "source_id": _stable_id("source", source_type, task_type, prompt, content),
        "source_type": source_type,
        "task_type": task_type,
    }


def _binding_rows(intent_labels: dict[str, int], *, repeats: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for repeat in range(repeats):
        for owner in PEOPLE:
            reviewer = PEOPLE[(PEOPLE.index(owner) + repeat + 3) % len(PEOPLE)]
            obj = OBJECTS[(PEOPLE.index(owner) + repeat) % len(OBJECTS)]
            date = DATES[(PEOPLE.index(owner) + repeat * 2) % len(DATES)]
            blocker = BLOCKERS[(PEOPLE.index(owner) + repeat * 3) % len(BLOCKERS)]
            source = f"{owner} will send the {obj} by {date}. {reviewer} will review it, and launch is currently blocked by {blocker}."
            variants = [
                (
                    "Action Item Agent",
                    "Extract concise action items from the user's provided text. Preserve names, objects, and dates exactly.",
                    "intent=action_items task=active_agent_action_items source_text_required=true",
                    "active_agent_action_items",
                    "action_items",
                    f"- {owner}: send the {obj} by {date}\n- {reviewer}: review the {obj}",
                ),
                (
                    "Checklist Agent",
                    "Convert the user's provided text into a concise checklist. Preserve objects, dates, and blockers exactly.",
                    "intent=checklist task=active_agent_checklist source_text_required=true",
                    "active_agent_checklist",
                    "checklist",
                    f"- Send the {obj} by {date}\n- Review the {obj}\n- Resolve {blocker}",
                ),
                (
                    "Risk Agent",
                    "List risks from the user's provided text. Preserve facts and do not invent unrelated tasks.",
                    "intent=risks task=active_agent_risks source_text_required=true",
                    "active_agent_risks",
                    "risks",
                    f"- {obj.title()} may miss the {date} deadline\n- {reviewer}'s review could delay launch\n- {blocker.title()} is still unresolved",
                ),
                (
                    "Subject Agent",
                    "Write a concise subject line from the user's provided text.",
                    "intent=subject task=active_agent_subject source_text_required=true",
                    "active_agent_subject",
                    "subject",
                    f"{obj.title()} Review and Launch Blocker",
                ),
                (
                    "Summary Agent",
                    "Summarize the user's provided text in one sentence. Preserve names, objects, dates, and blockers exactly.",
                    "intent=summary task=active_agent_summary source_text_required=true",
                    "active_agent_summary",
                    "summary",
                    source,
                ),
                (
                    "Professional Rewriter",
                    "Rewrite the user's provided text professionally. Preserve names, objects, dates, and blockers exactly.",
                    "intent=rewrite task=active_agent_rewrite source_text_required=true",
                    "active_agent_rewrite",
                    "rewrite",
                    source,
                ),
            ]
            for name, instruction, hint, task_type, intent, content in variants:
                rows.append(
                    _row(
                        prompt=_agent_prompt(name=name, instruction=instruction, task_hint=hint, user_text=source),
                        content=content,
                        task_type=task_type,
                        intent_label=intent,
                        intent_labels=intent_labels,
                        source_type="akv1_binding_curriculum",
                        weight=weight,
                        suffix=f"{index:06d}",
                    )
                )
                index += 1
    return rows


def _list_rows(intent_labels: dict[str, int], *, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lists = [
        ("pack laptop, charger, badge, and printed agenda", "- Pack laptop\n- Pack charger\n- Pack badge\n- Pack printed agenda"),
        ("submit payroll, reply to the client, and rename a folder", "1. Submit payroll\n2. Reply to client\n3. Rename a folder"),
        ("choose folders to index, remove private files, run the local import, and test retrieval", "1. Choose the folders to index.\n2. Remove files that should stay private.\n3. Run the local import.\n4. Test retrieval with a few queries."),
    ]
    for index, (source, content) in enumerate(lists):
        task_type = "active_agent_checklist" if content.startswith("-") else "active_agent_plan"
        intent = "checklist" if task_type.endswith("checklist") else "plan"
        rows.append(
            _row(
                prompt=_agent_prompt(
                    name="List Agent",
                    instruction="Convert the user's provided text into a clean ordered or bulleted list. Preserve every item.",
                    task_hint=f"intent={intent} task={task_type} source_text_required=true",
                    user_text=source,
                ),
                content=content,
                task_type=task_type,
                intent_label=intent,
                intent_labels=intent_labels,
                source_type="akv1_list_binding_curriculum",
                weight=weight,
                suffix=f"list_{index}",
            )
        )
    return rows


def _json_translation_rows(intent_labels: dict[str, int], *, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (instruction, payload) in enumerate(JSON_INTENTS):
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        rows.append(
            _row(
                prompt=_agent_prompt(
                    name="Intent JSON Agent",
                    instruction="Return only the compact JSON payload requested by the user. Preserve intent labels and field names exactly.",
                    task_hint="intent=json task=active_agent_json source_text_required=true",
                    user_text=instruction,
                ),
                content=content,
                task_type="active_agent_json",
                intent_label="json",
                intent_labels=intent_labels,
                source_type="akv1_json_intent_curriculum",
                weight=weight,
                suffix=f"json_{index}",
            )
        )
    for index, (source, language, translated) in enumerate(TRANSLATIONS):
        rows.append(
            _row(
                prompt=_agent_prompt(
                    name=f"{language} Translation Agent",
                    instruction=f"Translate the user's provided text into {language}. Return only the translation.",
                    task_hint="intent=translation task=active_agent_translation source_text_required=true",
                    user_text=source,
                ),
                content=translated,
                task_type="active_agent_translation",
                intent_label="translation",
                intent_labels=intent_labels,
                source_type="akv1_translation_curriculum",
                weight=weight,
                suffix=f"translation_{index}",
            )
        )
    return rows


def _copy_anchor(row: dict[str, Any], suffix: str, weight: float) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = _stable_id("anchor", str(row.get("example_id")), suffix)
    out["source_type"] = "akv1_binding_anchor_replay"
    out["decoder_loss_weight"] = float(weight)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    base_manifest_path = Path(args.base_manifest).expanduser().resolve()
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_train = _read_jsonl(Path(base_manifest["train_dataset_path"]))
    base_eval = _read_jsonl(Path(base_manifest["eval_dataset_path"]))
    intent_labels = {str(k): int(v) for k, v in dict(base_manifest.get("intent_labels", {})).items()}

    rows: list[dict[str, Any]] = []
    rows.extend(_binding_rows(intent_labels, repeats=int(args.binding_repeats), weight=float(args.binding_weight)))
    rows.extend(_list_rows(intent_labels, weight=float(args.binding_weight)))
    rows.extend(_json_translation_rows(intent_labels, weight=float(args.json_translation_weight)))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_train:
        buckets[str(row.get("task_type") or "")].append(row)
    anchors: list[dict[str, Any]] = []
    per_task = max(1, int(args.anchor_rows) // max(1, len(buckets)))
    for task_type in sorted(buckets):
        bucket = list(buckets[task_type])
        rng.shuffle(bucket)
        anchors.extend(bucket[:per_task])
    rng.shuffle(anchors)
    rows.extend(_copy_anchor(row, f"{index:05d}", float(args.anchor_weight)) for index, row in enumerate(anchors[: int(args.anchor_rows)]))
    rng.shuffle(rows)

    eval_rows = rows[: min(256, len(rows))]
    train_rows = rows

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    source_counts = Counter(str(row.get("source_type") or "unknown") for row in train_rows + eval_rows)
    task_type_counts = Counter(str(row.get("task_type") or "unknown") for row in train_rows + eval_rows)
    target_action_counts = Counter(str(row.get("action") or "unknown") for row in train_rows + eval_rows)
    manifest = dict(base_manifest)
    manifest.update(
        {
            "artifact_kind": "agentkernel_lite_encdec_akv1_binding_curriculum",
            "objective": "pocketpal_akv1_binding_curriculum",
            "source_manifest_path": str(base_manifest_path),
            "manifest_path": str((output_dir / "agentkernel_lite_encdec_dataset_manifest.json").resolve()),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
            "total_examples": len(train_rows) + len(eval_rows),
            "source_counts": dict(sorted(source_counts.items())),
            "task_type_counts": dict(sorted(task_type_counts.items())),
            "target_action_counts": dict(sorted(target_action_counts.items())),
            "weighting_policy": {
                "binding_repeats": int(args.binding_repeats),
                "binding_weight": float(args.binding_weight),
                "json_translation_weight": float(args.json_translation_weight),
                "anchor_rows": int(args.anchor_rows),
                "anchor_weight": float(args.anchor_weight),
            },
        }
    )
    (output_dir / "agentkernel_lite_encdec_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="tmp/pocketpal_stage67_structured_copy_decoder_v172d_akv1/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage73_akv1_binding_curriculum")
    parser.add_argument("--binding-repeats", type=int, default=12)
    parser.add_argument("--binding-weight", type=float, default=1.8)
    parser.add_argument("--json-translation-weight", type=float, default=1.6)
    parser.add_argument("--anchor-rows", type=int, default=9000)
    parser.add_argument("--anchor-weight", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
