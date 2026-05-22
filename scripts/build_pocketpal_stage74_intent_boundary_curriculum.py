#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _stable_id(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _encoder_text(
    *,
    agent_name: str,
    agent_instruction: str,
    intent: str,
    task: str,
    source_text: str,
    user_text: str | None = None,
) -> str:
    shown_user = user_text or source_text
    return (
        "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.\n"
        "<AK_AGENT_ACTIVE>\n"
        f"Agent name: {agent_name}\n"
        f"Agent instruction: {agent_instruction}\n"
        "Retrieval policy: auto\n"
        "Tool policy: ask_before_extensions\n"
        "Action policy: respond_or_ask\n"
        "The active agent instruction is the primary task contract for this turn.\n"
        "</AK_AGENT_ACTIVE>\n"
        f"<AK_TASK_HINT> intent={intent} task={task} source_text_required=true\n"
        "<AK_CONTEXT> Saved user data: none\n"
        "<AK_PROFILE> User text slots:\n"
        f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={source_text}\n"
        "Available placeholders for this turn: [[SOURCE_TEXT]].\n"
        "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.\n"
        "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.\n"
        "Use stale paper context only when the current user request asks about that paper or research evidence.\n"
        f"<AK_USER> {shown_user}\n"
        "Return compact JSON with the correct action and content for the active agent."
    )


def _base_row(
    *,
    labels: dict[str, int],
    intent: str,
    task_type: str,
    encoder_text: str,
    decoder_text: str,
    json_decoder_text: str,
    source_id: str,
    weight: float,
) -> dict[str, Any]:
    return {
        "action": "respond",
        "answer_confidence_target": None,
        "decoder_loss_weight": 0.0,
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": _stable_id(source_id + "\n" + encoder_text + "\n" + decoder_text),
        "intent_label": intent,
        "intent_label_id": int(labels[intent]),
        "json_decoder_text": json_decoder_text,
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
        "source_id": source_id,
        "source_type": "stage74_intent_boundary_minimal_pair",
        "split": "train",
        "task_type": task_type,
        "weight": float(weight),
    }


def _json_boundary_rows(labels: dict[str, int], repeats: int) -> list[dict[str, Any]]:
    prompts = [
        ("Translate this into Spanish.", "spanish", "La reunion se ha cambiado al viernes."),
        ("Please translate this into French.", "french", "La réunion a été déplacée à vendredi."),
        ("Translate the note to German: The report is ready.", "german", "Der Bericht ist fertig."),
        ("Turn this into Spanish: The build is still processing.", "spanish", "La compilacion aun se esta procesando."),
        ("Translate this sentence to Italian: The invoice is approved.", "italian", "La fattura e approvata."),
        ("Make this English sentence Spanish: The keys are in the box.", "spanish", "Las llaves estan en la caja."),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for idx, (source_text, language, translated) in enumerate(prompts):
            json_encoder = _encoder_text(
                agent_name="JSON Classifier",
                agent_instruction="Return only a compact JSON object describing the user's intent. Do not perform the requested transformation.",
                intent="json",
                task="active_agent_json",
                source_text=source_text,
            )
            rows.append(
                _base_row(
                    labels=labels,
                    intent="json",
                    task_type="active_agent_json",
                    encoder_text=json_encoder,
                    decoder_text=(
                        "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_json "
                        f"<AK_INTENT> translation <AK_FIELD> <AK_FIELD_NAME> target_language <AK_FIELD_VALUE> {language} <AK_END>"
                    ),
                    json_decoder_text=(
                        '{"action":"respond","content":"'
                        f'{{\\"intent\\":\\"translation\\",\\"target_language\\":\\"{language}\\"}}'
                        '","proposal_metadata":{"task_type":"active_agent_json"}}'
                    ),
                    source_id=f"stage74_json_contract_over_translate_{idx:02d}_{repeat:03d}",
                    weight=18.0,
                )
            )
            translation_encoder = _encoder_text(
                agent_name="Translation Agent",
                agent_instruction=f"Translate the user's English text into {language.title()}. Return the translated content, not a JSON intent classification.",
                intent="translation",
                task="active_agent_translation",
                source_text=source_text,
            )
            rows.append(
                _base_row(
                    labels=labels,
                    intent="translation",
                    task_type="active_agent_translation",
                    encoder_text=translation_encoder,
                    decoder_text=(
                        "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_translation "
                        f"<AK_CONTENT> {translated} </AK_CONTENT> <AK_END>"
                    ),
                    json_decoder_text=(
                        '{"action":"respond","content":"'
                        + translated.replace('"', '\\"')
                        + '","proposal_metadata":{"task_type":"active_agent_translation"}}'
                    ),
                    source_id=f"stage74_translate_contract_over_json_{idx:02d}_{repeat:03d}",
                    weight=12.0,
                )
            )
    return rows


def _summary_plan_boundary_rows(labels: dict[str, int], repeats: int) -> list[dict[str, Any]]:
    sources = [
        "The build uploaded, but Apple processing is still pending.",
        "The beta passed smoke tests, but TestFlight review has not started.",
        "The web bundle exports correctly, but mobile parity still needs a cache-bust.",
        "The dataset merged cleanly, but the intent head still confuses JSON and translation.",
        "The demo script runs locally, but the final artifact has not been signed.",
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for idx, source_text in enumerate(sources):
            summary_encoder = _encoder_text(
                agent_name="Summary Agent",
                agent_instruction="Create a short factual summary of the source text. Do not make a plan or add next steps.",
                intent="summary",
                task="active_agent_summary",
                source_text=source_text,
            )
            rows.append(
                _base_row(
                    labels=labels,
                    intent="summary",
                    task_type="active_agent_summary",
                    encoder_text=summary_encoder,
                    decoder_text=(
                        "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_summary "
                        "<AK_CONTENT> <AK_COPY_USER_SOURCE_1> </AK_CONTENT> <AK_END>"
                    ),
                    json_decoder_text=(
                        '{"action":"respond","content":"'
                        + source_text.replace('"', '\\"')
                        + '","proposal_metadata":{"task_type":"active_agent_summary"}}'
                    ),
                    source_id=f"stage74_summary_contract_over_plan_{idx:02d}_{repeat:03d}",
                    weight=14.0,
                )
            )
            plan_encoder = _encoder_text(
                agent_name="Planning Agent",
                agent_instruction="Create a concise plan with ordered next steps from the source text. Do not merely summarize.",
                intent="plan",
                task="active_agent_plan",
                source_text=source_text,
            )
            rows.append(
                _base_row(
                    labels=labels,
                    intent="plan",
                    task_type="active_agent_plan",
                    encoder_text=plan_encoder,
                    decoder_text=(
                        "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_plan "
                        "<AK_CONTENT> 1. Confirm the current status. 2. Identify the blocker. 3. Follow up on the next required action. </AK_CONTENT> <AK_END>"
                    ),
                    json_decoder_text=(
                        '{"action":"respond","content":"1. Confirm the current status. 2. Identify the blocker. 3. Follow up on the next required action.",'
                        '"proposal_metadata":{"task_type":"active_agent_plan"}}'
                    ),
                    source_id=f"stage74_plan_contract_over_summary_{idx:02d}_{repeat:03d}",
                    weight=10.0,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default=str(REPO_ROOT / "tmp/pocketpal_stage73_intent_balanced_repair_v285/agentkernel_lite_encdec_dataset_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tmp/pocketpal_stage74_intent_boundary_curriculum_v287"),
    )
    parser.add_argument("--json-repeats", type=int, default=18)
    parser.add_argument("--summary-plan-repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=28774)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    source_manifest_path = Path(args.source_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    labels = dict(source_manifest["intent_labels"])
    train_rows = _load_jsonl(Path(source_manifest["train_dataset_path"]))
    eval_rows = _load_jsonl(Path(source_manifest["eval_dataset_path"]))

    additions = _json_boundary_rows(labels, int(args.json_repeats))
    additions.extend(_summary_plan_boundary_rows(labels, int(args.summary_plan_repeats)))
    rng.shuffle(additions)
    merged_train = train_rows + additions
    rng.shuffle(merged_train)

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, merged_train)
    _write_jsonl(eval_path, eval_rows)

    manifest = deepcopy(source_manifest)
    manifest.update(
        {
            "artifact_kind": "agentkernel_lite_encdec_stage74_intent_boundary_curriculum",
            "dataset_objective": "pocketpal_stage74_intent_boundary_curriculum_v287",
            "objective": "pocketpal_stage74_intent_boundary_curriculum_v287",
            "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
            "source_manifest_path": str(source_manifest_path),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "train_examples": len(merged_train),
            "eval_examples": len(eval_rows),
            "stage74_additions": len(additions),
            "stage74_note": (
                "Minimal-pair active-agent contract curriculum: JSON-vs-translation and summary-vs-plan boundaries. "
                "Decoder loss should remain disabled for head-only runs."
            ),
            "stage74_weighting_policy": {
                "json_contract_weight": 18.0,
                "translation_contract_weight": 12.0,
                "summary_contract_weight": 14.0,
                "plan_contract_weight": 10.0,
                "json_repeats": int(args.json_repeats),
                "summary_plan_repeats": int(args.summary_plan_repeats),
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "train_examples": len(merged_train),
                "eval_examples": len(eval_rows),
                "stage74_additions": len(additions),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
