#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _agent_prompt(intent: str, agent_name: str, instruction: str, source: str) -> str:
    task = f"active_agent_{intent}"
    return (
        "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.\n"
        "<AK_AGENT_ACTIVE>\n"
        f"Agent name: {agent_name}\n"
        f"Agent instruction: {instruction}\n"
        "Retrieval policy: auto\n"
        "Tool policy: ask_before_extensions\n"
        "Action policy: respond_or_ask\n"
        "The active agent instruction is the primary task contract for this turn.\n"
        "</AK_AGENT_ACTIVE>\n"
        f"<AK_TASK_HINT> intent={intent} task={task} source_text_required=true\n"
        "<AK_CONTEXT> Saved user data: none\n"
        "<AK_PROFILE> User text slots:\n"
        f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={source}\n"
        "Available placeholders for this turn: [[SOURCE_TEXT]].\n"
        "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.\n"
        "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.\n"
        "Use stale paper context only when the current user request asks about that paper or research evidence.\n"
        f"<AK_USER> {source}\n"
        "Return compact JSON with the correct action and content for the active agent."
    )


def _row(source_id: str, intent: str, agent_name: str, instruction: str, source: str, content: str) -> dict[str, Any]:
    task = f"active_agent_{intent}"
    return {
        "source_id": source_id,
        "dataset_stage": "stage63_routing_boundary",
        "task_type": task,
        "intent_label_id": INTENT_LABELS[intent],
        "encoder_text": _agent_prompt(intent, agent_name, instruction, source),
        "decoder_prefix": "",
        "decoder_text": json.dumps(
            {"action": "respond", "content": content, "proposal_metadata": {"task_type": task}},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "expected_content": content,
    }


def _synthetic_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        (
            "json",
            "JSON Classifier",
            "Classify the user request as compact JSON.",
            "Find current news about TestFlight processing.",
            '{"intent":"web_search","freshness":"current"}',
        ),
        (
            "json",
            "JSON Classifier",
            "Return only a compact JSON object describing the user's intent.",
            "Find current news about App Store review delays.",
            '{"intent":"web_search","freshness":"current"}',
        ),
        (
            "rewrite",
            "Rewrite Agent",
            "Rewrite the user's message to be clear and polite. Return only the rewritten text.",
            "Could we move the meeting to Friday afternoon?",
            "Could we move the meeting to Friday afternoon?",
        ),
        (
            "rewrite",
            "Rewrite Agent",
            "Make the user's message sound professional without changing meaning.",
            "Can you send the notes after lunch?",
            "Could you send the notes after lunch?",
        ),
        (
            "extraction",
            "Extractor",
            "Extract the key question from the user text.",
            "Can you check if web search is active?",
            "Question: Can you check if web search is active?",
        ),
        (
            "extraction",
            "Extractor",
            "Extract the user question and return it as a concise field.",
            "Can you verify whether retrieval is enabled?",
            "Question: Can you verify whether retrieval is enabled?",
        ),
        (
            "brainstorm",
            "Brainstorm Agent",
            "Generate three concise ideas that fit the user's request.",
            "Ideas for making PocketPal feel more personal.",
            "1. Let users create custom agents\n2. Add local memory collections\n3. Offer per-agent tone and tool settings",
        ),
        (
            "plan",
            "Planner",
            "Create a concise sequence of steps for the user's goal.",
            "Ship the fixed rewrite agent to TestFlight.",
            "1. Verify the active-agent prompt path.\n2. Commit and push the fix.\n3. Run the TestFlight workflow.\n4. Install and test the processed build.",
        ),
    ]
    for repeat in range(max(1, int(repeats))):
        for idx, (intent, agent, instruction, source, content) in enumerate(cases):
            rows.append(_row(f"stage63_{intent}_{idx:02d}_repeat_{repeat:03d}", intent, agent, instruction, source, content))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json",
    )
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage63_routing_boundary_curriculum")
    parser.add_argument("--synthetic-repeats", type=int, default=600)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train_rows = _read_jsonl(Path(source_manifest["train_dataset_path"]))
    eval_rows = _read_jsonl(Path(source_manifest["eval_dataset_path"]))
    synthetic = _synthetic_rows(int(args.synthetic_repeats))
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows + synthetic)
    _write_jsonl(eval_path, eval_rows + synthetic[: min(len(synthetic), 256)])
    manifest = {
        **source_manifest,
        "dataset_objective": "pocketpal_stage63_routing_boundary_curriculum",
        "source_manifest": str(source_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "intent_labels": INTENT_LABELS,
        "stage63_synthetic_examples": len(synthetic),
        "train_examples": len(train_rows) + len(synthetic),
        "eval_examples": len(eval_rows) + min(len(synthetic), 256),
    }
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "train": manifest["train_examples"], "eval": manifest["eval_examples"]}, sort_keys=True))


if __name__ == "__main__":
    main()
