#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _slot_block(slots: dict[str, str]) -> str:
    if not slots:
        return "<AK_PROFILE> User text slots: none"
    lines = ["<AK_PROFILE> User text slots:"]
    for name, value in slots.items():
        lines.append(f"<AK_SLOT> <AK_SLOT_NAME>={name} <AK_SLOT_VALUE>={value}")
    lines.append(
        "When preserving exact user-provided details, write the matching [[SOURCE_TEXT]] or [[DATA_CONTEXT]] placeholder in the JSON content."
    )
    return "\n".join(lines)


def _prompt(
    *,
    agent_name: str,
    instruction: str,
    user_text: str,
    data_context: str,
    stale_context: str,
    slots: dict[str, str],
) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {agent_name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            "<AK_CONTEXT> Saved user data: none",
            "<AK_CONTEXT> User data pointers:",
            data_context,
            _slot_block(slots),
            f"<AK_CONTEXT> Stale selected paper context: {stale_context}",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _row(
    *,
    source_id: str,
    agent_name: str,
    instruction: str,
    user_text: str,
    data_context: str,
    stale_context: str,
    slots: dict[str, str],
    action: str,
    content: str,
    task_type: str,
    weight: float,
) -> dict[str, Any]:
    decoder = {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False),
        "encoder_text": _prompt(
            agent_name=agent_name,
            instruction=instruction,
            user_text=user_text,
            data_context=data_context,
            stale_context=stale_context,
            slots=slots,
        ),
        "example_id": hashlib.sha256(f"{source_id}\n{user_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage2_retrieval",
        "task_type": task_type,
        "weight": float(weight),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    instruction = (
        "Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. "
        "If no relevant saved data is available, ask one concise question."
    )
    data_items = [
        ("launch code", "Launch code is ORBIT-42 for the May TestFlight build."),
        ("vet appointment", "Maya's vet appointment is Thursday at 3 PM."),
        ("wifi password", "Studio Wi-Fi password is Maple-7782."),
        ("invoice", "Vendor invoice INV-2048 is blocked until finance approves $1,200."),
        ("design review", "Design review is Monday; send Priya the prototype link beforehand."),
        ("parking", "Office parking code is 1947# for the north garage."),
    ]
    stale_contexts = [
        "Selected paper [P1]: unrelated optimization notes from a previous research turn.",
        "Retrieved evidence [1]: unrelated web-search result about insurance markets.",
        "Selected paper [P1]: unrelated paper about transformer attention.",
        "none",
    ]
    index = 0
    for repeat in range(80):
        for topic, fact in data_items:
            data_context = f"[D1] saved note (note, chunk 1): {fact}"
            slots = {"SOURCE_TEXT": f"what is my {topic}", "DATA_CONTEXT": data_context}
            rows.append(
                _row(
                    source_id=f"stage2_use_data_{index:05d}",
                    agent_name="Saved Data Assistant",
                    instruction=instruction,
                    user_text=f"what is my {topic}",
                    data_context=data_context,
                    stale_context=stale_contexts[index % len(stale_contexts)],
                    slots=slots,
                    action="respond",
                    content="I found this in your saved data: [[DATA_CONTEXT]]",
                    task_type="use_user_data",
                    weight=5.0,
                )
            )
            index += 1
            rows.append(
                _row(
                    source_id=f"stage2_summarize_data_{index:05d}",
                    agent_name="Saved Data Assistant",
                    instruction=instruction,
                    user_text=f"summarize my saved note about {topic}",
                    data_context=data_context,
                    stale_context=stale_contexts[index % len(stale_contexts)],
                    slots=slots,
                    action="respond",
                    content="Your saved note says: [[DATA_CONTEXT]]",
                    task_type="use_user_data",
                    weight=5.0,
                )
            )
            index += 1

    missing_queries = [
        "what is my hotel confirmation code",
        "what time is my dentist appointment",
        "where did I save the beta invite list",
        "what is the client contract number",
    ]
    for repeat in range(80):
        for topic, _fact in data_items:
            query = f"what is my {topic}"
            rows.append(
                _row(
                    source_id=f"stage2_missing_known_shape_{repeat:03d}_{len(rows):05d}",
                    agent_name="Saved Data Assistant",
                    instruction=instruction,
                    user_text=query,
                    data_context="No saved user data sources.",
                    stale_context=stale_contexts[(repeat + len(topic)) % len(stale_contexts)],
                    slots={"SOURCE_TEXT": query},
                    action="ask_user",
                    content="I do not have relevant saved data for that. What should I look for or add?",
                    task_type="ask_missing_user_data",
                    weight=6.0,
                )
            )
    for repeat in range(120):
        for query in missing_queries:
            rows.append(
                _row(
                    source_id=f"stage2_missing_data_{repeat:03d}_{len(rows):05d}",
                    agent_name="Saved Data Assistant",
                    instruction=instruction,
                    user_text=query,
                    data_context="No saved user data sources.",
                    stale_context=stale_contexts[(repeat + len(query)) % len(stale_contexts)],
                    slots={"SOURCE_TEXT": query},
                    action="ask_user",
                    content="I do not have relevant saved data for that. What should I look for or add?",
                    task_type="ask_missing_user_data",
                    weight=4.0,
                )
            )

    casual_instruction = "Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it."
    for repeat in range(80):
        rows.append(
            _row(
                source_id=f"stage2_casual_retention_{repeat:03d}",
                agent_name="Casual Assistant",
                instruction=casual_instruction,
                user_text="How's it going?",
                data_context="No saved user data sources.",
                stale_context=stale_contexts[repeat % len(stale_contexts)],
                slots={"SOURCE_TEXT": "How's it going?"},
                action="respond",
                content="It's going well. What would you like help with?",
                task_type="agent_instruction_following",
                weight=3.0,
            )
        )
    return rows


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, eval_fraction: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pocketpal_stage2_retrieval_train.jsonl"
    eval_path = output_dir / "pocketpal_stage2_retrieval_eval.jsonl"
    eval_every = max(2, int(round(1.0 / max(0.01, min(0.5, float(eval_fraction))))))
    train_rows = [row for index, row in enumerate(rows) if index % eval_every != 0]
    eval_rows = [row for index, row in enumerate(rows) if index % eval_every == 0]
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / "pocketpal_stage2_retrieval_manifest.json").resolve()),
        "objective": "pocketpal_stage2_user_data_retrieval",
        "schema": {"encoder_text": "PocketPal active agent, user data pointers, and user request", "decoder_text": "compact JSON action decision"},
        "source_counts": {"pocketpal_stage2_retrieval": len(rows)},
        "target_action_counts": action_counts,
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / "pocketpal_stage2_retrieval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage2_retrieval_dataset")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(write_dataset(build_rows(), Path(args.output_dir), float(args.eval_fraction)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
