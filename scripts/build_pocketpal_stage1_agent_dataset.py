#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _agent_prompt(*, name: str, instruction: str, user_text: str, stale_context: str = "none") -> str:
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
            "<AK_CONTEXT> Saved user data: none",
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
    agent_instruction: str,
    user_text: str,
    action: str,
    content: str,
    task_type: str,
    weight: float = 1.0,
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
) -> dict[str, Any]:
    decoder = {
        "action": action,
        "content": content,
        "proposal_metadata": {"task_type": task_type},
    }
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False),
        "encoder_text": _agent_prompt(
            name=agent_name,
            instruction=agent_instruction,
            user_text=user_text,
            stale_context=stale_context,
        ),
        "example_id": hashlib.sha256(f"{source_id}\n{user_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage1_agent",
        "task_type": task_type,
        "weight": float(weight),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stale_contexts = [
        "Selected paper [P1]: unrelated research paper context.",
        "Selected paper [P1]: unrelated optimization notes from a previous research turn.",
        "Retrieved evidence [1]: unrelated web-search result about insurance markets.",
        "none",
    ]

    rewrite_instruction = (
        "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. "
        "If there is no editable text, ask for it."
    )
    names = [
        "john", "alex", "priya", "omar", "maya", "lee", "nina", "sam", "taylor", "riley",
        "jordan", "avery", "morgan", "casey", "devon", "elena", "marco", "lena", "noah", "zoe",
    ]
    artifacts = [
        "report", "budget summary", "release notes", "prototype link", "invoice", "contract draft",
        "client update", "launch plan", "design mockups", "test plan", "status memo", "roadmap",
    ]
    deadlines = ["friday", "thursday", "tomorrow morning", "before noon", "by end of day", "monday at 10"]
    reasons = [
        "the client is asking",
        "the review is starting",
        "engineering is blocked",
        "finance needs it",
        "the vendor is waiting",
        "testing starts soon",
    ]
    idx = 0
    for name in names:
        for artifact in artifacts:
            for deadline in deadlines:
                reason = reasons[(len(name) + len(artifact) + len(deadline)) % len(reasons)]
                variants = [
                    f"hey {name} i need the {artifact} by {deadline} because {reason}",
                    f"ask {name} for the {artifact} by {deadline} because {reason}",
                    f"tell {name} we need the {artifact} by {deadline} since {reason}",
                ]
                for user_text in variants:
                    pretty = name.capitalize()
                    content = (
                        f"Hi {pretty},\n\n"
                        f"Could you please send the {artifact} by {deadline}? {reason.capitalize()}.\n\n"
                        "Thank you."
                    )
                    rows.append(
                        _row(
                            source_id=f"stage1_rewrite_{idx:05d}",
                            agent_name="Professional Email Rewriter",
                            agent_instruction=rewrite_instruction,
                            user_text=user_text,
                            action="respond",
                            content=content,
                            task_type="agent_instruction_following",
                            weight=4.0,
                            stale_context=stale_contexts[idx % len(stale_contexts)],
                        )
                    )
                    idx += 1

    source_instruction = "Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording."
    source_texts = [
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "meeting moved to 11 and Priya needs the updated slides",
        "Maya needs the design review follow up tomorrow at 11",
        "legal needs one more day on the contract draft for Omar",
        "the prototype link is ready but battery usage is higher than expected",
        "client ACME asked for the Friday report before noon",
    ]
    for repeat in range(80):
        for source_index, user_text in enumerate(source_texts):
            rows.append(
                _row(
                    source_id=f"stage1_source_echo_{repeat:03d}_{source_index:02d}",
                    agent_name="Source Echo Agent",
                    agent_instruction=source_instruction,
                    user_text=user_text,
                    action="respond",
                    content=f"Source text: {user_text}",
                    task_type="source_echo",
                    weight=5.0,
                    stale_context=stale_contexts[(repeat + source_index) % len(stale_contexts)],
                )
            )

    casual_instruction = "Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it."
    casuals = [
        ("How's it going?", "It's going well. What would you like help with?"),
        ("hey how are you", "I'm doing well. What can I help with?"),
        ("thanks", "You're welcome."),
        ("what can you do", "I can help with writing, searching, summarizing, and organizing your local context."),
        ("can you help me today", "Yes. Tell me what you want to work on."),
    ]
    for repeat in range(160):
        for casual_index, (user_text, content) in enumerate(casuals):
            rows.append(
                _row(
                    source_id=f"stage1_casual_{repeat:03d}_{casual_index:02d}",
                    agent_name="Casual Assistant",
                    agent_instruction=casual_instruction,
                    user_text=user_text,
                    action="respond",
                    content=content,
                    task_type="agent_instruction_following",
                    weight=3.0,
                    stale_context=stale_contexts[(repeat + casual_index) % len(stale_contexts)],
                )
            )

    missing_texts = ["rewrite this", "make this professional", "polish this email", "turn this into a note"]
    for repeat in range(120):
        for missing_index, user_text in enumerate(missing_texts):
            rows.append(
                _row(
                    source_id=f"stage1_ask_missing_{repeat:03d}_{missing_index:02d}",
                    agent_name="Professional Email Rewriter",
                    agent_instruction=rewrite_instruction,
                    user_text=user_text,
                    action="ask_user",
                    content="What text should I rewrite?",
                    task_type="agent_ask_user",
                    weight=3.0,
                    stale_context=stale_contexts[(repeat + missing_index) % len(stale_contexts)],
                )
            )
    return rows


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, eval_fraction: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pocketpal_stage1_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_stage1_agent_eval.jsonl"
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
        "manifest_path": str((output_dir / "pocketpal_stage1_agent_manifest.json").resolve()),
        "objective": "pocketpal_stage1_agent_instruction_following",
        "schema": {
            "decoder_text": "compact JSON action decision",
            "encoder_text": "PocketPal active agent contract and user request",
        },
        "source_counts": {"pocketpal_stage1_agent": len(rows)},
        "target_action_counts": action_counts,
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / "pocketpal_stage1_agent_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage1_agent_dataset")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = write_dataset(build_rows(), Path(args.output_dir), float(args.eval_fraction))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
