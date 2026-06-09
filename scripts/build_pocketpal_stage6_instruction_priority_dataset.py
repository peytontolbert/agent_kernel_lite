#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _agent(name: str, instruction: str, action_policy: str = "respond_or_ask") -> tuple[str, str]:
    preamble = "\n".join(
        [
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            f"Action policy: {action_policy}",
            "The active agent instruction is the primary task contract for this turn. Apply it directly to the user request. Do not answer as the base assistant when an active agent is selected unless the active agent instruction asks for normal assistant chat.",
            "</AK_AGENT_ACTIVE>",
        ]
    )
    profile = "\n".join(
        [
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_name <AK_SLOT_VALUE>={name}",
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_instruction <AK_SLOT_VALUE>={instruction}",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_retrieval_policy <AK_SLOT_VALUE>=auto",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_tool_policy <AK_SLOT_VALUE>=ask_before_extensions",
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_action_policy <AK_SLOT_VALUE>={action_policy}",
        ]
    )
    return preamble, profile


def _prompt(user_text: str, agent_name: str, instruction: str, action_policy: str = "respond_or_ask") -> str:
    preamble, profile = _agent(agent_name, instruction, action_policy)
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
            "Return exactly this decision format: Action: respond, then Content: your direct answer.",
            "You are Agent Kernel Lite running entirely in this browser.",
            "Do not claim to execute, test, install, browse, or modify files.",
            "Mode: Chat. Reply like a helpful assistant. Use the strongest relevant evidence as support, not as the whole answer.",
            "Mode: Chat",
            preamble,
            f"Active agent selected: {agent_name}.",
            f"Follow this agent instruction unless it conflicts with user safety or the current user request: {instruction}",
            f"Agent policies: retrieval=auto tools=ask_before_extensions actions={action_policy}.",
            "<AK_PROFILE> PocketPal saved slots:",
            "No saved PocketPal slots.",
            "<AK_PROFILE> Active PocketPal agent:",
            profile,
            "<AK_PROFILE> PocketPal installed tools:",
            "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
            "<AK_MAX_SOURCES>=5",
            "When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.",
            "<AK_PROFILE> PocketPal local memory:",
            "No saved PocketPal memory.",
            "<AK_CONTEXT> User data pointers:",
            "No saved user data sources.",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "When preserving exact user-provided details, write the matching [[SOURCE_TEXT]] placeholder in the JSON content when exact copying is required.",
            "<AK_HISTORY> Recent conversation:",
            "No recent conversation.",
            "<AK_READING_NOTES> Semantic reading notes:",
            "No strong semantic match in retrieved evidence.",
            "<AK_CONTEXT> Research context:",
            "No research context was retrieved.",
            "<AK_ANSWER> Answer scaffold:",
            "Answer directly.",
            f"<AK_USER> {user_text}",
            "Return a structured decision that follows the active agent instruction.",
        ]
    )


def _row(source_id: str, agent_name: str, instruction: str, user_text: str, content: str, *, action: str = "respond", weight: float = 9.0) -> dict[str, Any]:
    decoder = {"action": action, "content": content, "proposal_metadata": {"task_type": "runtime_agent_instruction_priority"}}
    encoder_text = _prompt(user_text, agent_name, instruction)
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage6_instruction_priority",
        "task_type": "runtime_agent_instruction_priority",
        "weight": float(weight),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    agents: list[tuple[str, str, list[tuple[str, str]]]] = [
        (
            "Professional Email Rewriter",
            "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            [
                ("Hi how are you?", "Hello, I hope you are doing well."),
                ("hey john i need the report by friday because the client is asking", "Hi John, Could you please send the report by friday? The client is asking. Thank you."),
                ("thanks for the quick update", "Thank you for the quick update."),
                ("can we move the meeting to 3 because i have a conflict", "Hello, could we move the meeting to 3? I have a conflict. Thank you."),
            ],
        ),
        (
            "Formal Tone Agent",
            "Rewrite the user's text in a formal, polished tone while preserving the meaning.",
            [
                ("Hi how are you?", "Hello, I hope you are well."),
                ("this looks good to me", "This looks good to me."),
                ("we need the files today", "We need the files today."),
            ],
        ),
        (
            "Friendly Tone Agent",
            "Rewrite the user's text in a friendly, warm tone while preserving the meaning.",
            [
                ("Hi how are you?", "Hi, how are you doing?"),
                ("send me the notes", "Could you send me the notes when you have a chance?"),
                ("the draft is late", "The draft is running a bit late."),
            ],
        ),
        (
            "Bullet Summary Agent",
            "Turn the user's text into concise bullet points. Do not answer the text as a conversation.",
            [
                ("Hi how are you?", "- Greeting: Hi, how are you?"),
                ("launch is friday and QA needs the build today", "- Launch is friday\n- QA needs the build today"),
                ("client asked for pricing and timeline", "- Client asked for pricing\n- Client asked for timeline"),
            ],
        ),
        (
            "Grammar Fixer",
            "Fix grammar and clarity in the user's text. Preserve the original meaning.",
            [
                ("Hi how are you?", "Hi, how are you?"),
                ("we was waiting for the files", "We were waiting for the files."),
                ("she dont have the latest build", "She does not have the latest build."),
            ],
        ),
    ]
    for repeat in range(180):
        for agent_name, instruction, cases in agents:
            for case_index, (user_text, content) in enumerate(cases):
                rows.append(
                    _row(
                        source_id=f"stage6_{repeat:03d}_{agent_name.lower().replace(' ', '_')}_{case_index:02d}",
                        agent_name=agent_name,
                        instruction=instruction,
                        user_text=user_text,
                        content=content,
                    )
                )
    return rows


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key in ("train_dataset_path", "eval_dataset_path"):
        path = Path(str(manifest.get(key, "") or ""))
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, eval_fraction: float, name: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{name}_train.jsonl"
    eval_path = output_dir / f"{name}_eval.jsonl"
    eval_every = max(2, int(round(1.0 / max(0.01, min(0.5, float(eval_fraction))))))
    train_rows = [row for index, row in enumerate(rows) if index % eval_every != 0]
    eval_rows = [row for index, row in enumerate(rows) if index % eval_every == 0]
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
        source_counts[str(row.get("source_type", "unknown"))] = source_counts.get(str(row.get("source_type", "unknown")), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / f"{name}_manifest.json").resolve()),
        "objective": "pocketpal_stage6_active_agent_instruction_priority",
        "schema": {"encoder_text": "actual app-style active-agent prompt", "decoder_text": "compact JSON action decision"},
        "source_counts": source_counts,
        "target_action_counts": action_counts,
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage6_instruction_priority_dataset")
    parser.add_argument("--mixed-output-dir", default="")
    parser.add_argument("--include-manifest", action="append", default=[])
    parser.add_argument("--stage6-repeat", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    stage6_rows = build_rows()
    print(json.dumps(write_dataset(stage6_rows, Path(args.output_dir), float(args.eval_fraction), "pocketpal_stage6_instruction_priority"), indent=2, sort_keys=True))
    if str(args.mixed_output_dir).strip():
        mixed_rows: list[dict[str, Any]] = []
        for manifest_arg in args.include_manifest:
            mixed_rows.extend(_read_manifest_rows(Path(manifest_arg).expanduser().resolve()))
        repeated_rows: list[dict[str, Any]] = []
        for repeat in range(max(1, int(args.stage6_repeat))):
            for row in stage6_rows:
                next_row = dict(row)
                next_row["source_id"] = f"{row['source_id']}:repeat_{repeat:02d}"
                next_row["example_id"] = hashlib.sha256(f"{next_row['source_id']}\n{row['encoder_text']}\n{row['decoder_text']}".encode()).hexdigest()
                repeated_rows.append(next_row)
        mixed_rows.extend(repeated_rows)
        print(json.dumps(write_dataset(mixed_rows, Path(args.mixed_output_dir), float(args.eval_fraction), "pocketpal_stage6_mixed"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
