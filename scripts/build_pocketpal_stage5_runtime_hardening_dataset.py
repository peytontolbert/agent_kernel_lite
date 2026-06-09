#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _slot_block(user_text: str, data_context: str = "") -> str:
    lines = [
        "<AK_PROFILE> User text slots:",
        f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
    ]
    if data_context and data_context.lower() != "no saved user data sources.":
        lines.append(f"<AK_SLOT> <AK_SLOT_NAME>=DATA_CONTEXT <AK_SLOT_VALUE>={data_context}")
    lines.append(
        "When preserving exact user-provided details, write the matching [[NAME]], [[ITEM]], [[DEADLINE]], [[REASON]], [[SOURCE_TEXT]], or [[DATA_CONTEXT]] placeholder in the JSON content."
    )
    return "\n".join(lines)


def _runtime_prompt(
    *,
    user_text: str,
    active_agent: str = "No custom agent selected.",
    active_instruction: str = "",
    data_context: str = "No saved user data sources.",
    history: str = "No recent conversation.",
    reading_notes: str = "No strong semantic match in retrieved evidence.",
    research_context: str = "No research context was retrieved.",
    answer_scaffold: str = "Answer directly.",
) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
            "Return exactly this decision format: Action: respond, then Content: your direct answer.",
            "You are Agent Kernel Lite running entirely in this browser.",
            "Do not claim to execute, test, install, browse, or modify files.",
            "Mode: Chat. Reply like a helpful assistant. Use the strongest relevant evidence as support, not as the whole answer.",
            "Mode: Chat",
            active_instruction,
            "<AK_PROFILE> PocketPal saved slots:",
            "No saved PocketPal slots.",
            "<AK_PROFILE> Active PocketPal agent:",
            active_agent,
            "<AK_PROFILE> PocketPal installed tools:",
            "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
            "<AK_MAX_SOURCES>=5",
            "When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.",
            "<AK_PROFILE> PocketPal local memory:",
            "No saved PocketPal memory.",
            "<AK_CONTEXT> User data pointers:",
            data_context,
            _slot_block(user_text, data_context),
            "<AK_HISTORY> Recent conversation:",
            history,
            "<AK_READING_NOTES> Semantic reading notes:",
            reading_notes,
            "<AK_CONTEXT> Research context:",
            research_context,
            "<AK_ANSWER> Answer scaffold:",
            answer_scaffold,
            f"<AK_USER> {user_text}",
            "Return a structured decision with action=respond.",
        ]
    )


def _row(
    *,
    source_id: str,
    user_text: str,
    action: str,
    content: str,
    task_type: str,
    weight: float,
    metadata: dict[str, Any] | None = None,
    **prompt_kwargs: Any,
) -> dict[str, Any]:
    proposal_metadata = {"task_type": task_type}
    if metadata:
        proposal_metadata.update(metadata)
    decoder = {"action": action, "content": content, "proposal_metadata": proposal_metadata}
    encoder_text = _runtime_prompt(user_text=user_text, **prompt_kwargs)
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage5_runtime_hardening",
        "task_type": task_type,
        "weight": float(weight),
    }


def _agent_block(name: str, instruction: str, action_policy: str = "respond_or_ask") -> tuple[str, str]:
    active = "\n".join(
        [
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_name <AK_SLOT_VALUE>={name}",
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_instruction <AK_SLOT_VALUE>={instruction}",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_retrieval_policy <AK_SLOT_VALUE>=auto",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_tool_policy <AK_SLOT_VALUE>=ask_before_extensions",
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_action_policy <AK_SLOT_VALUE>={action_policy}",
        ]
    )
    instruction_block = "\n".join(
        [
            f"Active agent selected: {name}.",
            f"Follow this agent instruction unless it conflicts with user safety or the current user request: {instruction}",
            f"Agent policies: retrieval=auto tools=ask_before_extensions actions={action_policy}.",
        ]
    )
    return active, instruction_block


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    casuals = [
        ("Hi how are you?", "I'm doing well. What would you like help with?"),
        ("hi how are you", "I'm doing well. What would you like help with?"),
        ("hello", "Hi. What would you like help with?"),
        ("hey there", "Hey. What can I help with?"),
        ("thanks", "You're welcome."),
        ("what can you do?", "I can chat, rewrite text, use your saved data, and request web search when current information is needed."),
        ("can you help me today?", "Yes. Tell me what you want to work on."),
    ]
    for repeat in range(240):
        for index, (user_text, content) in enumerate(casuals):
            rows.append(
                _row(
                    source_id=f"stage5_casual_{repeat:03d}_{index:02d}",
                    user_text=user_text,
                    action="respond",
                    content=content,
                    task_type="runtime_plain_chat",
                    weight=9.0,
                )
            )

    rewrite_instruction = "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it."
    rewrite_agent, rewrite_block = _agent_block("Professional Email Rewriter", rewrite_instruction)
    rewrite_cases = [
        ("hey john i need the report by friday because the client is asking", "Hi John, Could you please send the report by friday? The client is asking. Thank you."),
        ("rewrite this", "What text should I rewrite?"),
        ("make this professional: nina please send the launch notes today", "Nina, please send the launch notes today."),
    ]
    for repeat in range(120):
        for index, (user_text, content) in enumerate(rewrite_cases):
            ask = user_text in {"rewrite this"}
            rows.append(
                _row(
                    source_id=f"stage5_rewrite_{repeat:03d}_{index:02d}",
                    user_text=user_text,
                    active_agent=rewrite_agent,
                    active_instruction=rewrite_block,
                    action="ask_user" if ask else "respond",
                    content=content,
                    task_type="runtime_agent_instruction",
                    weight=7.0,
                )
            )

    saved_instruction = "Use the user's saved data when it directly answers the request. If no relevant saved data is available, ask one concise question."
    saved_agent, saved_block = _agent_block("Saved Data Assistant", saved_instruction)
    data_context = "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."
    for repeat in range(120):
        rows.append(
            _row(
                source_id=f"stage5_saved_use_{repeat:03d}",
                user_text="what is my launch code",
                active_agent=saved_agent,
                active_instruction=saved_block,
                data_context=data_context,
                action="respond",
                content="I found this in your saved data: [[DATA_CONTEXT]]",
                task_type="runtime_use_user_data",
                weight=7.0,
            )
        )
        rows.append(
            _row(
                source_id=f"stage5_saved_missing_{repeat:03d}",
                user_text="what is my hotel confirmation code",
                active_agent=saved_agent,
                active_instruction=saved_block,
                action="ask_user",
                content="I do not have relevant saved data for that. What should I look for or add?",
                task_type="runtime_missing_user_data",
                weight=7.0,
            )
        )

    web_instruction = "When the user asks for current, recent, online, or web-backed information, request the web_search extension. Do not invent results before the extension runs."
    web_agent, web_block = _agent_block("Web Search Assistant", web_instruction, action_policy="allow_extension_requests")
    web_cases = [
        ("search the web for current TestFlight upload limits", "search the web for current TestFlight upload limits"),
        ("look up recent reviews for the Anker 737 power bank", "look up recent reviews for the Anker 737 power bank"),
    ]
    for repeat in range(100):
        for index, (user_text, query) in enumerate(web_cases):
            rows.append(
                _row(
                    source_id=f"stage5_web_{repeat:03d}_{index:02d}",
                    user_text=user_text,
                    active_agent=web_agent,
                    active_instruction=web_block,
                    action="extension_request",
                    content="Requesting approval to search the web.",
                    task_type="runtime_web_search_request",
                    metadata={
                        "extension_id": "web_search",
                        "capability": "web.search",
                        "query": query,
                        "max_sources": 5,
                        "requires_user_approval": True,
                    },
                    weight=7.0,
                )
            )
    return rows


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key in ("train_dataset_path", "eval_dataset_path"):
        path_value = str(manifest.get(key, "") or "")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
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
        "objective": "pocketpal_stage5_runtime_prompt_hardening",
        "schema": {"encoder_text": "actual app-style runtime prompt", "decoder_text": "compact JSON action decision"},
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
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage5_runtime_dataset")
    parser.add_argument("--mixed-output-dir", default="")
    parser.add_argument("--include-manifest", action="append", default=[])
    parser.add_argument("--stage5-repeat", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    stage5_rows = build_rows()
    print(json.dumps(write_dataset(stage5_rows, Path(args.output_dir), float(args.eval_fraction), "pocketpal_stage5_runtime"), indent=2, sort_keys=True))
    if str(args.mixed_output_dir).strip():
        mixed_rows: list[dict[str, Any]] = []
        for manifest_arg in args.include_manifest:
            mixed_rows.extend(_read_manifest_rows(Path(manifest_arg).expanduser().resolve()))
        repeated_rows: list[dict[str, Any]] = []
        for repeat in range(max(1, int(args.stage5_repeat))):
            for row in stage5_rows:
                next_row = dict(row)
                next_row["source_id"] = f"{row['source_id']}:repeat_{repeat:02d}"
                next_row["example_id"] = hashlib.sha256(f"{next_row['source_id']}\n{row['encoder_text']}\n{row['decoder_text']}".encode()).hexdigest()
                repeated_rows.append(next_row)
        mixed_rows.extend(repeated_rows)
        print(json.dumps(write_dataset(mixed_rows, Path(args.mixed_output_dir), float(args.eval_fraction), "pocketpal_stage5_mixed"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
