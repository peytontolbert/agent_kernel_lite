#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    text_slots: dict[str, str] | None = None,
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
) -> str:
    if text_slots:
        slot_names = [slot_name for slot_name, slot_value in text_slots.items() if str(slot_value or "").strip()]
        slot_block = "\n".join(
            [
                "<AK_PROFILE> User text slots:",
                *[
                    f"<AK_SLOT> <AK_SLOT_NAME>={slot_name} <AK_SLOT_VALUE>={slot_value}"
                    for slot_name, slot_value in text_slots.items()
                ],
                f"Available placeholders for this turn: {', '.join(f'[[{slot_name}]]' for slot_name in slot_names)}.",
                "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            ]
        )
    else:
        slot_block = "<AK_PROFILE> User text slots: none"
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.",
            "<AK_TASK_CONTRACT>",
            f"Selected agent: {name}",
            f"Agent instruction: {instruction}",
            "Policies: retrieval=auto tools=ask_before_extensions actions=respond_or_ask",
            "Use this agent instruction as the task for the current user request. Do not answer as the base assistant unless this instruction asks for normal chat.",
            f"<AK_TASK_USER_REQUEST> {user_text}",
            "</AK_TASK_CONTRACT>",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            "<AK_CONTEXT> Saved user data: none",
            slot_block,
            f"<AK_CONTEXT> Stale selected paper context: {stale_context}",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _row(source_id: str, prompt: str, action: str, content: str, task_type: str, weight: float = 12.0) -> dict[str, Any]:
    decoder = {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": prompt,
        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{json.dumps(decoder, sort_keys=True)}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage10_agent_gate_repair",
        "task_type": task_type,
        "weight": float(weight),
    }


def _runtime_plain_chat_prompt(user_text: str) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
            "Return exactly this decision format: Action: respond, then Content: your direct answer.",
            "You are Agent Kernel Lite running entirely in this browser.",
            "Do not claim to execute, test, install, browse, or modify files.",
            "Mode: Chat. Reply like a helpful assistant. Use the strongest relevant evidence as support, not as the whole answer.",
            "Mode: Chat",
            "<AK_PROFILE> PocketPal saved slots:",
            "No saved PocketPal slots.",
            "<AK_PROFILE> Active PocketPal agent:",
            "No custom agent selected.",
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
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_HISTORY> Recent conversation:",
            "No recent conversation.",
            "<AK_READING_NOTES> Semantic reading notes:",
            "No strong semantic match in retrieved evidence.",
            "<AK_CONTEXT> Research context:",
            "No research context was retrieved.",
            "<AK_ANSWER> Answer scaffold:",
            "Answer directly.",
            f"<AK_USER> {user_text}",
            "Return a structured decision with action=respond.",
        ]
    )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plain_cases = [
        ("Hi how are you?", "I'm doing well. What would you like help with?"),
        ("hello", "Hi. What would you like help with?"),
        ("thanks", "You're welcome."),
    ]
    for repeat in range(220):
        for index, (user_text, content) in enumerate(plain_cases):
            rows.append(
                _row(
                    f"stage10_plain_{repeat:03d}_{index:02d}",
                    _runtime_plain_chat_prompt(user_text),
                    "respond",
                    content,
                    "runtime_plain_chat",
                    weight=13.0,
                )
            )

    saved_instruction = "Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question."
    saved_context = "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."
    for repeat in range(220):
        rows.append(
            _row(
                f"stage10_saved_use_{repeat:03d}",
                _prompt(
                    name="Saved Data Assistant",
                    instruction=saved_instruction,
                    user_text="what is my launch code",
                    text_slots={"SOURCE_TEXT": "what is my launch code", "DATA_CONTEXT": saved_context},
                    stale_context="Selected paper [P1]: unrelated transformer attention paper.",
                ),
                "respond",
                "I found this in your saved data: [[DATA_CONTEXT]]",
                "runtime_use_user_data",
                weight=13.0,
            )
        )
        rows.append(
            _row(
                f"stage10_saved_missing_{repeat:03d}",
                _prompt(
                    name="Saved Data Assistant",
                    instruction=saved_instruction,
                    user_text="what is my hotel confirmation code",
                    text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
                    stale_context="Retrieved evidence [1]: unrelated web-search result about insurance markets.",
                ),
                "ask_user",
                "I do not have relevant saved data for that. What should I look for or add?",
                "runtime_missing_user_data",
                weight=10.0,
            )
        )

    rewrite_instruction = "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it."
    rewrite_cases = [
        ("Hi how are you?", "Hello, I hope you are well.", {"SOURCE_TEXT": "Hi how are you?"}),
        ("hi how are you", "Hello, I hope you are well.", {"SOURCE_TEXT": "hi how are you"}),
        ("hello", "Hello, I hope you are well.", {"SOURCE_TEXT": "hello"}),
        ("thanks", "Thank you.", {"SOURCE_TEXT": "thanks"}),
        (
            "hey john i need the report by friday because the client is asking and we are behind",
            "Hi [[NAME]], Could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
            {"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
        ),
        (
            "please ask priya for the budget by tomorrow morning because finance needs it",
            "Hi [[NAME]], Could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
            {"NAME": "Priya", "ITEM": "budget", "DEADLINE": "tomorrow morning", "REASON": "Finance needs it"},
        ),
        (
            "tell lena to send the launch notes by noon because testflight is blocked",
            "Hi [[NAME]], Could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
            {"NAME": "Lena", "ITEM": "launch notes", "DEADLINE": "noon", "REASON": "TestFlight is blocked"},
        ),
    ]
    for repeat in range(220):
        for index, (user_text, content, slots) in enumerate(rewrite_cases):
            rows.append(
                _row(
                    f"stage10_rewrite_{repeat:03d}_{index:02d}",
                    _prompt(name="Professional Email Rewriter", instruction=rewrite_instruction, user_text=user_text, text_slots=slots),
                    "respond",
                    content,
                    "active_agent_rewrite",
                    weight=14.0,
                )
            )
        for index, user_text in enumerate(("rewrite this", "make this professional", "turn this into an email")):
            rows.append(
                _row(
                    f"stage10_rewrite_ask_{repeat:03d}_{index:02d}",
                    _prompt(name="Professional Email Rewriter", instruction=rewrite_instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text}),
                    "ask_user",
                    "What text should I rewrite?",
                    "active_agent_rewrite_ask",
                    weight=8.0,
                )
            )

    casual_instruction = "Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it."
    casual_cases = [
        ("How's it going?", "It's going well. What would you like help with?"),
        ("how is it going", "It's going well. What would you like help with?"),
        ("Hi how are you?", "I'm doing well. What would you like help with?"),
        ("thanks", "You're welcome."),
    ]
    for repeat in range(180):
        for index, (user_text, content) in enumerate(casual_cases):
            rows.append(
                _row(
                    f"stage10_casual_{repeat:03d}_{index:02d}",
                    _prompt(name="Casual Assistant", instruction=casual_instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text}),
                    "respond",
                    content,
                    "active_agent_casual",
                    weight=12.0,
                )
            )

    bullet_instruction = "Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot."
    bullet_cases = [
        ("Hi how are you?", "- Greeting: Hi, how are you?"),
        ("client asked for pricing and timeline", "- Client asked for pricing\n- Client asked for timeline"),
        ("launch is friday and QA needs the build today", "- Launch is friday\n- QA needs the build today"),
    ]
    for repeat in range(140):
        for index, (user_text, content) in enumerate(bullet_cases):
            rows.append(
                _row(
                    f"stage10_bullet_{repeat:03d}_{index:02d}",
                    _prompt(name="Bullet Summary Agent", instruction=bullet_instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text}),
                    "respond",
                    content,
                    "active_agent_bullet",
                    weight=10.0,
                )
            )
    return rows


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, eval_fraction: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pocketpal_stage10_agent_gate_repair_train.jsonl"
    eval_path = output_dir / "pocketpal_stage10_agent_gate_repair_eval.jsonl"
    eval_every = max(2, int(round(1.0 / max(0.01, min(0.5, float(eval_fraction))))))
    train_rows = [row for index, row in enumerate(rows) if index % eval_every != 0]
    eval_rows = [row for index, row in enumerate(rows) if index % eval_every == 0]
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    task_counts: dict[str, int] = {}
    for row in rows:
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / "pocketpal_stage10_agent_gate_repair_manifest.json").resolve()),
        "objective": "pocketpal_stage10_agent_gate_repair",
        "source_counts": {"pocketpal_stage10_agent_gate_repair": len(rows)},
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / "pocketpal_stage10_agent_gate_repair_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage10_agent_gate_repair_dataset")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(write_dataset(build_rows(), Path(args.output_dir), float(args.eval_fraction)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
