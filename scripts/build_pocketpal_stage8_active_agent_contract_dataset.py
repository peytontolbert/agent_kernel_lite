#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _compact(value: object, limit: int = 900) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].strip()


def _agent_prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
    text_slots: dict[str, str] | None = None,
) -> str:
    if text_slots:
        slot_names = [str(slot_name) for slot_name, slot_value in text_slots.items() if str(slot_value or "").strip()]
        text_slot_block = "\n".join(
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
        text_slot_block = "<AK_PROFILE> User text slots: none"
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
            text_slot_block,
            f"<AK_CONTEXT> Stale selected paper context: {stale_context}",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _runtime_prompt(user_text: str, active_agent: str = "No custom agent selected.", active_instruction: str = "") -> str:
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


def _web_agent_prompt(user_text: str) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> <AK_EXTENSION> <AK_WEB_SEARCH> PocketPal user-configured agent example.",
            "<AK_AGENT_ACTIVE>",
            "Agent name: Web Search Assistant",
            "Agent instruction: When the user asks for current, recent, online, or web-backed information, request the web_search extension. Do not invent results before the extension runs.",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: allow_extension_requests",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            "<AK_PROFILE> PocketPal installed tools:",
            "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
            "<AK_MAX_SOURCES>=5",
            "When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.",
            "<AK_CONTEXT> User data pointers:",
            "No saved user data sources.",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research context.",
            "If web search is needed, return action=extension_request and include proposal_metadata.extension_id/capability/query/max_sources/requires_user_approval.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _row(source_id: str, encoder_text: str, action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None, weight: float = 10.0) -> dict[str, Any]:
    proposal_metadata = {"task_type": task_type}
    if metadata:
        proposal_metadata.update(metadata)
    decoder = {"action": action, "content": content, "proposal_metadata": proposal_metadata}
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{json.dumps(decoder, sort_keys=True)}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage8_active_agent_contract",
        "task_type": task_type,
        "weight": float(weight),
    }


def _active_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases: list[tuple[str, str, str, str]] = [
        ("Professional Email Rewriter", "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.", "Hi how are you?", "Hello, I hope you are well."),
        ("Formal Tone Agent", "Rewrite the user's text in a formal, polished tone while preserving the meaning.", "Hi how are you?", "Hello, I hope you are well."),
        ("Friendly Tone Agent", "Rewrite the user's text in a friendly, warm tone while preserving the meaning.", "Hi how are you?", "Hi, how are you doing?"),
        ("Bullet Summary Agent", "Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.", "Hi how are you?", "- Greeting: Hi, how are you?"),
        ("Question Extractor", "Extract the question the user is asking. Do not answer the question.", "Hi how are you?", "Question: How are you?"),
        ("Grammar Fixer", "Fix grammar and clarity in the user's text. Preserve the original meaning.", "Hi how are you?", "Hi, how are you?"),
        ("Uppercase Formatter", "Convert the user's text to uppercase. Do not answer the text.", "Hi how are you?", "HI HOW ARE YOU?"),
        ("JSON Labeler", "Classify the user's text with a short label and return only the label.", "Hi how are you?", "greeting"),
        ("Professional Email Rewriter", "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.", "thanks for getting back to me", "Thank you for getting back to me."),
        ("Professional Email Rewriter", "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.", "need the invoice today because accounting closes tomorrow", "Hello, please send the invoice today because accounting closes tomorrow. Thank you."),
        ("Bullet Summary Agent", "Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.", "need the invoice today because accounting closes tomorrow", "- Invoice is needed today\n- Accounting closes tomorrow"),
        ("Action Item Extractor", "Extract concrete action items from the user's text. Do not answer as a chatbot.", "need the invoice today because accounting closes tomorrow", "- Send the invoice today"),
        ("Tone Softener", "Rewrite the user's text to sound calmer and more diplomatic.", "this is wrong and you need to fix it now", "This looks incorrect. Could you please fix it as soon as possible?"),
        ("Concise Rewriter", "Rewrite the user's text to be shorter while preserving the meaning.", "can you please let me know when the build is ready for review", "Please let me know when the build is ready for review."),
        ("Translator", "Translate the user's English text into Spanish. Do not answer the text.", "Good morning", "Buenos dias."),
    ]
    for repeat in range(220):
        for index, (name, instruction, user_text, content) in enumerate(cases):
            prompt = _agent_prompt(name=name, instruction=instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text})
            rows.append(_row(f"stage8_active_direct_{repeat:03d}_{index:02d}", prompt, "respond", content, "active_agent_instruction_contract", weight=13.0))
            active_agent, active_instruction = _agent_block(name, instruction)
            runtime = _runtime_prompt(user_text, active_agent=active_agent, active_instruction=active_instruction)
            rows.append(_row(f"stage8_active_runtime_{repeat:03d}_{index:02d}", runtime, "respond", content, "active_agent_instruction_contract", weight=11.0))
    ask_instruction = "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it."
    for repeat in range(160):
        for index, user_text in enumerate(("rewrite this", "make this professional", "turn this into an email")):
            prompt = _agent_prompt(name="Professional Email Rewriter", instruction=ask_instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text})
            rows.append(_row(f"stage8_active_ask_{repeat:03d}_{index:02d}", prompt, "ask_user", "What text should I rewrite?", "active_agent_instruction_contract", weight=12.0))
    return rows


def _web_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    queries = [
        "search the web for current TestFlight upload limits",
        "look up recent reviews for the Anker 737 power bank",
        "find current iOS TestFlight beta tester limits",
        "search today's weather in Austin",
        "look up the latest Python release notes",
    ]
    for repeat in range(220):
        for index, query in enumerate(queries):
            rows.append(
                _row(
                    f"stage8_web_direct_{repeat:03d}_{index:02d}",
                    _web_agent_prompt(query),
                    "extension_request",
                    "Requesting approval to search the web.",
                    "runtime_web_search_request",
                    metadata={
                        "extension_id": "web_search",
                        "capability": "web.search",
                        "query": query,
                        "max_sources": 5,
                        "requires_user_approval": True,
                    },
                    weight=12.0,
                )
            )
            active_agent, active_instruction = _agent_block(
                "Web Search Assistant",
                "When the user asks for current, recent, online, or web-backed information, request the web_search extension. Do not invent results before the extension runs.",
                "allow_extension_requests",
            )
            rows.append(
                _row(
                    f"stage8_web_runtime_{repeat:03d}_{index:02d}",
                    _runtime_prompt(query, active_agent=active_agent, active_instruction=active_instruction),
                    "extension_request",
                    "Requesting approval to search the web.",
                    "runtime_web_search_request",
                    metadata={
                        "extension_id": "web_search",
                        "capability": "web.search",
                        "query": query,
                        "max_sources": 5,
                        "requires_user_approval": True,
                    },
                    weight=12.0,
                )
            )
    return rows


def _general_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plain_cases = [
        ("Hi how are you?", "I'm doing well. What would you like help with?"),
        ("hello", "Hi. What would you like help with?"),
        ("thanks", "You're welcome."),
    ]
    for repeat in range(220):
        for index, (user_text, content) in enumerate(plain_cases):
            rows.append(_row(f"stage8_plain_{repeat:03d}_{index:02d}", _runtime_prompt(user_text), "respond", content, "runtime_plain_chat", weight=10.0))

    saved_context = "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."
    saved_instruction = "Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question."
    for repeat in range(180):
        rows.append(
            _row(
                f"stage8_saved_use_{repeat:03d}",
                _agent_prompt(
                    name="Saved Data Assistant",
                    instruction=saved_instruction,
                    user_text="what is my launch code",
                    stale_context="Selected paper [P1]: unrelated transformer attention paper.",
                    text_slots={"SOURCE_TEXT": "what is my launch code", "DATA_CONTEXT": saved_context},
                ),
                "respond",
                "I found this in your saved data: [[DATA_CONTEXT]]",
                "runtime_use_user_data",
                weight=11.0,
            )
        )
        rows.append(
            _row(
                f"stage8_saved_missing_{repeat:03d}",
                _agent_prompt(
                    name="Saved Data Assistant",
                    instruction=saved_instruction,
                    user_text="what is my hotel confirmation code",
                    stale_context="Retrieved evidence [1]: unrelated web-search result about insurance markets.",
                    text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
                ),
                "ask_user",
                "I do not have relevant saved data for that. What should I look for or add?",
                "runtime_missing_user_data",
                weight=11.0,
            )
        )

    for repeat in range(180):
        user_text = "vendor invoice INV-2048 is blocked until finance approves $1,200"
        rows.append(
            _row(
                f"stage8_source_echo_{repeat:03d}",
                _agent_prompt(
                    name="Source Echo Agent",
                    instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
                    user_text=user_text,
                    text_slots={"SOURCE_TEXT": user_text},
                ),
                "respond",
                "Source text: [[SOURCE_TEXT]]",
                "source_echo",
                weight=10.0,
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
    source_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in rows:
        source_counts[str(row.get("source_type", "unknown"))] = source_counts.get(str(row.get("source_type", "unknown")), 0) + 1
        task_counts[str(row.get("task_type", "unknown"))] = task_counts.get(str(row.get("task_type", "unknown")), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / f"{name}_manifest.json").resolve()),
        "objective": "pocketpal_stage8_active_agent_contract",
        "source_counts": source_counts,
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage8_active_agent_contract_dataset")
    parser.add_argument("--mixed-output-dir", default="")
    parser.add_argument("--include-manifest", action="append", default=[])
    parser.add_argument("--stage8-repeat", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()
    stage8_rows = _general_rows() + _active_rows() + _web_rows()
    print(json.dumps(write_dataset(stage8_rows, Path(args.output_dir), float(args.eval_fraction), "pocketpal_stage8_active_agent_contract"), indent=2, sort_keys=True))
    if str(args.mixed_output_dir).strip():
        mixed_rows: list[dict[str, Any]] = []
        for manifest_arg in args.include_manifest:
            mixed_rows.extend(_read_manifest_rows(Path(manifest_arg).expanduser().resolve()))
        repeated: list[dict[str, Any]] = []
        for repeat in range(max(1, int(args.stage8_repeat))):
            for row in stage8_rows:
                next_row = dict(row)
                next_row["source_id"] = f"{row['source_id']}:repeat_{repeat:02d}"
                next_row["example_id"] = hashlib.sha256(f"{next_row['source_id']}\n{row['encoder_text']}\n{row['decoder_text']}".encode()).hexdigest()
                repeated.append(next_row)
        mixed_rows.extend(repeated)
        print(json.dumps(write_dataset(mixed_rows, Path(args.mixed_output_dir), float(args.eval_fraction), "pocketpal_stage8_mixed"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
