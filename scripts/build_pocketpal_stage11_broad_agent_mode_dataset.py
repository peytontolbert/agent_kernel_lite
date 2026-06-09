#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage10_agent_gate_repair_dataset import _runtime_plain_chat_prompt, _row


INTENT_LABELS: dict[str, int] = {
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


def _with_intent(row: dict[str, Any], intent: str) -> dict[str, Any]:
    row["intent_label"] = intent
    row["intent_label_id"] = INTENT_LABELS.get(intent, -1)
    return row


def _agent_prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    text_slots: dict[str, str] | None = None,
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
) -> str:
    if text_slots:
        slot_names = [str(slot_name) for slot_name, slot_value in text_slots.items() if str(slot_value or "").strip()]
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


def _web_prompt(user_text: str) -> str:
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
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research context.",
            "If web search is needed, return action=extension_request and include proposal_metadata.extension_id/capability/query/max_sources/requires_user_approval.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _web_row(source_id: str, query: str, weight: float = 12.0) -> dict[str, Any]:
    decoder = {
        "action": "extension_request",
        "content": "Requesting approval to search the web.",
        "proposal_metadata": {
            "capability": "web.search",
            "extension_id": "web_search",
            "max_sources": 5,
            "query": query,
            "requires_user_approval": True,
            "task_type": "runtime_web_search_request",
        },
    }
    prompt = _web_prompt(query)
    return {
        "action": "extension_request",
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": prompt,
        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{json.dumps(decoder, sort_keys=True)}".encode()).hexdigest(),
        "intent_label": "web_search",
        "intent_label_id": INTENT_LABELS["web_search"],
        "source_id": source_id,
        "source_type": "pocketpal_stage11_broad_agent_mode",
        "task_type": "runtime_web_search_request",
        "weight": float(weight),
    }


def _agent_rows() -> list[tuple[str, str, str, str, str]]:
    return [
        ("Quick Dinner Planner", "Turn the user's ingredients and constraints into a concise dinner plan. Do not ask unless a required constraint is missing.", "I have chicken, rice, broccoli, and 30 minutes. Make a simple dinner plan.", "Cook rice, saute chicken, steam broccoli, and serve everything together with a quick sauce. Start the rice first, season and cook the chicken while it simmers, then steam the broccoli for the last 5 minutes.", "plan"),
        ("Quick Dinner Planner", "Turn the user's ingredients and constraints into a concise dinner plan. Do not ask unless a required constraint is missing.", "I have pasta, tomatoes, spinach, and 20 minutes.", "Make tomato-spinach pasta. Boil the pasta, simmer tomatoes with seasoning, wilt the spinach into the sauce, then toss everything together.", "plan"),
        ("Action Item Extractor", "Extract concrete action items from the user's text as concise bullets. Preserve owners and deadlines.", "Maria owns the launch slides, Devin fixes the login bug by Thursday, and I will send the customer email tomorrow.", "- Maria: prepare the launch slides\n- Devin: fix the login bug by Thursday\n- Me: send the customer email tomorrow", "action_items"),
        ("Action Item Extractor", "Extract concrete action items from the user's text as concise bullets. Preserve owners and deadlines.", "Sam will book the room today and Priya will send notes on Friday.", "- Sam: book the room today\n- Priya: send notes on Friday", "action_items"),
        ("Friendly Tone Agent", "Rewrite the user's text in a friendly, warm tone while preserving the meaning.", "Your update is late and this is blocking my release.", "Thanks for the update. I am still waiting on this, and it is currently blocking my release. Could you please send it as soon as you can?", "rewrite"),
        ("Friendly Tone Agent", "Rewrite the user's text in a friendly, warm tone while preserving the meaning.", "Send me the draft now.", "Could you please send me the draft when you have a chance?", "rewrite"),
        ("Spanish Translator", "Translate the user's English text into Spanish. Do not answer the text.", "Please send me the invoice tomorrow morning.", "Por favor, enviame la factura manana por la manana.", "translation"),
        ("Spanish Translator", "Translate the user's English text into Spanish. Do not answer the text.", "Good morning, I need the report today.", "Buenos dias, necesito el informe hoy.", "translation"),
        ("Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "The meeting starts at three this afternoon.", "La reunion empieza a las tres esta tarde.", "translation"),
        ("Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Can you call me after lunch?", "Puedes llamarme despues del almuerzo?", "translation"),
        ("French Translator", "Translate the user's English text into French. Return only the translation.", "Please review the proposal before Friday.", "Veuillez examiner la proposition avant vendredi.", "translation"),
        ("German Translator", "Translate the user's English text into German. Return only the translation.", "The package should arrive tomorrow.", "Das Paket sollte morgen ankommen.", "translation"),
        ("Formal Tone Agent", "Rewrite the user's text in a formal, polished tone while preserving the meaning.", "this is wrong and needs to be fixed today", "This is incorrect and needs to be fixed today.", "rewrite"),
        ("Concise Rewriter", "Rewrite the user's text to be shorter while preserving the meaning.", "Can you please let me know when the build is ready for review?", "Please let me know when the build is ready for review.", "rewrite"),
        ("Grammar Fixer", "Fix grammar and clarity in the user's text. Preserve the original meaning.", "we was waiting for the files", "We were waiting for the files.", "rewrite"),
        ("Summary Agent", "Summarize the user's text in one concise sentence.", "The team finished onboarding, resolved two payment bugs, and still needs legal approval before launch.", "The team finished onboarding and fixed payment bugs, but launch still needs legal approval.", "summary"),
        ("Title Generator", "Create a short title for the user's text. Return only the title.", "Notes about fixing web search approval and clickable result links in PocketPal.", "PocketPal Web Search Fixes", "title"),
        ("Checklist Builder", "Convert the user's request into a concise checklist.", "Prepare the TestFlight build, verify web search, check links, and send notes.", "- Prepare the TestFlight build\n- Verify web search\n- Check links\n- Send notes", "checklist"),
        ("Risk Reviewer", "Identify the main risks in the user's plan as concise bullets.", "We will ship the beta tonight without retesting the agent flow or web search links.", "- Agent flow may still fail without retesting\n- Web search links may be broken\n- Shipping tonight leaves little time for rollback", "risks"),
        ("JSON Labeler", "Classify the user's text with a short JSON object containing intent and tone.", "Could you make this sound more professional?", "{\"intent\":\"rewrite\",\"tone\":\"professional\"}", "json"),
        ("Priority Sorter", "Sort the user's tasks from highest to lowest priority.", "Fix login crash, update button color, write launch tweet.", "1. Fix login crash\n2. Write launch tweet\n3. Update button color", "ranking"),
        ("Question Extractor", "Extract the question the user is asking. Do not answer the question.", "I am wondering whether we can upload the build today.", "Question: Can we upload the build today?", "extraction"),
        ("Email Subject Agent", "Write a concise professional email subject line for the user's message.", "Following up about the contract review due Friday.", "Subject: Follow-Up on Friday Contract Review", "subject"),
        ("Brainstorm Agent", "Generate three concise ideas that fit the user's request.", "Ideas for making PocketPal feel more personal.", "1. Let users create custom agents\n2. Add local memory collections\n3. Offer per-agent tone and tool settings", "brainstorm"),
        ("Professional Email Rewriter", "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.", "hey john i need the report by friday because the client is asking and we are behind", "Hi John, Could you please send the report by friday? The client is asking. Thank you.", "rewrite"),
        ("Source Echo Agent", "Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.", "vendor invoice INV-2048 is blocked until finance approves $1,200", "Source text: vendor invoice INV-2048 is blocked until finance approves $1,200", "source_echo"),
    ]


def _hard_negative_agent_rows() -> list[tuple[str, str, str, str, str]]:
    return [
        ("Clarifying Question Agent", "Ask one concise clarifying question when the user's request is missing the text, recipient, destination language, or needed constraint. Do not rewrite or translate.", "make this better", "What text would you like me to improve?", "ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question when the user's request is missing the text, recipient, destination language, or needed constraint. Do not rewrite or translate.", "translate it", "What text should I translate, and what language should I translate it into?", "ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question before acting when the user gives an ambiguous editing request. Do not assume missing details.", "send the email for me", "Who should receive the email, and what should it say?", "ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question before acting when the request depends on unavailable saved data.", "what was the hotel code", "I do not have the hotel code in the provided data. Where should I look for it?", "ask_user"),
        ("Question Extractor", "Extract the explicit question from the user's text. Do not answer, rewrite, or translate.", "Could you please send me the invoice tomorrow morning?", "Question: Could you please send me the invoice tomorrow morning?", "extraction"),
        ("Question Extractor", "Extract the explicit question from the user's text. Do not answer, rewrite, or translate.", "I need to know whether Maria can finish the slides by Friday.", "Question: Can Maria finish the slides by Friday?", "extraction"),
        ("Entity Extractor", "Extract names, dates, amounts, and objects from the user's text as compact bullets. Do not rewrite or translate.", "Hi John, please send the invoice for $1,200 by Friday.", "- Name: John\n- Object: invoice\n- Amount: $1,200\n- Date: Friday", "extraction"),
        ("Entity Extractor", "Extract names, dates, amounts, and objects from the user's text as compact bullets. Do not rewrite or translate.", "Priya booked Austin for May 18 and paid $420.", "- Name: Priya\n- Place: Austin\n- Date: May 18\n- Amount: $420", "extraction"),
        ("Brainstorm Agent", "Generate three concise ideas for the user's request. Do not rewrite, translate, or answer as a chat assistant.", "Ways to make a local personal agent feel useful every day.", "1. Let it remember reusable user preferences\n2. Give each custom agent clear tool permissions\n3. Surface quick actions from recent context", "brainstorm"),
        ("Brainstorm Agent", "Generate three concise ideas for the user's request. Do not rewrite, translate, or answer as a chat assistant.", "Ideas for a better web search flow inside PocketPal.", "1. Show source cards before synthesis\n2. Let users tune max source count\n3. Save useful results into local retrieval", "brainstorm"),
        ("Brainstorm Agent", "Generate three concise subject line ideas. Do not choose one unless asked.", "Follow-up about Friday's client report.", "1. Follow-Up on Friday Client Report\n2. Client Report Status for Friday\n3. Checking In on the Friday Report", "brainstorm"),
        ("Brainstorm Agent", "Generate three concise product names for the user's concept. Do not translate.", "A private phone assistant that uses local memory and tools.", "1. PocketPal\n2. LocalMate\n3. Sidekick Vault", "brainstorm"),
        ("Summary Agent", "Summarize the user's text in one concise sentence. Do not rewrite, translate, or invent details.", "Please send me the invoice tomorrow morning because finance needs it before lunch.", "Finance needs the invoice tomorrow morning before lunch.", "summary"),
        ("Summary Agent", "Summarize the user's text in one concise sentence. Do not rewrite, translate, or invent details.", "Maria owns launch slides, Devin fixes login by Thursday, and Sam sends the customer email tomorrow.", "Maria, Devin, and Sam each have launch tasks with near-term deadlines.", "summary"),
        ("Checklist Builder", "Convert the user's text into a checklist. Do not rewrite as prose or translate.", "Please review the proposal, confirm the budget, and send the invoice tomorrow.", "- Review the proposal\n- Confirm the budget\n- Send the invoice tomorrow", "checklist"),
        ("JSON Labeler", "Return a compact JSON object classifying the user's request. Do not rewrite or translate.", "Please make this sound more professional for a client.", "{\"intent\":\"rewrite\",\"tone\":\"professional\",\"audience\":\"client\"}", "json"),
        ("Priority Sorter", "Rank the user's tasks by urgency. Do not rewrite or translate.", "Fix the crash, update the docs, choose a nicer button color.", "1. Fix the crash\n2. Update the docs\n3. Choose a nicer button color", "ranking"),
        ("Risk Reviewer", "List the main risks in the user's plan as concise bullets. Do not rewrite or translate.", "Ship the TestFlight build without retesting the agent flow or web search links.", "- Agent flow may still fail in testing\n- Web search links may be broken\n- The build may need a fast rollback", "risks"),
        ("Spanish Translator", "Translate the user's English text into Spanish. Return only the translation.", "Could you please send me the invoice tomorrow morning?", "Podrias enviarme la factura manana por la manana?", "translation"),
        ("French Translator", "Translate the user's English text into French. Return only the translation.", "Could you please send me the invoice tomorrow morning?", "Pourriez-vous m'envoyer la facture demain matin?", "translation"),
    ]


def build_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for index, (name, instruction, user_text, content, task_type) in enumerate(_agent_rows()):
            rows.append(
                _with_intent(
                    _row(
                        f"stage11_agent_{repeat:03d}_{index:03d}",
                        _agent_prompt(name=name, instruction=instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text}),
                        "respond",
                        content,
                        f"broad_agent_{task_type}",
                        weight=12.0,
                    ),
                    task_type,
                )
            )
    for repeat in range(max(1, int(repeats))):
        for index, (name, instruction, user_text, content, task_type) in enumerate(_hard_negative_agent_rows()):
            rows.append(
                _with_intent(
                    _row(
                        f"stage11_hard_negative_{repeat:03d}_{index:03d}",
                        _agent_prompt(name=name, instruction=instruction, user_text=user_text, text_slots={"SOURCE_TEXT": user_text}),
                        "ask_user" if task_type == "ask_user" else "respond",
                        content,
                        f"broad_agent_hard_negative_{task_type}",
                        weight=18.0,
                    ),
                    task_type,
                )
            )
    plain_cases = [
        ("Hi how are you?", "I'm doing well. What would you like help with?"),
        ("hello", "Hi. What would you like help with?"),
        ("thanks", "You're welcome."),
        ("what can you do?", "I can chat, rewrite text, use your saved data, and request web search when current information is needed."),
    ]
    for repeat in range(max(1, int(repeats // 2))):
        for index, (user_text, content) in enumerate(plain_cases):
            rows.append(
                _with_intent(
                    _row(f"stage11_plain_{repeat:03d}_{index:03d}", _runtime_plain_chat_prompt(user_text), "respond", content, "runtime_plain_chat", weight=9.0),
                    "casual",
                )
            )
    gate_retention_cases = [
        (
            "Professional Email Rewriter",
            "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            "Hi how are you?",
            {"SOURCE_TEXT": "Hi how are you?"},
            "Hello, I hope you are well.",
            "rewrite",
        ),
        (
            "Bullet Summary Agent",
            "Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
            "Hi how are you?",
            {"SOURCE_TEXT": "Hi how are you?"},
            "- Greeting asking how the recipient is doing",
            "summary",
        ),
        (
            "Professional Email Rewriter",
            "Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            "hey john i need the report by friday because the client is asking and we are behind",
            {"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
            "Hi John, Could you please send the report by friday? The client is asking. Thank you.",
            "rewrite",
        ),
    ]
    for repeat in range(max(1, int(repeats // 2))):
        for index, (name, instruction, user_text, slots, content, intent) in enumerate(gate_retention_cases):
            rows.append(
                _with_intent(
                    _row(
                        f"stage11_gate_retention_{repeat:03d}_{index:03d}",
                        _agent_prompt(name=name, instruction=instruction, user_text=user_text, text_slots=slots),
                        "respond",
                        content,
                        f"runtime_gate_retention_{intent}",
                        weight=16.0,
                    ),
                    intent,
                )
            )
    saved_instruction = "Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question."
    saved_context = "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."
    for repeat in range(max(1, int(repeats // 2))):
        rows.append(
            _with_intent(
                _row(
                    f"stage11_saved_use_{repeat:03d}",
                    _agent_prompt(
                        name="Saved Data Assistant",
                        instruction=saved_instruction,
                        user_text="what is my launch code",
                        text_slots={"SOURCE_TEXT": "what is my launch code", "DATA_CONTEXT": saved_context},
                        stale_context="Selected paper [P1]: unrelated transformer attention paper.",
                    ),
                    "respond",
                    "I found this in your saved data: [[DATA_CONTEXT]]",
                    "runtime_use_user_data",
                    weight=10.0,
                ),
                "saved_data",
            )
        )
        rows.append(
            _with_intent(
                _row(
                    f"stage11_saved_missing_{repeat:03d}",
                    _agent_prompt(
                        name="Saved Data Assistant",
                        instruction=saved_instruction,
                        user_text="what is my hotel confirmation code",
                        text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
                        stale_context="Retrieved evidence [1]: unrelated web-search result about insurance markets.",
                    ),
                    "ask_user",
                    "I do not have relevant saved data for that. What should I look for or add?",
                    "runtime_missing_user_data",
                    weight=9.0,
                ),
                "ask_user",
            )
        )
    web_queries = [
        "search the web for current iPhone 17 release rumors",
        "look up recent reviews for the Anker 737 power bank",
        "find current iOS TestFlight beta tester limits",
        "search today's weather in Austin",
        "look up the latest Python release notes",
    ]
    for repeat in range(max(1, int(repeats // 2))):
        for index, query in enumerate(web_queries):
            rows.append(_web_row(f"stage11_web_{repeat:03d}_{index:03d}", query, weight=10.0))
    return rows


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, eval_fraction: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pocketpal_stage11_broad_agent_mode_train.jsonl"
    eval_path = output_dir / "pocketpal_stage11_broad_agent_mode_eval.jsonl"
    eval_mod = max(2, int(round(1.0 / max(0.01, min(0.5, float(eval_fraction))))))
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(str(row.get("example_id", "")).encode()).hexdigest()
        bucket = int(digest[:8], 16) % eval_mod
        if bucket == 0:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    task_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for row in rows:
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
        intent = str(row.get("intent_label", "") or "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / "pocketpal_stage11_broad_agent_mode_manifest.json").resolve()),
        "objective": "pocketpal_stage11_broad_agent_mode",
        "intent_labels": INTENT_LABELS,
        "intent_label_counts": intent_counts,
        "source_counts": {"pocketpal_stage11_broad_agent_mode": len(rows)},
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / "pocketpal_stage11_broad_agent_mode_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage11_broad_agent_mode_dataset")
    parser.add_argument("--repeats", type=int, default=220)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(write_dataset(build_rows(int(args.repeats)), Path(args.output_dir), float(args.eval_fraction)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
