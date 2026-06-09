#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.pocketpal_source_slots import compile_source_slots
    from scripts.pocketpal_source_slots import pointerize_exact_text
    from scripts.pocketpal_source_slots import source_slot_metadata
    from scripts.pocketpal_source_slots import source_slots_encoder_block
except ModuleNotFoundError:
    from pocketpal_source_slots import compile_source_slots
    from pocketpal_source_slots import pointerize_exact_text
    from pocketpal_source_slots import source_slot_metadata
    from pocketpal_source_slots import source_slots_encoder_block


AGENT_CASES: list[dict[str, Any]] = [
    {
        "agent_name": "Plain Rewriter",
        "agent_instruction": "Rewrite the user's text in clearer plain English. Preserve meaning and do not add commentary.",
        "user_request": "This update is kinda messy but I think users can figure it out after a while.",
        "content": "This update is somewhat unclear, but users should be able to understand it with time.",
        "retrieval_policy": "auto",
    },
    {
        "agent_name": "Checklist Planner",
        "agent_instruction": "Convert the user's request into a short actionable checklist. Keep each item concrete.",
        "user_request": "I need to prep the beta release and make sure the app doesn't break for testers.",
        "content": "- Confirm the beta build installs cleanly.\n- Smoke-test the main tester workflows.\n- Verify crash/log reporting is available.\n- Write short tester notes for known limitations.",
        "retrieval_policy": "auto",
    },
    {
        "agent_name": "Bug Triage",
        "agent_instruction": "Triage the user's bug report. Return likely cause, severity, and next diagnostic step.",
        "user_request": "The active agent button says active but the chat ignores the agent.",
        "content": "Likely cause: the selected agent state is visible in the UI but not being treated as the primary prompt contract during generation. Severity: high for agent workflows. Next diagnostic step: inspect the compiled prompt and confirm the active agent instruction appears before generic assistant behavior.",
        "retrieval_policy": "auto",
    },
    {
        "agent_name": "Source-Grounded Helper",
        "agent_instruction": "Answer using saved user data first. If the saved data is insufficient, say what is missing.",
        "user_request": "What budget did I set for the launch ads?",
        "content": "Your saved launch note says the launch ads budget is $1,200.",
        "retrieval_policy": "local_first",
        "user_data": "Launch notes: target release Friday. Launch ads budget: $1,200. Main channel: short video ads.",
        "retrieval_influenced": True,
    },
    {
        "agent_name": "Clarifier",
        "agent_instruction": "Ask one concise clarifying question when the user's request lacks the necessary input.",
        "user_request": "Make this better.",
        "action": "ask_user",
        "content": "What text should I improve?",
        "retrieval_policy": "auto",
    },
    {
        "agent_name": "Friendly Helper",
        "agent_instruction": "Be a concise friendly assistant. Answer casual messages normally and do not use stale paper context unless the user asks about papers.",
        "user_request": "How's it going?",
        "content": "It's going well. What would you like help with?",
        "retrieval_policy": "auto",
        "paper_context": "Selected paper [P1]: synthetic retrieval notes about unrelated optimization experiments.",
    },
    {
        "agent_name": "User Configured Text Agent",
        "agent_instruction": "Improve or reword text when the user provides text. If the user has not provided text to edit, ask for it.",
        "user_request": "How's it going?",
        "action": "ask_user",
        "content": "I'm ready. Send me the text you want improved or reworded.",
        "retrieval_policy": "auto",
        "paper_context": "Selected paper [P1]: stale research context that is unrelated to the user's current request.",
    },
    {
        "agent_name": "Calendar Assistant",
        "agent_instruction": "Prepare calendar actions, but request approval before using an extension.",
        "user_request": "Put launch review on my calendar tomorrow at 3.",
        "action": "extension_request",
        "content": "I can create that calendar event after you approve the calendar extension action.",
        "retrieval_policy": "none",
        "tool_policy": "ask_before_extensions",
        "action_policy": "allow_extension_requests",
        "proposal_metadata": {
            "task_type": "agent_extension_request",
            "extension_id": "calendar",
            "capability": "calendar.create_event",
            "requires_user_approval": True,
        },
    },
    {
        "agent_name": "Research Scout",
        "agent_instruction": "When the user asks for current papers or sources, gather context before answering.",
        "user_request": "Find recent papers about tiny on-device agent controllers.",
        "action": "gather_context",
        "content": "tiny on-device agent controllers",
        "retrieval_policy": "always",
    },
]


AGENT_INSTRUCTION_FAMILIES: list[dict[str, Any]] = [
    {
        "name": "Reworder",
        "instruction": "Reword the provided text. Preserve meaning and facts. If no editable text is provided, ask for the text.",
        "requests": [
            ("This wording sounds kinda rough and too long for testers.", "This wording is too long and unclear for testers.", "respond"),
            ("How's it going?", "Send me the text you want reworded.", "ask_user"),
            ("Make this better.", "What text should I improve?", "ask_user"),
        ],
    },
    {
        "name": "Friendly Assistant",
        "instruction": "Answer casual messages naturally. Keep replies short. Do not mention papers unless the user asks about research.",
        "requests": [
            ("How's it going?", "It's going well. What would you like help with?", "respond"),
            ("Thanks", "You're welcome.", "respond"),
            ("Can you help me?", "Yes. What do you want to work on?", "respond"),
        ],
    },
    {
        "name": "Checklist Agent",
        "instruction": "Turn the user's goal into a short checklist with concrete next actions.",
        "requests": [
            ("Prepare TestFlight for the next build.", "- Bump the iOS build number.\n- Sync the web assets into the mobile app.\n- Run the mobile checks.\n- Upload the archive to App Store Connect.", "respond"),
            ("Plan my morning app work.", "- Pick the highest-priority bug.\n- Reproduce it once.\n- Patch the smallest failing path.\n- Run the focused test.", "respond"),
        ],
    },
    {
        "name": "Clarifying Agent",
        "instruction": "Ask one concise clarifying question when the request is missing required input.",
        "requests": [
            ("Fix it.", "What specifically is broken?", "ask_user"),
            ("Rewrite this.", "What text should I rewrite?", "ask_user"),
            ("Make the agent better.", "What behavior should the agent improve first?", "ask_user"),
        ],
    },
    {
        "name": "Saved Data Agent",
        "instruction": "Use saved user data first. If saved data does not answer the question, say what is missing.",
        "requests": [
            ("What budget did I set for launch ads?", "Your saved launch notes say the launch ads budget is $1,200.", "respond"),
            ("Which channel did I pick?", "Your saved launch notes say the main channel is short video ads.", "respond"),
        ],
        "user_data": "Launch notes: launch ads budget $1,200. Main channel: short video ads. Release target: Friday.",
    },
]


def _generated_agent_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    stale_contexts = [
        "",
        "Selected paper [P1]: unrelated optimization notes from a previous research turn.",
        "Retrieved evidence [1]: unrelated web-search result about insurance markets.",
        "Selected paper [P1]: stale source about neural retrieval that the current user did not ask about.",
    ]
    for family_index, family in enumerate(AGENT_INSTRUCTION_FAMILIES):
        for request_index, (request, content, action) in enumerate(family["requests"]):
            for stale_index, stale_context in enumerate(stale_contexts):
                case: dict[str, Any] = {
                    "agent_name": f"{family['name']} {family_index + 1}",
                    "agent_instruction": family["instruction"],
                    "user_request": request,
                    "action": action,
                    "content": content,
                    "retrieval_policy": "local_first" if family.get("user_data") else "auto",
                    "paper_context": stale_context,
                    "user_data": family.get("user_data", ""),
                    "retrieval_influenced": bool(family.get("user_data") and action == "respond"),
                }
                cases.append(case)
    return cases


def _slot_instruction_cases() -> list[dict[str, Any]]:
    """Synthetic slot curriculum for user-defined agents, not a single task."""
    cases: list[dict[str, Any]] = []
    stale_contexts = [
        "",
        "Selected paper [P1]: unrelated optimization notes from a previous research turn.",
        "Retrieved evidence [1]: unrelated web-search result about insurance markets.",
    ]
    business_rewrites = [
        (
            "ask Alex for the launch notes by Thursday because review is starting",
            "Hi Alex,\n\nCould you please send the launch notes by Thursday? The review is starting soon.\n\nThank you.",
        ),
        (
            "tell Priya the demo moved to 11 and ask her to send the updated slides",
            "Hi Priya,\n\nThe demo has moved to 11:00. Could you please send the updated slides?\n\nThank you.",
        ),
        (
            "ask the design team if the new mockups are ready because engineering is blocked",
            "Hi Design Team,\n\nCould you please confirm whether the new mockups are ready? Engineering is currently blocked.\n\nThank you.",
        ),
        (
            "tell Omar the contract looks good but legal needs one more day",
            "Hi Omar,\n\nThe contract looks good, but Legal needs one more day to complete the review.\n\nThank you.",
        ),
    ]
    for index, (request, content) in enumerate(business_rewrites):
        for stale_index, stale_context in enumerate(stale_contexts):
            cases.append(
                {
                    "agent_name": f"Slot Tone Agent {index + 1}",
                    "agent_instruction": (
                        "Rewrite the user's provided text in the requested tone. Preserve names, dates, facts, and intent. "
                        "Do not use unrelated context. If there is no editable text, ask for it."
                    ),
                    "user_request": request,
                    "action": "respond",
                    "content": content,
                    "retrieval_policy": "auto",
                    "paper_context": stale_context,
                }
            )

    names = ["Alex", "Priya", "Omar", "Maya", "Lee", "Nina", "Sam", "Taylor"]
    artifacts = [
        ("launch notes", "the review is starting soon"),
        ("updated slides", "the demo was moved earlier"),
        ("budget summary", "planning is blocked"),
        ("prototype link", "testing starts tomorrow"),
        ("contract notes", "Legal needs the update"),
        ("release checklist", "the build review is today"),
    ]
    deadlines = ["Thursday", "tomorrow morning", "before noon", "by end of day", "early next week"]
    business_instructions = [
        (
            "Rewrite the user's text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            "email",
        ),
        (
            "Rewrite the user's text as a professional message. Preserve names, deadlines, facts, and intent. If there is no editable text, ask for it.",
            "message",
        ),
        (
            "Rewrite the user's text as a polite business email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            "email",
        ),
        (
            "Rewrite the user's text in a concise workplace tone. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            "status",
        ),
    ]
    generated_index = 0
    for instruction, shape in business_instructions:
        for name in names:
            for artifact, reason in artifacts:
                for deadline in deadlines:
                    request = f"ask {name} for the {artifact} {deadline} because {reason}"
                    if shape == "status":
                        content = f"Please send the {artifact} {deadline}; {reason}."
                    else:
                        content = (
                            f"Hi {name},\n\n"
                            f"Could you please send the {artifact} {deadline}? {reason[0].upper()}{reason[1:]}.\n\n"
                            "Thank you."
                        )
                    cases.append(
                        {
                            "agent_name": f"Parameterized Text Agent {generated_index % 17}",
                            "agent_instruction": instruction,
                            "user_request": request,
                            "action": "respond",
                            "content": content,
                            "retrieval_policy": "auto",
                            "paper_context": stale_contexts[generated_index % len(stale_contexts)],
                        }
                    )
                    generated_index += 1

    copy_names = [
        "alex", "priya", "omar", "maya", "lee", "nina", "sam", "taylor", "riley", "jordan",
        "avery", "morgan", "casey", "devon", "elena", "marco", "lena", "noah", "zoe", "victor",
        "harper", "quinn", "drew", "mina", "owen", "yasmin", "leo", "anika", "chris", "renee",
    ]
    copy_artifacts = [
        "report", "budget summary", "release notes", "prototype link", "invoice", "contract draft",
        "client update", "launch plan", "design mockups", "test plan", "status memo", "roadmap",
        "risk list", "pricing sheet", "meeting notes", "deployment checklist",
    ]
    copy_deadlines = [
        "friday", "thursday", "tomorrow morning", "before noon", "by end of day", "monday at 10",
        "early next week", "after the standup",
    ]
    copy_reasons = [
        "the client is asking",
        "the review is starting",
        "engineering is blocked",
        "finance needs it",
        "the vendor is waiting",
        "testing starts soon",
        "leadership wants an update",
        "legal needs one more pass",
    ]
    for name_index, name in enumerate(copy_names):
        for artifact_index, artifact in enumerate(copy_artifacts):
            for deadline_index, deadline in enumerate(copy_deadlines):
                reason = copy_reasons[(name_index + artifact_index + deadline_index) % len(copy_reasons)]
                request_forms = [
                    f"ask {name} for the {artifact} by {deadline} because {reason}",
                    f"hey {name} i need the {artifact} by {deadline} because {reason}",
                    f"tell {name} we need the {artifact} by {deadline} since {reason}",
                    f"can you ask {name} to send the {artifact} by {deadline} because {reason}",
                ]
                pretty_name = name.capitalize()
                for request in request_forms:
                    content = (
                        f"Hi {pretty_name},\n\n"
                        f"Could you please send the {artifact} by {deadline}? {reason.capitalize()}.\n\n"
                        "Thank you."
                    )
                    cases.append(
                        {
                            "agent_name": f"Copy-Preserving Text Agent {generated_index % 31}",
                            "agent_instruction": (
                                "Rewrite the user's provided text as a professional email. Preserve every name, object, "
                                "deadline, reason, and fact from the user text. Do not substitute example names or details. "
                                "If there is no editable text, ask for it."
                            ),
                            "user_request": request,
                            "action": "respond",
                            "content": content,
                            "retrieval_policy": "auto",
                            "paper_context": stale_contexts[generated_index % len(stale_contexts)],
                        }
                    )
                    generated_index += 1

    for name_index, name in enumerate(copy_names):
        for original_time in ["1pm", "2pm", "2:30pm", "3pm", "4:15pm"]:
            for alternative in ["tomorrow morning", "later today", "friday afternoon", "monday at 10"]:
                request = f"tell {name} i can't make the {original_time} today and ask if {alternative} works"
                content = (
                    f"Hi {name.capitalize()},\n\n"
                    f"I am unable to make the {original_time.upper()} meeting today. Would {alternative} work for you instead?\n\n"
                    "Thank you."
                )
                cases.append(
                    {
                        "agent_name": f"Copy-Preserving Text Agent {generated_index % 31}",
                        "agent_instruction": (
                            "Rewrite the user's provided text as a professional message. Preserve every name, time, "
                            "alternative, and fact from the user text. Do not substitute example names or details. "
                            "If there is no editable text, ask for it."
                        ),
                        "user_request": request,
                        "action": "respond",
                        "content": content,
                        "retrieval_policy": "auto",
                        "paper_context": stale_contexts[generated_index % len(stale_contexts)],
                    }
                )
                generated_index += 1

    reschedule_names = ["Riley", "Jordan", "Avery", "Morgan", "Casey", "Devon"]
    original_times = ["1:00 PM", "2:30 PM", "3:00 PM", "4:15 PM"]
    alternatives = ["tomorrow morning", "later this afternoon", "Friday afternoon", "Monday at 10:00"]
    for instruction, shape in business_instructions:
        for name in reschedule_names:
            for original_time in original_times:
                for alternative in alternatives:
                    request = f"tell {name} i can't make the {original_time} meeting and ask if {alternative} works"
                    if shape == "status":
                        content = f"I cannot make the {original_time} meeting. Would {alternative} work instead?"
                    else:
                        content = (
                            f"Hi {name},\n\n"
                            f"I am unable to make the {original_time} meeting. Would {alternative} work for you instead?\n\n"
                            "Thank you."
                        )
                    cases.append(
                        {
                            "agent_name": f"Parameterized Text Agent {generated_index % 17}",
                            "agent_instruction": instruction,
                            "user_request": request,
                            "action": "respond",
                            "content": content,
                            "retrieval_policy": "auto",
                            "paper_context": stale_contexts[generated_index % len(stale_contexts)],
                        }
                    )
                    generated_index += 1

    concise_rewrites = [
        ("the beta is late because two login bugs are still open", "Beta is delayed: two login bugs are still open."),
        ("we need to cut scope or the friday build probably misses testflight", "Cut scope or the Friday build may miss TestFlight."),
        ("the upload worked but app store processing has not finished yet", "Upload succeeded; App Store processing is still pending."),
    ]
    for request, content in concise_rewrites:
        cases.append(
            {
                "agent_name": "Concise Status Agent",
                "agent_instruction": "Rewrite the user's text as one concise status update. Preserve the facts and do not add commentary.",
                "user_request": request,
                "action": "respond",
                "content": content,
                "retrieval_policy": "auto",
            }
        )

    extraction_cases = [
        (
            "Book design review Monday, send Maya the prototype, and check whether TestFlight processed.",
            "- Book design review Monday.\n- Send Maya the prototype.\n- Check whether TestFlight processed.",
        ),
        (
            "Fix the agent active state, rerun the smoke test, then upload build 8.",
            "- Fix the agent active state.\n- Rerun the smoke test.\n- Upload build 8.",
        ),
    ]
    for request, content in extraction_cases:
        cases.append(
            {
                "agent_name": "Action Extractor",
                "agent_instruction": "Extract concrete action items from the user's text as a short checklist.",
                "user_request": request,
                "action": "respond",
                "content": content,
                "retrieval_policy": "auto",
            }
        )

    for request in [
        "How's it going?",
        "rewrite this",
        "make it sound better",
        "can you turn this into an email",
        "fix my wording",
        "write the professional version",
        "make this into a business note",
        "can you clean this up",
    ]:
        for instruction in [
            "Rewrite or transform the text the user provides. If the user does not provide editable text, ask for the text.",
            "Rewrite the user's text as a professional message. If no editable text is provided, ask for the text.",
            "Turn the user's provided text into a concise workplace update. If no editable text is provided, ask for the text.",
        ]:
            cases.append(
                {
                    "agent_name": "Generic Rewrite Agent",
                    "agent_instruction": instruction,
                    "user_request": request,
                    "action": "ask_user",
                    "content": "What text should I rewrite?",
                    "retrieval_policy": "auto",
                    "paper_context": "Selected paper [P1]: stale research context that is unrelated to this request.",
                }
            )
    missing_input_requests = [
        "how are you",
        "how's it going",
        "can you help",
        "make it professional",
        "turn this into a note",
        "rewrite as an email",
        "clean up the wording",
        "make this sound better",
        "professional version please",
        "fix this",
    ]
    missing_input_instructions = [
        "Rewrite the user's provided text as a professional message. If the user does not provide editable text, ask for the text.",
        "Rewrite the user's text as a professional email. Preserve the meaning and facts. If no editable text is provided, ask for the text.",
        "Rewrite the user's text in a concise workplace tone. If the user does not provide editable text, ask for the text.",
        "Transform the user's provided text according to the requested tone. If there is no editable text, ask for it.",
        "Edit the user's supplied draft while preserving facts. If no draft is supplied, ask for the draft.",
    ]
    for instruction_index, instruction in enumerate(missing_input_instructions):
        for request_index, request in enumerate(missing_input_requests):
            for stale_context in stale_contexts:
                cases.append(
                    {
                        "agent_name": f"Missing Input Transform Agent {instruction_index}",
                        "agent_instruction": instruction,
                        "user_request": request,
                        "action": "ask_user",
                        "content": "What text should I rewrite?",
                        "retrieval_policy": "auto",
                        "paper_context": stale_context,
                    }
                )
    return cases


def _source_pointer_cases() -> list[dict[str, Any]]:
    """Generic drills that teach source-slot copying without binding to one agent behavior."""
    cases: list[dict[str, Any]] = []
    instructions = [
        "Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
        "Extract the source text exactly as provided. Do not normalize names, dates, values, or punctuation.",
        "Prepare a response that includes the user's original wording verbatim, then add one concise note.",
    ]
    requests = [
        "maya needs the design review follow up tomorrow at 11",
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "send Priya the updated slides before noon because the demo moved",
        "the prototype link is ready but battery usage is higher than expected",
        "legal needs one more day on the contract draft for Omar",
    ]
    for instruction_index, instruction in enumerate(instructions):
        for request_index, request in enumerate(requests):
            cases.append(
                {
                    "agent_name": f"Source Slot Agent {instruction_index + 1}",
                    "agent_instruction": instruction,
                    "user_request": request,
                    "action": "respond",
                    "content": f"Source text: {request}\nNote: preserved exactly from the user's message.",
                    "retrieval_policy": "auto",
                    "paper_context": "",
                    "proposal_metadata": {"task_type": "source_slot_copy"},
                }
            )
    return cases


def _plain_chat_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    prompts = [
        ("Hi how are you?", "I'm doing well. What would you like help with?"),
        ("How's it going?", "It's going well. What would you like to work on?"),
        ("Can you help me?", "Yes. What do you want to work on?"),
        ("Thanks", "You're welcome."),
        ("What can you do?", "I can help chat, rewrite text, summarize, search when tools are available, and work with your local data."),
    ]
    for index, (user_text, content) in enumerate(prompts):
        encoder = "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
                "You are Agent Kernel Lite running entirely in this browser.",
                "Mode: Chat. Reply like a helpful assistant. Do not use stale research context unless the user asks about it.",
                "<AK_PROFILE> Active PocketPal agent:",
                "No custom agent selected.",
                "<AK_PROFILE> PocketPal installed tools:",
                "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
                "<AK_CONTEXT> User data pointers:",
                "No saved user data sources.",
                "<AK_CONTEXT> Research context:",
                "No research context was retrieved.",
                f"<AK_USER> {user_text}",
                "Return a structured decision with action=respond.",
            ]
        )
        cases.append(
            {
                "agent_name": f"Runtime Plain Chat {index}",
                "agent_instruction": "No custom agent selected.",
                "user_request": user_text,
                "action": "respond",
                "content": content,
                "encoder_text": encoder,
                "proposal_metadata": {"task_type": "runtime_plain_chat"},
                "weight": 8.0,
            }
        )
    return cases


def _web_extension_cases() -> list[dict[str, Any]]:
    topics = [
        "current TestFlight upload limits",
        "latest Apple developer TestFlight processing status",
        "current pricing for iCloud storage",
        "recent news about on-device AI models",
        "today's weather in San Francisco",
        "current App Store Connect status",
        "latest SwiftUI release notes",
        "recent papers about tiny agent controllers",
        "current exchange rate from USD to EUR",
        "latest browser WebGPU support on iPhone",
    ]
    cases: list[dict[str, Any]] = []
    for index, topic in enumerate(topics):
        user_request = f"search the web for {topic}"
        cases.append(
            {
                "agent_name": f"Web Search Assistant {index}",
                "agent_instruction": (
                    "When the user asks for current, recent, online, or web-backed information, request the web_search "
                    "extension. Do not invent results before the extension runs."
                ),
                "user_request": user_request,
                "action": "extension_request",
                "content": "I can search the web for that after you approve the web search action.",
                "retrieval_policy": "auto",
                "tool_policy": "ask_before_extensions",
                "action_policy": "allow_extension_requests",
                "proposal_metadata": {
                    "task_type": "web_search_request",
                    "extension_id": "web_search",
                    "capability": "web.search",
                    "query": user_request,
                    "max_sources": 5,
                    "requires_user_approval": True,
                },
                "weight": 8.0,
            }
        )
    return cases


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compact(value: object, *, limit: int = 4000) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _split_for_id(source_id: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _decision(row: dict[str, Any]) -> str:
    action = str(row.get("action") or "respond")
    source_slots = compile_source_slots(
        user_text=row.get("user_request", ""),
        user_data=row.get("user_data", ""),
        max_slots=8,
    )
    payload: dict[str, Any] = {
        "action": action,
        "content": pointerize_exact_text(_compact(row.get("content", ""), limit=7000), source_slots),
    }
    if row.get("retrieval_influenced"):
        payload["retrieval_influenced"] = True
    metadata = dict(row.get("proposal_metadata") or {})
    if metadata:
        payload["proposal_metadata"] = {**metadata, **source_slot_metadata(source_slots)}
    elif action != "respond":
        payload["proposal_metadata"] = {"task_type": f"agent_{action}"}
    else:
        payload["proposal_metadata"] = {"task_type": "agent_instruction_following"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _encoder(row: dict[str, Any]) -> str:
    if str(row.get("encoder_text", "") or "").strip():
        return str(row["encoder_text"])
    retrieval_policy = str(row.get("retrieval_policy") or "auto")
    tool_policy = str(row.get("tool_policy") or "ask_before_extensions")
    action_policy = str(row.get("action_policy") or "respond_or_ask")
    user_data = _compact(row.get("user_data", ""), limit=1000)
    paper_context = _compact(row.get("paper_context", ""), limit=1000)
    source_slots = compile_source_slots(
        user_text=row.get("user_request", ""),
        user_data=user_data,
        max_slots=8,
    )
    return "\n".join(
        part
        for part in [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {_compact(row.get('agent_name'), limit=120)}",
            f"Agent instruction: {_compact(row.get('agent_instruction'), limit=800)}",
            f"Retrieval policy: {retrieval_policy}",
            f"Tool policy: {tool_policy}",
            f"Action policy: {action_policy}",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            f"<AK_CONTEXT> Saved user data: {user_data}" if user_data else "<AK_CONTEXT> Saved user data: none",
            source_slots_encoder_block(source_slots),
            f"<AK_CONTEXT> Stale selected paper context: {paper_context}" if paper_context else "<AK_CONTEXT> Stale selected paper context: none",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {_compact(row.get('user_request'), limit=1600)}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
        if part
    )


def build_dataset(output_dir: Path, *, eval_fraction: float = 0.2) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_cases = (
        AGENT_CASES
        + _generated_agent_cases()
        + _slot_instruction_cases()
        + _source_pointer_cases()
        + _plain_chat_cases()
        + _web_extension_cases()
    )
    for index, case in enumerate(all_cases):
        agent_name = str(case.get("agent_name") or "PocketPal Agent")
        source_id = f"pocketpal_agent_quality_{index:04d}_{agent_name.lower().replace(' ', '_')}"
        action = str(case.get("action") or "respond")
        rows.append(
            {
                "source_id": source_id,
                "source_type": "pocketpal_agent_quality",
                "task_type": str((case.get("proposal_metadata") or {}).get("task_type") or "agent_instruction_following"),
                "action": action,
                "encoder_text": _encoder(case),
                "decoder_text": _decision(case),
                "weight": float(case.get("weight", 5.0 if str(case.get("source_type") or "pocketpal_agent_quality") == "pocketpal_agent_quality" else 4.0)),
                "metadata": {
                    "agent_name": agent_name,
                    "retrieval_policy": str(case.get("retrieval_policy") or "auto"),
                    "tool_policy": str(case.get("tool_policy") or "ask_before_extensions"),
                    "action_policy": str(case.get("action_policy") or "respond_or_ask"),
                },
            }
        )
    train_rows = [row for row in rows if _split_for_id(row["source_id"], eval_fraction) == "train"]
    eval_rows = [row for row in rows if _split_for_id(row["source_id"], eval_fraction) == "eval"]
    if not eval_rows and train_rows:
        eval_rows.append(train_rows.pop())

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pocketpal_agent_quality_train.jsonl"
    eval_path = output_dir / "pocketpal_agent_quality_eval.jsonl"
    for path, split_rows in [(train_path, train_rows), (eval_path, eval_rows)]:
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    action_counts: dict[str, int] = {}
    for row in rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_agent_quality",
        "dataset_format": "jsonl",
        "manifest_path": str(output_dir / "pocketpal_agent_quality_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"pocketpal_agent_quality": len(rows)},
        "target_action_counts": action_counts,
        "schema": {
            "encoder_text": "PocketPal active agent contract and user request",
            "decoder_text": "compact JSON action decision",
        },
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/pocketpal_agent_quality_dataset")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    args = parser.parse_args()
    manifest = build_dataset(Path(args.output_dir).expanduser().resolve(), eval_fraction=float(args.eval_fraction))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
