#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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

RESPOND_PREFIX = '{"action": "respond", "content": "'


def _task_for(intent: str) -> str:
    if intent == "rewrite":
        return "active_agent_rewrite"
    if intent == "web_search":
        return "runtime_web_search_request"
    return f"active_agent_{intent}"


def _stable_split(source_id: str) -> bool:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _active_prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    intent: str,
    slots: dict[str, str] | None = None,
    saved_data: str = "none",
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
    web_enabled: bool = False,
) -> str:
    slots = slots or {"SOURCE_TEXT": user_text}
    slot_names = [key for key, value in slots.items() if str(value or "").strip()]
    slot_lines = [
        "<AK_PROFILE> User text slots:",
        *[f"<AK_SLOT> <AK_SLOT_NAME>={key} <AK_SLOT_VALUE>={value}" for key, value in slots.items()],
        f"Available placeholders for this turn: {', '.join(f'[[{key}]]' for key in slot_names)}.",
        "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
    ]
    tool_lines: list[str] = []
    header = "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example."
    tool_policy = "ask_before_extensions"
    action_policy = "respond_or_ask"
    if web_enabled:
        header = "<AK_CHAT> <AK_RESPOND> <AK_EXTENSION> <AK_WEB_SEARCH> PocketPal user-configured agent example."
        tool_policy = "ask_before_extensions"
        action_policy = "allow_extension_requests"
        tool_lines = [
            "<AK_PROFILE> PocketPal installed tools:",
            "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
            "<AK_MAX_SOURCES>=5",
            "When current, recent, online, or web-backed information is needed, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true.",
        ]
    return "\n".join(
        [
            header,
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            f"Tool policy: {tool_policy}",
            f"Action policy: {action_policy}",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            f"<AK_TASK_HINT> intent={intent} task={_task_for(intent)} source_text_required={'false' if intent in {'casual', 'web_search'} else 'true'}",
            *tool_lines,
            f"<AK_CONTEXT> Saved user data: {saved_data}",
            *slot_lines,
            f"<AK_CONTEXT> Stale selected paper context: {stale_context}",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _decision(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "action": action,
        "content": content,
        "proposal_metadata": {"task_type": task_type},
    }
    if metadata:
        payload["proposal_metadata"].update(metadata)
    return _json(payload)


def _row(
    source_id: str,
    *,
    name: str,
    instruction: str,
    user_text: str,
    intent: str,
    content: str,
    action: str = "respond",
    metadata: dict[str, Any] | None = None,
    slots: dict[str, str] | None = None,
    saved_data: str = "none",
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
    web_enabled: bool = False,
    weight: float = 12.0,
) -> dict[str, Any]:
    task_type = _task_for(intent)
    prompt = _active_prompt(
        name=name,
        instruction=instruction,
        user_text=user_text,
        intent=intent,
        slots=slots,
        saved_data=saved_data,
        stale_context=stale_context,
        web_enabled=web_enabled,
    )
    decoder = _decision(action, content, task_type, metadata)
    return {
        "action": action,
        "decoder_prefix": RESPOND_PREFIX if action == "respond" else "",
        "decoder_text": decoder,
        "encoder_text": prompt,
        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{decoder}".encode("utf-8")).hexdigest(),
        "expected_content": content,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS.get(intent, -1),
        "source_id": source_id,
        "source_type": "pocketpal_stage25_diverse_instruction_curriculum",
        "task_type": task_type,
        "weight": float(weight),
    }


def _case_rows() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(intent: str, name: str, instructions: list[str], examples: list[tuple[str, str]], weight: float = 12.0) -> None:
        for instruction_index, instruction in enumerate(instructions):
            for example_index, (user_text, content) in enumerate(examples):
                cases.append(
                    _row(
                        f"stage25_{intent}_{instruction_index:02d}_{example_index:03d}",
                        name=name,
                        instruction=instruction,
                        user_text=user_text,
                        intent=intent,
                        content=content,
                        weight=weight,
                    )
                )

    add(
        "rewrite",
        "Professional Rewriter",
        [
            "Rewrite the user input for a professional email. Preserve facts and intent.",
            "Make the user's text professional and concise. Do not answer it.",
            "Convert the source text into polished workplace wording.",
        ],
        [
            ("hi how are you?", "Hello, I hope you are well."),
            ("hey maria can you send the slides today", "Hi Maria, could you please send the slides today?"),
            ("need the invoice asap", "Could you please send the invoice as soon as possible?"),
            ("thanks for fixing this yesterday", "Thank you for fixing this yesterday."),
            ("this is late and blocking us", "This is delayed and is currently blocking our work."),
            ("can we move the meeting to friday afternoon", "Could we move the meeting to Friday afternoon?"),
        ],
        weight=18.0,
    )
    add(
        "summary",
        "Summary Agent",
        [
            "Summarize the user's text in one concise sentence without changing facts.",
            "Create a short factual summary of the source text.",
            "Compress the text while preserving the important facts.",
        ],
        [
            ("The build uploaded, but Apple processing is still pending.", "The build uploaded, but Apple processing is still pending."),
            ("Design approved the search flow and asked us to make result links clickable.", "Design approved the search flow and requested clickable result links."),
            ("Payment bugs are fixed, but legal approval is still blocking launch.", "Payment bugs are fixed, but launch is waiting on legal approval."),
            ("Maria owns launch slides, Devin fixes login, and Priya sends notes Friday.", "Maria, Devin, and Priya own launch tasks with near-term deadlines."),
        ],
        weight=14.0,
    )
    add(
        "action_items",
        "Action Item Extractor",
        [
            "Extract action items, owners, and deadlines as bullets.",
            "Turn the text into concise task bullets. Preserve owners and dates.",
        ],
        [
            ("Maria owns launch slides and Devin fixes login by Thursday.", "- Maria: own launch slides\n- Devin: fix login by Thursday"),
            ("Sam books the room today and Priya sends notes Friday.", "- Sam: book the room today\n- Priya: send notes Friday"),
            ("I will update TestFlight, then Alex will verify the rewrite agent.", "- User: update TestFlight\n- Alex: verify the rewrite agent"),
            ("Finance approves the invoice Monday and Jordan sends the receipt.", "- Finance: approve the invoice Monday\n- Jordan: send the receipt"),
        ],
        weight=14.0,
    )
    add(
        "checklist",
        "Checklist Builder",
        [
            "Convert the user's text into a clean checklist.",
            "Make a concise checklist from the provided text.",
        ],
        [
            ("Review proposal, confirm budget, send invoice tomorrow.", "- Review proposal\n- Confirm budget\n- Send invoice tomorrow"),
            ("Before release, run tests, upload the build, and check TestFlight.", "- Run tests\n- Upload the build\n- Check TestFlight"),
            ("Pack laptop, charger, badge, and printed agenda.", "- Pack laptop\n- Pack charger\n- Pack badge\n- Pack printed agenda"),
        ],
        weight=13.0,
    )
    add(
        "extraction",
        "Extractor",
        [
            "Extract names, dates, amounts, and objects as compact bullets.",
            "Extract the explicit question. Do not answer it.",
        ],
        [
            ("Hi John, please send the invoice for $1,200 by Friday.", "- Name: John\n- Object: invoice\n- Amount: $1,200\n- Date: Friday"),
            ("Priya booked Austin for May 18 and paid $420.", "- Name: Priya\n- Place: Austin\n- Date: May 18\n- Amount: $420"),
            ("I was wondering whether we can upload the build today.", "Question: Can we upload the build today?"),
            ("Can you check if web search is active?", "Question: Can you check if web search is active?"),
        ],
        weight=13.0,
    )
    add(
        "translation",
        "Spanish Translator",
        ["Translate the user's English text into Spanish. Return only the translation."],
        [
            ("Please send the invoice tomorrow morning.", "Por favor, envia la factura manana por la manana."),
            ("The meeting has been moved to Friday.", "La reunion se ha cambiado al viernes."),
        ],
        weight=12.0,
    )
    add(
        "translation",
        "French Translator",
        ["Translate the source text into French. Return only the translation."],
        [
            ("Please review the proposal before Friday.", "Veuillez examiner la proposition avant vendredi."),
            ("Can you call me after lunch?", "Pouvez-vous m'appeler apres le dejeuner?"),
        ],
        weight=12.0,
    )
    add(
        "plan",
        "Planner",
        [
            "Turn the user's goal into a short practical plan.",
            "Create a concise sequence of steps for the user's goal.",
        ],
        [
            ("Ship the fixed rewrite agent to TestFlight.", "1. Verify the active-agent prompt path.\n2. Commit and push the fix.\n3. Run the TestFlight workflow.\n4. Install and test the processed build."),
            ("Prepare for the client review tomorrow.", "1. Confirm the agenda.\n2. Review open questions.\n3. Prepare supporting notes.\n4. Send the meeting reminder."),
            ("Organize my local documents for retrieval.", "1. Choose the folders to index.\n2. Remove files that should stay private.\n3. Run the local import.\n4. Test retrieval with a few queries."),
        ],
        weight=12.0,
    )
    add(
        "risks",
        "Risk Reviewer",
        [
            "List the main risks in the user's plan as concise bullets.",
            "Identify likely risks without inventing extra context.",
        ],
        [
            ("Ship the beta tonight without retesting agent flow or web search links.", "- Agent flow may still fail without retesting\n- Web search links may be broken\n- Shipping tonight leaves little time for rollback"),
            ("Delete old checkpoints before confirming the latest model exports.", "- The latest export may be missing\n- Recovery will be harder if the checkpoint is needed\n- Evaluation results may become harder to reproduce"),
        ],
        weight=12.0,
    )
    add(
        "ranking",
        "Priority Sorter",
        [
            "Sort the user's tasks from highest to lowest priority.",
            "Rank the tasks by urgency.",
        ],
        [
            ("Fix login crash, update docs, choose nicer button color.", "1. Fix login crash\n2. Update docs\n3. Choose nicer button color"),
            ("Submit payroll, rename a folder, reply to client.", "1. Submit payroll\n2. Reply to client\n3. Rename a folder"),
        ],
        weight=11.0,
    )
    add(
        "brainstorm",
        "Brainstorm Agent",
        [
            "Generate three concise ideas that fit the user's request.",
            "Brainstorm three useful options. Do not over-explain.",
        ],
        [
            ("Ideas for making PocketPal feel more personal.", "1. Let users create custom agents\n2. Add local memory collections\n3. Offer per-agent tone and tool settings"),
            ("Ways to make web search easier in the app.", "1. Add a search button in chat\n2. Show source cards with clickable links\n3. Let users set the max source count"),
        ],
        weight=11.0,
    )
    add(
        "subject",
        "Subject Line Agent",
        [
            "Write a concise professional email subject line for the user's message.",
            "Create only the email subject line.",
        ],
        [
            ("Following up about the contract review due Friday.", "Follow-Up on Friday Contract Review"),
            ("Can we meet tomorrow to discuss the launch plan?", "Meeting Request: Launch Plan Discussion"),
            ("Reminder that the invoice needs approval today.", "Invoice Approval Reminder"),
        ],
        weight=12.0,
    )
    add(
        "json",
        "JSON Classifier",
        [
            "Classify the user request as compact JSON.",
            "Return only a compact JSON object describing the user's intent.",
        ],
        [
            ("Please make this more professional.", '{"intent":"rewrite","tone":"professional"}'),
            ("Find current news about TestFlight processing.", '{"intent":"web_search","freshness":"current"}'),
            ("Translate this into Spanish.", '{"intent":"translation","target_language":"spanish"}'),
            ("Extract the owner and deadline.", '{"intent":"extraction","fields":["owner","deadline"]}'),
            ("Rank these tasks by urgency.", '{"intent":"ranking","criterion":"urgency"}'),
        ],
        weight=9.0,
    )

    saved_data = "[D1] saved note: Launch code is ORBIT-42 for the May TestFlight build."
    cases.extend(
        [
            _row(
                "stage25_saved_000",
                name="Memory Agent",
                instruction="Use saved user data when it directly answers the request.",
                user_text="what is my launch code",
                intent="saved_data",
                content="Your launch code is ORBIT-42 for the May TestFlight build.",
                saved_data=saved_data,
                weight=13.0,
            ),
            _row(
                "stage25_saved_001",
                name="Memory Agent",
                instruction="Use saved user data when it directly answers the request.",
                user_text="what hotel code did we save",
                intent="ask_user",
                content="I do not have the hotel code in the provided data. Where should I look for it?",
                saved_data=saved_data,
                weight=13.0,
            ),
            _row(
                "stage25_source_echo_000",
                name="Source Echo Agent",
                instruction="Return the exact source text with a short label. Preserve all wording.",
                user_text="vendor invoice INV-2048 is blocked until finance approves $1,200",
                intent="source_echo",
                content="Source text: vendor invoice INV-2048 is blocked until finance approves $1,200",
                weight=12.0,
            ),
            _row(
                "stage25_ask_000",
                name="Clarifying Question Agent",
                instruction="Ask one concise clarifying question when required text is missing.",
                user_text="make this better",
                intent="ask_user",
                content="What text would you like me to improve?",
                slots={},
                weight=13.0,
            ),
            _row(
                "stage25_web_000",
                name="Web Search Assistant",
                instruction="When the user asks for current online information, request web search. Do not invent results.",
                user_text="search the web for current TestFlight processing delays",
                intent="web_search",
                action="extension_request",
                content="Requesting approval to search the web.",
                metadata={
                    "capability": "web.search",
                    "extension_id": "web_search",
                    "max_sources": 5,
                    "query": "current TestFlight processing delays",
                    "requires_user_approval": True,
                },
                web_enabled=True,
                weight=13.0,
            ),
            _row(
                "stage25_web_001",
                name="Web Search Assistant",
                instruction="Request web search when fresh sources are needed. Do not answer before results arrive.",
                user_text="look up today's Apple developer system status",
                intent="web_search",
                action="extension_request",
                content="Requesting approval to search the web.",
                metadata={
                    "capability": "web.search",
                    "extension_id": "web_search",
                    "max_sources": 5,
                    "query": "today's Apple developer system status",
                    "requires_user_approval": True,
                },
                web_enabled=True,
                weight=13.0,
            ),
        ]
    )
    cases.extend(_generated_control_rows())
    return cases


def _generated_control_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan"]
    objects = ["launch memo", "budget sheet", "security review", "client deck", "invoice packet", "release notes"]
    dates = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow"]
    blockers = ["legal approval", "finance review", "QA signoff", "design feedback", "vendor confirmation", "manager approval"]
    for index in range(48):
        owner = names[index % len(names)]
        reviewer = names[(index + 3) % len(names)]
        item = objects[index % len(objects)]
        date = dates[index % len(dates)]
        blocker = blockers[index % len(blockers)]
        user_text = f"{owner} will send the {item} by {date}, {reviewer} will review it, and launch is blocked by {blocker}."
        rows.extend(
            [
                _row(
                    f"stage25_control_summary_{index:03d}",
                    name="Summary Agent",
                    instruction="Summarize the source text in one concise sentence. Do not make a checklist.",
                    user_text=user_text,
                    intent="summary",
                    content=f"{owner} will send the {item} by {date}, {reviewer} will review it, and {blocker} is blocking launch.",
                    weight=15.0,
                ),
                _row(
                    f"stage25_control_actions_{index:03d}",
                    name="Action Item Agent",
                    instruction="Extract concrete action items as bullets with owners. Do not summarize as prose.",
                    user_text=user_text,
                    intent="action_items",
                    content=f"- {owner}: send the {item} by {date}\n- {reviewer}: review the {item}",
                    weight=15.0,
                ),
                _row(
                    f"stage25_control_checklist_{index:03d}",
                    name="Checklist Agent",
                    instruction="Convert the text into a checklist. Do not include owner labels unless needed.",
                    user_text=user_text,
                    intent="checklist",
                    content=f"- Send the {item} by {date}\n- Review the {item}\n- Resolve {blocker}",
                    weight=14.0,
                ),
                _row(
                    f"stage25_control_extract_{index:03d}",
                    name="Entity Extractor",
                    instruction="Extract people, dates, objects, and blockers as compact bullets.",
                    user_text=user_text,
                    intent="extraction",
                    content=f"- Owner: {owner}\n- Reviewer: {reviewer}\n- Object: {item}\n- Date: {date}\n- Blocker: {blocker}",
                    weight=14.0,
                ),
                _row(
                    f"stage25_control_rewrite_{index:03d}",
                    name="Professional Rewriter",
                    instruction="Rewrite the user's text as a polished workplace update. Do not extract bullets.",
                    user_text=user_text,
                    intent="rewrite",
                    content=f"{owner} will send the {item} by {date}. {reviewer} will review it, and launch is currently blocked by {blocker}.",
                    weight=15.0,
                ),
                _row(
                    f"stage25_control_subject_{index:03d}",
                    name="Subject Line Agent",
                    instruction="Write only a concise email subject line for the source text.",
                    user_text=user_text,
                    intent="subject",
                    content=f"{item.title()} Review and Launch Blocker",
                    weight=13.0,
                ),
                _row(
                    f"stage25_control_risk_{index:03d}",
                    name="Risk Agent",
                    instruction="List the main risks as concise bullets. Do not rewrite or summarize.",
                    user_text=user_text,
                    intent="risks",
                    content=f"- {item.title()} may miss the {date} deadline\n- {reviewer}'s review could delay launch\n- {blocker.title()} is still unresolved",
                    weight=13.0,
                ),
            ]
        )
    return rows


def build_rows(repeats: int) -> list[dict[str, Any]]:
    base = _case_rows()
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for row in base:
            copied = dict(row)
            copied["source_id"] = f"{row['source_id']}_repeat_{repeat:03d}"
            copied["example_id"] = hashlib.sha256(
                f"{copied['source_id']}\n{copied['encoder_text']}\n{copied['decoder_text']}".encode("utf-8")
            ).hexdigest()
            rows.append(copied)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage25_diverse_instruction_curriculum_v1")
    parser.add_argument("--repeats", type=int, default=80)
    args = parser.parse_args()

    rows = build_rows(args.repeats)
    train_rows = [row for row in rows if not _stable_split(str(row["source_id"]))]
    eval_rows = [row for row in rows if _stable_split(str(row["source_id"]))]
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "pocketpal_stage25_diverse_instruction_train.jsonl"
    eval_path = output_dir / "pocketpal_stage25_diverse_instruction_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage25_diverse_instruction_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "decoder_prefix": RESPOND_PREFIX,
        "objective": "pocketpal_stage25_diverse_instruction_curriculum",
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "manifest_path": str(manifest_path),
        "intent_labels": INTENT_LABELS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
