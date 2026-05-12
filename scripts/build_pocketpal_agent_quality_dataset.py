#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    payload: dict[str, Any] = {
        "action": action,
        "content": _compact(row.get("content", ""), limit=7000),
    }
    if row.get("retrieval_influenced"):
        payload["retrieval_influenced"] = True
    metadata = dict(row.get("proposal_metadata") or {})
    if metadata:
        payload["proposal_metadata"] = metadata
    elif action != "respond":
        payload["proposal_metadata"] = {"task_type": f"agent_{action}"}
    else:
        payload["proposal_metadata"] = {"task_type": "agent_instruction_following"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _encoder(row: dict[str, Any]) -> str:
    retrieval_policy = str(row.get("retrieval_policy") or "auto")
    tool_policy = str(row.get("tool_policy") or "ask_before_extensions")
    action_policy = str(row.get("action_policy") or "respond_or_ask")
    user_data = _compact(row.get("user_data", ""), limit=1000)
    paper_context = _compact(row.get("paper_context", ""), limit=1000)
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
            f"<AK_CONTEXT> Stale selected paper context: {paper_context}" if paper_context else "<AK_CONTEXT> Stale selected paper context: none",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {_compact(row.get('user_request'), limit=1600)}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
        if part
    )


def build_dataset(output_dir: Path, *, eval_fraction: float = 0.2) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_cases = AGENT_CASES + _generated_agent_cases()
    for index, case in enumerate(all_cases):
        source_id = f"pocketpal_agent_quality_{index:04d}_{case['agent_name'].lower().replace(' ', '_')}"
        action = str(case.get("action") or "respond")
        rows.append(
            {
                "source_id": source_id,
                "source_type": "pocketpal_agent_quality",
                "task_type": str((case.get("proposal_metadata") or {}).get("task_type") or "agent_instruction_following"),
                "action": action,
                "encoder_text": _encoder(case),
                "decoder_text": _decision(case),
                "weight": 5.0 if str(case.get("source_type") or "pocketpal_agent_quality") == "pocketpal_agent_quality" else 4.0,
                "metadata": {
                    "agent_name": case["agent_name"],
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
