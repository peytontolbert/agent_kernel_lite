#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from build_pocketpal_stage10_agent_gate_repair_dataset import _row
from build_pocketpal_stage11_broad_agent_mode_dataset import (
    _agent_prompt,
    _with_intent,
    build_rows as build_stage11_rows,
    write_dataset,
)


def _repair_specs() -> list[tuple[str, str, str, str, str, str]]:
    return [
        ("Clarifying Question Agent", "Ask one concise clarifying question when the user has not provided the required text, recipient, language, source, or constraint. Do not complete the task from guesses.", "make this sound better", "What text would you like me to improve?", "ask_user", "confusion_repair_ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question when the user has not provided the required text, recipient, language, source, or constraint. Do not complete the task from guesses.", "turn it into bullets", "What text should I turn into bullets?", "ask_user", "confusion_repair_ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question when the user has not provided the required text, recipient, language, source, or constraint. Do not complete the task from guesses.", "translate this", "What text should I translate, and what language should I use?", "ask_user", "confusion_repair_ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question when the user has not provided the required text, recipient, language, source, or constraint. Do not complete the task from guesses.", "write the email", "Who should receive the email, and what should it say?", "ask_user", "confusion_repair_ask_user"),
        ("Clarifying Question Agent", "Ask one concise clarifying question if the user asks about unavailable saved data. Do not use stale research context.", "what is my reservation code", "I do not have that reservation code in the provided data. Where should I look for it?", "ask_user", "confusion_repair_ask_user"),
        ("Action Item Extractor", "Extract concrete action items with owners and deadlines. Do not extract every entity separately.", "Maria will update the slides today, Devin fixes the login bug by Thursday, and Sam sends notes Friday.", "- Maria: update the slides today\n- Devin: fix the login bug by Thursday\n- Sam: send notes Friday", "action_items", "confusion_repair_action_items"),
        ("Action Item Extractor", "Extract concrete action items with owners and deadlines. Do not extract every entity separately.", "I will call finance tomorrow, Priya will review the contract, and Omar will book the room.", "- Me: call finance tomorrow\n- Priya: review the contract\n- Omar: book the room", "action_items", "confusion_repair_action_items"),
        ("Action Item Extractor", "Extract concrete action items with owners and deadlines. Do not extract every entity separately.", "Lena owns the launch checklist, Max handles QA by Monday, and I will send the beta notes.", "- Lena: own the launch checklist\n- Max: handle QA by Monday\n- Me: send the beta notes", "action_items", "confusion_repair_action_items"),
        ("Entity Extractor", "Extract names, dates, places, amounts, and objects. Do not turn them into action items.", "Maria mentioned Austin on May 18 with a $420 hotel charge.", "- Name: Maria\n- Place: Austin\n- Date: May 18\n- Amount: $420\n- Object: hotel charge", "extraction", "confusion_repair_extraction"),
        ("Entity Extractor", "Extract names, dates, places, amounts, and objects. Do not turn them into action items.", "The invoice INV-2048 for $1,200 is due Friday for John.", "- ID: INV-2048\n- Amount: $1,200\n- Date: Friday\n- Name: John\n- Object: invoice", "extraction", "confusion_repair_extraction"),
        ("Entity Extractor", "Extract names, dates, places, amounts, and objects. Do not turn them into action items.", "Priya booked Delta 482 to Seattle next Tuesday.", "- Name: Priya\n- Flight: Delta 482\n- Place: Seattle\n- Date: next Tuesday", "extraction", "confusion_repair_extraction"),
        ("Email Subject Agent", "Write exactly one concise professional email subject line. Do not brainstorm multiple options.", "Following up about the Friday client report.", "Subject: Follow-Up on Friday Client Report", "subject", "confusion_repair_subject"),
        ("Email Subject Agent", "Write exactly one concise professional email subject line. Do not brainstorm multiple options.", "Confirming the updated contract review timeline.", "Subject: Updated Contract Review Timeline", "subject", "confusion_repair_subject"),
        ("Email Subject Agent", "Write exactly one concise professional email subject line. Do not brainstorm multiple options.", "Requesting approval for the May TestFlight build.", "Subject: Approval Request for May TestFlight Build", "subject", "confusion_repair_subject"),
        ("Brainstorm Agent", "Generate three concise ideas. Do not return a single subject line.", "Subject line ideas for the Friday client report.", "1. Friday Client Report Follow-Up\n2. Client Report Status for Friday\n3. Checking In on the Friday Report", "brainstorm", "confusion_repair_brainstorm"),
        ("Brainstorm Agent", "Generate three concise ideas. Do not return a single subject line.", "Ideas for improving PocketPal's retrieval tab.", "1. Add source health indicators\n2. Let users pin trusted folders\n3. Show recent retrieval misses", "brainstorm", "confusion_repair_brainstorm"),
        ("Brainstorm Agent", "Generate three concise ideas. Do not return a single subject line.", "Three ways to make custom agents easier to configure.", "1. Use guided templates\n2. Preview tool permissions\n3. Add test prompts before saving", "brainstorm", "confusion_repair_brainstorm"),
        ("Title Generator", "Create exactly one short title for the user's text. Do not brainstorm multiple options.", "Notes about fixing web search approval and clickable result links in PocketPal.", "PocketPal Web Search Fixes", "title", "confusion_repair_title"),
        ("Title Generator", "Create exactly one short title for the user's text. Do not brainstorm multiple options.", "A plan for training the local controller with weighted intent repair examples.", "Weighted Intent Repair Plan", "title", "confusion_repair_title"),
        ("Title Generator", "Create exactly one short title for the user's text. Do not brainstorm multiple options.", "Ideas and notes from testing custom agents, retrieval, and local web search.", "PocketPal Agent Testing Notes", "title", "confusion_repair_title"),
        ("Title Generator", "Create exactly one short title for the user's text. Do not brainstorm multiple options.", "Summary of mobile sync, TestFlight blockers, and model validation results.", "Mobile Sync and Model Validation", "title", "confusion_repair_title"),
        ("Checklist Builder", "Convert the text into a checklist. Do not translate it.", "Please review the proposal, confirm the budget, and send the invoice tomorrow morning.", "- Review the proposal\n- Confirm the budget\n- Send the invoice tomorrow morning", "checklist", "confusion_repair_checklist"),
        ("Checklist Builder", "Convert the text into a checklist. Do not translate it.", "Before launch, test web search, verify links, update screenshots, and send notes.", "- Test web search\n- Verify links\n- Update screenshots\n- Send notes", "checklist", "confusion_repair_checklist"),
        ("Summary Agent", "Summarize the user's text in one concise sentence. Do not rewrite it as a polished message.", "Please send the invoice tomorrow morning because finance needs it before lunch.", "Finance needs the invoice tomorrow morning before lunch.", "summary", "confusion_repair_summary"),
        ("Summary Agent", "Summarize the user's text in one concise sentence. Do not rewrite it as a polished message.", "The team fixed search links, synced mobile assets, and still needs a TestFlight upload.", "Search links and mobile assets are fixed, but TestFlight upload is still pending.", "summary", "confusion_repair_summary"),
        ("Professional Email Rewriter", "Rewrite the user's text as a professional email. Do not summarize it.", "send me the invoice tomorrow morning finance needs it before lunch", "Could you please send me the invoice tomorrow morning? Finance needs it before lunch. Thank you.", "rewrite", "confusion_repair_rewrite"),
        ("Professional Email Rewriter", "Rewrite the user's text as a professional email. Do not summarize it.", "john the report is late and the client needs it friday", "Hi John, the report is currently late, and the client needs it by Friday. Could you please send an update? Thank you.", "rewrite", "confusion_repair_rewrite"),
    ]


def build_rows(base_repeats: int, repair_repeats: int) -> list[dict[str, Any]]:
    rows = build_stage11_rows(int(base_repeats))
    for repeat in range(max(1, int(repair_repeats))):
        for index, (name, instruction, user_text, content, intent, task_type) in enumerate(_repair_specs()):
            action = "ask_user" if intent == "ask_user" else "respond"
            rows.append(
                _with_intent(
                    _row(
                        f"stage17_{task_type}_{repeat:03d}_{index:03d}",
                        _agent_prompt(
                            name=name,
                            instruction=instruction,
                            user_text=user_text,
                            text_slots={"SOURCE_TEXT": user_text},
                        ),
                        action,
                        content,
                        task_type,
                        weight=22.0,
                    ),
                    intent,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage17_intent_confusion_repair_dataset_v1")
    parser.add_argument("--base-repeats", type=int, default=220)
    parser.add_argument("--repair-repeats", type=int, default=260)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()
    manifest = write_dataset(
        build_rows(int(args.base_repeats), int(args.repair_repeats)),
        Path(args.output_dir),
        float(args.eval_fraction),
    )
    print(__import__("json").dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
