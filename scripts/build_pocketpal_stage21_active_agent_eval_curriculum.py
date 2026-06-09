#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage11_broad_agent_mode_dataset import INTENT_LABELS


RESPOND_PREFIX = '{"action": "respond", "content": "'


def _task_for(intent: str) -> str:
    return "active_agent_rewrite" if intent == "rewrite" else f"active_agent_{intent}"


def _prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    intent: str,
    saved_data: str = "none",
) -> str:
    task = _task_for(intent)
    source_required = "true" if intent != "casual" else "false"
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
            f"<AK_TASK_HINT> intent={intent} task={task} source_text_required={source_required}",
            f"<AK_CONTEXT> Saved user data: {saved_data}",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _json_decoder(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "action": action,
        "content": content,
        "proposal_metadata": {"task_type": task_type},
    }
    if metadata:
        payload["proposal_metadata"].update(metadata)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _row(
    source_id: str,
    *,
    name: str,
    instruction: str,
    user_text: str,
    content: str,
    intent: str,
    action: str = "respond",
    saved_data: str = "none",
    weight: float = 10.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = _task_for(intent)
    prompt = _prompt(
        name=name,
        instruction=instruction,
        user_text=user_text,
        intent=intent,
        saved_data=saved_data,
    )
    decoder_text = _json_decoder(action, content, task, metadata)
    return {
        "action": action,
        "decoder_prefix": RESPOND_PREFIX if action == "respond" else "",
        "decoder_text": decoder_text,
        "encoder_text": prompt,
        "example_id": hashlib.sha256(f"{source_id}\n{prompt}\n{decoder_text}".encode()).hexdigest(),
        "expected_content": content,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS.get(intent, -1),
        "source_id": source_id,
        "source_type": "pocketpal_stage21_active_agent_eval_curriculum",
        "task_type": task,
        "weight": float(weight),
    }


def _base_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = [
        {
            "name": "Professional Email Rewriter",
            "instruction": "Rewrite the user input for a professional email.",
            "intent": "rewrite",
            "pairs": [
                ("hi how are you?", "Hello, I hope you are well."),
                ("hey john i need the report by friday", "Hi John, could you please send the report by Friday?"),
                ("can you send me the invoice today", "Could you please send me the invoice today?"),
                ("need those docs asap", "Could you please send those documents as soon as possible?"),
                ("thanks for helping me yesterday", "Thank you for your help yesterday."),
            ],
        },
        {
            "name": "Concise Summary Agent",
            "instruction": "Summarize the user's text concisely without changing facts.",
            "intent": "summary",
            "pairs": [
                ("The build uploaded, but Apple processing is still pending.", "The build uploaded, but Apple processing is still pending."),
                ("Payment bugs are fixed and legal approval is the only launch blocker.", "Payment bugs are fixed; launch is waiting on legal approval."),
                ("We met with design, reviewed the search flow, and agreed to make links clickable.", "Design reviewed the search flow and agreed links should be clickable."),
            ],
        },
        {
            "name": "Action Item Agent",
            "instruction": "Extract action items, owners, and deadlines as bullets.",
            "intent": "action_items",
            "pairs": [
                ("Maria owns launch slides and Devin fixes login by Thursday.", "- Maria: own launch slides\n- Devin: fix login by Thursday"),
                ("Sam books the room today and Priya sends notes Friday.", "- Sam: book the room today\n- Priya: send notes Friday"),
                ("I will update TestFlight, then Alex will verify the rewrite agent.", "- User: update TestFlight\n- Alex: verify the rewrite agent"),
            ],
        },
        {
            "name": "Checklist Agent",
            "instruction": "Turn the user's text into a clean checklist.",
            "intent": "checklist",
            "pairs": [
                ("Review proposal, confirm budget, send invoice tomorrow.", "- Review proposal\n- Confirm budget\n- Send invoice tomorrow"),
                ("Before release, run tests, upload the build, and check TestFlight.", "- Run tests\n- Upload the build\n- Check TestFlight"),
            ],
        },
        {
            "name": "Question Extractor",
            "instruction": "Extract the explicit question from the user's text. Do not answer it.",
            "intent": "extraction",
            "pairs": [
                ("I was wondering whether we can upload the TestFlight build today.", "Question: Can we upload the TestFlight build today?"),
                ("Can you check if the web search button is active?", "Question: Can you check if the web search button is active?"),
            ],
        },
        {
            "name": "Subject Line Agent",
            "instruction": "Write a concise professional email subject line for the user's message.",
            "intent": "subject",
            "pairs": [
                ("Following up about the contract review due Friday.", "Follow-Up on Friday Contract Review"),
                ("Can we meet tomorrow to discuss the launch plan?", "Meeting Request: Launch Plan Discussion"),
            ],
        },
        {
            "name": "Spanish Translation Agent",
            "instruction": "Translate the user's English text into Spanish.",
            "intent": "translation",
            "pairs": [
                ("Please send the invoice tomorrow morning.", "Por favor, envia la factura manana por la manana."),
                ("The meeting has been moved to Friday.", "La reunion se ha cambiado al viernes."),
            ],
        },
        {
            "name": "JSON Classifier",
            "instruction": "Classify the user request as compact JSON.",
            "intent": "json",
            "pairs": [
                ("Please make this more professional.", '{"intent":"rewrite","tone":"professional"}'),
                ("Find current news about Apple TestFlight.", '{"intent":"web_search","freshness":"current"}'),
                ("Translate this into Spanish.", '{"intent":"translation","target_language":"spanish"}'),
                ("Summarize these meeting notes.", '{"intent":"summary","format":"concise"}'),
                ("Extract the owner and deadline.", '{"intent":"extraction","fields":["owner","deadline"]}'),
                ("Make a checklist from this note.", '{"intent":"checklist","format":"bullets"}'),
                ("Rank these tasks by urgency.", '{"intent":"ranking","criterion":"urgency"}'),
                ("Write a subject line for this email.", '{"intent":"subject","format":"email_subject"}'),
                ("Use my saved launch code.", '{"intent":"saved_data","data_needed":"launch_code"}'),
                ("Search online for current pricing.", '{"intent":"web_search","freshness":"current"}'),
            ],
        },
        {
            "name": "Planning Agent",
            "instruction": "Turn the user's goal into a short practical plan.",
            "intent": "plan",
            "pairs": [
                ("Ship the fixed rewrite agent to TestFlight.", "1. Verify the active-agent prompt path.\n2. Commit and push the fix.\n3. Run the TestFlight workflow.\n4. Install and test the processed build."),
                ("Prepare for the client review tomorrow.", "1. Confirm the agenda.\n2. Review open questions.\n3. Prepare supporting notes.\n4. Send the meeting reminder."),
            ],
        },
        {
            "name": "Memory Agent",
            "instruction": "Use saved user data when it directly answers the request.",
            "intent": "saved_data",
            "saved_data": "[D1] saved note: Launch code is ORBIT-42 for the May TestFlight build.",
            "pairs": [
                ("what is my launch code", "Your launch code is ORBIT-42 for the May TestFlight build."),
                ("remind me what code we saved for launch", "The saved launch code is ORBIT-42."),
            ],
        },
    ]
    for spec_index, spec in enumerate(specs):
        for pair_index, (user_text, content) in enumerate(spec["pairs"]):
            rows.append(
                _row(
                    f"stage21_base_{spec_index:03d}_{pair_index:03d}",
                    name=str(spec["name"]),
                    instruction=str(spec["instruction"]),
                    user_text=user_text,
                    content=content,
                    intent=str(spec["intent"]),
                    saved_data=str(spec.get("saved_data", "none")),
                    weight=18.0,
                )
            )
    return rows


def build_rows(repeats: int) -> list[dict[str, Any]]:
    base = _base_cases()
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for row in base:
            copied = dict(row)
            copied["source_id"] = f"{row['source_id']}_repeat_{repeat:03d}"
            copied["example_id"] = hashlib.sha256(
                f"{copied['source_id']}\n{copied['encoder_text']}\n{copied['decoder_text']}".encode()
            ).hexdigest()
            rows.append(copied)
    return rows


def _is_eval_row(row: dict[str, Any]) -> bool:
    digest = hashlib.sha256(str(row["source_id"]).encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage21_active_agent_eval_curriculum_v1")
    parser.add_argument("--repeats", type=int, default=120)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(args.repeats)
    train_rows = [row for row in rows if not _is_eval_row(row)]
    eval_rows = [row for row in rows if _is_eval_row(row)]
    train_path = output_dir / "pocketpal_stage21_active_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_stage21_active_agent_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage21_active_agent_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "decoder_prefix": RESPOND_PREFIX,
        "objective": "pocketpal_stage21_active_agent_eval_curriculum",
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
