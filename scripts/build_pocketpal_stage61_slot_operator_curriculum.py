#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from build_pocketpal_stage11_broad_agent_mode_dataset import INTENT_LABELS


RESPOND_PREFIX = '{"action": "respond", "content": "'


def _task_for(intent: str) -> str:
    return "active_agent_rewrite" if intent == "rewrite" else f"active_agent_{intent}"


def _json_decoder(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "action": action,
        "content": content,
        "proposal_metadata": {"task_type": task_type},
    }
    if metadata:
        payload["proposal_metadata"].update(metadata)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _negative(action: str, content: str, task_type: str, wrong_intent: str = "active_agent_summary") -> str:
    return _json_decoder(action, content, wrong_intent, {"negative_reason": "wrong_operator_or_slot_binding"})


def _prompt(
    *,
    agent_name: str,
    instruction: str,
    intent: str,
    user_text: str,
    saved_data: str = "none",
    extra_slots: dict[str, str] | None = None,
) -> str:
    task = _task_for(intent)
    slots = {"SOURCE_TEXT": user_text}
    if extra_slots:
        slots.update(extra_slots)
    slot_lines = [f"<AK_SLOT> <AK_SLOT_NAME>={name} <AK_SLOT_VALUE>={value}" for name, value in slots.items()]
    slot_names = ", ".join(f"[[{name}]]" for name in slots)
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent turn.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {agent_name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary contract. Follow it over stale context.",
            "</AK_AGENT_ACTIVE>",
            f"<AK_TASK_HINT> intent={intent} task={task} output=json source_text_required=true",
            f"<AK_CONTEXT> Saved user data: {saved_data}",
            "<AK_PROFILE> User text slots:",
            *slot_lines,
            f"Available placeholders for this turn: {slot_names}.",
            "Bind output only to the available slots and saved data. Do not invent names, dates, items, citations, or paper facts.",
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Ignore stale paper context unless the user explicitly asks about that paper.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with action=respond and content that satisfies the active agent instruction.",
        ]
    )


def _row(
    source_id: str,
    *,
    agent_name: str,
    instruction: str,
    intent: str,
    user_text: str,
    content: str,
    saved_data: str = "none",
    extra_slots: dict[str, str] | None = None,
    weight: float = 8.0,
    negative_content: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = _task_for(intent)
    encoder_text = _prompt(
        agent_name=agent_name,
        instruction=instruction,
        intent=intent,
        user_text=user_text,
        saved_data=saved_data,
        extra_slots=extra_slots,
    )
    decoder_text = _json_decoder("respond", content, task, metadata)
    return {
        "action": "respond",
        "answer_confidence_target": 0.92,
        "decoder_prefix": RESPOND_PREFIX,
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{decoder_text}".encode("utf-8")).hexdigest(),
        "expected_content": content,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS.get(intent, -1),
        "needs_verification_target": 0.2,
        "negative_decoder_text": _negative("respond", negative_content or user_text, "active_agent_summary"),
        "negative_loss_weight": 0.18,
        "ood_evidence_target": 0.05,
        "ood_query_target": 0.05,
        "paper_action_validity_target": 0.95,
        "query_confidence_target": 0.15,
        "retrieval_coverage_target": 0.1,
        "retrieval_doc_text": content,
        "retrieval_loss_weight": 0.2,
        "retrieval_query_text": user_text,
        "source_id": source_id,
        "source_type": "pocketpal_stage61_slot_operator_curriculum",
        "split": "train",
        "state_text": f"intent={intent}\nsource={user_text}\ncontent={content}",
        "task_type": task,
        "weight": float(weight),
    }


def _rewrite_cases() -> list[tuple[str, str]]:
    return [
        ("hi how are you?", "Hello, I hope you are well."),
        ("i need the deck by friday", "Could you please send the deck by Friday?"),
        ("sorry this is late but i fixed it", "I apologize for the delay, but I have resolved it."),
        ("can you send me the invoice today", "Could you please send me the invoice today?"),
        ("this is blocked because legal has not approved it", "This is currently blocked because legal approval is still pending."),
        ("hey alex the login bug is still happening", "Hi Alex, the login issue is still occurring."),
        ("need your answer asap", "Could you please provide your response as soon as possible?"),
        ("thanks for helping yesterday", "Thank you for your help yesterday."),
        ("we should move the meeting to monday", "I recommend moving the meeting to Monday."),
        ("the client is waiting on the budget sheet", "The client is waiting for the budget sheet."),
    ]


def _summary_cases() -> list[tuple[str, str]]:
    return [
        ("The build uploaded, but Apple processing is still pending.", "The build uploaded, but Apple processing is still pending."),
        ("Payment bugs are fixed and legal approval is the only launch blocker.", "Payment bugs are fixed; launch is waiting on legal approval."),
        ("Design reviewed the search flow and agreed links should be clickable.", "Design reviewed the search flow and agreed links should be clickable."),
        ("The contract was revised, finance approved the budget, and procurement still needs to sign.", "The contract and budget are ready; procurement still needs to sign."),
        ("Avery finished onboarding notes, but the training video still needs captions.", "Onboarding notes are done; the training video still needs captions."),
    ]


def _action_cases() -> list[tuple[str, str]]:
    return [
        ("Maria owns launch slides and Devin fixes login by Thursday.", "- Maria: own launch slides\n- Devin: fix login by Thursday"),
        ("Sam books the room today and Priya sends notes Friday.", "- Sam: book the room today\n- Priya: send notes Friday"),
        ("I will update TestFlight, then Alex will verify the rewrite agent.", "- User: update TestFlight\n- Alex: verify the rewrite agent"),
        ("Jordan sends the budget sheet Tuesday and Riley reviews it Wednesday.", "- Jordan: send the budget sheet Tuesday\n- Riley: review it Wednesday"),
    ]


def _checklist_cases() -> list[tuple[str, str]]:
    return [
        ("Review proposal, confirm budget, send invoice tomorrow.", "- Review proposal\n- Confirm budget\n- Send invoice tomorrow"),
        ("Before release, run tests, upload the build, and check TestFlight.", "- Run tests\n- Upload the build\n- Check TestFlight"),
        ("Pack charger, print ticket, call the hotel.", "- Pack charger\n- Print ticket\n- Call the hotel"),
    ]


def _extract_cases() -> list[tuple[str, str]]:
    return [
        ("I was wondering whether we can upload the TestFlight build today.", "Question: Can we upload the TestFlight build today?"),
        ("Can you check if the web search button is active?", "Question: Can you check if the web search button is active?"),
        ("Owner is Blake, deadline is next Tuesday, blocker is finance review.", "Owner: Blake\nDeadline: next Tuesday\nBlocker: finance review"),
        ("The hotel confirmation code is HZ-492 and check-in is Friday.", "Confirmation code: HZ-492\nCheck-in: Friday"),
    ]


def _translation_cases() -> list[tuple[str, str]]:
    return [
        ("Please send the invoice tomorrow morning.", "Por favor, envia la factura manana por la manana."),
        ("The meeting has been moved to Friday.", "La reunion se ha cambiado al viernes."),
        ("Thank you for your help yesterday.", "Gracias por tu ayuda ayer."),
        ("Can we review the launch plan today?", "Podemos revisar el plan de lanzamiento hoy?"),
    ]


def _json_cases() -> list[tuple[str, str]]:
    return [
        ("Please make this more professional.", '{"intent":"rewrite","tone":"professional"}'),
        ("Find current news about Apple TestFlight.", '{"intent":"web_search","freshness":"current"}'),
        ("Translate this into Spanish.", '{"intent":"translation","target_language":"spanish"}'),
        ("Summarize these meeting notes.", '{"intent":"summary","format":"concise"}'),
        ("Extract the owner and deadline.", '{"intent":"extraction","fields":["owner","deadline"]}'),
        ("Make a checklist from this note.", '{"intent":"checklist","format":"bullets"}'),
    ]


def _plan_cases() -> list[tuple[str, str]]:
    return [
        ("Ship the fixed rewrite agent to TestFlight.", "1. Verify the active-agent prompt path.\n2. Commit and push the fix.\n3. Run the TestFlight workflow.\n4. Install and test the processed build."),
        ("Prepare for the client review tomorrow.", "1. Confirm the agenda.\n2. Review open questions.\n3. Prepare supporting notes.\n4. Send the meeting reminder."),
        ("Organize the launch checklist.", "1. List required launch tasks.\n2. Assign owners.\n3. Confirm deadlines.\n4. Track unresolved blockers."),
    ]


def _search_cases() -> list[tuple[str, str]]:
    return [
        ("search current TestFlight processing status", '{"query":"current TestFlight processing status","max_sources":5}'),
        ("look up latest Apple developer program fees", '{"query":"latest Apple developer program fees","max_sources":5}'),
        ("find recent news about on-device AI models", '{"query":"recent news about on-device AI models","max_sources":5}'),
    ]


def _memory_cases() -> list[tuple[str, str, str]]:
    saved = "[D1] saved note: Launch code is ORBIT-42 for the May TestFlight build."
    return [
        ("what is my launch code", "Your launch code is ORBIT-42 for the May TestFlight build.", saved),
        ("remind me what code we saved for launch", "The saved launch code is ORBIT-42.", saved),
    ]


def build_rows(repeats: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    specs: list[tuple[str, str, str, list[tuple[str, str]]]] = [
        ("Professional Rewrite Agent", "Rewrite the user's text as a professional message. Preserve the meaning.", "rewrite", _rewrite_cases()),
        ("Concise Summary Agent", "Summarize the user's text concisely without adding facts.", "summary", _summary_cases()),
        ("Action Item Agent", "Extract action items, owners, and deadlines as bullets.", "action_items", _action_cases()),
        ("Checklist Agent", "Turn the user's text into a clean checklist.", "checklist", _checklist_cases()),
        ("Extraction Agent", "Extract the requested facts or explicit question. Do not answer questions.", "extraction", _extract_cases()),
        ("Spanish Translator", "Translate the user's English text into Spanish.", "translation", _translation_cases()),
        ("JSON Classifier", "Classify the user request as compact JSON only.", "json", _json_cases()),
        ("Planning Agent", "Turn the user's goal into a short practical plan.", "plan", _plan_cases()),
        ("Web Search Request Builder", "Convert the user's web-search need into compact JSON with query and max_sources.", "web_search", _search_cases()),
    ]
    instruction_variants = [
        "{instruction}",
        "{instruction} Use the SOURCE_TEXT slot as the only source text.",
        "Your job: {instruction} Return only the requested result.",
        "{instruction} Ignore stale paper context.",
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for spec_index, (agent_name, instruction, intent, cases) in enumerate(specs):
            for case_index, (user_text, content) in enumerate(cases):
                variant = rng.choice(instruction_variants).format(instruction=instruction)
                rows.append(
                    _row(
                        f"stage61_{intent}_{spec_index:02d}_{case_index:03d}_r{repeat:04d}",
                        agent_name=agent_name,
                        instruction=variant,
                        intent=intent,
                        user_text=user_text,
                        content=content,
                        weight=7.0 if intent in {"json", "plan", "translation", "extraction"} else 4.0,
                        negative_content=rng.choice([item[1] for _name, _inst, _intent, source_cases in specs for item in source_cases]),
                        metadata={"stage": "stage61_slot_operator", "repeat": repeat},
                    )
                )
        for case_index, (user_text, content, saved_data) in enumerate(_memory_cases()):
            rows.append(
                _row(
                    f"stage61_saved_data_{case_index:03d}_r{repeat:04d}",
                    agent_name="Memory Agent",
                    instruction="Use saved user data when it directly answers the request.",
                    intent="saved_data",
                    user_text=user_text,
                    content=content,
                    saved_data=saved_data,
                    weight=5.0,
                    negative_content="I do not have that saved.",
                    metadata={"stage": "stage61_slot_operator", "repeat": repeat},
                )
            )
    return rows


def _is_eval_row(row: dict[str, Any]) -> bool:
    digest = hashlib.sha256(str(row["source_id"]).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 20 == 0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage61_slot_operator_curriculum")
    parser.add_argument("--repeats", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    rows = build_rows(repeats=args.repeats, seed=args.seed)
    train_rows = [row for row in rows if not _is_eval_row(row)]
    eval_rows = [row for row in rows if _is_eval_row(row)]
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest: dict[str, Any] = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "decoder_prefix": RESPOND_PREFIX,
        "objective": "pocketpal_stage61_slot_operator_curriculum",
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
