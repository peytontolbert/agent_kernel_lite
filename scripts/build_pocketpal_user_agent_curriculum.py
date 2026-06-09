#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _compact(value: object, *, limit: int = 5000) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].rstrip()


def _decision(action: str, content: str, task_type: str) -> str:
    return json.dumps(
        {
            "action": action,
            "content": _compact(content, limit=4000),
            "proposal_metadata": {"task_type": task_type},
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _encoder(agent_name: str, instruction: str, user_request: str, *, stale_context: str = "") -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent curriculum.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {_compact(agent_name, limit=120)}",
            f"Agent instruction: {_compact(instruction, limit=800)}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            "<AK_CONTEXT> Saved user data: none",
            f"<AK_CONTEXT> Stale selected paper context: {_compact(stale_context, limit=500) if stale_context else 'none'}",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {_compact(user_request, limit=1600)}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _row(agent_name: str, instruction: str, user_request: str, action: str, content: str, task_type: str, index: int) -> dict[str, Any]:
    stale = [
        "",
        "Selected paper [P1]: stale retrieval notes about unrelated optimization experiments.",
        "Retrieved evidence [1]: unrelated search result from a previous turn.",
    ][index % 3]
    encoder_text = _encoder(agent_name, instruction, user_request, stale_context=stale)
    decoder_text = _decision(action, content, task_type)
    example_id = hashlib.sha256(f"{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
    return {
        "example_id": example_id,
        "source_id": f"user_agent_curriculum:{index:06d}",
        "source_type": "pocketpal_user_agent_curriculum",
        "task_type": task_type,
        "action": action,
        "encoder_text": encoder_text,
        "decoder_text": decoder_text,
        "weight": 4.0 if action == "ask_user" else 2.0,
    }


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def build_examples(max_examples: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    subjects = [
        ("login bug", "Users are getting signed out after refreshing the dashboard."),
        ("beta release", "The build uploaded successfully, but TestFlight processing is still pending."),
        ("invoice delay", "The vendor is waiting because the invoice has not been paid yet."),
        ("design review", "The mockups are ready, but engineering needs final notes before implementation."),
        ("support queue", "Three urgent tickets are blocked on account verification."),
        ("launch ads", "The first video ad is performing well, but the budget needs approval."),
        ("calendar change", "The 2pm meeting conflicts with a customer call."),
        ("prototype test", "The latest prototype works, but battery usage is higher than expected."),
    ]
    tones = [
        ("plain English", lambda text: text),
        ("concise workplace update", lambda text: text.split(", but ")[0] + "."),
        ("friendly note", lambda text: f"Quick update: {text}"),
        ("executive summary", lambda text: f"Summary: {text}"),
    ]
    names = ["alex", "priya", "omar", "maya", "lee", "nina", "sam", "taylor", "riley", "jordan"]
    index = 0
    while len(examples) < max_examples:
        subject, text = subjects[index % len(subjects)]
        tone_name, tone_fn = tones[(index // len(subjects)) % len(tones)]
        name = names[(index // 7) % len(names)]
        pretty_name = name.capitalize()
        task_family = index % 10
        if task_family == 0:
            instruction = f"Rewrite the user's provided text in {tone_name}. Preserve facts and do not add details."
            request = text
            content = tone_fn(text)
            examples.append(_row("Rewriter", instruction, request, "respond", content, "agent_rewrite", index))
        elif task_family == 1:
            instruction = "Summarize the user's provided text in one sentence. Preserve the main fact."
            request = f"{text} Please keep it short."
            content = text
            examples.append(_row("Summarizer", instruction, request, "respond", content, "agent_summarize", index))
        elif task_family == 2:
            instruction = "Turn the user's text into a short checklist of concrete actions."
            request = f"{text} Next we need to confirm owner, deadline, and follow-up."
            content = "- Confirm the owner.\n- Set the deadline.\n- Send the follow-up."
            examples.append(_row("Checklist Agent", instruction, request, "respond", content, "agent_checklist", index))
        elif task_family == 3:
            instruction = "Classify the user's text as bug, scheduling, finance, launch, or support. Return only the label."
            request = text
            if "invoice" in text or "budget" in text:
                content = "finance"
            elif "meeting" in text or "review" in text:
                content = "scheduling"
            elif "ticket" in text:
                content = "support"
            elif "bug" in subject or "signed out" in text:
                content = "bug"
            else:
                content = "launch"
            examples.append(_row("Classifier", instruction, request, "respond", content, "agent_classify", index))
        elif task_family == 4:
            instruction = "Extract the person, topic, and requested follow-up from the user's text."
            request = f"tell {name} about {subject} and ask for an update tomorrow"
            content = f"person: {pretty_name}\ntopic: {subject}\nfollow_up: ask for an update tomorrow"
            examples.append(_row("Extractor", instruction, request, "respond", content, "agent_extract", index))
        elif task_family == 5:
            instruction = "Triage the user's bug report. Return likely cause, severity, and next diagnostic step."
            request = f"{text} It started after the latest build."
            content = "Likely cause: recent build regression. Severity: medium until reproduced. Next diagnostic step: reproduce once and inspect the changed login/session path."
            examples.append(_row("Bug Triage", instruction, request, "respond", content, "agent_triage", index))
        elif task_family == 6:
            instruction = "Write a clear title for the user's text. Return only the title."
            request = text
            content = subject.title()
            examples.append(_row("Title Agent", instruction, request, "respond", content, "agent_title", index))
        elif task_family == 7:
            instruction = "Convert the user's rough note into a polite message. Preserve the recipient and the request."
            request = f"hey {name} can you send an update about {subject} tomorrow"
            content = f"Hi {pretty_name},\n\nCould you please send an update about {subject} tomorrow?\n\nThank you."
            examples.append(_row("Message Agent", instruction, request, "respond", content, "agent_message", index))
        elif task_family == 8:
            instruction = "Ask one concise clarifying question when the user does not provide the text or details needed for the task."
            request = ["make it better", "rewrite this", "turn it into a checklist", "what do you think?", "how's it going?"][(index // 10) % 5]
            content = "What text should I work on?"
            examples.append(_row("Clarifier", instruction, request, "ask_user", content, "agent_ask_missing_input", index))
        else:
            instruction = "Answer casual messages normally. Keep replies short and do not use stale context unless asked."
            request = ["thanks", "how's it going?", "can you help me?", "ok"][(index // 10) % 4]
            content = {
                "thanks": "You're welcome.",
                "how's it going?": "It's going well. What would you like help with?",
                "can you help me?": "Yes. What do you want to work on?",
                "ok": "Okay.",
            }[request]
            examples.append(_row("Friendly Assistant", instruction, request, "respond", content, "agent_casual", index))
        index += 1
    return examples[:max_examples]


def build_dataset(output_dir: Path, *, max_examples: int, eval_fraction: float) -> dict[str, Any]:
    rows = build_examples(max_examples)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        row["split"] = _hash_split(str(row["example_id"]), eval_fraction)
        if row["split"] == "eval":
            eval_rows.append(row)
        else:
            train_rows.append(row)
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "pocketpal_user_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_user_agent_eval.jsonl"
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
        task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_user_agent_curriculum",
        "manifest_path": str(output_dir / "pocketpal_user_agent_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"pocketpal_user_agent_curriculum": len(rows)},
        "target_action_counts": dict(sorted(action_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
        "schema": {
            "encoder_text": "PocketPal active agent instruction plus user request",
            "decoder_text": "compact JSON action/content decision",
        },
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/pocketpal_user_agent_curriculum")
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    args = parser.parse_args()
    print(json.dumps(build_dataset(Path(args.output_dir).resolve(), max_examples=args.max_examples, eval_fraction=args.eval_fraction), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
