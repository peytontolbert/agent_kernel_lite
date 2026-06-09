#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib import request as urlrequest

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


AGENT_OPERATIONS = [
    ("Rewriter", "Rewrite the user's text in {style}. Preserve names, dates, facts, and intent. If no editable text is provided, ask for it."),
    ("Summarizer", "Summarize the user's text as {format}. Preserve the main facts."),
    ("Planner", "Turn the user's request into {format} with concrete next actions."),
    ("Classifier", "Classify the user's text using the categories implied by the request. Return only the best label unless the instruction asks for more."),
    ("Extractor", "Extract the key fields from the user's text as {format}. Preserve source wording for names, dates, and values."),
    ("Reviewer", "Review the user's text for {domain}. Return the main issue and one practical improvement."),
    ("Explainer", "Explain the user's text in {style}. Keep it grounded in the provided text."),
    ("Draft Writer", "Draft a response in {style} using only the details the user provided. If details are missing, ask one concise question."),
    ("Decision Helper", "Given the user's situation, recommend the next step in {format}. Do not invent missing facts."),
    ("Tone Converter", "Convert the user's text to {style}. Preserve the meaning and all concrete details."),
]

STYLES = [
    "plain English",
    "a professional tone",
    "a friendly tone",
    "a concise workplace tone",
    "an executive tone",
    "a careful technical tone",
    "a customer-support tone",
    "a neutral factual tone",
]

FORMATS = [
    "one sentence",
    "three bullets",
    "short key-value lines",
    "a compact paragraph",
    "a short checklist",
    "a label plus one reason",
]

DOMAINS = [
    "clarity",
    "risk",
    "tone",
    "correctness",
    "support triage",
    "release readiness",
    "privacy",
    "actionability",
]


def build_agent_specs() -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for operation_name, template in AGENT_OPERATIONS:
        for style in STYLES:
            for output_format in FORMATS:
                for domain in DOMAINS[:4]:
                    instruction = template.format(style=style, format=output_format, domain=domain)
                    specs.append((f"{operation_name} Agent", instruction))
    specs.extend(
        [
            ("Clarifier", "Ask one concise clarifying question when the user's request lacks the text, details, or goal needed to complete the task."),
            ("Casual Assistant", "Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it."),
            ("Tool-Safe Assistant", "If the request requires an external tool or private data, ask before taking action. Otherwise answer directly."),
        ]
    )
    return specs

USER_TEXTS = [
    "hey john i need the report by friday because the client is asking and we are behind",
    "tell sarah i cannot make the 2pm today and ask if tomorrow morning works",
    "the beta release needs an owner deadline and follow up note",
    "users are getting signed out after refreshing the dashboard",
    "the vendor is waiting because the invoice has not been paid yet",
    "how's it going?",
    "rewrite this",
    "launch ads are working but budget approval is still missing",
    "extract person maya topic design review follow up tomorrow",
    "the prototype works but battery usage is higher than expected",
    "make this sound better",
    "support has three urgent tickets blocked on account verification",
    "meeting moved to 11 and priya needs the updated slides",
    "legal needs one more day on the contract draft",
    "thanks",
]


def _compact(value: object, *, limit: int = 5000) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].rstrip()


def _chat(endpoint: str, model: str, prompt: str, *, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write compact PocketPal training labels. Return only valid JSON with keys action and content. "
                    "Use action respond or ask_user. Treat the user request itself as the source text. "
                    "Only ask a question when the source text or target is genuinely missing. Do not include chain of thought."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    message = parsed["choices"][0]["message"]
    return str(message.get("content") or "")


def _parse_teacher(text: str) -> dict[str, str] | None:
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    action = str(parsed.get("action", "respond") or "respond")
    if action not in {"respond", "ask_user"}:
        action = "respond"
    content = _compact(parsed.get("content", ""), limit=3000)
    if not content or content.strip(". ") == "" or len(content) < 4:
        return None
    return {"action": action, "content": content}


def _encoder(agent_name: str, instruction: str, user_text: str) -> str:
    slots = compile_source_slots(user_text=user_text, max_slots=8)
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal teacher-distilled user agent example.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {agent_name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            "<AK_CONTEXT> Saved user data: none",
            source_slots_encoder_block(slots),
            "<AK_CONTEXT> Stale selected paper context: none",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _teacher_prompt(agent_name: str, instruction: str, user_text: str) -> str:
    return (
        f"Active PocketPal agent: {agent_name}\n"
        f"Agent instruction: {instruction}\n"
        f"User request: {user_text}\n\n"
        "Produce the assistant result for this active agent. "
        "The user request is the source text to transform, classify, summarize, extract, review, explain, or plan from. "
        "Do not ask for more details just because more context could be helpful. "
        "Ask one concise clarifying question only when the user literally omitted the text or required target, such as 'rewrite this', 'make this better', or 'translate it'. "
        "Return JSON only: {\"action\":\"respond|ask_user\",\"content\":\"...\"}."
    )


def generate(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = int(args.max_examples) * 3
    while len(rows) < int(args.max_examples) and attempts < max_attempts:
        attempts += 1
        agent_name, instruction = rng.choice(build_agent_specs())
        user_text = rng.choice(USER_TEXTS)
        prompt = _teacher_prompt(agent_name, instruction, user_text)
        try:
            answer = _parse_teacher(_chat(str(args.endpoint), str(args.model), prompt, timeout=int(args.timeout)))
        except Exception as exc:
            print(json.dumps({"event": "teacher_error", "error": str(exc), "attempt": attempts}), flush=True)
            time.sleep(0.5)
            continue
        if answer is None:
            continue
        encoder_text = _encoder(agent_name, instruction, user_text)
        source_slots = compile_source_slots(user_text=user_text, max_slots=8)
        content = pointerize_exact_text(answer["content"], source_slots)
        decoder_text = json.dumps(
            {
                "action": answer["action"],
                "content": content,
                "proposal_metadata": {
                    "task_type": "teacher_user_agent",
                    **source_slot_metadata(source_slots),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        example_id = hashlib.sha256(f"{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
        if any(row["example_id"] == example_id for row in rows):
            continue
        rows.append(
            {
                "example_id": example_id,
                "source_id": f"teacher_user_agent:{len(rows):06d}",
                "source_type": "pocketpal_teacher_user_agent",
                "task_type": "teacher_user_agent",
                "action": answer["action"],
                "encoder_text": encoder_text,
                "decoder_text": decoder_text,
                "weight": 2.5,
            }
        )
        if len(rows) % 25 == 0:
            print(json.dumps({"event": "generated", "rows": len(rows)}), flush=True)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        bucket = int(hashlib.sha256(row["example_id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        row["split"] = "eval" if bucket < float(args.eval_fraction) else "train"
        (eval_rows if row["split"] == "eval" else train_rows).append(row)
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())
    train_path = output_dir / "pocketpal_teacher_user_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_teacher_user_agent_eval.jsonl"
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_teacher_user_agent",
        "manifest_path": str(output_dir / "pocketpal_teacher_user_agent_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"pocketpal_teacher_user_agent": len(rows)},
        "target_action_counts": {
            "ask_user": sum(1 for row in rows if row["action"] == "ask_user"),
            "respond": sum(1 for row in rows if row["action"] == "respond"),
        },
        "task_type_counts": {"teacher_user_agent": len(rows)},
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", default="artifacts/pocketpal_teacher_user_agent_v1")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--timeout", type=int, default=60)
    print(json.dumps(generate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
