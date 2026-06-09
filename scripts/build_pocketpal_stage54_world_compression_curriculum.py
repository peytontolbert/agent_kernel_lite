#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


INTENT_LABELS = {
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

NAMES = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Morgan", "Nora", "Priya", "Riley"]
OBJECTS = ["budget sheet", "client deck", "invoice packet", "launch memo", "release notes", "security review"]
DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow"]
BLOCKERS = ["design feedback", "finance review", "legal approval", "manager approval", "QA signoff", "vendor confirmation"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _decision(content: str, task_type: str) -> str:
    return json.dumps(
        {"action": "respond", "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _prompt(agent_name: str, instruction: str, user_text: str, source_text: str, task_type: str, intent: str) -> str:
    return (
        "<AK_CHAT> <AK_RESPOND> PocketPal world-compression active-agent example.\n"
        "<AK_AGENT_ACTIVE>\n"
        f"Agent name: {agent_name}\n"
        f"Agent instruction: {instruction}\n"
        "Retrieval policy: current_context_only\n"
        "Tool policy: no_tools\n"
        "Action policy: respond_or_ask\n"
        "</AK_AGENT_ACTIVE>\n"
        f"<AK_TASK_HINT> intent={intent} task={task_type} latent_state_supervised=true\n"
        "<AK_CONTEXT> Stale context: Blake owns a launch memo due Monday and finance review is blocking launch.\n"
        "<AK_PROFILE> User text slots:\n"
        f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={source_text}\n"
        "Equivalent wording should map to the same task state. Use SOURCE_TEXT only.\n"
        f"<AK_USER> {user_text}\n"
        "Return compact JSON with action=respond and content only."
    )


def _row(
    *,
    source_id: str,
    task_type: str,
    intent: str,
    contrastive_label_id: int,
    agent_name: str,
    instruction: str,
    user_text: str,
    source_text: str,
    expected: str,
    weight: float,
) -> dict[str, Any]:
    encoder_text = _prompt(agent_name, instruction, user_text, source_text, task_type, intent)
    decoder_text = _decision(expected, task_type)
    return {
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{decoder_text}".encode("utf-8")).hexdigest(),
        "source_type": "pocketpal_stage54_world_compression_curriculum",
        "source_id": source_id,
        "task_type": task_type,
        "encoder_text": encoder_text,
        "decoder_text": decoder_text,
        "expected_content": expected,
        "action": "respond",
        "source_action": "respond",
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS[intent],
        "contrastive_label_id": int(contrastive_label_id),
        "retrieval_query_text": "",
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "negative_decoder_text": "",
        "negative_loss_weight": 0.0,
        "weight": float(weight),
        "query_confidence_target": 0.95,
        "retrieval_coverage_target": 1.0,
        "ood_query_target": 0.0,
        "ood_evidence_target": 0.0,
        "answer_confidence_target": 0.95,
        "needs_verification_target": 0.1,
    }


def _task_outputs(owner: str, reviewer: str, obj: str, date: str, blocker: str) -> dict[str, tuple[str, str, str]]:
    source = f"{owner} will send the {obj} by {date}. {reviewer} will review it, and {blocker} is blocking launch."
    extraction = f"- Owner: {owner}\n- Reviewer: {reviewer}\n- Object: {obj}\n- Date: {date}\n- Blocker: {blocker}"
    plan = f"1. {owner} sends the {obj} by {date}.\n2. {reviewer} reviews the {obj}.\n3. Resolve {blocker} before launch."
    compact = json.dumps({"intent": "extraction", "owner": owner, "reviewer": reviewer, "object": obj, "date": date, "blocker": blocker}, sort_keys=True, separators=(",", ":"))
    return {
        "active_agent_extraction": ("extraction", "Extract owner, reviewer, object, date, and blocker from SOURCE_TEXT.", extraction),
        "active_agent_json": ("json", "Return only a compact JSON object with intent, owner, reviewer, object, date, and blocker from SOURCE_TEXT.", compact),
        "active_agent_plan": ("plan", "Make a short launch plan from SOURCE_TEXT. Use only stated facts.", plan),
    }


def _paraphrases(owner: str, reviewer: str, obj: str, date: str, blocker: str) -> list[str]:
    return [
        f"{owner} will send the {obj} by {date}. {reviewer} will review it, and {blocker} is blocking launch.",
        f"Launch note: {owner} owns the {obj} due {date}; {reviewer} reviews it; blocker is {blocker}.",
        f"Before launch, {owner} needs to deliver the {obj} by {date}. Review is with {reviewer}. Waiting on {blocker}.",
    ]


def _simple_rows(rng: random.Random, start_state_id: int, weight: float, examples: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    state_id = start_state_id
    for index in range(examples):
        owner, reviewer = rng.sample(NAMES, 2)
        obj = rng.choice(OBJECTS)
        date = rng.choice(DATES)
        blocker = rng.choice(BLOCKERS)
        paraphrases = _paraphrases(owner, reviewer, obj, date, blocker)
        outputs = _task_outputs(owner, reviewer, obj, date, blocker)
        for task_type, (intent, instruction, expected) in outputs.items():
            label = state_id
            state_id += 1
            for variant_index, source in enumerate(paraphrases):
                rows.append(
                    _row(
                        source_id=f"stage54_state_{index:05d}_{task_type}_{variant_index}",
                        task_type=task_type,
                        intent=intent,
                        contrastive_label_id=label,
                        agent_name=f"{intent.title()} Agent",
                        instruction=instruction,
                        user_text=source,
                        source_text=source,
                        expected=expected,
                        weight=weight,
                    )
                )
        # Translation and brainstorm are simpler but still get same-state paraphrases.
        phrase = rng.choice([
            ("Can you call me after lunch?", "Pouvez-vous m'appeler apres le dejeuner?"),
            ("Please review the proposal before Friday.", "Veuillez examiner la proposition avant vendredi."),
        ])
        label = state_id
        state_id += 1
        for variant_index, user_text in enumerate([phrase[0], f"Translate this into French: {phrase[0]}", f"SOURCE_TEXT: {phrase[0]}"]):
            rows.append(
                _row(
                    source_id=f"stage54_translate_{index:05d}_{variant_index}",
                    task_type="active_agent_translation",
                    intent="translation",
                    contrastive_label_id=label,
                    agent_name="French Translator",
                    instruction="Translate SOURCE_TEXT into French. Return only the translation.",
                    user_text=user_text,
                    source_text=phrase[0],
                    expected=phrase[1],
                    weight=weight,
                )
            )
        ideas = "1. Add a search button in chat\n2. Show source cards with clickable links\n3. Let users set the max source count"
        label = state_id
        state_id += 1
        for variant_index, user_text in enumerate(["Make web search easier in chat.", "Ideas for better web search UX.", "Brainstorm source cards and max source count controls."]):
            rows.append(
                _row(
                    source_id=f"stage54_brainstorm_{index:05d}_{variant_index}",
                    task_type="active_agent_brainstorm",
                    intent="brainstorm",
                    contrastive_label_id=label,
                    agent_name="Brainstorm Agent",
                    instruction="Brainstorm three concrete product ideas from SOURCE_TEXT.",
                    user_text=user_text,
                    source_text=user_text,
                    expected=ideas,
                    weight=weight,
                )
            )
    return rows, state_id


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    rows, _ = _simple_rows(rng, 0, float(args.weight), int(args.examples))
    rng.shuffle(rows)
    eval_count = max(1, int(len(rows) * float(args.eval_ratio)))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    counts: dict[str, int] = {}
    for row in rows:
        task = str(row["task_type"])
        counts[task] = counts.get(task, 0) + 1
    manifest = {
        "objective": "pocketpal_stage54_world_compression_curriculum",
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(rows),
        "intent_labels": INTENT_LABELS,
        "task_type_counts": counts,
        "contrastive_label": "same latent world/task state across paraphrases",
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage54_world_compression_curriculum")
    parser.add_argument("--examples", type=int, default=1400)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--weight", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=54)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
