#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

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

DEFAULT_SPECS = [
    # Small, high-signal alignment and reasoning sources first.
    "nvidia/HelpSteer2::train:ranking:1200",
    "nvidia/Nemotron-SFT-Math-v3::train:summary:1200",
    "nvidia/AceMath-Instruct-Training-Data::math_sft:summary:1200",
    "nvidia/OpenMathInstruct-2::train:summary:1200",
    # Agent/tool/retrieval shaped sources. These may fail if gated or renamed; failures are recorded.
    "nvidia/When2Call::mcq:plan:1200",
    "nvidia/Retrieval-Synthetic-NVDocs-v1::train:summary:1200",
    "nvidia/SWE-Zero-openhands-trajectories::train:checklist:800",
    "nvidia/SWE-Hero-openhands-trajectories::train:checklist:800",
    "nvidia/Nemotron-Terminal-Corpus:skill_based_mixed:train:checklist:800",
]


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _compact(value: object, *, limit: int = 1800) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _ak_decision(task_type: str, content: str) -> str:
    return (
        f"<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> {task_type} "
        f"<AK_CONTENT> {_compact(content, limit=1400)} </AK_CONTENT> <AK_END>"
    )


def _json_decision(task_type: str, content: str) -> str:
    return json.dumps(
        {"action": "respond", "content": _compact(content, limit=1400), "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _task_for(intent: str) -> str:
    return f"active_agent_{intent}" if intent in INTENT_LABELS else "active_agent_summary"


def _prompt(*, intent: str, task_type: str, user: str, context: str = "", dataset_name: str = "") -> str:
    lines = [
        "<AK_CHAT> <AK_RESPOND> PocketPal NVIDIA curriculum example.",
        "<AK_AGENT_ACTIVE>",
        "Agent name: Local Skill Synthesizer",
        "Agent instruction: Answer using the provided high-quality source signal. Prefer compact, structured reasoning over style.",
        "Retrieval policy: auto",
        "Tool policy: ask_before_extensions",
        "Action policy: respond_or_ask",
        "The active agent instruction is the primary task contract for this turn.",
        "</AK_AGENT_ACTIVE>",
        f"<AK_TASK_HINT> intent={intent} task={task_type} source_text_required=true",
        "<AK_CONTEXT> Saved user data: none",
    ]
    if dataset_name:
        lines.append(f"<AK_CONTEXT> Source dataset: {dataset_name}")
    if context:
        lines.append(f"<AK_CONTEXT> Retrieved/high-quality source: {_compact(context, limit=1200)}")
    lines.extend(
        [
            f"<AK_USER> {_compact(user, limit=1200)}",
            "Return AK structured tokens for the active agent decision.",
        ]
    )
    return "\n".join(lines)


def _row(
    *,
    dataset_name: str,
    source_id: str,
    intent: str,
    user: str,
    content: str,
    context: str = "",
    retrieval_doc: str = "",
    weight: float = 3.0,
) -> dict[str, Any] | None:
    intent = intent if intent in INTENT_LABELS else "summary"
    task_type = _task_for(intent)
    user = _compact(user, limit=1200)
    content = _compact(content, limit=1400)
    if not user or not content:
        return None
    encoder_text = _prompt(intent=intent, task_type=task_type, user=user, context=context, dataset_name=dataset_name)
    decoder_text = _ak_decision(task_type, content)
    retrieval_doc = _compact(retrieval_doc or content, limit=1600)
    return {
        "action": "respond",
        "decoder_loss_weight": 1.0,
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": _stable_id("nvidia_hf", dataset_name, source_id, encoder_text, decoder_text),
        "expected_content": content,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS[intent],
        "json_decoder_text": _json_decision(task_type, content),
        "negative_decoder_text": None,
        "negative_loss_weight": None,
        "retrieval_doc_text": retrieval_doc,
        "retrieval_loss_weight": 1.0 if retrieval_doc else 0.0,
        "retrieval_query_text": user,
        "source_id": source_id,
        "source_type": "nvidia_hf_curriculum",
        "split": "train",
        "state_text": f"dataset={dataset_name}; intent={intent}; content={content}",
        "task_type": task_type,
        "weight": float(weight),
    }


def _flatten_strings(value: Any, *, limit: int = 12) -> list[str]:
    out: list[str] = []
    stack = [value]
    while stack and len(out) < limit:
        item = stack.pop(0)
        if item is None:
            continue
        if isinstance(item, str):
            text = _compact(item, limit=1200)
            if text:
                out.append(text)
        elif isinstance(item, dict):
            for key in ("content", "text", "value", "answer", "response", "solution", "problem", "question", "prompt", "instruction"):
                if key in item:
                    stack.append(item[key])
            if not out:
                stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float, bool)):
            out.append(str(item))
    return out


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            text = " ".join(_flatten_strings(row[key], limit=4))
            if text:
                return text
    return ""


def _messages_to_pair(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""
    prompt_parts: list[str] = []
    answer_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").lower()
        text = _compact(msg.get("content") or msg.get("value") or msg.get("text"), limit=1200)
        if not text:
            continue
        if "assistant" in role or role in {"gpt", "model"}:
            answer_parts.append(text)
        else:
            prompt_parts.append(text)
    return " ".join(prompt_parts), " ".join(answer_parts)


def _quality_score(row: dict[str, Any]) -> float:
    keys = ("helpfulness", "correctness", "coherence", "complexity", "verbosity", "quality", "score", "rating")
    vals: list[float] = []
    for key in keys:
        try:
            vals.append(float(row[key]))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def _extract_pair(row: dict[str, Any], *, fallback_intent: str) -> tuple[str, str, str, str]:
    qa_pairs = row.get("deduplicated_qa_pairs")
    if not qa_pairs and isinstance(row.get("qa_generation"), dict):
        qa_pairs = row["qa_generation"].get("pairs")
    if isinstance(qa_pairs, list) and qa_pairs:
        pair = next((item for item in qa_pairs if isinstance(item, dict)), None)
        if isinstance(pair, dict):
            question = _first_text(pair, ("question", "query", "prompt"))
            answer = _first_text(pair, ("answer", "response", "solution"))
            context = _first_text(row, ("text", "sections_structured", "chunks", "document_artifacts"))
            if question and answer:
                return question, answer, context, context or answer

    for msg_key in ("messages", "conversations", "conversation"):
        if msg_key in row:
            prompt, answer = _messages_to_pair(row[msg_key])
            if prompt and answer:
                return prompt, answer, "", answer
            if prompt:
                side_answer = _first_text(
                    row,
                    (
                        "response",
                        "answer",
                        "output",
                        "solution",
                        "generated_solution",
                        "expected_answer",
                        "completion",
                        "final_answer",
                    ),
                )
                if side_answer:
                    return prompt, side_answer, "", side_answer

    prompt = _first_text(
        row,
        (
            "prompt",
            "instruction",
            "question",
            "problem",
            "query",
            "input",
            "task",
            "title",
            "text",
        ),
    )
    answer = _first_text(
        row,
        (
            "response",
            "answer",
            "output",
            "solution",
            "generated_solution",
            "expected_answer",
            "completion",
            "final_answer",
            "chosen",
            "label",
        ),
    )
    context = _first_text(row, ("context", "document", "doc", "passage", "retrieved_context", "source"))
    retrieval_doc = _first_text(row, ("positive", "positive_doc", "document", "doc", "passage", "retrieved_context"))

    if fallback_intent == "ranking":
        responses = []
        for key in ("response", "responses", "chosen", "rejected"):
            if key in row:
                responses.extend(_flatten_strings(row[key], limit=6))
        if prompt and responses:
            answer = max(responses, key=len)
            score = _quality_score(row)
            return prompt, f"Preferred response score {score:.2f}: {answer}", context, retrieval_doc or answer

    if not answer and context:
        answer = context
    return prompt, answer, context, retrieval_doc or answer


def _parse_spec(spec: str) -> tuple[str, str | None, str, str, int]:
    parts = spec.split(":")
    while len(parts) < 5:
        parts.append("")
    name, config, split, intent, limit_text = parts[:5]
    split = split or "train"
    intent = intent or "summary"
    limit = int(limit_text or 1000)
    return name, config or None, split, intent, limit


def _load_stream(name: str, config: str | None, split: str):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        return load_dataset(name, config, **kwargs)
    return load_dataset(name, **kwargs)


def _build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    specs = args.dataset_spec or DEFAULT_SPECS
    for spec in specs:
        name, config, split, intent, limit = _parse_spec(spec)
        report: dict[str, Any] = {"dataset": name, "config": config, "split": split, "intent": intent, "limit": limit}
        try:
            stream = _load_stream(name, config, split)
            count = 0
            kept = 0
            for row_index, source_row in enumerate(stream):
                if count >= limit:
                    break
                if not isinstance(source_row, dict):
                    continue
                prompt, answer, context, retrieval_doc = _extract_pair(source_row, fallback_intent=intent)
                built = _row(
                    dataset_name=name if not config else f"{name}/{config}",
                    source_id=f"{name}:{config or 'default'}:{split}:{row_index}",
                    intent=intent,
                    user=prompt,
                    content=answer,
                    context=context,
                    retrieval_doc=retrieval_doc,
                    weight=args.weight,
                )
                count += 1
                if built is None:
                    continue
                kept += 1
                rows.append(built)
            report.update({"status": "ok", "scanned": count, "kept": kept})
        except Exception as exc:
            report.update({"status": "failed", "error": repr(exc)})
        reports.append(report)

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[row["example_id"]] = row
    rows = list(dedup.values())
    rng.shuffle(rows)
    docs = [str(row.get("retrieval_doc_text") or "") for row in rows if str(row.get("retrieval_doc_text") or "").strip()]
    for index, row in enumerate(rows):
        negatives: list[str] = []
        if docs:
            offset = (index * 7 + 3) % len(docs)
            for step in range(len(docs)):
                candidate = docs[(offset + step * 11) % len(docs)]
                if candidate and candidate != row.get("retrieval_doc_text"):
                    negatives.append(candidate)
                if len(negatives) >= int(args.negatives):
                    break
        row["retrieval_negative_doc_texts"] = json.dumps(negatives, ensure_ascii=False)
    return rows, reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_nvidia_hf_curriculum"))
    parser.add_argument(
        "--dataset-spec",
        action="append",
        help="Dataset spec as name:config:split:intent:max_rows. Empty config is allowed, e.g. nvidia/HelpSteer2::train:ranking:1000",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8751)
    parser.add_argument("--weight", type=float, default=3.0)
    args = parser.parse_args()

    rows, reports = _build_rows(args)
    output_dir = Path(args.output_dir).resolve()
    eval_count = max(1, int(len(rows) * float(args.eval_ratio))) if rows else 0
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_nvidia_hf_curriculum",
        "dataset_reports": reports,
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_nvidia_hf_curriculum",
        "source_datasets": args.dataset_spec or DEFAULT_SPECS,
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
        "total_examples": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
