#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


AK_USER = "<AK_USER>"
AK_CHAT = "<AK_CHAT>"
AK_LOOP = "<AK_LOOP>"
AK_PLAN = "<AK_PLAN>"
AK_STATE = "<AK_STATE>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_ACTIVE_CONTEXT = "<AK_ACTIVE_CONTEXT>"
AK_EVIDENCE = "<AK_EVIDENCE>"
AK_EVIDENCE_ID = "<AK_EVIDENCE_ID>"
AK_TITLE = "<AK_TITLE>"
AK_PAPER_ID = "<AK_PAPER_ID>"
AK_CATEGORY = "<AK_CATEGORY>"
AK_YEAR = "<AK_YEAR>"
AK_ABSTRACT = "<AK_ABSTRACT>"
AK_CANDIDATE = "<AK_CANDIDATE>"
AK_SELECTED_PAPER = "<AK_SELECTED_PAPER>"
AK_CONTEXT_ID = "<AK_CONTEXT_ID>"
AK_TARGET_CONTEXT = "<AK_TARGET_CONTEXT>"
AK_RERANK = "<AK_RERANK>"
AK_GATHER_CONTEXT = "<AK_GATHER_CONTEXT>"
AK_RESPOND = "<AK_RESPOND>"
AK_USE_CONTEXT = "<AK_USE_CONTEXT>"
AK_NO_RETRIEVAL = "<AK_NO_RETRIEVAL>"
AK_ANSWER = "<AK_ANSWER>"
AK_CITE = "<AK_CITE>"


def _state(*, selected_context: bool = False, retrieval: str = "none", mode: str = "chat") -> str:
    return (
        f"{AK_LOOP} {AK_STATE} mode={mode} "
        f"selected_context={'1' if selected_context else '0'} "
        f"retrieval={retrieval}"
    )


def _compact(value: object, *, limit: int = 1200) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _term_tokens(*values: object, limit: int = 7) -> list[str]:
    text = " ".join(str(value or "") for value in values)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    skip = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "paper",
        "study",
        "studies",
        "using",
        "based",
        "toward",
    }
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        token = token.strip("-")
        if token in skip or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _first_sentences(text: str, *, limit: int = 360) -> str:
    text = _compact(text, limit=1400)
    if not text:
        return ""
    pieces: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in ".!?":
            sentence = "".join(current).strip()
            if sentence:
                pieces.append(sentence)
            current = []
            if len(" ".join(pieces)) >= limit:
                break
    if not pieces and current:
        pieces.append("".join(current).strip())
    return " ".join(pieces).strip()[:limit].rstrip()


def _extract_tagged_field(text: str, label: str) -> str:
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)(?=^\s*[A-Z_ ]+\s*:|\Z)"
    )
    match = pattern.search(str(text or ""))
    return _compact(match.group(1), limit=1400) if match else ""


def _line(action: str, content: str) -> str:
    return f"Action: {action}\nContent: {_compact(content, limit=900)}"


def _paper_row(payload: dict[str, Any], *, source_file: str, row_index: int) -> dict[str, Any] | None:
    tagged_text = str(payload.get("text_a", "") or payload.get("text_b", "") or "")
    title = _compact(payload.get("title", "") or _extract_tagged_field(tagged_text, "TITLE") or _extract_tagged_field(tagged_text, "Title"), limit=180)
    abstract = _compact(payload.get("abstract", "") or _extract_tagged_field(tagged_text, "ABSTRACT") or _extract_tagged_field(tagged_text, "Abstract"), limit=1200)
    text = _compact(payload.get("text", payload.get("full_text", "")), limit=1400)
    summary = _first_sentences(abstract or text, limit=360)
    if not title or len(summary) < 70:
        return None
    return {
        "paper_id": _compact(
            payload.get("paper_id", payload.get("canonical_paper_id", payload.get("arxiv_id", payload.get("id", "")))),
            limit=80,
        ),
        "title": title,
        "summary": summary,
        "categories": _compact(payload.get("categories", payload.get("primary_category", "")), limit=120),
        "year": _compact(payload.get("year", payload.get("published_year", payload.get("update_date", ""))), limit=40),
        "source_file": source_file,
        "row_index": row_index,
    }


def _read_paper_rows(path: Path, *, max_examples: int, max_files: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("loop curriculum parquet loading requires pyarrow") from exc
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    columns = [
        "paper_id",
        "canonical_paper_id",
        "arxiv_id",
        "id",
        "text_a",
        "text_b",
        "title",
        "abstract",
        "text",
        "full_text",
        "categories",
        "primary_category",
        "year",
        "published_year",
        "update_date",
    ]
    rows: list[dict[str, Any]] = []
    for file_path in paths:
        parquet_file = pq.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        has_structured_text = bool({"text_a", "text_b"} & set(read_columns))
        has_title = "title" in read_columns or has_structured_text
        has_body = bool({"abstract", "text", "full_text", "text_a", "text_b"} & set(read_columns))
        if not has_title or not has_body:
            continue
        for batch in parquet_file.iter_batches(batch_size=512, columns=read_columns):
            for row_index, payload in enumerate(batch.to_pylist()):
                row = _paper_row(payload, source_file=file_path.name, row_index=row_index)
                if not row:
                    continue
                rows.append(row)
                if max_examples > 0 and len(rows) >= max_examples:
                    return rows
    return rows


def _candidate_block(candidates: list[dict[str, Any]]) -> str:
    parts = []
    for index, row in enumerate(candidates, start=1):
        meta = " | ".join(item for item in [row["paper_id"], row["categories"], str(row["year"])] if item)
        parts.append(
            "\n".join(
                [
                    f"{AK_CANDIDATE} {AK_EVIDENCE_ID} P{index}",
                    f"{AK_TITLE} {row['title']}",
                    f"{AK_PAPER_ID} {row['paper_id']}",
                    f"{AK_CATEGORY} {row['categories']}",
                    f"{AK_YEAR} {row['year']}",
                    f"{AK_ABSTRACT} {row['summary']}",
                    meta,
                ]
            ).strip()
        )
    return "\n\n".join(parts)


def _active_context(row: dict[str, Any]) -> str:
    meta = " | ".join(item for item in [row["paper_id"], row["categories"], str(row["year"])] if item)
    return (
        f"{AK_CONTEXT_ID} P1\n"
        f"{AK_SELECTED_PAPER} {AK_EVIDENCE_ID} P1\n"
        f"{AK_TITLE} {row['title']}\n"
        f"{AK_PAPER_ID} {row['paper_id']}\n"
        f"{AK_CATEGORY} {row['categories']}\n"
        f"{AK_YEAR} {row['year']}\n"
        f"{AK_EVIDENCE} {AK_ABSTRACT} [P1]: {row['summary']}\n"
        f"{meta}"
    )


def _negative(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return rows[(index + max(1, len(rows) // 3)) % len(rows)]


def _examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if len(rows) > 1:
            neg = _negative(rows, index)
            candidates = [row, neg] if index % 2 == 0 else [neg, row]
            label = "P1" if candidates[0] is row else "P2"
            query = " ".join(_term_tokens(row["title"], row["summary"], limit=7)) or row["title"]
            examples.append(
                {
                    "source_type": "agentkernel_loop_curriculum",
                    "source_id": f"{row['paper_id'] or index}:rerank_copy",
                    "task_type": "rerank_candidates",
                    "encoder_text": (
                        f"{AK_CHAT} {AK_PLAN} {AK_GATHER_CONTEXT} {AK_RERANK}\n"
                        f"{_state(selected_context=False, retrieval='candidates')}\n"
                        f"{AK_USER} {query}\n"
                        f"{AK_CONTEXT}\n{_candidate_block(candidates)}"
                    ),
                    "decoder_text": _line("gather_context", f"selected_candidate_id={label}"),
                    "action": "gather_context",
                    "weight": 2.0,
                }
            )

        evidence_prompt = (
            f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_ANSWER}\n"
            f"{_state(selected_context=False, retrieval='ranked')}\n"
            f"{AK_USER} What is the main contribution of {row['title']}?\n"
            f"{AK_EVIDENCE} {AK_EVIDENCE_ID} P1\n"
            f"{AK_TITLE} {row['title']}\n"
            f"{AK_PAPER_ID} {row['paper_id']}\n"
            f"{AK_CATEGORY} {row['categories']}\n"
            f"{AK_YEAR} {row['year']}\n"
            f"{AK_ABSTRACT} {row['summary']}\n"
            "Cite [1] for supported claims."
        )
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:grounded_answer_copy",
                "task_type": "grounded_answer_copy",
                "encoder_text": evidence_prompt,
                "decoder_text": _line(
                    "respond",
                    f"Based on [1], the retrieved evidence focuses on this contribution: {row['summary']} The answer is grounded in the retrieved evidence [1].",
                ),
                "action": "respond",
                "weight": 2.4,
            }
        )

        active = _active_context(row)
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:selected_context_answer_copy",
                "task_type": "selected_context_answer",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT}\n"
                    f"{_state(selected_context=True, retrieval='active')}\n"
                    f"{AK_CONTEXT}\n{active}\n"
                    f"{AK_USER} Tell me more about this paper please."
                ),
                "decoder_text": _line(
                    "respond",
                    f"The selected paper [P1] focuses on this: {row['summary']} I would read it around the problem it studies, the method or argument it uses, and the supported takeaway in the active evidence [P1].",
                ),
                "action": "respond",
                "weight": 2.6,
            }
        )
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:selected_context_plan_copy",
                "task_type": "selected_context_followup_plan",
                "encoder_text": (
                    f"{AK_CHAT} {AK_PLAN} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER}\n"
                    f"{_state(selected_context=True, retrieval='active')}\n"
                    f"{AK_CONTEXT}\n{active}\n"
                    f"{AK_USER} Explain this paper in more detail."
                ),
                "decoder_text": _line("respond", f"{AK_USE_CONTEXT} {AK_TARGET_CONTEXT}=P1 {AK_NO_RETRIEVAL}"),
                "action": "respond",
                "weight": 2.0,
            }
        )
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:evidence_binding_copy",
                "task_type": "evidence_binding_copy",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_ANSWER}\n"
                    f"{_state(selected_context=False, retrieval='ranked')}\n"
                    f"{AK_USER} Explain the most relevant retrieved source for {' '.join(_term_tokens(row['title'], row['summary'], limit=5))}.\n"
                    f"{AK_CONTEXT}\n{_candidate_block([row])}"
                ),
                "decoder_text": _line(
                    "respond",
                    f"The most relevant retrieved source is [1]. It matters here because {row['summary']} My answer is grounded in that evidence item rather than a title or claim outside the provided context.",
                ),
                "action": "respond",
                "weight": 3.0,
            }
        )
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:title_field_binding",
                "task_type": "title_field_binding",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_ANSWER}\n"
                    f"{_state(selected_context=False, retrieval='ranked')}\n"
                    f"{AK_USER} Answer using the retrieved evidence item.\n"
                    f"{AK_CONTEXT}\n{_candidate_block([row])}"
                ),
                "decoder_text": _line(
                    "respond",
                    f"The retrieved evidence item is [1]. The relevant support is: {row['summary']}",
                ),
                "action": "respond",
                "weight": 4.0,
            }
        )
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:selected_title_field_binding",
                "task_type": "selected_title_field_binding",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT}\n"
                    f"{_state(selected_context=True, retrieval='active')}\n"
                    f"{AK_CONTEXT}\n{active}\n"
                    f"{AK_USER} What paper is this, and what is its main point?"
                ),
                "decoder_text": _line(
                    "respond",
                    f"This is the selected paper [P1]. Its main point is: {row['summary']}",
                ),
                "action": "respond",
                "weight": 4.0,
            }
        )

        topic = " ".join(_term_tokens(row["title"], limit=4)) or row["title"]
        examples.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"{row['paper_id'] or index}:direct_chat",
                "task_type": "direct_chat",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND}\n"
                    f"{_state(selected_context=False, retrieval='none')}\n"
                    f"{AK_USER} Tell me about {topic} in plain language."
                ),
                "decoder_text": _line(
                    "respond",
                    f"{topic} is a research topic about how systems, methods, or models address that problem. In plain language, I would explain the core idea first, then separate the method, evidence, and limitations. If you want a grounded literature answer, I can gather papers next.",
                ),
                "action": "respond",
                "weight": 1.5,
            }
        )
    return examples


def _seed_chat_examples() -> list[dict[str, Any]]:
    seeds = [
        (
            "multi-agent intelligence",
            "Multi-agent intelligence is about systems where multiple agents coordinate, communicate, compete, or divide work. The important questions are how they share information, avoid conflicting actions, plan jointly, and produce better results than a single agent.",
        ),
        (
            "multi-agent LLM systems",
            "Multi-agent LLM systems use several language-model agents with different roles or tools. A good answer should separate orchestration, communication, verification, memory, and evaluation, because more agents only help when the coordination loop is reliable.",
        ),
        (
            "retrieval augmented generation",
            "Retrieval augmented generation improves answers by fetching relevant evidence before generation. The model should use retrieved context for grounded claims, cite or surface the evidence when available, and avoid presenting weak retrieval as certainty.",
        ),
    ]
    rows = []
    for index, (topic, answer) in enumerate(seeds):
        rows.append(
            {
                "source_type": "agentkernel_loop_curriculum",
                "source_id": f"seed_direct_chat:{index}",
                "task_type": "direct_chat",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND}\n"
                    f"{_state(selected_context=False, retrieval='none')}\n"
                    f"{AK_USER} Tell me about {topic} in plain language."
                ),
                "decoder_text": _line("respond", answer),
                "action": "respond",
                "weight": 2.0,
            }
        )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    paper_path = Path(args.paper_text_path).expanduser().resolve()
    rows = _read_paper_rows(paper_path, max_examples=int(args.max_paper_text_examples), max_files=int(args.max_files))
    examples = [*_seed_chat_examples(), *_examples(rows)]
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in examples:
        split = _hash_split(str(row["source_id"]), float(args.eval_fraction))
        clean = {**row, "split": split}
        if split == "eval":
            eval_rows.append(clean)
        else:
            train_rows.append(clean)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    task_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in examples:
        task = str(row.get("task_type", "") or "")
        action = str(row.get("action", "") or "")
        task_counts[task] = task_counts.get(task, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "chat",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "paper_text_path": str(paper_path),
        "total_examples": len(examples),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"agentkernel_loop_curriculum": len(examples)},
        "task_type_counts": dict(sorted(task_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
        "schema": {
            "encoder_text": "AgentKernel Lite loop/action prompt",
            "decoder_text": "line protocol action/content target",
            "weight": "weighted loss multiplier",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-text-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-paper-text-examples", type=int, default=20000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
