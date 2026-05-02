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
AK_STATE = "<AK_STATE>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_ACTIVE_CONTEXT = "<AK_ACTIVE_CONTEXT>"
AK_EVIDENCE = "<AK_EVIDENCE>"
AK_CANDIDATE = "<AK_CANDIDATE>"
AK_SELECTED_PAPER = "<AK_SELECTED_PAPER>"
AK_CONTEXT_ID = "<AK_CONTEXT_ID>"
AK_QUERY_REWRITE = "<AK_QUERY_REWRITE>"
AK_RERANK = "<AK_RERANK>"
AK_GATHER_CONTEXT = "<AK_GATHER_CONTEXT>"
AK_RESPOND = "<AK_RESPOND>"
AK_USE_CONTEXT = "<AK_USE_CONTEXT>"
AK_ANSWER = "<AK_ANSWER>"
AK_CITE = "<AK_CITE>"


def _compact(value: object, *, limit: int = 1200) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].rstrip()


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _line(action: str, content: str) -> str:
    return f"Action: {action}\nContent: {_compact(content, limit=900)}"


def _first_sentence(text: str, *, limit: int = 320) -> str:
    text = _compact(text, limit=1200)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if len(sentence) >= 40:
            return sentence[:limit].rstrip()
    return text[:limit].rstrip()


def _terms(title: str, abstract: str, *, limit: int = 7) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", f"{title} {abstract}")
    skip = {"the", "and", "for", "with", "from", "this", "that", "paper", "study", "studies"}
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        key = token.lower().strip("-")
        if key in skip or key in seen:
            continue
        seen.add(key)
        out.append(token.strip("-"))
        if len(out) >= limit:
            break
    return " ".join(out)


def _paper_id(row: dict[str, Any], fallback: str) -> str:
    return _compact(
        row.get("paper_id", row.get("canonical_paper_id", row.get("arxiv_id", row.get("id", fallback)))),
        limit=96,
    )


def _read_papers(path: Path, *, max_papers: int, max_files: int) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("decoder repair curriculum requires pyarrow") from exc
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    columns = ["paper_id", "canonical_paper_id", "arxiv_id", "id", "title", "abstract", "categories", "update_date"]
    rows: list[dict[str, str]] = []
    for file_path in paths:
        parquet_file = pq.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if "title" not in read_columns or "abstract" not in read_columns:
            continue
        row_index = 0
        for batch in parquet_file.iter_batches(batch_size=1024, columns=read_columns):
            for payload in batch.to_pylist():
                title = _compact(payload.get("title", ""), limit=180)
                abstract = _compact(payload.get("abstract", ""), limit=900)
                if not title or len(abstract) < 80:
                    row_index += 1
                    continue
                rows.append(
                    {
                        "paper_id": _paper_id(payload, f"{file_path.name}:{row_index}"),
                        "title": title,
                        "abstract": abstract,
                        "summary": _first_sentence(abstract),
                        "categories": _compact(payload.get("categories", ""), limit=80),
                        "date": _compact(payload.get("update_date", ""), limit=32),
                    }
                )
                row_index += 1
                if max_papers > 0 and len(rows) >= max_papers:
                    return rows
    return rows


def _candidate_block(candidates: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for index, row in enumerate(candidates, start=1):
        meta = " | ".join(part for part in [row["paper_id"], row["categories"], row["date"]] if part)
        parts.append(f"{AK_CANDIDATE} P{index}: {row['title']}\n{meta}\n{row['summary']}")
    return "\n\n".join(parts)


def _paper_examples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    n = len(rows)
    for index, row in enumerate(rows):
        terms = _terms(row["title"], row["abstract"])
        query = terms or row["title"]
        examples.append(
            {
                "source_type": "agentkernel_decoder_repair_curriculum",
                "source_id": f"{row['paper_id']}:query_rewrite",
                "task_type": "query_rewrite_clean",
                "encoder_text": (
                    f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_QUERY_REWRITE} "
                    "Rewrite the user request into a compact research-library search query. Do not answer yet. "
                    f"{AK_USER} Find papers about {query}. Return action=gather_context."
                ),
                "decoder_text": _line("gather_context", query),
                "action": "gather_context",
                "weight": 1.0,
            }
        )
        if n > 1:
            neg = rows[(index + max(1, n // 2)) % n]
            candidates = [row, neg] if index % 2 == 0 else [neg, row]
            label = f"P{candidates.index(row) + 1}"
            examples.append(
                {
                    "source_type": "agentkernel_decoder_repair_curriculum",
                    "source_id": f"{row['paper_id']}:rerank",
                    "task_type": "rerank_clean",
                    "encoder_text": (
                        f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_RERANK} "
                        "Select the candidate that best matches the user request. Do not invent a paper outside the candidate list. "
                        f"{AK_USER} {query} {AK_CONTEXT} Candidates:\n{_candidate_block(candidates)} "
                        "Return action=gather_context and the selected candidate id."
                    ),
                    "decoder_text": _line("gather_context", f"selected_candidate_id={label}"),
                    "action": "gather_context",
                    "weight": 1.2,
                }
            )
        examples.append(
            {
                "source_type": "agentkernel_decoder_repair_curriculum",
                "source_id": f"{row['paper_id']}:answer",
                "task_type": "answer_from_evidence_clean",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_ANSWER} AgentKernel Lite research answer training example. "
                    f"{AK_USER} What is the main contribution of {row['title']}? "
                    f"{AK_EVIDENCE} [1]: {row['title']} {row['summary']} "
                    "Answer directly, cite [1] for supported claims, and do not list unrelated papers."
                ),
                "decoder_text": _line(
                    "respond",
                    f"Based on [1], {row['title']} focuses on this: {row['summary']} This answer is grounded in the retrieved evidence [1].",
                ),
                "action": "respond",
                "weight": 1.7,
            }
        )
        examples.append(
            {
                "source_type": "agentkernel_decoder_repair_curriculum",
                "source_id": f"{row['paper_id']}:selected_followup",
                "task_type": "selected_context_answer_clean",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_ANSWER}\n"
                    "Answer the user's follow-up from the paper that is already loaded in chat. Do not say retrieval failed and do not introduce unrelated papers.\n"
                    f"{AK_CONTEXT} Active context:\nContext id P1\nSelected paper P1: {row['title']}\n"
                    f"{row['paper_id']} | {row['categories']} | {row['date']}\n{AK_EVIDENCE} [P1]: {row['summary']}\n"
                    f"{AK_USER} Tell me more about this paper please.\nAnswer directly, cite [P1] for supported claims, and keep the response conversational."
                ),
                "decoder_text": _line(
                    "respond",
                    f"The selected paper [P1], {row['title']}, is about this: {row['summary']} A useful way to read it is to focus on the problem, the method or argument, and the supported takeaway in the active evidence [P1].",
                ),
                "action": "respond",
                "weight": 1.8,
            }
        )
    return examples


def _seed_examples() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:query_rewrite_multi_agent_planning",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_QUERY_REWRITE} "
                "AgentKernel Lite retrieval training example. Rewrite the user request into a compact "
                "research-library search query. Do not answer yet. "
                f"{AK_USER} Find the best papers about multi-agent planning with grounded evidence. "
                "Return a structured decision with action=gather_context."
            ),
            "decoder_text": _line("gather_context", "multi-agent planning grounded evidence"),
            "action": "gather_context",
            "weight": 2.0,
        },
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:rerank_neural_retrieval",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_RERANK} AgentKernel Lite retrieval training example. "
                "Select the candidate that best matches the user request. Do not invent a paper outside the candidate list. "
                f"{AK_USER} neural retrieval scientific assistant {AK_CONTEXT} Candidates: "
                f"{AK_CANDIDATE} P1: Neural Retrieval for Scientific Assistants cs.AI | 2026 "
                "This paper studies neural retrieval for scientific assistant systems, candidate ranking, and grounded answer synthesis from long research documents. "
                f"{AK_CANDIDATE} P2: Distributed Storage Recovery cs.DC | 2026 "
                "This paper studies recovery algorithms for replicated storage systems after partial node failures. "
                "Return a structured decision with action=gather_context and the selected candidate id."
            ),
            "decoder_text": _line("gather_context", "selected_candidate_id=P1"),
            "action": "gather_context",
            "weight": 2.0,
        },
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:answer_neural_retrieval",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND} {AK_ANSWER} AgentKernel Lite research answer training example. "
                f"{AK_USER} What is the main contribution of Neural Retrieval for Scientific Assistants? "
                f"{AK_EVIDENCE} [1]: Neural Retrieval for Scientific Assistants This paper studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents. "
                "Answer directly, cite [1] for supported claims, and do not list unrelated papers."
            ),
            "decoder_text": _line(
                "respond",
                "Based on [1], Neural Retrieval for Scientific Assistants studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents. This answer is grounded in the retrieved evidence [1].",
            ),
            "action": "respond",
            "weight": 2.4,
        },
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:selected_neural_retrieval",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_ANSWER}\n"
                "AgentKernel Lite selected-context answer training example.\n"
                "Answer the user's follow-up from the paper that is already loaded in chat. Do not say retrieval failed and do not introduce unrelated papers.\n"
                f"{AK_CONTEXT} Active context:\nContext id P1\nSelected paper P1: Neural Retrieval for Scientific Assistants\n"
                f"2501.00001 | cs.AI | 2026\n{AK_EVIDENCE} [P1]: This paper studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents.\n"
                f"{AK_USER} Tell me more about this paper please.\nAnswer directly, cite [P1] for supported claims, and keep the response conversational."
            ),
            "decoder_text": _line(
                "respond",
                "The selected paper [P1], Neural Retrieval for Scientific Assistants, focuses on neural retrieval for scientific assistant systems. It connects candidate ranking with grounded answer synthesis over long research documents, so the main takeaway is that retrieval quality matters directly for the assistant response [P1].",
            ),
            "action": "respond",
            "weight": 2.4,
        },
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:direct_chat_multi_agent",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND} AgentKernel Lite chat example. "
                f"{AK_USER} Tell me about multi-agent intelligence in plain language. Return a structured decision with action=respond."
            ),
            "decoder_text": _line(
                "respond",
                "Multi-agent intelligence is about systems where multiple agents coordinate, communicate, compete, or divide work. In plain language, the core question is how a group of agents can share information and make better decisions than one agent alone.",
            ),
            "action": "respond",
            "weight": 2.0,
        },
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:web_compiled_direct_chat",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND}\n"
                "AgentKernel Lite browser chat example.\n"
                "You cannot execute, test, install, browse, or modify files from this environment.\n"
                "Return exactly this decision format: Action: respond, then Content: your direct answer.\n"
                "Agent loop: step 1 of 5; mode=chat; code_execution=disabled; target_language=Auto.\n"
                "Mode: Chat. Reply as a helpful assistant first. Use retrieved evidence as support, cite evidence numbers only when they improve clarity, and do not turn the answer into a paper list.\n"
                f"{AK_USER} Tell me about multi-agent intelligence in plain language.\n"
                "Return a structured decision with action=respond."
            ),
            "decoder_text": _line(
                "respond",
                "Multi-agent intelligence is about systems where multiple agents coordinate, communicate, compete, or divide work. In plain language, the core question is how a group of agents can share information and make better decisions than one agent alone.",
            ),
            "action": "respond",
            "weight": 2.6,
        },
        {
            "source_type": "agentkernel_decoder_repair_curriculum",
            "source_id": "probe:web_compiled_grounded_answer",
            "task_type": "probe_repair",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_ANSWER}\n"
                "AgentKernel Lite browser grounded-answer example.\n"
                "Return exactly this decision format: Action: respond, then Content: your direct answer.\n"
                "You are Agent Kernel Lite running entirely in this browser.\n"
                "You cannot execute, test, install, browse, or modify files from this environment.\n"
                "Use retrieved evidence when it is relevant, and say when it is not enough. Do not invent citations.\n"
                "Answer the user's question directly before mentioning sources. The interface displays paper titles and PDF links separately, so do not copy source metadata unless the user asks for it.\n"
                "Mode: Chat. Reply as a helpful assistant first. Use retrieved evidence as support, cite evidence numbers only when they improve clarity, and do not turn the answer into a paper list.\n"
                "Agent loop: step 1 of 5; mode=chat; code_execution=disabled; target_language=Auto.\n\n"
                "Context packet:\n"
                "{\"request_id\":\"browser-1\",\"retrieval\":{\"branch_scoped\":[{\"span_id\":\"span-1\",\"source_id\":\"2501.00001\",\"text\":\"This paper studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents.\",\"metadata\":{\"title\":\"Neural Retrieval for Scientific Assistants\"}}]}}\n\n"
                f"{AK_CONTEXT} Retrieved evidence:\n"
                f"{AK_EVIDENCE} [1]: Neural Retrieval for Scientific Assistants 2501.00001 This paper studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents.\n\n"
                f"{AK_ANSWER} Answer scaffold:\n"
                "This paper studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents.\n\n"
                f"{AK_USER} What is the main contribution of this paper?\n"
                "Answer directly, cite evidence like [1] only for supported claims, and keep the response conversational."
            ),
            "decoder_text": _line(
                "respond",
                "Based on [1], the paper studies neural retrieval for scientific assistant systems, especially candidate ranking and grounded answer synthesis from long research documents. The main contribution is connecting retrieval quality to the assistant's final grounded response [1].",
            ),
            "action": "respond",
            "weight": 2.8,
        },
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    paper_root = Path(args.paper_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_papers(paper_root, max_papers=int(args.max_papers), max_files=int(args.max_files))
    seed_examples: list[dict[str, Any]] = []
    for repeat in range(max(1, int(args.seed_repeat))):
        for row in _seed_examples():
            seed_examples.append({**row, "source_id": f"{row['source_id']}:repeat{repeat}"})
    examples = [*seed_examples, *_paper_examples(rows)]
    include_actions = {item.strip() for item in str(args.include_actions).split(",") if item.strip()}
    include_tasks = {item.strip() for item in str(args.include_task_types).split(",") if item.strip()}
    if include_actions:
        examples = [row for row in examples if str(row.get("action", "")) in include_actions]
    if include_tasks:
        examples = [row for row in examples if str(row.get("task_type", "")) in include_tasks]
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in examples:
        key = hashlib.sha256(f"{row['encoder_text']}\n-->\n{row['decoder_text']}".encode("utf-8")).hexdigest()
        split = _hash_split(key, float(args.eval_fraction))
        clean = {**row, "example_id": key, "split": split}
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
        (eval_rows if split == "eval" else train).append(clean)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    for path, split_rows in ((train_path, train), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "decoder_repair",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "paper_root": str(paper_root),
        "total_examples": len(examples),
        "train_examples": len(train),
        "eval_examples": len(eval_rows),
        "source_counts": {"agentkernel_decoder_repair_curriculum": len(examples)},
        "task_type_counts": dict(sorted(task_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", default="/arxiv/huggingface/paper_text_1m_dedup_v1")
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/decoder_repair_curriculum")
    parser.add_argument("--max-papers", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--seed-repeat", type=int, default=1)
    parser.add_argument("--include-actions", default="")
    parser.add_argument("--include-task-types", default="")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
