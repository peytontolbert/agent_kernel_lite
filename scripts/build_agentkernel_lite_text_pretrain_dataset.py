#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compact(value: object, *, limit: int = 4000) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _repetition_score(text: str) -> float:
    words = [word.lower() for word in text.split() if word.strip()]
    if len(words) < 12:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - (len(set(bigrams)) / max(1, len(bigrams)))


def _sentences(text: str) -> list[str]:
    cleaned = _compact(text, limit=5000)
    if cleaned.lower().startswith("abstract:"):
        cleaned = cleaned[len("abstract:") :].strip()
    if cleaned.lower().startswith("title:"):
        marker_index = cleaned.lower().find(" abstract:")
        if marker_index >= 0:
            cleaned = cleaned[marker_index + len(" abstract:"):].strip()
    for marker in (" References ", " ACKNOWLEDG", " Appendix "):
        index = cleaned.find(marker)
        if index > 240:
            cleaned = cleaned[:index].strip()
    pieces = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = []
    for piece in pieces:
        sentence = piece.strip()
        if len(sentence) < 40:
            continue
        if len(sentence) > 420:
            sentence = sentence[:420].rsplit(" ", 1)[0].strip()
        if _repetition_score(sentence) > 0.35:
            continue
        sentences.append(sentence)
    return sentences


def _summary_from_text(*, title: str, text: str, categories: str) -> str:
    sentences = _sentences(text)
    if sentences:
        return " ".join(sentences[:2])[:520].rstrip()
    topic = title or "the retrieved paper"
    suffix = f" in {categories}" if categories else ""
    return (
        f"The retrieved evidence discusses {topic}{suffix}. "
        "A grounded answer should cite [1] and avoid claims beyond the excerpt."
    )


def _first_sentence(text: str) -> str:
    sentences = _sentences(text)
    return sentences[0] if sentences else _summary_from_text(title="", text=text, categories="")


def _short_takeaway(summary: str, *, limit: int = 260) -> str:
    sentences = _sentences(summary)
    takeaway = sentences[0] if sentences else _compact(summary, limit=limit)
    return takeaway[:limit].rstrip().rstrip(".")


def _assistant_chat_response(*, title: str, summary: str) -> str:
    takeaway = _short_takeaway(summary)
    if takeaway.lower().startswith("in this paper "):
        takeaway = takeaway[len("in this paper ") :].strip()
    return (
        f"The main idea is that {takeaway[0].lower() + takeaway[1:] if takeaway else title}. "
        "In practical terms, that gives a concrete research result rather than just a keyword match. "
        "I would treat this as a scoped answer from the retrieved abstract, not a full literature review [1]."
    )


def _assistant_takeaway_response(*, summary: str) -> str:
    takeaway = _short_takeaway(summary)
    return (
        f"The useful takeaway is: {takeaway} "
        "That is the part I would carry forward for a quick chat answer. For stronger confidence, I would compare it with more retrieved sources [1]."
    )


def _assistant_deep_research_response(*, title: str, summary: str) -> str:
    return (
        f"For a deeper read, I would start with this claim from [1]: {summary} "
        "My synthesis is that this source appears relevant, but this single abstract is only one evidence item. "
        "The next step would be to inspect the full paper and nearby work before making a broad conclusion."
    )


def _evidence_text_from_row(row: dict[str, Any], *, max_evidence_chars: int, prefer_abstract: bool) -> str:
    abstract = _compact(row.get("abstract", ""), limit=max_evidence_chars)
    body = _compact(row.get("text", ""), limit=max_evidence_chars)
    if prefer_abstract and len(abstract) >= 120:
        return f"Abstract: {abstract}"
    return abstract if len(abstract) >= 120 else body


def _continuation_pair(text: str) -> tuple[str, str] | None:
    sentences = _sentences(text)
    if len(sentences) < 4:
        return None
    context = " ".join(sentences[:2])[:700].rstrip()
    target = " ".join(sentences[2:4])[:480].rstrip()
    if len(context) < 120 or len(target) < 80:
        return None
    return context, target


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _examples_from_row(
    row: dict[str, Any],
    *,
    source_file: str,
    row_index: int,
    max_evidence_chars: int,
    prefer_abstract: bool,
    include_continuation: bool,
    include_copy_tasks: bool,
    include_chat_tasks: bool,
) -> list[dict[str, Any]]:
    raw_text = _evidence_text_from_row(
        row,
        max_evidence_chars=max_evidence_chars,
        prefer_abstract=prefer_abstract,
    )
    if len(raw_text) < 240:
        return []
    title = _compact(row.get("title", ""), limit=180) or "the retrieved paper"
    paper_id = _compact(row.get("paper_id", ""), limit=80)
    categories = _compact(row.get("categories", ""), limit=120)
    year = _compact(row.get("year", ""), limit=20)
    chunk_index = row.get("chunk_index", row_index)
    meta = " | ".join(part for part in [f"arXiv:{paper_id}" if paper_id else "", categories, year] if part)
    source_base = f"{source_file}:{chunk_index}:{row_index}"
    summary = _summary_from_text(title=title, text=raw_text, categories=categories)
    first_sentence = _first_sentence(raw_text)
    answer_scaffold = _short_takeaway(summary, limit=360)

    evidence_header = f"[1] {title}\n{meta}\n{raw_text}"
    rows = [
        {
            "source_type": "research_evidence_answer",
            "source_id": f"{source_base}:answer",
            "encoder_text": (
                "AgentKernel Lite text pretraining.\n"
                "Task: answer the user from retrieved research evidence.\n"
                f"User request: Explain the main point of {title} in one concise paragraph. Cite [1].\n"
                "Retrieved evidence:\n"
                f"{evidence_header}\n"
                f"Answer scaffold: {answer_scaffold}\n"
                "Answer in plain text. Reply like a helpful chat assistant. Cite [1] for claims from the evidence. Do not copy source metadata."
            ),
            "decoder_text": f"The main point is that {summary} This is a scoped summary from the retrieved evidence [1].",
            "action": "respond",
            "source_action": "respond",
            "extension_capability": "",
            "benchmark_family": "research_text_pretrain",
            "difficulty": "evidence_answer",
            "paper_id": paper_id,
            "weight": 1.0,
        },
        {
            "source_type": "research_evidence_synthesis",
            "source_id": f"{source_base}:synthesis",
            "encoder_text": (
                "AgentKernel Lite text pretraining.\n"
                "Task: synthesize retrieved evidence and mention uncertainty.\n"
                f"User request: What does this evidence say about {title}, and what is one limitation?\n"
                "Retrieved evidence:\n"
                f"{evidence_header}\n"
                f"Answer scaffold: {answer_scaffold}\n"
                "Answer in plain text. Reply like a helpful chat assistant. Cite [1] for claims from the evidence. Do not copy source metadata."
            ),
            "decoder_text": (
                f"The evidence supports this synthesis: {summary} "
                "The limitation is that this is one retrieved excerpt, so broader claims need more evidence [1]."
            ),
            "action": "respond",
            "source_action": "respond",
            "extension_capability": "",
            "benchmark_family": "research_text_pretrain",
            "difficulty": "evidence_synthesis",
            "paper_id": paper_id,
            "weight": 1.0,
        },
    ]
    if include_chat_tasks:
        rows.extend(
            [
                {
                    "source_type": "research_assistant_chat_answer",
                    "source_id": f"{source_base}:assistant_chat",
                    "encoder_text": (
                        "AgentKernel Lite assistant chat example.\n"
                        "Mode: chat.\n"
                        "Task: answer the user conversationally from retrieved evidence.\n"
                        f"User request: Can you explain what {title} is about in plain English?\n"
                        "Retrieved evidence:\n"
                        f"{evidence_header}\n"
                        f"Answer scaffold: {answer_scaffold}\n"
                        "Answer the user first. Use [1] only as a lightweight citation. Do not list paper metadata."
                    ),
                    "decoder_text": _assistant_chat_response(title=title, summary=summary),
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "research_text_pretrain",
                    "difficulty": "assistant_chat",
                    "paper_id": paper_id,
                    "weight": 1.4,
                },
                {
                    "source_type": "research_assistant_takeaway",
                    "source_id": f"{source_base}:assistant_takeaway",
                    "encoder_text": (
                        "AgentKernel Lite assistant chat example.\n"
                        "Mode: chat.\n"
                        "Task: give a concise useful takeaway from retrieved evidence.\n"
                        f"User request: What should I take away from {title}?\n"
                        "Retrieved evidence:\n"
                        f"{evidence_header}\n"
                        f"Answer scaffold: {answer_scaffold}\n"
                        "Answer naturally. Mention uncertainty if only one source is available. Do not copy source metadata."
                    ),
                    "decoder_text": _assistant_takeaway_response(summary=summary),
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "research_text_pretrain",
                    "difficulty": "assistant_takeaway",
                    "paper_id": paper_id,
                    "weight": 1.3,
                },
                {
                    "source_type": "research_assistant_deep_research",
                    "source_id": f"{source_base}:assistant_deep_research",
                    "encoder_text": (
                        "AgentKernel Lite assistant chat example.\n"
                        "Mode: deep_research.\n"
                        "Task: produce a careful research-style answer from retrieved evidence.\n"
                        f"User request: Do a deeper pass on {title}.\n"
                        "Retrieved evidence:\n"
                        f"{evidence_header}\n"
                        f"Answer scaffold: {summary}\n"
                        "Answer as an assistant. Separate supported evidence from uncertainty. Do not copy source metadata."
                    ),
                    "decoder_text": _assistant_deep_research_response(title=title, summary=summary),
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "research_text_pretrain",
                    "difficulty": "assistant_deep_research",
                    "paper_id": paper_id,
                    "weight": 1.2,
                },
            ]
        )
    if include_copy_tasks:
        rows.extend(
            [
                {
                    "source_type": "research_source_copy",
                    "source_id": f"{source_base}:source_copy",
                    "encoder_text": (
                        "AgentKernel Lite text pretraining.\n"
                        "Task: identify the retrieved source exactly.\n"
                        f"User request: What is source [1]?\n"
                        "Retrieved evidence:\n"
                        f"{evidence_header}\n"
                        "Answer in plain text. Copy the title and arXiv id from the retrieved evidence."
                    ),
                    "decoder_text": (
                        f"Source [1] is {title}"
                        + (f" (arXiv:{paper_id})." if paper_id else ".")
                    ),
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "research_text_pretrain",
                    "difficulty": "source_copy",
                    "paper_id": paper_id,
                    "weight": 1.2,
                },
                {
                    "source_type": "research_key_evidence_extract",
                    "source_id": f"{source_base}:key_evidence",
                    "encoder_text": (
                        "AgentKernel Lite text pretraining.\n"
                        "Task: extract the key evidence from the retrieved source.\n"
                        f"User request: Give the key evidence from [1] for {title}.\n"
                        "Retrieved evidence:\n"
                        f"{evidence_header}\n"
                        "Answer in plain text. Cite [1]. Do not copy source metadata."
                    ),
                    "decoder_text": f"Key evidence from [1]: {first_sentence}",
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "research_text_pretrain",
                    "difficulty": "key_evidence_extract",
                    "paper_id": paper_id,
                    "weight": 1.1,
                },
            ]
        )
    continuation = _continuation_pair(raw_text) if include_continuation else None
    if continuation is not None:
        context, target = continuation
        rows.append(
            {
                "source_type": "research_continuation",
                "source_id": f"{source_base}:continuation",
                "encoder_text": (
                    "AgentKernel Lite text pretraining.\n"
                    "Task: continue the research passage in the same technical style.\n"
                    f"Title: {title}\n"
                    f"Context: {context}\n"
                    "Continuation:"
                ),
                "decoder_text": target,
                "action": "respond",
                "source_action": "respond",
                "extension_capability": "",
                "benchmark_family": "research_text_pretrain",
                "difficulty": "continuation",
                "paper_id": paper_id,
                "weight": 0.7,
            }
        )
    return rows


def build_dataset(
    *,
    repo_root: Path,
    paper_chunk_root: Path,
    output_dir: Path,
    max_examples: int,
    max_files: int,
    eval_fraction: float,
    max_evidence_chars: int,
    prefer_abstract: bool,
    include_continuation: bool,
    include_copy_tasks: bool,
    include_chat_tasks: bool,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("building paper text pretraining data requires pyarrow") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(paper_chunk_root.glob("*.parquet"))
    if max_files > 0:
        paths = paths[:max_files]
    columns = ["text", "abstract", "paper_id", "title", "categories", "year", "chunk_index"]
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    total = 0
    seen_keys: set[str] = set()
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if "text" not in read_columns:
            continue
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=512, columns=read_columns):
            for raw_row in batch.to_pylist():
                for row in _examples_from_row(
                    raw_row,
                    source_file=path.name,
                    row_index=row_offset,
                    max_evidence_chars=max_evidence_chars,
                    prefer_abstract=prefer_abstract,
                    include_continuation=include_continuation,
                    include_copy_tasks=include_copy_tasks,
                    include_chat_tasks=include_chat_tasks,
                ):
                    key = hashlib.sha256(
                        f"{row['encoder_text']}\n-->\n{row['decoder_text']}".encode("utf-8")
                    ).hexdigest()
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    row["example_id"] = key
                    row["split"] = _hash_split(key, eval_fraction)
                    if row["split"] == "eval":
                        eval_rows.append(row)
                    else:
                        train_rows.append(row)
                    source_type = str(row["source_type"])
                    source_counts[source_type] = source_counts.get(source_type, 0) + 1
                    total += 1
                    if max_examples > 0 and total >= max_examples:
                        break
                row_offset += 1
                if max_examples > 0 and total >= max_examples:
                    break
            if max_examples > 0 and total >= max_examples:
                break
        if max_examples > 0 and total >= max_examples:
            break
    if not train_rows and eval_rows:
        train_rows.append(eval_rows.pop())
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "text",
        "repo_root": str(repo_root),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "target_action_counts": {"respond": len(train_rows) + len(eval_rows)},
        "source_action_counts": {"respond": len(train_rows) + len(eval_rows)},
        "extension_counts": {},
        "prefer_abstract": bool(prefer_abstract),
        "include_continuation": bool(include_continuation),
        "include_copy_tasks": bool(include_copy_tasks),
        "include_chat_tasks": bool(include_chat_tasks),
        "schema": {
            "encoder_text": "research text/evidence pretraining input for the encoder",
            "decoder_text": "plain text target for the decoder",
            "weight": "loss weight; trainer may ignore until weighted loss is enabled",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--paper-chunk-root", required=True)
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/text_pretrain_dataset")
    parser.add_argument("--max-examples", type=int, default=50000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--max-evidence-chars", type=int, default=1800)
    parser.add_argument("--prefer-abstract", type=int, choices=(0, 1), default=1)
    parser.add_argument("--include-continuation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--include-copy-tasks", type=int, choices=(0, 1), default=0)
    parser.add_argument("--include-chat-tasks", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest = build_dataset(
        repo_root=repo_root,
        paper_chunk_root=Path(args.paper_chunk_root).resolve(),
        output_dir=(repo_root / args.output_dir).resolve(),
        max_examples=int(args.max_examples),
        max_files=int(args.max_files),
        eval_fraction=float(args.eval_fraction),
        max_evidence_chars=int(args.max_evidence_chars),
        prefer_abstract=bool(args.prefer_abstract),
        include_continuation=bool(args.include_continuation),
        include_copy_tasks=bool(args.include_copy_tasks),
        include_chat_tasks=bool(args.include_chat_tasks),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
