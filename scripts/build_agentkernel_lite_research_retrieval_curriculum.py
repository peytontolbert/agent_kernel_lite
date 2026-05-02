#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


AK_USER = "<AK_USER>"
AK_CHAT = "<AK_CHAT>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_ACTIVE_CONTEXT = "<AK_ACTIVE_CONTEXT>"
AK_EVIDENCE = "<AK_EVIDENCE>"
AK_CANDIDATE = "<AK_CANDIDATE>"
AK_SELECTED_PAPER = "<AK_SELECTED_PAPER>"
AK_CONTEXT_ID = "<AK_CONTEXT_ID>"
AK_TARGET_CONTEXT = "<AK_TARGET_CONTEXT>"
AK_RERANK = "<AK_RERANK>"
AK_ABSTRACT_RETRIEVAL = "<AK_ABSTRACT_RETRIEVAL>"
AK_GATHER_CONTEXT = "<AK_GATHER_CONTEXT>"
AK_RESPOND = "<AK_RESPOND>"
AK_USE_CONTEXT = "<AK_USE_CONTEXT>"
AK_FULL_TEXT = "<AK_FULL_TEXT>"
AK_LOAD_FULL_TEXT = "<AK_LOAD_FULL_TEXT>"
AK_NO_RETRIEVAL = "<AK_NO_RETRIEVAL>"
AK_ANSWER = "<AK_ANSWER>"
AK_CITE = "<AK_CITE>"
AK_RETRIEVAL_PAIR = "<AK_RETRIEVAL_PAIR>"
AK_DECISION = "<AK_DECISION>"
AK_NEED_MEMORY = "<AK_NEED_MEMORY>"
AK_MEMORY_TYPE = "<AK_MEMORY_TYPE>"
AK_MEMORY_QUERY = "<AK_MEMORY_QUERY>"
AK_USE_MEMORY = "<AK_USE_MEMORY>"


PARQUET_COLUMNS = [
    "example_id",
    "split",
    "source_type",
    "source_id",
    "task_type",
    "encoder_text",
    "decoder_text",
    "action",
    "weight",
    "retrieval_query_text",
    "retrieval_doc_text",
    "retrieval_loss_weight",
    "query_confidence_target",
    "retrieval_coverage_target",
    "ood_query_target",
    "ood_evidence_target",
    "answer_confidence_target",
    "needs_verification_target",
    "paper_action_validity_target",
]

POLICY_TARGET_COLUMNS = (
    "query_confidence_target",
    "retrieval_coverage_target",
    "ood_query_target",
    "ood_evidence_target",
    "answer_confidence_target",
    "needs_verification_target",
    "paper_action_validity_target",
)


def _policy_targets(
    *,
    query_confidence: float,
    retrieval_coverage: float,
    ood_query: float,
    ood_evidence: float,
    answer_confidence: float | None,
    needs_verification: float,
    paper_action_validity: float,
) -> dict[str, float | None]:
    return {
        "query_confidence_target": float(query_confidence),
        "retrieval_coverage_target": float(retrieval_coverage),
        "ood_query_target": float(ood_query),
        "ood_evidence_target": float(ood_evidence),
        "answer_confidence_target": None if answer_confidence is None else float(answer_confidence),
        "needs_verification_target": float(needs_verification),
        "paper_action_validity_target": float(paper_action_validity),
    }


def _compact(value: object, *, limit: int = 1600) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _sentences(text: str, *, limit: int = 6) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", _compact(text, limit=5000))
    out: list[str] = []
    for piece in pieces:
        sentence = piece.strip()
        if len(sentence) < 40:
            continue
        if len(sentence) > 520:
            sentence = sentence[:520].rsplit(" ", 1)[0].strip()
        out.append(sentence)
        if len(out) >= limit:
            break
    return out


def _summary(text: str, *, limit: int = 520) -> str:
    sentences = _sentences(text, limit=3)
    if not sentences:
        return _compact(text, limit=limit)
    return " ".join(sentences)[:limit].rstrip()


def _query_terms(title: str, abstract: str, *, limit: int = 8) -> str:
    text = f"{title} {abstract}"
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        key = token.lower().strip("-")
        if key in seen:
            continue
        seen.add(key)
        out.append(token.strip("-"))
        if len(out) >= limit:
            break
    return " ".join(out)


def _line(action: str, content: str) -> str:
    return f"Action: {action}\nContent: {_compact(content, limit=1200)}"


def _paper_id(row: dict[str, Any], fallback: str) -> str:
    return _compact(
        row.get("paper_id", row.get("canonical_paper_id", row.get("arxiv_id", row.get("id", fallback)))),
        limit=96,
    )


def _abstract_row(payload: dict[str, Any], *, source_file: str, row_index: int) -> dict[str, Any] | None:
    title = _compact(payload.get("title", ""), limit=220)
    abstract = _compact(payload.get("abstract", ""), limit=1400)
    if not title or len(abstract) < 120:
        return None
    return {
        "paper_id": _paper_id(payload, f"{source_file}:{row_index}"),
        "title": title,
        "abstract": abstract,
        "categories": _compact(payload.get("categories", payload.get("primary_category", "")), limit=160),
        "year": _compact(payload.get("year", payload.get("published_year", payload.get("update_date", ""))), limit=48),
        "source_file": source_file,
        "row_index": row_index,
    }


def _fulltext_row(payload: dict[str, Any], *, source_file: str, row_index: int) -> dict[str, Any] | None:
    title = _compact(payload.get("title", ""), limit=220)
    text = _compact(payload.get("text", ""), limit=2200)
    target = _compact(payload.get("target", ""), limit=1200)
    if not title or len(text) < 220:
        return None
    return {
        "paper_id": _paper_id(payload, f"{source_file}:{row_index}"),
        "title": title,
        "text": text,
        "target": target,
        "categories": _compact(payload.get("categories", ""), limit=160),
        "year": _compact(payload.get("year", ""), limit=48),
        "source_file": source_file,
        "row_index": row_index,
        "chunk_index": int(payload.get("chunk_index", row_index) or row_index),
    }


def _read_abstract_rows(path: Path, *, max_papers: int, max_files: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("building research retrieval curriculum requires pyarrow") from exc
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    columns = [
        "paper_id",
        "canonical_paper_id",
        "arxiv_id",
        "id",
        "title",
        "abstract",
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
        if "title" not in read_columns or "abstract" not in read_columns:
            continue
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=2048, columns=read_columns):
            for payload in batch.to_pylist():
                row = _abstract_row(payload, source_file=file_path.name, row_index=row_offset)
                row_offset += 1
                if row is None:
                    continue
                rows.append(row)
                if max_papers > 0 and len(rows) >= max_papers:
                    return rows
    return rows


def _iter_fulltext_rows(path: Path, *, max_chunks: int, max_files: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("building research retrieval curriculum requires pyarrow") from exc
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    columns = ["id", "paper_id", "title", "text", "target", "categories", "year", "chunk_index"]
    emitted = 0
    for file_path in paths:
        parquet_file = pq.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if "title" not in read_columns or "text" not in read_columns:
            continue
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=512, columns=read_columns):
            for payload in batch.to_pylist():
                row = _fulltext_row(payload, source_file=file_path.name, row_index=row_offset)
                row_offset += 1
                if row is None:
                    continue
                yield row
                emitted += 1
                if max_chunks > 0 and emitted >= max_chunks:
                    return


def _doc_text(row: dict[str, Any]) -> str:
    meta = " | ".join(part for part in [row.get("paper_id", ""), row.get("categories", ""), row.get("year", "")] if part)
    return _compact(f"{row['title']}\n{meta}\n{row['abstract']}", limit=1700)


def _candidate_block(candidates: list[dict[str, Any]]) -> str:
    parts = []
    for index, row in enumerate(candidates, start=1):
        meta = " | ".join(part for part in [row.get("paper_id", ""), row.get("categories", ""), row.get("year", "")] if part)
        parts.append(f"{AK_CANDIDATE} P{index}: {row['title']}\n{meta}\n{_summary(row['abstract'], limit=360)}")
    return "\n\n".join(parts)


def _abstract_examples(rows: list[dict[str, Any]], *, candidates_per_query: int) -> Iterator[dict[str, Any]]:
    n = len(rows)
    if n < 2:
        return
    for index, row in enumerate(rows):
        query_terms = _query_terms(row["title"], row["abstract"], limit=8)
        variant = index % 4
        if variant == 0:
            query = row["title"]
        elif variant == 1:
            query = f"find research papers about {query_terms}"
        elif variant == 2:
            query = f"what paper should I read for {query_terms}"
        else:
            query = f"explain work related to {query_terms}"

        candidates = [row]
        stride = max(1, n // max(2, candidates_per_query + 2))
        offset = stride
        while len(candidates) < max(2, candidates_per_query):
            candidate = rows[(index + offset) % n]
            offset += stride
            if candidate["paper_id"] == row["paper_id"]:
                continue
            candidates.append(candidate)
        rotation = index % len(candidates)
        rotated = candidates[rotation:] + candidates[:rotation]
        label = f"P{rotated.index(row) + 1}"
        source_id = f"{row['paper_id']}:{index}:abstract_retrieval"
        yield {
            "source_type": "agentkernel_research_retrieval_curriculum",
            "source_id": source_id,
            "task_type": "abstract_retrieval_rerank",
            "encoder_text": (
                f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_ABSTRACT_RETRIEVAL} {AK_RERANK}\n"
                "Select the paper candidate that best matches the user request. Return only the selected candidate id in the action content.\n"
                f"{AK_USER} {query}\n"
                f"{AK_CONTEXT} Candidate abstracts:\n{_candidate_block(rotated)}"
            ),
            "decoder_text": _line(
                "gather_context",
                f"{AK_DECISION} {AK_NEED_MEMORY} {AK_MEMORY_TYPE} paper {AK_MEMORY_QUERY} {query} selected_candidate_id={label}",
            ),
            "action": "gather_context",
            "weight": 1.8,
            "retrieval_query_text": f"{AK_RETRIEVAL_PAIR} {query}",
            "retrieval_doc_text": f"{AK_RETRIEVAL_PAIR} {_doc_text(row)}",
            "retrieval_loss_weight": 1.0,
            **_policy_targets(
                query_confidence=0.9,
                retrieval_coverage=0.82,
                ood_query=0.04,
                ood_evidence=0.08,
                answer_confidence=None,
                needs_verification=0.18,
                paper_action_validity=1.0,
            ),
        }


def _active_abstract_context(row: dict[str, Any]) -> str:
    meta = " | ".join(part for part in [row.get("paper_id", ""), row.get("categories", ""), row.get("year", "")] if part)
    return (
        f"{AK_CONTEXT_ID} P1\n"
        f"{AK_SELECTED_PAPER} P1: {row['title']}\n"
        f"{meta}\n"
        f"{AK_EVIDENCE} [P1 abstract]: {_summary(row['text'], limit=360)}"
    )


def _active_fulltext_context(row: dict[str, Any]) -> str:
    meta = " | ".join(part for part in [row.get("paper_id", ""), row.get("categories", ""), row.get("year", "")] if part)
    continuation = f"\n{row['target']}" if row.get("target") else ""
    full_text = _compact(f"{row['text']}{continuation}", limit=2600)
    return (
        f"{AK_CONTEXT_ID} P1\n"
        f"{AK_SELECTED_PAPER} P1: {row['title']}\n"
        f"{meta}\n"
        f"{AK_FULL_TEXT} [P1 excerpt]: {full_text}"
    )


def _fulltext_answer(row: dict[str, Any]) -> str:
    summary = _summary(f"{row['text']} {row.get('target', '')}", limit=640)
    return (
        f"The selected paper [P1], {row['title']}, is saying this in the loaded full-text excerpt: {summary} "
        "I would treat that as the local evidence for this follow-up, rather than starting a new paper search [P1]."
    )


def _fulltext_examples(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        source_base = f"{row['paper_id']}:{row['source_file']}:{row['chunk_index']}"
        yield {
            "source_type": "agentkernel_research_retrieval_curriculum",
            "source_id": f"{source_base}:load_full_text_plan",
            "task_type": "selected_paper_load_fulltext_plan",
            "encoder_text": (
                f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_LOAD_FULL_TEXT}\n"
                "The user is asking a detailed follow-up about the currently selected paper, but only abstract context is loaded. Load full-text context for that selected paper.\n"
                f"{AK_CONTEXT} Active context:\n{_active_abstract_context(row)}\n"
                f"{AK_USER} Explain this paper in more detail, including the method and what the evidence supports."
            ),
            "decoder_text": _line(
                "gather_context",
                f"{AK_DECISION} {AK_NEED_MEMORY} {AK_MEMORY_TYPE} paper_full_text {AK_LOAD_FULL_TEXT} {AK_TARGET_CONTEXT}=P1 reason=selected_paper_followup_requires_full_text",
            ),
            "action": "gather_context",
            "weight": 2.2,
            **_policy_targets(
                query_confidence=0.92,
                retrieval_coverage=0.38,
                ood_query=0.03,
                ood_evidence=0.18,
                answer_confidence=None,
                needs_verification=0.82,
                paper_action_validity=1.0,
            ),
        }
        yield {
            "source_type": "agentkernel_research_retrieval_curriculum",
            "source_id": f"{source_base}:full_text_followup_answer",
            "task_type": "selected_paper_fulltext_answer",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT} {AK_FULL_TEXT}\n"
                "The user is asking about the paper already loaded in chat. Answer from the loaded full-text excerpt and do not retrieve new papers.\n"
                f"{AK_CONTEXT} Active context:\n{_active_fulltext_context(row)}\n"
                f"{AK_USER} Tell me more about this paper please.\n"
                f"Answer directly and cite {AK_CITE} [P1]."
            ),
            "decoder_text": _line("respond", f"{AK_DECISION} {AK_USE_MEMORY} {AK_TARGET_CONTEXT}=P1 {_fulltext_answer(row)}"),
            "action": "respond",
            "weight": 2.8,
            **_policy_targets(
                query_confidence=0.94,
                retrieval_coverage=0.9,
                ood_query=0.03,
                ood_evidence=0.06,
                answer_confidence=0.78,
                needs_verification=0.28,
                paper_action_validity=1.0,
            ),
        }
        yield {
            "source_type": "agentkernel_research_retrieval_curriculum",
            "source_id": f"{source_base}:full_text_method_answer",
            "task_type": "selected_paper_fulltext_answer",
            "encoder_text": (
                f"{AK_CHAT} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT} {AK_FULL_TEXT}\n"
                "The user asks for method-level detail about the selected paper. Use the loaded full-text excerpt.\n"
                f"{AK_CONTEXT} Active context:\n{_active_fulltext_context(row)}\n"
                f"{AK_USER} What is the method or core argument in this paper?"
            ),
            "decoder_text": _line("respond", f"{AK_DECISION} {AK_USE_MEMORY} {AK_TARGET_CONTEXT}=P1 {_fulltext_answer(row)}"),
            "action": "respond",
            "weight": 2.6,
            **_policy_targets(
                query_confidence=0.94,
                retrieval_coverage=0.88,
                ood_query=0.03,
                ood_evidence=0.07,
                answer_confidence=0.76,
                needs_verification=0.32,
                paper_action_validity=1.0,
            ),
        }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


class DatasetWriter:
    def __init__(self, output_dir: Path, *, output_format: str, parquet_shard_size: int) -> None:
        self.output_dir = output_dir
        self.output_format = str(output_format)
        self.parquet_shard_size = max(1, int(parquet_shard_size))
        self.counts = {"train": 0, "eval": 0}
        self.shard_counts = {"train": 0, "eval": 0}
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
        self.handles: dict[str, Any] = {}
        if self.output_format == "jsonl":
            self.train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
            self.eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
            self.handles["train"] = self.train_path.open("w", encoding="utf-8")
            self.handles["eval"] = self.eval_path.open("w", encoding="utf-8")
        elif self.output_format == "parquet":
            self.train_path = output_dir / "train"
            self.eval_path = output_dir / "eval"
            self.train_path.mkdir(parents=True, exist_ok=True)
            self.eval_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"unknown output format: {self.output_format}")

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = {key: row.get(key) for key in PARQUET_COLUMNS}
        normalized["weight"] = float(normalized.get("weight") or 1.0)
        normalized["retrieval_loss_weight"] = float(normalized.get("retrieval_loss_weight") or 0.0)
        for key in POLICY_TARGET_COLUMNS:
            value = normalized.get(key)
            if value is None or value == "":
                normalized[key] = None
            else:
                try:
                    normalized[key] = float(value)
                except (TypeError, ValueError):
                    normalized[key] = None
        return normalized

    def _write_parquet_shard(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("parquet output requires pyarrow") from exc
        shard_index = self.shard_counts[split]
        path = (self.train_path if split == "train" else self.eval_path) / f"part-{shard_index:05d}.parquet"
        schema = pa.schema(
            [
                pa.field("example_id", pa.string()),
                pa.field("split", pa.string()),
                pa.field("source_type", pa.string()),
                pa.field("source_id", pa.string()),
                pa.field("task_type", pa.string()),
                pa.field("encoder_text", pa.string()),
                pa.field("decoder_text", pa.string()),
                pa.field("action", pa.string()),
                pa.field("weight", pa.float32()),
                pa.field("retrieval_query_text", pa.string()),
                pa.field("retrieval_doc_text", pa.string()),
                pa.field("retrieval_loss_weight", pa.float32()),
                pa.field("query_confidence_target", pa.float32()),
                pa.field("retrieval_coverage_target", pa.float32()),
                pa.field("ood_query_target", pa.float32()),
                pa.field("ood_evidence_target", pa.float32()),
                pa.field("answer_confidence_target", pa.float32()),
                pa.field("needs_verification_target", pa.float32()),
                pa.field("paper_action_validity_target", pa.float32()),
            ]
        )
        table = pa.Table.from_pylist([self._normalize(row) for row in rows], schema=schema)
        pq.write_table(table, path, compression="zstd")
        self.buffers[split] = []
        self.shard_counts[split] += 1

    def write(self, split: str, row: dict[str, Any]) -> None:
        if split not in {"train", "eval"}:
            raise ValueError(f"unknown split: {split}")
        self.counts[split] += 1
        if self.output_format == "jsonl":
            self.handles[split].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            return
        self.buffers[split].append(row)
        if len(self.buffers[split]) >= self.parquet_shard_size:
            self._write_parquet_shard(split)

    def close(self) -> None:
        if self.output_format == "jsonl":
            for handle in self.handles.values():
                handle.close()
            return
        self._write_parquet_shard("train")
        self._write_parquet_shard("eval")


def build(args: argparse.Namespace) -> dict[str, Any]:
    abstract_root = Path(args.abstract_paper_root).expanduser().resolve()
    fulltext_root = Path(args.fulltext_chunk_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    abstract_rows = _read_abstract_rows(
        abstract_root,
        max_papers=int(args.max_abstract_papers),
        max_files=int(args.max_abstract_files),
    )
    fulltext_rows = _iter_fulltext_rows(
        fulltext_root,
        max_chunks=int(args.max_fulltext_chunks),
        max_files=int(args.max_fulltext_files),
    )
    examples = itertools.chain(
        _abstract_examples(abstract_rows, candidates_per_query=int(args.candidates_per_query)),
        _fulltext_examples(fulltext_rows),
    )
    seen: set[str] = set()
    writer = DatasetWriter(
        output_dir,
        output_format=str(args.output_format),
        parquet_shard_size=int(args.parquet_shard_size),
    )
    task_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    retrieval_pair_count = 0
    try:
        for row in examples:
            key = hashlib.sha256(f"{row['encoder_text']}\n-->\n{row['decoder_text']}".encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            split = _hash_split(key, float(args.eval_fraction))
            clean = {**row, "example_id": key, "split": split}
            if clean.get("retrieval_query_text") and clean.get("retrieval_doc_text"):
                retrieval_pair_count += 1
            task = str(clean.get("task_type", "") or "")
            action = str(clean.get("action", "") or "")
            task_counts[task] = task_counts.get(task, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
            writer.write(split, clean)
    finally:
        writer.close()

    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "chat_retrieval_contrastive",
        "decoder_format": "line",
        "dataset_format": writer.output_format,
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(writer.train_path),
        "eval_dataset_path": str(writer.eval_path),
        "abstract_paper_root": str(abstract_root),
        "fulltext_chunk_root": str(fulltext_root),
        "total_examples": writer.counts["train"] + writer.counts["eval"],
        "train_examples": writer.counts["train"],
        "eval_examples": writer.counts["eval"],
        "train_shards": int(writer.shard_counts["train"]),
        "eval_shards": int(writer.shard_counts["eval"]),
        "retrieval_pair_count": int(retrieval_pair_count),
        "source_counts": {"agentkernel_research_retrieval_curriculum": writer.counts["train"] + writer.counts["eval"]},
        "task_type_counts": dict(sorted(task_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
        "schema": {
            "encoder_text": "AgentKernel Lite loop/action prompt",
            "decoder_text": "line protocol action/content target",
            "retrieval_query_text": "optional encoder-side retrieval query for contrastive loss",
            "retrieval_doc_text": "optional encoder-side positive paper abstract for contrastive loss",
            "retrieval_loss_weight": "optional contrastive loss weight for query/document pairs",
            "weight": "decoder loss multiplier",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstract-paper-root", default="/arxiv/huggingface/paper_text_1m_dedup_v1")
    parser.add_argument("--fulltext-chunk-root", default="/data/tmp/p1_full_paper_lm_hf_all_chunks")
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/research_retrieval_curriculum")
    parser.add_argument("--max-abstract-papers", type=int, default=100000)
    parser.add_argument("--max-abstract-files", type=int, default=0)
    parser.add_argument("--max-fulltext-chunks", type=int, default=50000)
    parser.add_argument("--max-fulltext-files", type=int, default=0)
    parser.add_argument("--candidates-per-query", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--output-format", choices=("parquet", "jsonl"), default="parquet")
    parser.add_argument("--parquet-shard-size", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
