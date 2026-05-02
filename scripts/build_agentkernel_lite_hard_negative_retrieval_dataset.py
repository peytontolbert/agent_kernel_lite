#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_agentkernel_lite_memory_quantization import _batched_embed, _embed_doc, _embed_query
from scripts.export_agentkernel_lite_neural_memory_pack import _compact, _doc_text, _iter_paper_rows, _load_model


def _paper_query_text(row: dict[str, Any]) -> str:
    title = _compact(row.get("title", ""), limit=220)
    categories = _compact(row.get("categories", ""), limit=120)
    if categories:
        return f"{title}\ncategory: {categories}"
    return title


def _training_row(row: dict[str, Any], doc_text: str, negatives: list[str]) -> dict[str, Any]:
    title = _compact(row.get("title", ""), limit=220)
    categories = _compact(row.get("categories", ""), limit=120)
    query = _paper_query_text(row)
    return {
        "encoder_text": (
            "<AK_USER> Find the paper that best matches this research request.\n"
            f"{query}\n"
            "<AK_RETRIEVE> <AK_RET_PAPERS> <AK_RET_SEMANTIC>"
        ),
        "decoder_text": "<AK_GATHER_CONTEXT> <AK_RETRIEVE> <AK_RET_PAPERS> <AK_CONF_MEDIUM>",
        "action": "gather_context",
        "task_type": "hard_negative_retrieval",
        "weight": 0.0,
        "retrieval_query_text": query,
        "retrieval_doc_text": doc_text,
        "retrieval_negative_doc_texts": json.dumps(negatives, ensure_ascii=False),
        "retrieval_loss_weight": 1.0,
        "query_confidence_target": 0.80,
        "retrieval_coverage_target": 0.85,
        "ood_query_target": 0.10,
        "ood_evidence_target": 0.10,
        "answer_confidence_target": 0.75,
        "needs_verification_target": 0.20,
        "paper_action_validity_target": 1.0,
        "metadata": {
            "paper_id": row.get("paper_id", ""),
            "title": title,
            "categories": categories,
            "source_file": row.get("source_file", ""),
            "row_offset": row.get("row_offset", 0),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(str(args.device))
    model, tokenizer, _manifest = _load_model(Path(args.bundle_dir).resolve(), repo_root=repo_root, device=device)
    paper_rows = list(
        _iter_paper_rows(
            Path(args.paper_dataset).resolve(),
            max_rows=int(args.max_rows),
            max_files=int(args.max_files),
        )
    )
    if len(paper_rows) < 2:
        raise SystemExit("need at least two paper rows to mine hard negatives")
    query_texts = [_paper_query_text(row) for row in paper_rows]
    doc_texts = [_doc_text(row) for row in paper_rows]
    queries = _batched_embed(
        _embed_query,
        model,
        tokenizer,
        query_texts,
        max_tokens=int(args.max_query_tokens),
        batch_size=int(args.batch_size),
        device=device,
    )
    docs = _batched_embed(
        _embed_doc,
        model,
        tokenizer,
        doc_texts,
        max_tokens=int(args.max_doc_tokens),
        batch_size=int(args.batch_size),
        device=device,
    )
    scores = queries @ docs.T
    np.fill_diagonal(scores, -np.inf)
    negative_count = max(1, int(args.negative_count))
    order = np.argsort(-scores, axis=1)[:, :negative_count]
    rows = [
        _training_row(paper_rows[index], doc_texts[index], [doc_texts[int(candidate)] for candidate in order[index]])
        for index in range(len(paper_rows))
    ]
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(rows)
    eval_count = min(max(1, int(round(len(rows) * float(args.eval_fraction)))), max(1, len(rows) // 5))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_hard_negative_retrieval_dataset",
        "objective": "hard_negative_retrieval",
        "manifest_path": str(output_dir / "dataset_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "paper_dataset": str(Path(args.paper_dataset).resolve()),
        "bundle_dir": str(Path(args.bundle_dir).resolve()),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "negative_count": negative_count,
        "max_query_tokens": int(args.max_query_tokens),
        "max_doc_tokens": int(args.max_doc_tokens),
        "seed": int(args.seed),
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--paper-dataset", default="/arxiv/huggingface/paper_text_1m_dedup_v1")
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/hard_negative_retrieval_dataset")
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
