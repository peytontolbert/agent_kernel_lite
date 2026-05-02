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

from scripts.export_agentkernel_lite_neural_memory_pack import (
    _doc_text,
    _encode_batch,
    _iter_paper_rows,
    _load_model,
    _quantize_vectors,
    _quantize_vectors_ternary,
    _quantize_vectors_ternary_grouped,
    _quantize_vectors_ternary_grouped_signed,
    _repo_root,
)


def _embed_query(model, tokenizer, texts: list[str], *, max_tokens: int, device: torch.device) -> np.ndarray:
    ids, mask = _encode_batch(tokenizer, texts, max_tokens=max_tokens, device=device)
    with torch.no_grad():
        if hasattr(model, "retrieval_query_embedding"):
            embeddings = model.retrieval_query_embedding(ids, mask)
        else:
            hidden = model.encode(ids, mask)
            weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            embeddings = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def _embed_doc(model, tokenizer, texts: list[str], *, max_tokens: int, device: torch.device) -> np.ndarray:
    ids, mask = _encode_batch(tokenizer, texts, max_tokens=max_tokens, device=device)
    with torch.no_grad():
        if hasattr(model, "retrieval_doc_embedding"):
            embeddings = model.retrieval_doc_embedding(ids, mask)
        else:
            hidden = model.encode(ids, mask)
            weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            embeddings = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def _batched_embed(fn, model, tokenizer, texts: list[str], *, max_tokens: int, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for offset in range(0, len(texts), batch_size):
        chunks.append(fn(model, tokenizer, texts[offset : offset + batch_size], max_tokens=max_tokens, device=device))
    return np.concatenate(chunks, axis=0)


def _score_int8(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    quantized, scales = _quantize_vectors(docs)
    return queries @ (quantized.astype(np.float32) * scales[:, None]).T


def _unpack_ternary(packed: np.ndarray, *, dimension: int) -> np.ndarray:
    rows = packed.shape[0]
    unpacked = np.zeros((rows, dimension), dtype=np.float32)
    for shift in range(4):
        values = ((packed >> (shift * 2)) & 3)[:, : ((dimension + 3 - shift) // 4)]
        decoded = np.where(values == 1, 1.0, np.where(values == 2, -1.0, 0.0)).astype(np.float32)
        unpacked[:, shift::4] = decoded[:, : unpacked[:, shift::4].shape[1]]
    return unpacked


def _score_ternary(queries: np.ndarray, docs: np.ndarray, *, threshold_ratio: float) -> np.ndarray:
    packed, scales, _density = _quantize_vectors_ternary(docs, threshold_ratio=threshold_ratio)
    decoded = _unpack_ternary(packed, dimension=docs.shape[1]) * scales[:, None]
    return queries @ decoded.T


def _score_ternary_grouped(
    queries: np.ndarray,
    docs: np.ndarray,
    *,
    threshold_ratio: float,
    group_size: int,
) -> np.ndarray:
    packed, scales, _density = _quantize_vectors_ternary_grouped(
        docs,
        threshold_ratio=threshold_ratio,
        group_size=group_size,
    )
    decoded = _unpack_ternary(packed, dimension=docs.shape[1])
    for group_index in range(scales.shape[1]):
        start = group_index * group_size
        end = min(docs.shape[1], start + group_size)
        decoded[:, start:end] *= scales[:, group_index : group_index + 1]
    return queries @ decoded.T


def _score_ternary_grouped_signed(
    queries: np.ndarray,
    docs: np.ndarray,
    *,
    threshold_ratio: float,
    group_size: int,
) -> np.ndarray:
    packed, scales, _density = _quantize_vectors_ternary_grouped_signed(
        docs,
        threshold_ratio=threshold_ratio,
        group_size=group_size,
    )
    decoded = _unpack_ternary(packed, dimension=docs.shape[1])
    group_count = scales.shape[1] // 2
    for group_index in range(group_count):
        start = group_index * group_size
        end = min(docs.shape[1], start + group_size)
        block = decoded[:, start:end]
        pos = block > 0
        neg = block < 0
        block[pos] = np.take(scales[:, group_index * 2], np.nonzero(pos)[0])
        block[neg] = -np.take(scales[:, group_index * 2 + 1], np.nonzero(neg)[0])
        decoded[:, start:end] = block
    return queries @ decoded.T


def _decode_ternary_simple_docs(docs: np.ndarray, *, threshold_ratio: float) -> np.ndarray:
    packed, scales, _density = _quantize_vectors_ternary(docs, threshold_ratio=threshold_ratio)
    return _unpack_ternary(packed, dimension=docs.shape[1]) * scales[:, None]


def _decode_ternary_grouped_signed_docs(
    docs: np.ndarray,
    *,
    threshold_ratio: float,
    group_size: int,
) -> np.ndarray:
    packed, scales, _density = _quantize_vectors_ternary_grouped_signed(
        docs,
        threshold_ratio=threshold_ratio,
        group_size=group_size,
    )
    decoded = _unpack_ternary(packed, dimension=docs.shape[1])
    group_count = scales.shape[1] // 2
    for group_index in range(group_count):
        start = group_index * group_size
        end = min(docs.shape[1], start + group_size)
        block = decoded[:, start:end]
        pos = block > 0
        neg = block < 0
        block[pos] = np.take(scales[:, group_index * 2], np.nonzero(pos)[0])
        block[neg] = -np.take(scales[:, group_index * 2 + 1], np.nonzero(neg)[0])
        decoded[:, start:end] = block
    return decoded


def _score_with_sparse_residual(
    queries: np.ndarray,
    docs: np.ndarray,
    decoded: np.ndarray,
    *,
    residual_dims: int,
) -> np.ndarray:
    if residual_dims <= 0:
        return queries @ decoded.T
    residual = docs - decoded
    keep = min(int(residual_dims), docs.shape[1])
    if keep >= docs.shape[1]:
        return queries @ docs.T
    top_indices = np.argpartition(np.abs(residual), -keep, axis=1)[:, -keep:]
    corrected = decoded.copy()
    row_indices = np.arange(docs.shape[0])[:, None]
    corrected[row_indices, top_indices] += residual[row_indices, top_indices]
    return queries @ corrected.T


def _fit_query_calibration(
    queries: np.ndarray,
    decoded_docs: np.ndarray,
    float_scores: np.ndarray,
    *,
    train_fraction: float,
    max_pairs: int,
    ridge: float,
    seed: int,
) -> np.ndarray:
    train_rows = max(1, int(round(queries.shape[0] * max(0.0, min(1.0, train_fraction)))))
    pair_count = train_rows * decoded_docs.shape[0]
    rng = np.random.default_rng(seed)
    if max_pairs > 0 and pair_count > max_pairs:
        flat = rng.choice(pair_count, size=max_pairs, replace=False)
        query_indices = flat // decoded_docs.shape[0]
        doc_indices = flat % decoded_docs.shape[0]
    else:
        query_indices, doc_indices = np.divmod(np.arange(pair_count), decoded_docs.shape[0])
    features = queries[query_indices] * decoded_docs[doc_indices]
    targets = float_scores[query_indices, doc_indices]
    lhs = features.T @ features
    lhs += np.eye(lhs.shape[0], dtype=np.float32) * float(ridge)
    rhs = features.T @ targets
    weights = np.linalg.solve(lhs.astype(np.float64), rhs.astype(np.float64)).astype(np.float32)
    return np.clip(weights, -4.0, 4.0).astype(np.float32)


def _rank_metrics(scores: np.ndarray, *, top_k: int) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1)
    labels = np.arange(scores.shape[0])
    top1 = float(np.mean(order[:, 0] == labels))
    topk = float(np.mean([label in row[:top_k] for label, row in zip(labels, order, strict=True)]))
    reciprocal_ranks = []
    for label, row in zip(labels, order, strict=True):
        rank = int(np.nonzero(row == label)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
    return {
        "self_top1": top1,
        f"self_top{top_k}": topk,
        "self_mrr": float(np.mean(reciprocal_ranks)),
    }


def _overlap_metrics(candidate_scores: np.ndarray, reference_scores: np.ndarray, *, top_k: int) -> dict[str, Any]:
    candidate_order = np.argsort(-candidate_scores, axis=1)[:, :top_k]
    reference_order = np.argsort(-reference_scores, axis=1)[:, :top_k]
    overlaps = []
    exact_top1 = []
    for candidate, reference in zip(candidate_order, reference_order, strict=True):
        overlaps.append(len(set(candidate.tolist()) & set(reference.tolist())) / max(1, top_k))
        exact_top1.append(candidate[0] == reference[0])
    return {
        f"float_top{top_k}_overlap": float(np.mean(overlaps)),
        "float_top1_agreement": float(np.mean(exact_top1)),
    }


def _format_bytes_per_row(*, format_name: str, dim: int, group_size: int) -> int:
    if format_name == "float":
        return dim * 4
    if format_name == "int8":
        return dim + 4
    if format_name == "ternary":
        return (dim + 3) // 4 + 4
    group_count = (dim + group_size - 1) // group_size
    if format_name == "ternary_grouped":
        return (dim + 3) // 4 + group_count * 4
    if format_name in {"ternary_grouped_signed", "ternary_grouped_signed_calibrated"}:
        return (dim + 3) // 4 + group_count * 8
    if format_name.startswith("ternary_residual"):
        residual_dims = int(format_name.rsplit("_", 1)[-1])
        return (dim + 3) // 4 + 4 + residual_dims * 4
    if format_name.startswith("ternary_grouped_signed_residual"):
        residual_dims = int(format_name.rsplit("_", 1)[-1])
        return (dim + 3) // 4 + group_count * 8 + residual_dims * 4
    raise ValueError(f"unknown format: {format_name}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    cache_path = Path(str(args.embedding_cache)).expanduser().resolve() if str(args.embedding_cache).strip() else None
    if cache_path and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            queries = payload["queries"].astype(np.float32)
            docs = payload["docs"].astype(np.float32)
        rows = [{} for _ in range(int(queries.shape[0]))]
    else:
        device = torch.device(str(args.device))
        model, tokenizer, _manifest = _load_model(Path(args.bundle_dir).resolve(), repo_root=repo_root, device=device)
        rows = list(
            _iter_paper_rows(
                Path(args.paper_root).expanduser().resolve(),
                max_rows=int(args.max_rows),
                max_files=int(args.max_files),
            )
        )
        if not rows:
            raise SystemExit("no paper rows found")
        query_texts = [str(row["title"]) for row in rows]
        doc_texts = [_doc_text(row) for row in rows]
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
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, queries=queries, docs=docs)
    float_scores = queries @ docs.T
    top_k = int(args.top_k)
    dim = int(docs.shape[1])
    group_size = int(args.ternary_group_size)
    result: dict[str, Any] = {
        "bundle_dir": str(Path(args.bundle_dir).resolve()),
        "paper_root": str(Path(args.paper_root).expanduser().resolve()),
        "rows": len(rows),
        "dim": dim,
        "top_k": top_k,
        "threshold_ratio": float(args.ternary_threshold_ratio),
        "ternary_group_size": group_size,
        "embedding_cache": str(cache_path) if cache_path else "",
        "formats": {
            "float": _rank_metrics(float_scores, top_k=top_k),
        },
    }
    result["formats"]["float"]["bytes_per_row"] = _format_bytes_per_row(
        format_name="float",
        dim=dim,
        group_size=group_size,
    )
    scored_formats = [
        ("int8", _score_int8(queries, docs)),
        ("ternary", _score_ternary(queries, docs, threshold_ratio=float(args.ternary_threshold_ratio))),
        (
            "ternary_grouped",
            _score_ternary_grouped(
                queries,
                docs,
                threshold_ratio=float(args.ternary_threshold_ratio),
                group_size=int(args.ternary_group_size),
            ),
        ),
        (
            "ternary_grouped_signed",
            _score_ternary_grouped_signed(
                queries,
                docs,
                threshold_ratio=float(args.ternary_threshold_ratio),
                group_size=int(args.ternary_group_size),
            ),
        ),
    ]
    for residual_dims in [int(value) for value in str(args.residual_dims).split(",") if value.strip()]:
        decoded_simple = _decode_ternary_simple_docs(docs, threshold_ratio=float(args.ternary_threshold_ratio))
        scored_formats.append(
            (
                f"ternary_residual_{residual_dims}",
                _score_with_sparse_residual(queries, docs, decoded_simple, residual_dims=residual_dims),
            )
        )
        decoded_signed = _decode_ternary_grouped_signed_docs(
            docs,
            threshold_ratio=float(args.ternary_threshold_ratio),
            group_size=int(args.ternary_group_size),
        )
        scored_formats.append(
            (
                f"ternary_grouped_signed_residual_{residual_dims}",
                _score_with_sparse_residual(queries, docs, decoded_signed, residual_dims=residual_dims),
            )
        )
    if bool(args.calibrate_query_scales):
        decoded_signed = _decode_ternary_grouped_signed_docs(
            docs,
            threshold_ratio=float(args.ternary_threshold_ratio),
            group_size=int(args.ternary_group_size),
        )
        weights = _fit_query_calibration(
            queries,
            decoded_signed,
            float_scores,
            train_fraction=float(args.calibration_train_fraction),
            max_pairs=int(args.calibration_max_pairs),
            ridge=float(args.calibration_ridge),
            seed=int(args.seed),
        )
        scored_formats.append(("ternary_grouped_signed_calibrated", (queries * weights[None, :]) @ decoded_signed.T))
    for name, scores in scored_formats:
        metrics = _rank_metrics(scores, top_k=top_k)
        metrics.update(_overlap_metrics(scores, float_scores, top_k=top_k))
        metrics["bytes_per_row"] = _format_bytes_per_row(format_name=name, dim=dim, group_size=group_size)
        metrics["bytes_vs_int8"] = metrics["bytes_per_row"] / _format_bytes_per_row(
            format_name="int8",
            dim=dim,
            group_size=group_size,
        )
        result["formats"][name] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare int8/simple ternary/grouped ternary paper-memory ranking loss.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--paper-root", default="/arxiv/huggingface/paper_text_1m_dedup_v1")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ternary-threshold-ratio", type=float, default=0.20)
    parser.add_argument("--ternary-group-size", type=int, default=16)
    parser.add_argument("--embedding-cache", default="")
    parser.add_argument("--calibrate-query-scales", type=int, default=0)
    parser.add_argument("--calibration-train-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-max-pairs", type=int, default=32768)
    parser.add_argument("--calibration-ridge", type=float, default=1e-3)
    parser.add_argument("--residual-dims", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    result = evaluate(args)
    if str(args.output_json).strip():
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
