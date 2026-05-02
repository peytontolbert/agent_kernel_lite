#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_paths(repo_root: Path) -> None:
    model_stack = repo_root / "other_repos" / "model-stack"
    for path in (repo_root, model_stack):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _compact(value: object, *, limit: int = 1600) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())[:limit].strip()


def _iter_paper_rows(path: Path, *, max_rows: int, max_files: int) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

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
    emitted = 0
    for file_path in paths:
        parquet_file = pq.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if "title" not in read_columns or "abstract" not in read_columns:
            continue
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=2048, columns=read_columns):
            for payload in batch.to_pylist():
                title = _compact(payload.get("title", ""), limit=220)
                abstract = _compact(payload.get("abstract", ""), limit=1400)
                if not title or len(abstract) < 80:
                    row_offset += 1
                    continue
                paper_id = _compact(
                    payload.get(
                        "paper_id",
                        payload.get("canonical_paper_id", payload.get("arxiv_id", payload.get("id", ""))),
                    ),
                    limit=96,
                )
                yield {
                    "paper_id": paper_id or f"{file_path.name}:{row_offset}",
                    "title": title,
                    "abstract": abstract,
                    "categories": _compact(payload.get("categories", payload.get("primary_category", "")), limit=160),
                    "year": _compact(payload.get("year", payload.get("published_year", payload.get("update_date", ""))), limit=48),
                    "source_file": file_path.name,
                    "row_offset": row_offset,
                }
                emitted += 1
                row_offset += 1
                if max_rows > 0 and emitted >= max_rows:
                    return


def _load_tokenizer(manifest: dict[str, Any]):
    from scripts.sample_agentkernel_lite_encdec import ByteTokenizer, HuggingFaceTokenizer, TokenizersBpe

    tokenizer_kind = str(manifest.get("tokenizer_kind", "byte") or "byte").lower()
    tokenizer_dir = Path(str(manifest.get("tokenizer_dir", "") or ""))
    if tokenizer_kind == "byte":
        return ByteTokenizer()
    if tokenizer_kind == "agentkernel-bpe":
        return TokenizersBpe(tokenizer_dir / "tokenizer.json")
    return HuggingFaceTokenizer(tokenizer_dir, str(manifest.get("tokenizer_name", "")))


def _materialize_lazy_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _load_model(bundle_dir: Path, *, repo_root: Path, device: torch.device):
    _install_paths(repo_root)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    manifest = json.loads((bundle_dir / "agentkernel_lite_encdec_manifest.json").read_text(encoding="utf-8"))
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    model.to(device).eval()
    return model, tokenizer, manifest


def _encode_batch(tokenizer, texts: list[str], *, max_tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    ids: list[list[int]] = []
    masks: list[list[int]] = []
    for text in texts:
        row = tokenizer.encode(text, max_length=max_tokens)[:max_tokens]
        mask = [1] * len(row)
        while len(row) < max_tokens:
            row.append(pad_id)
            mask.append(0)
        ids.append(row)
        masks.append(mask)
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _doc_text(row: dict[str, Any]) -> str:
    meta = " | ".join(part for part in [row.get("paper_id", ""), row.get("categories", ""), row.get("year", "")] if part)
    return _compact(f"{row['title']}\n{meta}\n{row['abstract']}", limit=1800)


def _quantize_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.maximum(np.max(np.abs(vectors), axis=1), 1e-6).astype(np.float32)
    scales = (max_abs / 127.0).astype(np.float32)
    quantized = np.clip(np.round(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return quantized, scales


def _pack_ternary_codes(codes: np.ndarray) -> np.ndarray:
    if codes.ndim != 2:
        raise ValueError("ternary codes must be a rank-2 matrix")
    encoded = np.zeros(codes.shape, dtype=np.uint8)
    encoded[codes > 0] = 1
    encoded[codes < 0] = 2
    rows, dim = encoded.shape
    packed_dim = (dim + 3) // 4
    packed = np.zeros((rows, packed_dim), dtype=np.uint8)
    for shift in range(4):
        chunk = encoded[:, shift::4]
        packed[:, : chunk.shape[1]] |= chunk << (shift * 2)
    return packed


def _quantize_vectors_ternary(vectors: np.ndarray, *, threshold_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_abs = np.maximum(np.max(np.abs(vectors), axis=1), 1e-6).astype(np.float32)
    threshold = (max_abs * max(0.0, float(threshold_ratio))).astype(np.float32)
    codes = np.where(vectors > threshold[:, None], 1, np.where(vectors < -threshold[:, None], -1, 0)).astype(np.int8)
    nonzero = codes != 0
    nonzero_counts = np.maximum(nonzero.sum(axis=1), 1).astype(np.float32)
    scales = (np.abs(vectors) * nonzero).sum(axis=1).astype(np.float32) / nonzero_counts
    scales = np.maximum(scales, 1e-6).astype(np.float32)
    packed = _pack_ternary_codes(codes)
    density = nonzero.mean(axis=1).astype(np.float32)
    return packed, scales, density


def _quantize_vectors_ternary_grouped(
    vectors: np.ndarray,
    *,
    threshold_ratio: float,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_size <= 0:
        raise ValueError("ternary group size must be positive")
    rows, dim = vectors.shape
    group_count = (dim + group_size - 1) // group_size
    codes = np.zeros((rows, dim), dtype=np.int8)
    scales = np.zeros((rows, group_count), dtype=np.float32)
    density = np.zeros((rows, group_count), dtype=np.float32)
    for group_index in range(group_count):
        start = group_index * group_size
        end = min(dim, start + group_size)
        block = vectors[:, start:end]
        max_abs = np.maximum(np.max(np.abs(block), axis=1), 1e-6).astype(np.float32)
        threshold = (max_abs * max(0.0, float(threshold_ratio))).astype(np.float32)
        block_codes = np.where(block > threshold[:, None], 1, np.where(block < -threshold[:, None], -1, 0)).astype(np.int8)
        nonzero = block_codes != 0
        nonzero_counts = np.maximum(nonzero.sum(axis=1), 1).astype(np.float32)
        scales[:, group_index] = (np.abs(block) * nonzero).sum(axis=1).astype(np.float32) / nonzero_counts
        density[:, group_index] = nonzero.mean(axis=1).astype(np.float32)
        codes[:, start:end] = block_codes
    scales = np.maximum(scales, 1e-6).astype(np.float32)
    packed = _pack_ternary_codes(codes)
    return packed, scales, density


def _quantize_vectors_ternary_grouped_signed(
    vectors: np.ndarray,
    *,
    threshold_ratio: float,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_size <= 0:
        raise ValueError("ternary group size must be positive")
    rows, dim = vectors.shape
    group_count = (dim + group_size - 1) // group_size
    codes = np.zeros((rows, dim), dtype=np.int8)
    scales = np.zeros((rows, group_count, 2), dtype=np.float32)
    density = np.zeros((rows, group_count), dtype=np.float32)
    for group_index in range(group_count):
        start = group_index * group_size
        end = min(dim, start + group_size)
        block = vectors[:, start:end]
        max_abs = np.maximum(np.max(np.abs(block), axis=1), 1e-6).astype(np.float32)
        threshold = (max_abs * max(0.0, float(threshold_ratio))).astype(np.float32)
        block_codes = np.where(block > threshold[:, None], 1, np.where(block < -threshold[:, None], -1, 0)).astype(np.int8)
        pos_mask = block_codes > 0
        neg_mask = block_codes < 0
        pos_count = np.maximum(pos_mask.sum(axis=1), 1).astype(np.float32)
        neg_count = np.maximum(neg_mask.sum(axis=1), 1).astype(np.float32)
        scales[:, group_index, 0] = (block * pos_mask).sum(axis=1).astype(np.float32) / pos_count
        scales[:, group_index, 1] = ((-block) * neg_mask).sum(axis=1).astype(np.float32) / neg_count
        density[:, group_index] = (pos_mask | neg_mask).mean(axis=1).astype(np.float32)
        codes[:, start:end] = block_codes
    scales = np.maximum(scales, 1e-6).astype(np.float32)
    packed = _pack_ternary_codes(codes)
    return packed, scales.reshape(rows, group_count * 2), density


def _float32_to_float16_bytes(values: np.ndarray) -> bytes:
    return values.astype(np.float16).tobytes(order="C")


def _sparse_residual(
    vectors: np.ndarray,
    decoded: np.ndarray,
    *,
    residual_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    if residual_dims <= 0:
        raise ValueError("residual dims must be positive")
    residual = vectors - decoded
    keep = min(int(residual_dims), vectors.shape[1])
    indices = np.argpartition(np.abs(residual), -keep, axis=1)[:, -keep:].astype(np.uint16)
    values = np.take_along_axis(residual, indices.astype(np.int64), axis=1).astype(np.float16)
    return indices, values


def _decode_ternary_grouped_signed(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    dim: int,
    group_size: int,
) -> np.ndarray:
    rows = packed.shape[0]
    decoded = np.zeros((rows, dim), dtype=np.float32)
    for shift in range(4):
        view = decoded[:, shift::4]
        values = ((packed >> (shift * 2)) & 3)[:, : view.shape[1]]
        view[:, :] = np.where(values == 1, 1.0, np.where(values == 2, -1.0, 0.0)).astype(np.float32)
    group_count = scales.shape[1] // 2
    for group_index in range(group_count):
        start = group_index * group_size
        end = min(dim, start + group_size)
        block = decoded[:, start:end]
        pos = block > 0
        neg = block < 0
        row_index = np.nonzero(pos)[0]
        block[pos] = np.take(scales[:, group_index * 2], row_index)
        row_index = np.nonzero(neg)[0]
        block[neg] = -np.take(scales[:, group_index * 2 + 1], row_index)
        decoded[:, start:end] = block
    return decoded


def export_pack(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(str(args.device))
    model, tokenizer, model_manifest = _load_model(Path(args.bundle_dir).resolve(), repo_root=repo_root, device=device)
    rows = list(
        _iter_paper_rows(
            Path(args.paper_root).expanduser().resolve(),
            max_rows=int(args.max_rows),
            max_files=int(args.max_files),
        )
    )
    if not rows:
        raise SystemExit("no paper rows found for memory pack export")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    vectors: list[np.ndarray] = []
    with metadata_path.open("w", encoding="utf-8") as meta_handle, torch.no_grad():
        for offset in range(0, len(rows), int(args.batch_size)):
            batch = rows[offset : offset + int(args.batch_size)]
            texts = [_doc_text(row) for row in batch]
            ids, mask = _encode_batch(tokenizer, texts, max_tokens=int(args.max_doc_tokens), device=device)
            if hasattr(model, "retrieval_doc_embedding"):
                embeddings = model.retrieval_doc_embedding(ids, mask)
            else:
                hidden = model.encode(ids, mask)
                weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                embeddings = torch.nn.functional.normalize(pooled.float(), dim=-1)
            embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
            vectors.append(embeddings_np)
            for index, row in enumerate(batch, start=offset):
                meta_handle.write(
                    json.dumps(
                        {
                            "memory_id": f"P{index + 1}",
                            "paper_id": row["paper_id"],
                            "title": row["title"],
                            "abstract": row["abstract"],
                            "categories": row["categories"],
                            "year": row["year"],
                            "source_file": row["source_file"],
                            "row_offset": row["row_offset"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    matrix = np.concatenate(vectors, axis=0)
    vector_format = str(args.vector_format).strip().lower()
    if vector_format not in {
        "int8",
        "ternary",
        "ternary_grouped",
        "ternary_grouped_signed",
        "ternary_grouped_signed_residual",
        "both",
    }:
        raise ValueError(f"unsupported vector format: {args.vector_format}")
    vector_entries: dict[str, Any] = {}
    if vector_format in {"int8", "both"}:
        quantized, scales = _quantize_vectors(matrix)
        vector_path = output_dir / "vectors.i8.bin"
        scale_path = output_dir / "scales.f32.bin"
        vector_path.write_bytes(quantized.tobytes(order="C"))
        scale_path.write_bytes(scales.tobytes(order="C"))
        vector_entries["int8"] = {
            "vector_dtype": "int8_symmetric_per_vector",
            "vector_path": vector_path.name,
            "scale_path": scale_path.name,
            "similarity": "dot_product_dequantized_l2",
        }
    if vector_format in {"ternary", "both"}:
        ternary, ternary_scales, ternary_density = _quantize_vectors_ternary(
            matrix,
            threshold_ratio=float(args.ternary_threshold_ratio),
        )
        ternary_path = output_dir / "vectors.t2.bin"
        ternary_scale_path = output_dir / "ternary_scales.f32.bin"
        ternary_density_path = output_dir / "ternary_density.f32.bin"
        ternary_path.write_bytes(ternary.tobytes(order="C"))
        ternary_scale_path.write_bytes(ternary_scales.tobytes(order="C"))
        ternary_density_path.write_bytes(ternary_density.tobytes(order="C"))
        vector_entries["ternary"] = {
            "vector_dtype": "ternary_packed_2bit_per_vector",
            "vector_path": ternary_path.name,
            "scale_path": ternary_scale_path.name,
            "density_path": ternary_density_path.name,
            "threshold_ratio": float(args.ternary_threshold_ratio),
            "packed_dim": int(ternary.shape[1]),
            "similarity": "dot_product_dequantized_ternary_l2",
        }
    if vector_format in {"ternary_grouped", "both"}:
        grouped, grouped_scales, grouped_density = _quantize_vectors_ternary_grouped(
            matrix,
            threshold_ratio=float(args.ternary_threshold_ratio),
            group_size=int(args.ternary_group_size),
        )
        grouped_path = output_dir / "vectors.t2.grouped.bin"
        grouped_scale_path = output_dir / "ternary_group_scales.f32.bin"
        grouped_density_path = output_dir / "ternary_group_density.f32.bin"
        grouped_path.write_bytes(grouped.tobytes(order="C"))
        grouped_scale_path.write_bytes(grouped_scales.tobytes(order="C"))
        grouped_density_path.write_bytes(grouped_density.tobytes(order="C"))
        vector_entries["ternary_grouped"] = {
            "vector_dtype": "ternary_packed_2bit_group_scaled",
            "vector_path": grouped_path.name,
            "scale_path": grouped_scale_path.name,
            "density_path": grouped_density_path.name,
            "threshold_ratio": float(args.ternary_threshold_ratio),
            "group_size": int(args.ternary_group_size),
            "group_count": int(grouped_scales.shape[1]),
            "packed_dim": int(grouped.shape[1]),
            "similarity": "dot_product_dequantized_ternary_grouped_l2",
        }
    if vector_format in {"ternary_grouped_signed", "both"}:
        signed, signed_scales, signed_density = _quantize_vectors_ternary_grouped_signed(
            matrix,
            threshold_ratio=float(args.ternary_threshold_ratio),
            group_size=int(args.ternary_group_size),
        )
        signed_path = output_dir / "vectors.t2.grouped_signed.bin"
        signed_scale_dtype = str(args.ternary_scale_dtype).lower()
        signed_scale_path = output_dir / f"ternary_group_signed_scales.{signed_scale_dtype}.bin"
        signed_density_path = output_dir / "ternary_group_signed_density.f32.bin"
        signed_path.write_bytes(signed.tobytes(order="C"))
        if signed_scale_dtype == "f16":
            signed_scale_path.write_bytes(_float32_to_float16_bytes(signed_scales))
        else:
            signed_scale_path.write_bytes(signed_scales.tobytes(order="C"))
        signed_density_path.write_bytes(signed_density.tobytes(order="C"))
        vector_entries["ternary_grouped_signed"] = {
            "vector_dtype": "ternary_packed_2bit_group_signed_scaled",
            "vector_path": signed_path.name,
            "scale_path": signed_scale_path.name,
            "scale_dtype": signed_scale_dtype,
            "density_path": signed_density_path.name,
            "threshold_ratio": float(args.ternary_threshold_ratio),
            "group_size": int(args.ternary_group_size),
            "group_count": int(signed_density.shape[1]),
            "packed_dim": int(signed.shape[1]),
            "scale_components": 2,
            "similarity": "dot_product_dequantized_ternary_grouped_signed_l2",
        }
    if vector_format in {"ternary_grouped_signed_residual", "both"}:
        signed, signed_scales, signed_density = _quantize_vectors_ternary_grouped_signed(
            matrix,
            threshold_ratio=float(args.ternary_threshold_ratio),
            group_size=int(args.ternary_group_size),
        )
        decoded = _decode_ternary_grouped_signed(
            signed,
            signed_scales,
            dim=int(matrix.shape[1]),
            group_size=int(args.ternary_group_size),
        )
        residual_indices, residual_values = _sparse_residual(
            matrix,
            decoded,
            residual_dims=int(args.ternary_residual_dims),
        )
        signed_path = output_dir / "vectors.t2.grouped_signed_residual.bin"
        signed_scale_dtype = str(args.ternary_scale_dtype).lower()
        signed_scale_path = output_dir / f"ternary_group_signed_residual_scales.{signed_scale_dtype}.bin"
        signed_density_path = output_dir / "ternary_group_signed_residual_density.f32.bin"
        residual_index_path = output_dir / "ternary_residual_indices.u16.bin"
        residual_value_path = output_dir / "ternary_residual_values.f16.bin"
        signed_path.write_bytes(signed.tobytes(order="C"))
        if signed_scale_dtype == "f16":
            signed_scale_path.write_bytes(_float32_to_float16_bytes(signed_scales))
        else:
            signed_scale_path.write_bytes(signed_scales.tobytes(order="C"))
        signed_density_path.write_bytes(signed_density.tobytes(order="C"))
        residual_index_path.write_bytes(residual_indices.tobytes(order="C"))
        residual_value_path.write_bytes(residual_values.tobytes(order="C"))
        vector_entries["ternary_grouped_signed_residual"] = {
            "vector_dtype": "ternary_packed_2bit_group_signed_scaled_sparse_residual",
            "vector_path": signed_path.name,
            "scale_path": signed_scale_path.name,
            "scale_dtype": signed_scale_dtype,
            "density_path": signed_density_path.name,
            "residual_index_path": residual_index_path.name,
            "residual_value_path": residual_value_path.name,
            "residual_index_dtype": "uint16",
            "residual_value_dtype": "float16",
            "residual_dims": int(args.ternary_residual_dims),
            "threshold_ratio": float(args.ternary_threshold_ratio),
            "group_size": int(args.ternary_group_size),
            "group_count": int(signed_density.shape[1]),
            "packed_dim": int(signed.shape[1]),
            "scale_components": 2,
            "similarity": "dot_product_dequantized_ternary_grouped_signed_residual_l2",
        }
    primary = (
        "ternary"
        if vector_format == "ternary"
        else "ternary_grouped"
        if vector_format == "ternary_grouped"
        else "ternary_grouped_signed"
        if vector_format == "ternary_grouped_signed"
        else "ternary_grouped_signed_residual"
        if vector_format == "ternary_grouped_signed_residual"
        else "int8"
    )
    primary_entry = vector_entries[primary]
    manifest_path = output_dir / "memory_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_neural_memory_pack",
        "format_version": 2,
        "manifest_path": str(manifest_path),
        "source_model_manifest": str(Path(args.bundle_dir).resolve() / "agentkernel_lite_encdec_manifest.json"),
        "source_browser_manifest": str(model_manifest.get("browser_bitnet_manifest_path", "")),
        "paper_root": str(Path(args.paper_root).expanduser().resolve()),
        "row_count": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "primary_vector_format": primary,
        "vector_formats": vector_entries,
        "vector_dtype": primary_entry["vector_dtype"],
        "vector_path": primary_entry["vector_path"],
        "scale_path": primary_entry["scale_path"],
        "metadata_path": metadata_path.name,
        "query_runtime": {
            "model": "agentkernel_lite_bitnet_encoder",
            "method": "retrievalQueryEmbedding",
            "similarity": primary_entry["similarity"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact browser memory vectors from AgentKernel Lite retrieval heads.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--paper-root", default="/arxiv/huggingface/paper_text_1m_dedup_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument(
        "--vector-format",
        choices=(
            "int8",
            "ternary",
            "ternary_grouped",
            "ternary_grouped_signed",
            "ternary_grouped_signed_residual",
            "both",
        ),
        default="int8",
    )
    parser.add_argument("--ternary-threshold-ratio", type=float, default=0.20)
    parser.add_argument("--ternary-group-size", type=int, default=16)
    parser.add_argument("--ternary-scale-dtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--ternary-residual-dims", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(export_pack(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
