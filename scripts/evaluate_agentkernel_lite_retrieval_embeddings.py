#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import torch
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_paths(repo_root: Path) -> None:
    model_stack = repo_root / "other_repos" / "model-stack"
    for path in (repo_root, model_stack):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _materialize_lazy_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir() and any(path.glob("*.parquet")):
        import pyarrow.parquet as pq

        columns = ["retrieval_query_text", "retrieval_doc_text", "task_type", "source_id"]
        for shard in sorted(path.glob("*.parquet")):
            parquet_file = pq.ParquetFile(shard)
            available = set(parquet_file.schema_arrow.names)
            read_columns = [column for column in columns if column in available]
            for batch in parquet_file.iter_batches(batch_size=2048, columns=read_columns):
                yield from batch.to_pylist()
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_tokenizer(manifest: dict[str, Any]):
    from scripts.sample_agentkernel_lite_encdec import ByteTokenizer, HuggingFaceTokenizer, TokenizersBpe

    tokenizer_kind = str(manifest.get("tokenizer_kind", "byte") or "byte").lower()
    tokenizer_dir = Path(str(manifest.get("tokenizer_dir", "") or ""))
    if tokenizer_kind == "byte":
        return ByteTokenizer()
    if tokenizer_kind == "agentkernel-bpe":
        return TokenizersBpe(tokenizer_dir / "tokenizer.json")
    return HuggingFaceTokenizer(tokenizer_dir, str(manifest.get("tokenizer_name", "")))


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
        row = tokenizer.encode(text, max_length=max_tokens)
        row = row[:max_tokens]
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


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _embed(model, tokenizer, texts: list[str], *, max_tokens: int, device: torch.device) -> torch.Tensor:
    ids, mask = _encode_batch(tokenizer, texts, max_tokens=max_tokens, device=device)
    if hasattr(model, "retrieval_query_embedding") and hasattr(model, "retrieval_doc_embedding"):
        # Caller decides whether these are query or doc strings. For retrieval
        # quality we need both asymmetric heads, so this helper remains a
        # fallback for old bundles.
        hidden = model.encode(ids, mask)
        return F.normalize(_mean_pool(hidden, mask), dim=-1)
    hidden = model.encode(ids, mask)
    return F.normalize(_mean_pool(hidden, mask), dim=-1)


def _embed_query(model, tokenizer, texts: list[str], *, max_tokens: int, device: torch.device) -> torch.Tensor:
    ids, mask = _encode_batch(tokenizer, texts, max_tokens=max_tokens, device=device)
    if hasattr(model, "retrieval_query_embedding"):
        return model.retrieval_query_embedding(ids, mask)
    hidden = model.encode(ids, mask)
    return F.normalize(_mean_pool(hidden, mask), dim=-1)


def _embed_doc(model, tokenizer, texts: list[str], *, max_tokens: int, device: torch.device) -> torch.Tensor:
    ids, mask = _encode_batch(tokenizer, texts, max_tokens=max_tokens, device=device)
    if hasattr(model, "retrieval_doc_embedding"):
        return model.retrieval_doc_embedding(ids, mask)
    hidden = model.encode(ids, mask)
    return F.normalize(_mean_pool(hidden, mask), dim=-1)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(str(args.device))
    model, tokenizer, manifest = _load_model(Path(args.bundle_dir).resolve(), repo_root=repo_root, device=device)
    if str(args.dataset_manifest).strip():
        dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    else:
        dataset_manifest = json.loads(Path(str(manifest["dataset_manifest_path"])).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in _iter_rows(Path(str(dataset_manifest["eval_dataset_path"]))):
        query = str(row.get("retrieval_query_text", "") or "").strip()
        doc = str(row.get("retrieval_doc_text", "") or "").strip()
        if query and doc:
            rows.append(row)
            if int(args.limit) > 0 and len(rows) >= int(args.limit):
                break
    if not rows:
        return {"bundle_dir": str(args.bundle_dir), "retrieval_pairs": 0}

    top1 = 0
    mrr_total = 0.0
    batches = 0
    evaluated = 0
    with torch.no_grad():
        for offset in range(0, len(rows), int(args.batch_size)):
            batch = rows[offset : offset + int(args.batch_size)]
            if len(batch) < 2:
                continue
            queries = [str(row["retrieval_query_text"]) for row in batch]
            docs = [str(row["retrieval_doc_text"]) for row in batch]
            query_embeddings = _embed_query(
                model,
                tokenizer,
                queries,
                max_tokens=int(args.max_query_tokens),
                device=device,
            )
            doc_embeddings = _embed_doc(
                model,
                tokenizer,
                docs,
                max_tokens=int(args.max_doc_tokens),
                device=device,
            )
            scores = query_embeddings @ doc_embeddings.transpose(0, 1)
            ranks = torch.argsort(scores, dim=1, descending=True)
            labels = torch.arange(scores.shape[0], device=device)
            top1 += int((ranks[:, 0] == labels).sum().detach().cpu().item())
            for index in range(scores.shape[0]):
                rank = int((ranks[index] == index).nonzero(as_tuple=False)[0, 0].detach().cpu().item()) + 1
                mrr_total += 1.0 / rank
            evaluated += int(scores.shape[0])
            batches += 1
    return {
        "bundle_dir": str(Path(args.bundle_dir).resolve()),
        "dataset_manifest": str(Path(str(dataset_manifest["manifest_path"])).resolve())
        if dataset_manifest.get("manifest_path")
        else "",
        "retrieval_pairs": len(rows),
        "evaluated_pairs": evaluated,
        "batches": batches,
        "batch_size": int(args.batch_size),
        "top1_accuracy": top1 / evaluated if evaluated else None,
        "mean_reciprocal_rank": mrr_total / evaluated if evaluated else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    result = evaluate(args)
    if str(args.output_json).strip():
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
