#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_sampler():
    path = REPO_ROOT / "scripts" / "sample_agentkernel_lite_encdec.py"
    spec = importlib.util.spec_from_file_location("sample_agentkernel_lite_encdec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sampler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("retrieval_query_text") and row.get("retrieval_doc_text") and row.get("retrieval_negative_doc_texts"):
            out.append(row)
    return out


def _score_rows(
    *,
    bundle_dir: Path,
    rows: list[dict[str, Any]],
    device_name: str,
    max_rows: int,
    row_offset: int,
    margin_threshold: float,
    query_tokens: int,
    doc_tokens: int,
) -> list[dict[str, Any]]:
    sampler = _load_sampler()
    sampler._install_paths(REPO_ROOT)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(device_name)
    model.to(device).eval()

    def encode(text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [int(item) for item in tokenizer.encode(str(text), max_length=max_len)]
        ids = ids[:max_len]
        mask = [1] * len(ids)
        pad = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        while len(ids) < max_len:
            ids.append(pad)
            mask.append(0)
        return (
            torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([mask], dtype=torch.long, device=device),
        )

    hard: list[dict[str, Any]] = []
    seen = 0
    scored = 0
    with torch.no_grad():
        for row in rows:
            if seen < row_offset:
                seen += 1
                continue
            if scored >= max_rows:
                break
            seen += 1
            scored += 1
            try:
                negatives = json.loads(str(row.get("retrieval_negative_doc_texts") or "[]"))
            except json.JSONDecodeError:
                negatives = []
            negatives = [str(item) for item in negatives[:4] if str(item).strip()]
            if not negatives:
                continue
            q_ids, q_mask = encode(str(row["retrieval_query_text"]), query_tokens)
            query = model.retrieval_query_embedding(q_ids, q_mask)
            scores: list[float] = []
            for doc in [str(row["retrieval_doc_text"]), *negatives]:
                d_ids, d_mask = encode(doc, doc_tokens)
                doc_embedding = model.retrieval_doc_embedding(d_ids, d_mask)
                scores.append(float((query @ doc_embedding.transpose(0, 1)).item()))
            margin = scores[0] - max(scores[1:])
            if margin < margin_threshold:
                copied = deepcopy(row)
                copied["retrieval_loss_weight"] = max(float(copied.get("retrieval_loss_weight", 1.0) or 1.0), 2.0)
                copied["stage87_mined_margin"] = margin
                copied["stage87_mined_scores"] = json.dumps(scores)
                copied["source_id"] = f"{copied.get('source_id') or copied.get('example_id')}_stage87_margin_repair"
                hard.append(copied)
    return hard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage86_retrieval_hard_negative_corpus/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--bundle-dir", default=str(REPO_ROOT / "artifacts/pocketpal_controller_100m_v312a_retrieval_only_hardneg_from_v311"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_stage87_retrieval_margin_repair_corpus"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-mine-rows", type=int, default=8192)
    parser.add_argument("--train-mine-row-offset", type=int, default=0)
    parser.add_argument("--margin-threshold", type=float, default=0.04)
    parser.add_argument("--duplicate-hard-rows", type=int, default=2)
    parser.add_argument("--query-tokens", type=int, default=192)
    parser.add_argument("--doc-tokens", type=int, default=320)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train_rows = list(_iter_jsonl(Path(source_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(source_manifest["eval_dataset_path"])))
    train_retrieval = _candidate_rows(train_rows)
    mined = _score_rows(
        bundle_dir=Path(args.bundle_dir).resolve(),
        rows=train_retrieval,
        device_name=str(args.device),
        max_rows=int(args.max_train_mine_rows),
        row_offset=int(args.train_mine_row_offset),
        margin_threshold=float(args.margin_threshold),
        query_tokens=int(args.query_tokens),
        doc_tokens=int(args.doc_tokens),
    )
    repair_rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(args.duplicate_hard_rows))):
        for row in mined:
            copied = deepcopy(row)
            copied["source_id"] = f"{row.get('source_id')}_dup{repeat:02d}"
            repair_rows.append(copied)
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    merged_train = [*train_rows, *repair_rows]
    _write_jsonl(train_path, merged_train)
    _write_jsonl(eval_path, eval_rows)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "artifact_kind": "agentkernel_lite_encdec_stage87_retrieval_margin_repair_corpus",
            "base_manifest": str(source_manifest_path),
            "eval_dataset_path": str(eval_path),
            "eval_examples": len(eval_rows),
            "manifest_path": str(manifest_path),
            "mined_hard_retrieval_rows": len(mined),
            "mined_train_retrieval_rows_scored": min(int(args.max_train_mine_rows), len(train_retrieval)),
            "mined_train_retrieval_row_offset": int(args.train_mine_row_offset),
            "objective": "pocketpal_stage87_retrieval_margin_repair_corpus",
            "repair_rows_added": len(repair_rows),
            "source_bundle": str(Path(args.bundle_dir).resolve()),
            "train_dataset_path": str(train_path),
            "train_examples": len(merged_train),
            "total_examples": len(merged_train) + len(eval_rows),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "train_examples": len(merged_train),
                "eval_examples": len(eval_rows),
                "mined_hard_retrieval_rows": len(mined),
                "repair_rows_added": len(repair_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
