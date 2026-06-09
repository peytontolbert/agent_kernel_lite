#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("encoder_text") or "").strip() or not str(row.get("decoder_text") or "").strip():
                continue
            rows.append(row)
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def _bucket_add(bucket: dict[str, float], *, nll: float, tokens: int, correct: int, bytes_: int) -> None:
    bucket["nll_sum"] = bucket.get("nll_sum", 0.0) + float(nll)
    bucket["tokens"] = bucket.get("tokens", 0.0) + float(tokens)
    bucket["correct"] = bucket.get("correct", 0.0) + float(correct)
    bucket["bytes"] = bucket.get("bytes", 0.0) + float(bytes_)
    bucket["examples"] = bucket.get("examples", 0.0) + 1.0


def _bucket_finish(bucket: dict[str, float]) -> dict[str, float]:
    tokens = max(float(bucket.get("tokens", 0.0)), 1.0)
    bytes_ = max(float(bucket.get("bytes", 0.0)), 1.0)
    nll = float(bucket.get("nll_sum", 0.0)) / tokens
    return {
        "examples": int(bucket.get("examples", 0.0)),
        "tokens": int(bucket.get("tokens", 0.0)),
        "avg_nll_nats": nll,
        "perplexity": math.exp(min(20.0, nll)),
        "bits_per_token": nll / math.log(2.0),
        "bits_per_byte": float(bucket.get("nll_sum", 0.0)) / math.log(2.0) / bytes_,
        "token_accuracy": float(bucket.get("correct", 0.0)) / tokens,
    }


class _CallableTokenizerAdapter:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.bos_token_id = int(getattr(tokenizer, "bos_token_id", 1) or 1)
        self.eos_token_id = int(getattr(tokenizer, "eos_token_id", 2) or 2)
        self.unk_token_id = int(getattr(tokenizer, "unk_token_id", 3) or 3)
        self.vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        ids = [int(item) for item in self.tokenizer.encode(str(text), max_length=int(max_length))]
        if truncation:
            ids = ids[: int(max_length)]
        attention = [1] * len(ids)
        if padding == "max_length":
            while len(ids) < int(max_length):
                ids.append(self.pad_token_id)
                attention.append(0)
        return {"input_ids": ids, "attention_mask": attention}

    def decode(self, ids: list[int]) -> str:
        try:
            return str(self.tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=True))
        except TypeError:
            return str(self.tokenizer.decode([int(token_id) for token_id in ids]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split", choices=("eval", "train"), default="eval")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--max-decoder-tokens", type=int, default=384)
    parser.add_argument("--max-retrieval-query-tokens", type=int, default=128)
    parser.add_argument("--max-retrieval-doc-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sampler = _load_module(repo_root / "scripts" / "sample_agentkernel_lite_encdec.py", "sample_agentkernel_lite_encdec")
    sampler._install_paths(repo_root)
    trainer = _load_module(repo_root / "scripts" / "train_agentkernel_lite_encdec.py", "train_agentkernel_lite_encdec")

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = sampler._load_manifest(bundle_dir)
    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    dataset_key = "eval_dataset_path" if str(args.split) == "eval" else "train_dataset_path"
    rows = _iter_rows(Path(str(dataset_manifest[dataset_key])), int(args.limit))
    if not rows:
        raise SystemExit("no decoder rows to evaluate")

    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _CallableTokenizerAdapter(sampler._load_tokenizer(manifest))
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    decoder_start_token_id = int(getattr(tokenizer, "bos_token_id", 1) or 1)
    encoded_rows: list[dict[str, torch.Tensor]] = [
        trainer._encode_encdec_row(
            row,
            tokenizer=tokenizer,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_decoder_tokens=int(args.max_decoder_tokens),
            max_retrieval_query_tokens=int(args.max_retrieval_query_tokens),
            max_retrieval_doc_tokens=int(args.max_retrieval_doc_tokens),
            max_retrieval_negatives=0,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
        for row in rows
    ]

    total: dict[str, float] = {}
    by_task: dict[str, dict[str, float]] = {}
    by_source_type: dict[str, dict[str, float]] = {}
    batch_size = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            row_batch = rows[start : start + batch_size]
            encoded_batch = encoded_rows[start : start + batch_size]
            batch = {
                key: torch.stack([item[key] for item in encoded_batch]).to(device)
                for key in ("enc_input_ids", "enc_attention_mask", "dec_input_ids", "labels")
            }
            logits = model(batch["enc_input_ids"], batch["dec_input_ids"], batch["enc_attention_mask"], None)
            labels = batch["labels"]
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)
            predictions = torch.argmax(logits, dim=-1)
            valid = labels != -100
            for index, row in enumerate(row_batch):
                mask = valid[index]
                token_count = int(mask.sum().detach().cpu().item())
                if token_count <= 0:
                    continue
                nll_sum = float((losses[index] * mask.to(dtype=losses.dtype)).sum().detach().cpu().item())
                correct = int(((predictions[index] == labels[index]) & mask).sum().detach().cpu().item())
                byte_count = len(str(row.get("decoder_text") or "").encode("utf-8"))
                _bucket_add(total, nll=nll_sum, tokens=token_count, correct=correct, bytes_=byte_count)
                task = str(row.get("task_type") or "unknown")
                source_type = str(row.get("source_type") or row.get("source_dataset") or "unknown")
                _bucket_add(by_task.setdefault(task, {}), nll=nll_sum, tokens=token_count, correct=correct, bytes_=byte_count)
                _bucket_add(
                    by_source_type.setdefault(source_type, {}),
                    nll=nll_sum,
                    tokens=token_count,
                    correct=correct,
                    bytes_=byte_count,
                )

    result = {
        "bundle_dir": str(bundle_dir),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "dataset_path": str(Path(str(dataset_manifest[dataset_key])).resolve()),
        "split": str(args.split),
        "device": str(device),
        "overall": _bucket_finish(total),
        "by_task": {key: _bucket_finish(value) for key, value in sorted(by_task.items())},
        "by_source_type": {key: _bucket_finish(value) for key, value in sorted(by_source_type.items())},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
