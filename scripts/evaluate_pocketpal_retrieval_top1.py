#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sampler(repo_root: Path):
    path = repo_root / "scripts" / "sample_agentkernel_lite_encdec.py"
    spec = importlib.util.spec_from_file_location("sample_agentkernel_lite_encdec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sampler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    repo_root = _repo_root()
    sampler = _load_sampler(repo_root)
    sampler._install_paths(repo_root)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    if not hasattr(model, "retrieval_query_embedding") or not hasattr(model, "retrieval_doc_embedding"):
        raise RuntimeError("bundle does not expose retrieval embedding heads")
    device = torch.device(str(args.device))
    model.to(device).eval()

    rows = []
    with Path(args.eval_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("retrieval_query_text") and row.get("retrieval_doc_text") and row.get("retrieval_negative_doc_texts"):
                rows.append(row)
            if len(rows) >= int(args.max_examples):
                break

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

    correct = 0
    total = 0
    margins: list[float] = []
    examples = []
    with torch.no_grad():
        for row in rows:
            try:
                negatives = json.loads(str(row.get("retrieval_negative_doc_texts") or "[]"))
            except json.JSONDecodeError:
                negatives = []
            negatives = [str(item) for item in negatives[:4] if str(item).strip()]
            if not negatives:
                continue
            q_ids, q_mask = encode(str(row["retrieval_query_text"]), 192)
            query = model.retrieval_query_embedding(q_ids, q_mask)
            docs = [str(row["retrieval_doc_text"]), *negatives]
            scores = []
            for doc in docs:
                d_ids, d_mask = encode(doc, 320)
                doc_embedding = model.retrieval_doc_embedding(d_ids, d_mask)
                scores.append(float((query @ doc_embedding.transpose(0, 1)).item()))
            order = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
            passed = order[0] == 0
            correct += 1 if passed else 0
            total += 1
            margin = scores[0] - max(scores[1:])
            margins.append(margin)
            if len(examples) < 8:
                examples.append({"passed": passed, "margin": margin, "scores": scores, "source_id": row.get("source_id")})
    summary = {
        "bundle_dir": str(bundle_dir),
        "evaluated": total,
        "top1": correct,
        "accuracy": float(correct) / float(total or 1),
        "mean_margin": sum(margins) / float(len(margins) or 1),
        "min_margin": min(margins) if margins else None,
        "examples": examples,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        Path(args.output_json).expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
