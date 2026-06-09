#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sampler(repo_root: Path):
    path = repo_root / "scripts" / "sample_agentkernel_lite_encdec.py"
    spec = importlib.util.spec_from_file_location("sample_agentkernel_lite_encdec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sampler script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("intent_label_id", -1) or -1) < 0:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split", choices=("eval", "train"), default="eval")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sampler = _load_sampler(repo_root)
    sampler._install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = sampler._load_manifest(bundle_dir)
    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    dataset_key = "eval_dataset_path" if str(args.split) == "eval" else "train_dataset_path"
    rows = _load_rows(Path(str(dataset_manifest[dataset_key])), int(args.limit))
    if not rows:
        raise SystemExit("no intent-labeled rows to evaluate")

    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()
    if not hasattr(model, "agent_intent_logits") or getattr(model, "agent_intent_head", None) is None:
        raise SystemExit("bundle does not expose an agent intent head")

    label_names = {
        int(value): str(key)
        for key, value in (dataset_manifest.get("intent_labels", {}) or {}).items()
    }
    total = 0
    correct = 0
    confusion: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            target = int(row["intent_label_id"])
            ids = tokenizer.encode(str(row["encoder_text"]), max_length=int(args.max_encoder_tokens))
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            logits = model.agent_intent_logits(input_ids, mask)
            if logits is None:
                raise SystemExit("agent intent head returned no logits")
            pred = int(torch.argmax(logits[0]).detach().cpu().item())
            target_name = label_names.get(target, str(target))
            pred_name = label_names.get(pred, str(pred))
            confusion.setdefault(target_name, {})
            confusion[target_name][pred_name] = confusion[target_name].get(pred_name, 0) + 1
            total += 1
            correct += 1 if pred == target else 0
            if len(examples) < 25 and pred != target:
                examples.append(
                    {
                        "source_id": row.get("source_id"),
                        "task_type": row.get("task_type"),
                        "target": target_name,
                        "predicted": pred_name,
                    }
                )
    result = {
        "bundle_dir": str(bundle_dir),
        "dataset_path": str(Path(str(dataset_manifest[dataset_key])).resolve()),
        "split": str(args.split),
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "confusion": confusion,
        "sample_errors": examples,
    }
    if str(args.output_json).strip():
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
