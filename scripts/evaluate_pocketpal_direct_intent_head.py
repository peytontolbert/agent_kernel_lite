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


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _task_to_intent(task_type: str, intent_labels: dict[str, int]) -> str:
    task = str(task_type or "")
    prefix = "active_agent_"
    if task.startswith(prefix):
        task = task[len(prefix) :]
    aliases = {
        "action_items": "action_items",
        "json": "json",
        "plan": "plan",
        "translation": "translation",
        "brainstorm": "brainstorm",
    }
    return aliases.get(task, task if task in intent_labels else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument(
        "--direct-dataset-manifest",
        default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json",
    )
    parser.add_argument(
        "--intent-label-manifest",
        default="tmp/pocketpal_stage61_slot_operator_curriculum/agentkernel_lite_encdec_dataset_manifest.json",
    )
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

    direct_manifest = json.loads(Path(args.direct_dataset_manifest).read_text(encoding="utf-8"))
    label_manifest = json.loads(Path(args.intent_label_manifest).read_text(encoding="utf-8"))
    intent_labels = {str(k): int(v) for k, v in (label_manifest.get("intent_labels", {}) or {}).items()}
    label_names = {v: k for k, v in intent_labels.items()}
    rows = _read_jsonl(Path(direct_manifest["eval_dataset_path"]), int(args.limit))

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()
    if not hasattr(model, "agent_intent_logits") or getattr(model, "agent_intent_head", None) is None:
        raise SystemExit("bundle does not expose an agent intent head")

    total = 0
    correct = 0
    by_intent: dict[str, dict[str, Any]] = {}
    sample_errors: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            target_name = _task_to_intent(str(row.get("task_type", "")), intent_labels)
            if not target_name:
                continue
            target = int(intent_labels[target_name])
            ids = tokenizer.encode(str(row.get("encoder_text", "")), max_length=int(args.max_encoder_tokens))
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            logits = model.agent_intent_logits(input_ids, mask)
            if logits is None:
                raise SystemExit("agent intent head returned no logits")
            pred = int(torch.argmax(logits[0]).detach().cpu().item())
            pred_name = label_names.get(pred, str(pred))
            total += 1
            correct += 1 if pred == target else 0
            bucket = by_intent.setdefault(target_name, {"total": 0, "correct": 0, "predictions": {}})
            bucket["total"] += 1
            bucket["correct"] += 1 if pred == target else 0
            bucket["predictions"][pred_name] = int(bucket["predictions"].get(pred_name, 0)) + 1
            if pred != target and len(sample_errors) < 25:
                sample_errors.append(
                    {
                        "source_id": row.get("source_id"),
                        "task_type": row.get("task_type"),
                        "target": target_name,
                        "predicted": pred_name,
                    }
                )
    for bucket in by_intent.values():
        bucket["accuracy"] = bucket["correct"] / max(int(bucket["total"]), 1)
    result = {
        "bundle_dir": str(bundle_dir),
        "direct_dataset_manifest": str(Path(args.direct_dataset_manifest).resolve()),
        "intent_label_manifest": str(Path(args.intent_label_manifest).resolve()),
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "by_intent": by_intent,
        "sample_errors": sample_errors,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
