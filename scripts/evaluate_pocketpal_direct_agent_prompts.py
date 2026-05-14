#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
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


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9$-]+", str(text or ""))
        if len(token) > 2 and token.lower() not in {"the", "and", "for", "you", "your", "with", "that", "this"}
    ]


def _looks_malformed(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if re.search(r"[{}[\]\"]", raw) and re.search(
        r"\b(action|proposal|metadata|respond|retrieval|memory|slot|policy|capability)\b",
        raw,
        flags=re.I,
    ):
        return True
    if re.search(r"</?AK_[A-Z0-9_]+>|AK_[A-Z0-9_]+|\b(?:RESPOND|RETRIEVAL|ACTION|GOAL|CONF|MEMORY|SLOT|EXTENSION|CAPABILITY)\b", raw):
        return True
    if re.search(r"\b[a-z]{2,}[A-Z]{2,}[a-zA-Z]{2,}\b", raw):
        return True
    if len(raw) > 120 and not re.search(r"[.!?\n]", raw):
        return True
    return False


def _score_output(output: str, expected: str) -> tuple[bool, float, list[str]]:
    failures: list[str] = []
    if _looks_malformed(output):
        failures.append("malformed")
    output_tokens = set(_content_tokens(output))
    expected_tokens = set(_content_tokens(expected))
    overlap = len(output_tokens & expected_tokens)
    recall = overlap / float(len(expected_tokens) or 1)
    if recall < 0.45:
        failures.append(f"low_recall:{recall:.2f}")
    return not failures, recall, failures


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    sampler = _load_sampler(repo_root)
    sampler._install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    rows = _iter_jsonl(Path(dataset_manifest["eval_dataset_path"]))
    if int(args.max_examples) > 0:
        rows = rows[: int(args.max_examples)]

    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    passed = 0
    by_task: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    recalls: list[float] = []
    for row in rows:
        output = sampler._generate(
            model,
            tokenizer,
            str(row["encoder_text"]),
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        ok, recall, row_failures = _score_output(output, str(row["decoder_text"]))
        recalls.append(recall)
        task = str(row.get("task_type", "unknown") or "unknown")
        bucket = by_task.setdefault(task, {"total": 0, "passed": 0, "recall_sum": 0.0})
        bucket["total"] += 1
        bucket["passed"] += 1 if ok else 0
        bucket["recall_sum"] += recall
        if ok:
            passed += 1
        elif len(failures) < int(args.max_failures):
            failures.append(
                {
                    "source_id": row.get("source_id", ""),
                    "task_type": task,
                    "expected": str(row["decoder_text"]),
                    "output": output,
                    "failures": row_failures,
                    "recall": recall,
                }
            )
    for bucket in by_task.values():
        bucket["pass_rate"] = bucket["passed"] / float(bucket["total"] or 1)
        bucket["mean_recall"] = bucket["recall_sum"] / float(bucket["total"] or 1)
        del bucket["recall_sum"]
    summary = {
        "bundle_dir": str(bundle_dir),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "device": str(device),
        "examples": len(rows),
        "passed": passed,
        "pass_rate": passed / float(len(rows) or 1),
        "mean_recall": sum(recalls) / float(len(recalls) or 1),
        "by_task": by_task,
        "failures": failures,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.65)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=120)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
