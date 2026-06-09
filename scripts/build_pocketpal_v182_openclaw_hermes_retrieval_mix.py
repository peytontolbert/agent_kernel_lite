#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _retrieval_rows(path: Path, *, max_rows: int, seed: int, split: str) -> list[dict[str, Any]]:
    df = pd.read_parquet(path, engine="pyarrow")
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    rows: list[dict[str, Any]] = []
    for index, row in df.reset_index(drop=True).iterrows():
        negatives = row.get("retrieval_negative_doc_texts", [])
        if not isinstance(negatives, str):
            negatives = json.dumps(list(negatives) if isinstance(negatives, (list, tuple)) else negatives, ensure_ascii=False)
        rows.append(
            {
                "action": "gather_context",
                "decoder_text": "<AK_GATHER_CONTEXT> <AK_RETRIEVE> <AK_RET_SKILLS> <AK_CONF_MEDIUM>",
                "encoder_text": str(row["encoder_text"]),
                "example_id": f"v182_retrieval_{split}_{index:06d}",
                "expected_content": "<AK_GATHER_CONTEXT>",
                "intent_label": "web_search",
                "intent_label_id": 4,
                "negative_decoder_text": "",
                "negative_loss_weight": 0.0,
                "query_confidence_target": float(row.get("query_confidence_target", 0.05) or 0.05),
                "retrieval_coverage_target": float(row.get("retrieval_coverage_target", 0.85) or 0.85),
                "ood_query_target": float(row.get("ood_query_target", 0.05) or 0.05),
                "ood_evidence_target": float(row.get("ood_evidence_target", 0.05) or 0.05),
                "answer_confidence_target": float(row.get("answer_confidence_target", 0.05) or 0.05),
                "needs_verification_target": float(row.get("needs_verification_target", 0.35) or 0.35),
                "paper_action_validity_target": float(row.get("paper_action_validity_target", 1.0) or 1.0),
                "retrieval_doc_text": str(row["retrieval_doc_text"]),
                "retrieval_loss_weight": float(row.get("retrieval_loss_weight", 0.05) or 0.05),
                "retrieval_negative_doc_texts": negatives,
                "retrieval_query_text": str(row["retrieval_query_text"]),
                "source_id": str(row.get("source_id", f"v182_retrieval_{index:06d}")),
                "source_type": "openclaw_hermes_harness_skill_retrieval",
                "split": split,
                "task_type": "harness_skill_retrieval",
                "weight": 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-train", default="/data/agentkernel/artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset/train/part-00000.parquet")
    parser.add_argument("--retrieval-eval", default="/data/agentkernel/artifacts/agentkernel_lite_encdec/harness_skill_retrieval_dataset/eval/part-00000.parquet")
    parser.add_argument("--gate-manifest", default="tmp/pocketpal_v181_greeting_slot_boundary/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-retrieval", type=int, default=24000)
    parser.add_argument("--max-eval-retrieval", type=int, default=2000)
    parser.add_argument("--gate-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1820)
    args = parser.parse_args()

    gate_manifest = json.loads(Path(args.gate_manifest).read_text(encoding="utf-8"))
    train_rows = _retrieval_rows(Path(args.retrieval_train), max_rows=int(args.max_train_retrieval), seed=int(args.seed), split="train")
    eval_rows = _retrieval_rows(Path(args.retrieval_eval), max_rows=int(args.max_eval_retrieval), seed=int(args.seed) + 1, split="eval")

    for repeat in range(int(args.gate_repeat)):
        for row in _iter_jsonl(Path(gate_manifest["train_dataset_path"])):
            protected = dict(row)
            protected["example_id"] = f"{protected.get('example_id', protected.get('source_id', 'gate'))}_v182_{repeat:02d}"
            protected["source_type"] = "v181_generation_protection_replay"
            train_rows.append(protected)
        for row in _iter_jsonl(Path(gate_manifest["eval_dataset_path"])):
            protected = dict(row)
            protected["example_id"] = f"{protected.get('example_id', protected.get('source_id', 'gate'))}_v182_{repeat:02d}"
            protected["source_type"] = "v181_generation_protection_replay"
            eval_rows.append(protected)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": {"rewrite": 2, "web_search": 4, "casual": 5, "ask_user": 8, "summary": 9},
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v182_openclaw_hermes_retrieval_mix",
        "retrieval_sources": [
            "/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet",
            str(Path(args.retrieval_train).resolve()),
        ],
        "total_examples": len(train_rows) + len(eval_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
