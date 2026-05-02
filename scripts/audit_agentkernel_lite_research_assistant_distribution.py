#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir() and any(path.glob("*.parquet")):
        import pyarrow.parquet as pq

        for shard in sorted(path.glob("*.parquet")):
            parquet_file = pq.ParquetFile(shard)
            for batch in parquet_file.iter_batches(batch_size=4096):
                yield from batch.to_pylist()
        return
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=4096):
            yield from batch.to_pylist()
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _classify(row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type", "") or "").lower()
    action = str(row.get("action", "") or "").lower()
    encoder = str(row.get("encoder_text", "") or "").lower()
    decoder = str(row.get("decoder_text", "") or "").lower()
    if "selected_context_retrieve_new" in task_type or "<ak_retrieve_new>" in encoder or "retrieve_new" in decoder:
        return "selected_context_retrieve_new"
    if "selected" in task_type or "selected_context=1" in encoder or "<ak_selected_paper>" in encoder:
        if action == "gather_context":
            return "selected_context_retrieve_new"
        return "selected_context_answer"
    if action == "gather_context":
        return "paper_retrieval_request"
    if "retrieval=none" in encoder or "direct_chat" in task_type or "no_retrieval" in task_type:
        return "direct_chat_no_papers"
    if "<ak_context>" in encoder or "retrieval=ranked" in encoder or "evidence" in task_type:
        return "grounded_answer_with_papers"
    return "other_respond"


def _pct(value: int, total: int) -> float:
    return round((value / total * 100.0) if total else 0.0, 3)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    paths = [
        Path(str(manifest.get("train_dataset_path", ""))),
        Path(str(manifest.get("eval_dataset_path", ""))),
    ]
    counts: dict[str, int] = {}
    weighted: dict[str, float] = {}
    action_counts: dict[str, int] = {}
    total = 0
    total_weight = 0.0
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_rows(path):
            bucket = _classify(row)
            action = str(row.get("action", "") or "unknown")
            weight = float(row.get("weight", 1.0) or 1.0)
            counts[bucket] = counts.get(bucket, 0) + 1
            weighted[bucket] = weighted.get(bucket, 0.0) + weight
            action_counts[action] = action_counts.get(action, 0) + 1
            total += 1
            total_weight += weight
    desired = {
        "direct_chat_no_papers": "ordinary questions answer without retrieval",
        "paper_retrieval_request": "paper/source/literature requests choose gather_context",
        "grounded_answer_with_papers": "ranked paper evidence is synthesized into an answer",
        "selected_context_answer": "loaded paper followups answer from active context without fresh retrieval",
        "selected_context_retrieve_new": "loaded paper plus related/new literature request gathers new context",
    }
    return {
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "total_rows": total,
        "total_weight": round(total_weight, 3),
        "bucket_counts": dict(sorted(counts.items())),
        "bucket_percentages": {key: _pct(value, total) for key, value in sorted(counts.items())},
        "bucket_weight": {key: round(value, 3) for key, value in sorted(weighted.items())},
        "bucket_weight_percentages": {
            key: round((value / total_weight * 100.0) if total_weight else 0.0, 3)
            for key, value in sorted(weighted.items())
        },
        "action_counts": dict(sorted(action_counts.items())),
        "required_behavior_buckets": desired,
        "missing_required_buckets": [key for key in desired if counts.get(key, 0) <= 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    result = audit(args)
    if str(args.output_json).strip():
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
