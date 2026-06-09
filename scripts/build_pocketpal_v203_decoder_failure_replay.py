#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


WEAK_TASKS = {
    "active_agent_brainstorm",
    "active_agent_extraction",
    "active_agent_json",
    "active_agent_plan",
    "active_agent_translation",
}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize(
    row: dict[str, Any],
    *,
    suffix: str,
    weight: float,
    negative: str = "",
    split: str | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = f"{out.get('example_id')}_{suffix}"
    out["source_id"] = f"{out.get('source_id')}_{suffix}"
    out["source_type"] = f"{out.get('source_type', 'unknown')}_{suffix}"
    if split:
        out["split"] = split
    out["weight"] = float(weight)
    out["retrieval_loss_weight"] = 0.0
    if negative:
        out["negative_decoder_text"] = negative
        out["negative_loss_weight"] = 1.0
    else:
        out["negative_decoder_text"] = ""
        out["negative_loss_weight"] = 0.0
    return out


def _sample_by_task(
    rows: list[dict[str, Any]],
    *,
    rng: random.Random,
    tasks: set[str],
    per_task: int,
    weight: float,
    suffix_prefix: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task = str(row.get("task_type") or "")
        if task in tasks:
            by_task.setdefault(task, []).append(row)
    for task in sorted(tasks):
        candidates = list(by_task.get(task, []))
        if not candidates:
            continue
        rng.shuffle(candidates)
        for index, row in enumerate(candidates[: int(per_task)]):
            out.append(_normalize(row, suffix=f"{suffix_prefix}_{task}_{index:04d}", weight=weight))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--failure-json", default="tmp/v192b_direct_agent_prompts_all_failures.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument("--broad-sample", type=int, default=16000)
    parser.add_argument("--weak-per-task", type=int, default=1800)
    parser.add_argument("--failure-weight", type=float, default=80.0)
    parser.add_argument("--weak-weight", type=float, default=26.0)
    parser.add_argument("--broad-min-weight", type=float, default=2.0)
    parser.add_argument("--broad-max-weight", type=float, default=14.0)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    train_rows = list(_iter_jsonl(Path(broad_manifest["train_dataset_path"])))
    eval_rows_source = list(_iter_jsonl(Path(broad_manifest["eval_dataset_path"])))
    by_source_id = {str(row.get("source_id") or ""): row for row in train_rows + eval_rows_source}

    train_out: list[dict[str, Any]] = []
    eval_out: list[dict[str, Any]] = []

    failures = json.loads(Path(args.failure_json).read_text(encoding="utf-8")).get("failures", [])
    for index, failure in enumerate(failures):
        source_id = str(failure.get("source_id") or "")
        row = by_source_id.get(source_id)
        if not row:
            continue
        negative = str(failure.get("output") or "").strip()
        train_out.append(
            _normalize(
                row,
                suffix=f"v203_failure_{index:03d}",
                weight=float(args.failure_weight),
                negative=negative,
                split="train",
            )
        )
        eval_out.append(
            _normalize(
                row,
                suffix=f"v203_eval_failure_{index:03d}",
                weight=float(args.failure_weight),
                negative=negative,
                split="eval",
            )
        )

    rng.shuffle(train_rows)
    for index, row in enumerate(train_rows[: int(args.broad_sample)]):
        row_weight = float(row.get("weight") or 1.0)
        train_out.append(
            _normalize(
                row,
                suffix=f"v203_broad_{index:05d}",
                weight=min(max(row_weight, float(args.broad_min_weight)), float(args.broad_max_weight)),
                split="train",
            )
        )

    train_out.extend(
        _sample_by_task(
            train_rows,
            rng=rng,
            tasks=WEAK_TASKS,
            per_task=int(args.weak_per_task),
            weight=float(args.weak_weight),
            suffix_prefix="v203_weak",
        )
    )
    eval_out.extend(
        _sample_by_task(
            eval_rows_source,
            rng=rng,
            tasks=WEAK_TASKS,
            per_task=min(256, int(args.weak_per_task)),
            weight=float(args.weak_weight),
            suffix_prefix="v203_eval_weak",
        )
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_out)
    _write_jsonl(eval_path, eval_out)

    all_rows = train_out + eval_out
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_out),
        "failure_json": str(Path(args.failure_json).expanduser().resolve()),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v203_decoder_failure_replay",
        "source_manifest": str(Path(args.broad_manifest).expanduser().resolve()),
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_out),
        "weak_tasks": sorted(WEAK_TASKS),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
