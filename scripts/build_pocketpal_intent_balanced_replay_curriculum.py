#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _task(row: dict[str, Any]) -> str:
    return str(row.get("task_type") or "unknown").strip()


def _clone(row: dict[str, Any], *, source_id_suffix: str, source_type: str, weight: float) -> dict[str, Any]:
    patched = dict(row)
    source_id = str(patched.get("source_id") or source_type)
    patched["source_id"] = f"{source_id}::{source_id_suffix}"
    patched["source_type"] = source_type
    patched["weight"] = float(weight)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--failure-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--broad-per-task", type=int, default=640)
    parser.add_argument("--focus-per-task", type=int, default=256)
    parser.add_argument("--failure-repeat", type=int, default=12)
    parser.add_argument("--broad-weight", type=float, default=1.0)
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=1.25)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    base_manifest_path = Path(args.base_manifest).resolve()
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    train_rows = _iter_jsonl(Path(base["train_dataset_path"]))
    eval_rows = _iter_jsonl(Path(base["eval_dataset_path"]))
    failure_report = json.loads(Path(args.failure_json).read_text(encoding="utf-8"))
    failures = list(failure_report.get("failures") or [])
    failure_ids = {str(row.get("source_id") or "") for row in failures if row.get("source_id")}
    failure_tasks = {_task(row) for row in failures if row.get("task_type")}
    predicted_tasks = {
        f"active_agent_{row.get('route_intent')}"
        for row in failures
        if row.get("route_intent") and row.get("route_intent") != "web_search"
    }
    focus_tasks = sorted((failure_tasks | predicted_tasks) - {"active_agent_unknown"})

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_task[_task(row)].append(row)

    out_rows: list[dict[str, Any]] = []

    for task in sorted(by_task):
        rows = list(by_task[task])
        rng.shuffle(rows)
        for index, row in enumerate(rows[: max(0, int(args.broad_per_task))]):
            out_rows.append(
                _clone(
                    row,
                    source_id_suffix=f"intent_broad_anchor::{index:04d}",
                    source_type="intent_balanced_broad_anchor",
                    weight=float(args.broad_weight),
                )
            )

    for task in focus_tasks:
        rows = list(by_task.get(task) or [])
        rng.shuffle(rows)
        for index, row in enumerate(rows[: max(0, int(args.focus_per_task))]):
            out_rows.append(
                _clone(
                    row,
                    source_id_suffix=f"intent_focus_anchor::{index:04d}",
                    source_type="intent_balanced_focus_anchor",
                    weight=float(args.focus_weight),
                )
            )

    replay_rows = [row for row in eval_rows if str(row.get("source_id") or "") in failure_ids]
    for row in replay_rows:
        for repeat in range(max(0, int(args.failure_repeat))):
            out_rows.append(
                _clone(
                    row,
                    source_id_suffix=f"intent_failure_replay::{repeat:03d}",
                    source_type="intent_balanced_failure_replay",
                    weight=float(args.failure_weight),
                )
            )

    rng.shuffle(out_rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    tokenizer_dir = output_dir / "tokenizer"
    _write_jsonl(train_path, out_rows)
    _write_jsonl(eval_path, eval_rows)
    if "tokenizer_path" in base:
        src_tokenizer = Path(base["tokenizer_path"]).resolve()
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tokenizer, tokenizer_dir / "tokenizer.json")

    manifest = dict(base)
    task_counts = Counter(_task(row) for row in out_rows)
    source_counts = Counter(str(row.get("source_type") or "") for row in out_rows)
    manifest.update(
        {
            "artifact_kind": "agentkernel_lite_encdec_intent_balanced_replay_manifest",
            "dataset_objective": "pocketpal_intent_balanced_replay",
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "base_manifest_path": str(base_manifest_path),
            "failure_json_path": str(Path(args.failure_json).resolve()),
            "focus_tasks": focus_tasks,
            "failure_examples": len(replay_rows),
            "train_examples": len(out_rows),
            "task_type_counts": dict(sorted(task_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
        }
    )
    if (tokenizer_dir / "tokenizer.json").exists():
        manifest["tokenizer_path"] = str(tokenizer_dir / "tokenizer.json")
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "train_examples": len(out_rows),
                "failure_examples": len(replay_rows),
                "focus_tasks": focus_tasks,
                "source_counts": dict(sorted(source_counts.items())),
                "task_type_counts": dict(sorted(task_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
