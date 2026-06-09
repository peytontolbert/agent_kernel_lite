#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any


WEAK_TASKS = {
    "active_agent_translation": 900,
    "active_agent_checklist": 900,
    "active_agent_risks": 800,
    "active_agent_summary": 800,
    "active_agent_subject": 650,
    "active_agent_json": 600,
    "active_agent_brainstorm": 400,
    "active_agent_plan": 700,
    "active_agent_ranking": 350,
    "active_agent_extraction": 900,
    "active_agent_action_items": 500,
    "active_agent_rewrite": 500,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _stable_id(row: dict[str, Any], suffix: str) -> str:
    seed = json.dumps(
        {
            "source": row.get("source_id") or row.get("example_id"),
            "task": row.get("task_type"),
            "decoder": row.get("decoder_text"),
            "suffix": suffix,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _copy(row: dict[str, Any], *, suffix: str, source_type: str, weight: float, tags: list[str] | None = None) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = _stable_id(row, suffix)
    out["source_id"] = str(row.get("source_id") or row.get("example_id") or out["example_id"])
    out["source_type"] = source_type
    out["decoder_loss_weight"] = float(weight)
    if tags:
        out["failure_tags"] = list(tags)
    return out


def _by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in (row.get("source_id"), row.get("example_id")):
            if key:
                result.setdefault(str(key), row)
    return result


def _sample_task_rows(
    rows: list[dict[str, Any]],
    *,
    rng: random.Random,
    task_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("task_type") or "") == task_type]
    rng.shuffle(candidates)
    return candidates[: min(limit, len(candidates))]


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    base_manifest_path = Path(args.base_manifest).expanduser().resolve()
    direct_eval_path = Path(args.direct_eval_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_train = _read_jsonl(Path(base_manifest["train_dataset_path"]))
    base_eval = _read_jsonl(Path(base_manifest["eval_dataset_path"]))
    base_all = base_train + base_eval
    source_index = _by_source(base_all)

    direct_eval = json.loads(direct_eval_path.read_text(encoding="utf-8"))
    failures = [item for item in direct_eval.get("failures", []) if isinstance(item, dict)]

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    missing_failure_sources: list[str] = []
    failure_task_counts: Counter[str] = Counter()

    for failure_index, failure in enumerate(failures):
        source_id = str(failure.get("source_id") or "")
        row = source_index.get(source_id)
        if row is None:
            missing_failure_sources.append(source_id)
            continue
        task_type = str(row.get("task_type") or failure.get("task_type") or "")
        failure_task_counts[task_type] += 1
        tags = [str(tag) for tag in failure.get("failures", [])]
        for repeat in range(int(args.failure_repeats)):
            train_rows.append(
                _copy(
                    row,
                    suffix=f"failure_{failure_index:03d}_{repeat:03d}",
                    source_type="akv1_broad_direct_failure_replay",
                    weight=float(args.failure_weight),
                    tags=tags,
                )
            )
        eval_rows.append(
            _copy(
                row,
                suffix=f"failure_eval_{failure_index:03d}",
                source_type="akv1_broad_direct_failure_eval",
                weight=1.0,
                tags=tags,
            )
        )

    for task_type, limit in WEAK_TASKS.items():
        task_rows = _sample_task_rows(base_train, rng=rng, task_type=task_type, limit=int(limit * float(args.weak_task_scale)))
        for index, row in enumerate(task_rows):
            train_rows.append(
                _copy(
                    row,
                    suffix=f"weak_{task_type}_{index:05d}",
                    source_type=f"akv1_weak_task_focus_{task_type}",
                    weight=float(args.weak_task_weight),
                )
            )

    task_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_train:
        task_buckets[str(row.get("task_type") or "")].append(row)
    anchor_rows: list[dict[str, Any]] = []
    per_task_anchor = max(1, int(args.anchor_rows) // max(1, len(task_buckets)))
    for task_type in sorted(task_buckets):
        bucket = list(task_buckets[task_type])
        rng.shuffle(bucket)
        anchor_rows.extend(bucket[:per_task_anchor])
    rng.shuffle(anchor_rows)
    anchor_rows = anchor_rows[: int(args.anchor_rows)]
    for index, row in enumerate(anchor_rows):
        train_rows.append(
            _copy(
                row,
                suffix=f"anchor_{index:05d}",
                source_type="akv1_broad_anchor_replay",
                weight=float(args.anchor_weight),
            )
        )

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    source_counts = Counter(str(row.get("source_type") or "unknown") for row in train_rows + eval_rows)
    task_type_counts = Counter(str(row.get("task_type") or "unknown") for row in train_rows + eval_rows)
    target_action_counts = Counter(str(row.get("action") or "unknown") for row in train_rows + eval_rows)
    manifest = dict(base_manifest)
    manifest.update(
        {
            "artifact_kind": "agentkernel_lite_encdec_akv1_broad_failure_curriculum",
            "objective": "pocketpal_akv1_broad_failure_curriculum",
            "source_manifest_path": str(base_manifest_path),
            "direct_eval_json": str(direct_eval_path),
            "manifest_path": str((output_dir / "agentkernel_lite_encdec_dataset_manifest.json").resolve()),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
            "total_examples": len(train_rows) + len(eval_rows),
            "failure_count": len(failures),
            "matched_failure_count": len(eval_rows),
            "missing_failure_sources": missing_failure_sources,
            "failure_task_counts": dict(sorted(failure_task_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "task_type_counts": dict(sorted(task_type_counts.items())),
            "target_action_counts": dict(sorted(target_action_counts.items())),
            "weighting_policy": {
                "failure_repeats": int(args.failure_repeats),
                "failure_weight": float(args.failure_weight),
                "weak_task_weight": float(args.weak_task_weight),
                "anchor_rows": int(args.anchor_rows),
                "anchor_weight": float(args.anchor_weight),
            },
        }
    )
    (output_dir / "agentkernel_lite_encdec_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="tmp/pocketpal_stage67_structured_copy_decoder_v172d_akv1/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--direct-eval-json", default="artifacts/pocketpal_promotion_gates/v277a/direct_akv1_structured_keep_special.json")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage69_akv1_broad_failure_curriculum")
    parser.add_argument("--failure-repeats", type=int, default=18)
    parser.add_argument("--failure-weight", type=float, default=3.0)
    parser.add_argument("--weak-task-scale", type=float, default=1.0)
    parser.add_argument("--weak-task-weight", type=float, default=1.4)
    parser.add_argument("--anchor-rows", type=int, default=4200)
    parser.add_argument("--anchor-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
