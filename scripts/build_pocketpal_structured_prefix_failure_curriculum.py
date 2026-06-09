#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from pocketpal_structured_decode import CONTENT, CONTENT_END, END, STRUCTURED


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prefix(task_type: str) -> str:
    return f"{STRUCTURED} <AK_ACTION_RESPOND> <AK_TASK_TYPE> {task_type} {CONTENT}"


def _repair_row(
    row: dict[str, Any],
    *,
    failure: dict[str, Any],
    label: str,
    index: int,
    repeat: int,
    weight: float,
) -> dict[str, Any]:
    task_type = str(row.get("task_type") or failure.get("task_type") or "active_agent_summary")
    expected = str(failure.get("expected") or row.get("expected_content") or "").strip()
    out = deepcopy(row)
    out["decoder_train_prefix"] = _prefix(task_type)
    out["decoder_text"] = f"{expected} {CONTENT_END} {END}".strip() if expected else f"{CONTENT_END} {END}"
    out["expected_content"] = expected
    out["negative_decoder_text"] = ""
    out["negative_loss_weight"] = 0.0
    out["decoder_loss_weight"] = 1.0
    out["weight"] = float(weight)
    out["source_type"] = f"structured_prefix_failure_repair_{label}"
    out["split"] = "train"
    out["example_id"] = _stable_id("structured_prefix_failure", label, index, repeat, out.get("source_id"), expected)
    out["source_id"] = f"{out.get('source_id', 'row')}_structured_prefix_failure_{label}_{index:05d}_{repeat:02d}"
    return out


def _anchor_row(row: dict[str, Any], *, label: str, index: int, weight_cap: float) -> dict[str, Any]:
    out = deepcopy(row)
    out.pop("decoder_train_prefix", None)
    out["example_id"] = _stable_id("structured_prefix_anchor", label, index, out.get("source_id"), out.get("encoder_text"), out.get("decoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_structured_prefix_anchor_{label}_{index:05d}"
    out["source_type"] = f"structured_prefix_anchor_{label}"
    out["split"] = "train"
    out["weight"] = min(max(float(out.get("weight") or 1.0), 1.0), float(weight_cap))
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    base_manifest_path = Path(args.base_manifest).expanduser().resolve()
    base_manifest = _load_manifest(base_manifest_path)
    train_rows = list(_iter_jsonl(Path(base_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(base_manifest["eval_dataset_path"])))
    by_source = {str(row.get("source_id") or ""): row for row in train_rows + eval_rows}
    task_filter = {item.strip() for item in str(args.task_filter or "").split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    repair_count = 0
    for failure_path in [Path(path).expanduser().resolve() for path in args.failure_json]:
        if not failure_path.exists():
            continue
        failures = json.loads(failure_path.read_text(encoding="utf-8")).get("failures", [])
        label = failure_path.parent.name
        for failure in failures:
            source_id = str(failure.get("source_id") or "")
            row = by_source.get(source_id)
            if row is None:
                continue
            task_type = str(row.get("task_type") or failure.get("task_type") or "")
            if task_filter and task_type not in task_filter:
                continue
            for repeat in range(max(1, int(args.failure_repeat))):
                rows.append(
                    _repair_row(
                        row,
                        failure=failure,
                        label=label,
                        index=repair_count,
                        repeat=repeat,
                        weight=float(args.failure_weight),
                    )
                )
            repair_count += 1

    anchors = list(train_rows)
    rng.shuffle(anchors)
    rows.extend(
        _anchor_row(row, label="broad", index=index, weight_cap=float(args.anchor_weight_cap))
        for index, row in enumerate(anchors[: max(0, int(args.anchor_rows))])
    )
    rng.shuffle(rows)

    out_eval: list[dict[str, Any]] = []
    for index, row in enumerate(eval_rows[: min(len(eval_rows), int(args.eval_rows))]):
        copied = deepcopy(row)
        copied.pop("decoder_train_prefix", None)
        copied["split"] = "eval"
        copied["example_id"] = _stable_id("structured_prefix_eval", index, copied.get("source_id"), copied.get("encoder_text"))
        out_eval.append(copied)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    train_count = _write_jsonl(train_path, rows)
    eval_count = _write_jsonl(eval_path, out_eval)
    manifest = {
        **base_manifest,
        "artifact_kind": "agentkernel_lite_structured_prefix_failure_curriculum",
        "objective": "pocketpal_structured_prefix_failure_repair",
        "source_manifest_path": str(base_manifest_path),
        "source_failure_json": [str(Path(path).expanduser().resolve()) for path in args.failure_json],
        "task_filter": sorted(task_filter),
        "failure_repair_examples": int(repair_count),
        "failure_repeat": int(args.failure_repeat),
        "failure_weight": float(args.failure_weight),
        "anchor_rows": int(args.anchor_rows),
        "anchor_weight_cap": float(args.anchor_weight_cap),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "total_examples": int(train_count + eval_count),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--failure-json", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--failure-repeat", type=int, default=3)
    parser.add_argument("--failure-weight", type=float, default=28.0)
    parser.add_argument("--anchor-rows", type=int, default=22000)
    parser.add_argument("--anchor-weight-cap", type=float, default=10.0)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=115)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
