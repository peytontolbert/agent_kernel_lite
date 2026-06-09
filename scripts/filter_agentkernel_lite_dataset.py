#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _allowed(row: dict[str, Any], *, actions: set[str], task_types: set[str], max_decoder_chars: int) -> bool:
    if actions and str(row.get("action", "") or "") not in actions:
        return False
    if task_types and str(row.get("task_type", "") or "") not in task_types:
        return False
    decoder_text = str(row.get("decoder_text", "") or "")
    if max_decoder_chars > 0 and len(decoder_text) > max_decoder_chars:
        return False
    return bool(str(row.get("encoder_text", "") or "").strip() and decoder_text.strip())


def filter_dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actions = {item for item in str(args.keep_action or "").split(",") if item}
    task_types = {item for item in str(args.keep_task_type or "").split(",") if item}
    rows = [
        row
        for source in (Path(str(manifest["train_dataset_path"])), Path(str(manifest["eval_dataset_path"])))
        for row in _iter_jsonl(source)
        if _allowed(row, actions=actions, task_types=task_types, max_decoder_chars=int(args.max_decoder_chars))
    ]
    for row in rows:
        key = str(row.get("example_id") or row.get("source_id") or f"{row.get('encoder_text')}->{row.get('decoder_text')}")
        row["split"] = _hash_split(key, float(args.eval_fraction))

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    output_manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("action", "") or "unknown")
        task_type = str(row.get("task_type", "") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    output_manifest = {
        **manifest,
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": str(args.objective or manifest.get("objective", "chat")),
        "manifest_path": str(output_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "source_manifest_path": str(manifest_path),
        "filter": {
            "keep_action": sorted(actions),
            "keep_task_type": sorted(task_types),
            "max_decoder_chars": int(args.max_decoder_chars),
        },
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "target_action_counts": dict(sorted(action_counts.items())),
        "source_action_counts": dict(sorted(action_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-action", default="")
    parser.add_argument("--keep-task-type", default="")
    parser.add_argument("--max-decoder-chars", type=int, default=5000)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    parser.add_argument("--objective", default="chat")
    print(json.dumps(filter_dataset(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
