#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ATTRACTORS = [
    "could you please send those documents",
    "design reviewed the search flow",
    "maria: own launch slides",
    "maria owns launch slides",
    "follow-up on friday contract review",
    "follow up on friday contract review",
]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _hash_split(row: dict[str, Any], eval_fraction: float) -> str:
    key = str(row.get("example_id") or row.get("source_id") or f"{row.get('encoder_text')}->{row.get('decoder_text')}")
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _blocked(row: dict[str, Any], attractors: list[str], fields: list[str]) -> str | None:
    haystack = "\n".join(str(row.get(field, "") or "") for field in fields).lower()
    for attractor in attractors:
        if attractor.lower() in haystack:
            return attractor
    return None


def filter_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    attractors = [item.strip() for item in (args.attractor or DEFAULT_ATTRACTORS) if item.strip()]
    fields = [item.strip() for item in str(args.fields).split(",") if item.strip()]
    if not fields:
        fields = ["decoder_text"]

    kept: list[dict[str, Any]] = []
    removed_counts: dict[str, int] = {}
    for source in (Path(str(manifest["train_dataset_path"])), Path(str(manifest["eval_dataset_path"]))):
        for row in _iter_jsonl(source):
            reason = _blocked(row, attractors, fields)
            if reason:
                removed_counts[reason] = removed_counts.get(reason, 0) + 1
                continue
            if not str(row.get("encoder_text", "") or "").strip() or not str(row.get("decoder_text", "") or "").strip():
                removed_counts["empty_encoder_or_decoder"] = removed_counts.get("empty_encoder_or_decoder", 0) + 1
                continue
            row["split"] = _hash_split(row, float(args.eval_fraction))
            kept.append(row)

    train_rows = [row for row in kept if row["split"] == "train"]
    eval_rows = [row for row in kept if row["split"] == "eval"]
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    output_manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in kept:
        action = str(row.get("action", "") or "unknown")
        task_type = str(row.get("task_type", "") or "unknown")
        source_type = str(row.get("source_type", "") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
        source_counts[source_type] = source_counts.get(source_type, 0) + 1

    output_manifest = {
        **manifest,
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": str(args.objective or manifest.get("objective", "chat")),
        "manifest_path": str(output_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "source_manifest_path": str(manifest_path),
        "filter": {
            "blocked_attractors": attractors,
            "fields": fields,
            "removed_counts": dict(sorted(removed_counts.items())),
        },
        "total_examples": len(kept),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "target_action_counts": dict(sorted(action_counts.items())),
        "source_action_counts": dict(sorted(action_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attractor", action="append")
    parser.add_argument("--fields", default="decoder_text")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--objective", default="")
    print(json.dumps(filter_manifest(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
