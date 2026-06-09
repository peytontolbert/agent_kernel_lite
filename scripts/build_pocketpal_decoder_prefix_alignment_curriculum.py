#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


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


def _json_payload(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_string_suffix(value: str) -> str:
    encoded = json.dumps(str(value or ""), ensure_ascii=False)
    return encoded[1:-1]


def _completion_suffix(row: dict[str, Any]) -> str | None:
    prefix = str(row.get("decoder_prefix") or "").strip()
    if not prefix or '"content"' not in prefix:
        return None
    payload = _json_payload(str(row.get("decoder_text") or row.get("json_decoder_text") or ""))
    if payload is None:
        return None
    content = str(row.get("expected_content") or payload.get("content") or "").strip()
    if not content:
        return None
    metadata = payload.get("proposal_metadata")
    suffix = _json_string_suffix(content) + '"'
    if isinstance(metadata, dict) and metadata:
        suffix += ',"proposal_metadata":' + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    suffix += "}"
    return suffix


def _prefix_row(row: dict[str, Any], *, index: int, weight: float) -> dict[str, Any] | None:
    prefix = str(row.get("decoder_prefix") or "").strip()
    suffix = _completion_suffix(row)
    if not prefix or suffix is None:
        return None
    out = deepcopy(row)
    out["decoder_train_prefix"] = prefix
    out["decoder_text"] = suffix
    out["negative_decoder_text"] = ""
    out["negative_loss_weight"] = 0.0
    out["decoder_loss_weight"] = 1.0
    out["weight"] = float(weight)
    out["source_type"] = f"{out.get('source_type', 'row')}_json_prefix_alignment"
    out["split"] = "train"
    out["example_id"] = _stable_id("json_prefix_alignment", index, out.get("source_id"), prefix, suffix)
    out["source_id"] = f"{out.get('source_id', 'row')}_json_prefix_alignment_{index:05d}"
    return out


def _negative_suffix(output: str, *, prefix: str) -> str:
    raw = str(output or "").strip()
    prefix = str(prefix or "").strip()
    if prefix and raw.startswith(prefix):
        return raw[len(prefix):].strip()
    payload = _json_payload(raw)
    if payload is not None and "content" in payload:
        metadata = payload.get("proposal_metadata")
        suffix = _json_string_suffix(str(payload.get("content") or "")) + '"'
        if isinstance(metadata, dict) and metadata:
            suffix += ',"proposal_metadata":' + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        suffix += "}"
        return suffix
    return raw


def _failure_row(
    row: dict[str, Any],
    *,
    failure: dict[str, Any],
    label: str,
    index: int,
    repeat: int,
    weight: float,
) -> dict[str, Any] | None:
    converted = _prefix_row(row, index=index, weight=float(weight))
    if converted is None:
        return None
    prefix = str(converted.get("decoder_train_prefix") or "")
    negative = _negative_suffix(str(failure.get("output") or ""), prefix=prefix)
    if negative:
        converted["negative_decoder_train_prefix"] = prefix
        converted["negative_decoder_text"] = negative
        converted["negative_loss_weight"] = 1.0
    expected = str(failure.get("expected") or "").strip()
    if expected:
        metadata = _json_payload(str(row.get("decoder_text") or row.get("json_decoder_text") or "")) or {}
        suffix = _json_string_suffix(expected) + '"'
        proposal_metadata = metadata.get("proposal_metadata")
        if isinstance(proposal_metadata, dict) and proposal_metadata:
            suffix += ',"proposal_metadata":' + json.dumps(proposal_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        suffix += "}"
        converted["decoder_text"] = suffix
        converted["expected_content"] = expected
    converted["source_type"] = f"{converted.get('source_type', 'row')}_sampled_failure_{label}"
    converted["example_id"] = _stable_id("json_prefix_failure", label, index, repeat, row.get("source_id"), converted.get("decoder_text"), negative)
    converted["source_id"] = f"{row.get('source_id', 'row')}_json_prefix_failure_{label}_{index:05d}_{repeat:02d}"
    return converted


def _anchor_row(row: dict[str, Any], *, index: int, weight_cap: float) -> dict[str, Any]:
    out = deepcopy(row)
    out.pop("decoder_train_prefix", None)
    out["weight"] = min(max(float(out.get("weight") or 1.0), 1.0), float(weight_cap))
    out["split"] = "train"
    out["source_type"] = f"{out.get('source_type', 'row')}_prefix_alignment_anchor"
    out["example_id"] = _stable_id("prefix_alignment_anchor", index, out.get("source_id"), out.get("encoder_text"), out.get("decoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_prefix_alignment_anchor_{index:05d}"
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_manifest = _load_manifest(source_manifest_path)
    train_rows = _read_jsonl(Path(source_manifest["train_dataset_path"]))
    eval_rows = _read_jsonl(Path(source_manifest["eval_dataset_path"]))
    by_source = {str(row.get("source_id") or ""): row for row in train_rows + eval_rows}
    task_filter = {item.strip() for item in str(args.task_filter or "").split(",") if item.strip()}

    candidates = [
        row
        for row in train_rows
        if (not task_filter or str(row.get("task_type") or "") in task_filter)
        and _completion_suffix(row) is not None
    ]
    rng.shuffle(candidates)
    if int(args.balanced_per_task) > 0:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            grouped.setdefault(str(row.get("task_type") or "unknown"), []).append(row)
        candidates = []
        for task_rows in grouped.values():
            candidates.extend(task_rows[: int(args.balanced_per_task)])
        rng.shuffle(candidates)
    candidates = candidates[: max(0, int(args.prefix_rows))]

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        converted = _prefix_row(row, index=index, weight=float(args.prefix_weight))
        if converted is not None:
            rows.append(converted)

    failure_examples = 0
    for failure_path in [Path(path).expanduser().resolve() for path in list(args.failure_json or [])]:
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
                converted = _failure_row(
                    row,
                    failure=failure,
                    label=label,
                    index=failure_examples,
                    repeat=repeat,
                    weight=float(args.failure_weight),
                )
                if converted is not None:
                    rows.append(converted)
            failure_examples += 1

    anchors = list(train_rows)
    rng.shuffle(anchors)
    rows.extend(
        _anchor_row(row, index=index, weight_cap=float(args.anchor_weight_cap))
        for index, row in enumerate(anchors[: max(0, int(args.anchor_rows))])
    )
    rng.shuffle(rows)

    out_eval: list[dict[str, Any]] = []
    for index, row in enumerate(eval_rows[: min(len(eval_rows), int(args.eval_rows))]):
        copied = deepcopy(row)
        copied.pop("decoder_train_prefix", None)
        copied["split"] = "eval"
        copied["example_id"] = _stable_id("prefix_alignment_eval", index, copied.get("source_id"), copied.get("encoder_text"))
        out_eval.append(copied)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    train_count = _write_jsonl(train_path, rows)
    eval_count = _write_jsonl(eval_path, out_eval)
    task_counts = Counter(str(row.get("task_type") or "unknown") for row in rows)
    source_counts = Counter(str(row.get("source_type") or "unknown") for row in rows)
    manifest = {
        **source_manifest,
        "artifact_kind": "agentkernel_lite_decoder_prefix_alignment_curriculum",
        "objective": "pocketpal_decoder_prefix_alignment",
        "source_manifest_path": str(source_manifest_path),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "total_examples": int(train_count + eval_count),
        "prefix_alignment_examples": len(candidates),
        "sampled_failure_examples": int(failure_examples),
        "source_failure_json": [str(Path(path).expanduser().resolve()) for path in list(args.failure_json or [])],
        "source_counts": dict(sorted(source_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
        "prefix_alignment_policy": {
            "task_filter": sorted(task_filter),
            "prefix_rows": int(args.prefix_rows),
            "balanced_per_task": int(args.balanced_per_task),
            "prefix_weight": float(args.prefix_weight),
            "failure_repeat": int(args.failure_repeat),
            "failure_weight": float(args.failure_weight),
            "anchor_rows": int(args.anchor_rows),
            "anchor_weight_cap": float(args.anchor_weight_cap),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--prefix-rows", type=int, default=18000)
    parser.add_argument("--balanced-per-task", type=int, default=1600)
    parser.add_argument("--prefix-weight", type=float, default=2.0)
    parser.add_argument("--failure-json", action="append", default=[])
    parser.add_argument("--failure-repeat", type=int, default=1)
    parser.add_argument("--failure-weight", type=float, default=4.0)
    parser.add_argument("--anchor-rows", type=int, default=32000)
    parser.add_argument("--anchor-weight-cap", type=float, default=6.0)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=405)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
