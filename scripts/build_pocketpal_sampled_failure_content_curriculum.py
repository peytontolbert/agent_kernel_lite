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


CONTENT_ONLY_INSTRUCTION = (
    "Return only the final content text. Do not emit JSON, action names, task labels, "
    "AK control tokens, or proposal metadata."
)


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


def _content_prompt(text: str) -> str:
    lines = str(text or "").strip().splitlines()
    if lines and (
        "Return compact JSON" in lines[-1]
        or "Return AK structured tokens" in lines[-1]
        or "Return a structured decision" in lines[-1]
    ):
        lines = lines[:-1]
    lines.append(CONTENT_ONLY_INSTRUCTION)
    return "\n".join(lines).strip()


def _failure_row(row: dict[str, Any], *, failure: dict[str, Any], index: int, repeat: int, weight: float) -> dict[str, Any] | None:
    expected = str(failure.get("expected") or row.get("expected_content") or "").strip()
    if not expected:
        return None
    negative = str(failure.get("output") or "").strip()
    out = deepcopy(row)
    out.pop("decoder_prefix", None)
    out.pop("decoder_train_prefix", None)
    out["encoder_text"] = _content_prompt(str(row.get("encoder_text") or ""))
    out["decoder_text"] = expected
    out["expected_content"] = expected
    out["negative_decoder_text"] = negative
    out["negative_loss_weight"] = 1.0 if negative else 0.0
    out["decoder_loss_weight"] = 1.0
    out["weight"] = float(weight)
    out["split"] = "train"
    out["source_type"] = f"{out.get('source_type', 'row')}_sampled_failure_content"
    out["example_id"] = _stable_id("sampled_failure_content", index, repeat, out.get("source_id"), expected, negative)
    out["source_id"] = f"{out.get('source_id', 'row')}_sampled_failure_content_{index:05d}_{repeat:02d}"
    return out


def _anchor_row(row: dict[str, Any], *, index: int, weight_cap: float) -> dict[str, Any]:
    out = deepcopy(row)
    out.pop("decoder_train_prefix", None)
    out["split"] = "train"
    out["weight"] = min(max(float(out.get("weight") or 1.0), 1.0), float(weight_cap))
    out["source_type"] = f"{out.get('source_type', 'row')}_sampled_failure_content_anchor"
    out["example_id"] = _stable_id("sampled_failure_content_anchor", index, out.get("source_id"), out.get("encoder_text"), out.get("decoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_sampled_failure_content_anchor_{index:05d}"
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_manifest = _load_manifest(source_manifest_path)
    train_rows = list(_iter_jsonl(Path(source_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(source_manifest["eval_dataset_path"])))
    by_source = {str(row.get("source_id") or ""): row for row in train_rows + eval_rows}
    task_filter = {item.strip() for item in str(args.task_filter or "").split(",") if item.strip()}
    source_prefix_filter = {item.strip() for item in str(args.source_prefix_filter or "").split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    failure_examples = 0
    for failure_path in [Path(path).expanduser().resolve() for path in list(args.failure_json or [])]:
        if not failure_path.exists():
            continue
        failures = json.loads(failure_path.read_text(encoding="utf-8")).get("failures", [])
        for failure in failures:
            source_id = str(failure.get("source_id") or "")
            row = by_source.get(source_id)
            if row is None:
                continue
            task_type = str(row.get("task_type") or failure.get("task_type") or "")
            if task_filter and task_type not in task_filter:
                continue
            if source_prefix_filter and not any(source_id.startswith(prefix) for prefix in source_prefix_filter):
                continue
            for repeat in range(max(1, int(args.failure_repeat))):
                converted = _failure_row(row, failure=failure, index=failure_examples, repeat=repeat, weight=float(args.failure_weight))
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
        copied["split"] = "eval"
        copied["example_id"] = _stable_id("sampled_failure_content_eval", index, copied.get("source_id"), copied.get("encoder_text"))
        out_eval.append(copied)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    train_count = _write_jsonl(train_path, rows)
    eval_count = _write_jsonl(eval_path, out_eval)
    manifest = {
        **source_manifest,
        "artifact_kind": "agentkernel_lite_sampled_failure_content_curriculum",
        "objective": "pocketpal_sampled_failure_content",
        "content_only_instruction": CONTENT_ONLY_INSTRUCTION,
        "source_manifest_path": str(source_manifest_path),
        "source_failure_json": [str(Path(path).expanduser().resolve()) for path in list(args.failure_json or [])],
        "source_prefix_filter": sorted(source_prefix_filter),
        "task_filter": sorted(task_filter),
        "failure_examples": int(failure_examples),
        "failure_repeat": int(args.failure_repeat),
        "failure_weight": float(args.failure_weight),
        "anchor_rows": int(args.anchor_rows),
        "anchor_weight_cap": float(args.anchor_weight_cap),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "total_examples": int(train_count + eval_count),
        "source_counts": dict(sorted(Counter(str(row.get("source_type") or "unknown") for row in rows).items())),
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "unknown") for row in rows).items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--failure-json", action="append", default=[])
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--source-prefix-filter", default="skill:,concept:,paper:")
    parser.add_argument("--failure-repeat", type=int, default=1)
    parser.add_argument("--failure-weight", type=float, default=4.0)
    parser.add_argument("--anchor-rows", type=int, default=38000)
    parser.add_argument("--anchor-weight-cap", type=float, default=5.0)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=408)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
