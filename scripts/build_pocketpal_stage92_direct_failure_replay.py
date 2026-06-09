#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decoder_with_expected_content(row: dict[str, Any], expected: str) -> str:
    expected = str(expected or "").strip()
    raw = str(row.get("decoder_text") or "").strip()
    if not expected:
        return raw
    if "<AK_CONTENT>" in raw:
        if "</AK_CONTENT>" in raw:
            return re.sub(
                r"<AK_CONTENT>\s*.*?\s*</AK_CONTENT>",
                f"<AK_CONTENT> {expected} </AK_CONTENT>",
                raw,
                count=1,
                flags=re.S,
            )
        before, _, _after = raw.partition("<AK_CONTENT>")
        suffix = " <AK_END>" if "<AK_END>" not in raw else ""
        return f"{before}<AK_CONTENT> {expected} </AK_CONTENT>{suffix}".strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return expected
    if not isinstance(parsed, dict):
        return expected
    if isinstance(parsed.get("decision_packet"), dict) and isinstance(parsed["decision_packet"].get("decision"), dict):
        parsed["decision_packet"]["decision"]["content"] = expected
    else:
        parsed["content"] = expected
        parsed.setdefault("action", str(row.get("action") or "respond"))
        parsed.setdefault("proposal_metadata", {"task_type": str(row.get("task_type") or "")})
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def _repair_row(
    row: dict[str, Any],
    *,
    failure: dict[str, Any],
    label: str,
    index: int,
    weight: float,
) -> dict[str, Any]:
    out = deepcopy(row)
    expected = str(failure.get("expected") or "").strip()
    if expected:
        out["decoder_text"] = _decoder_with_expected_content(out, expected)
        out["expected_content"] = expected
    negative = str(failure.get("output") or "").strip()
    if negative:
        out["negative_decoder_text"] = negative
        out["negative_loss_weight"] = 1.0
    out["decoder_loss_weight"] = 1.0
    out["example_id"] = _stable_id("stage92", label, index, out.get("source_id"), expected, negative)
    out["source_id"] = f"{out.get('source_id', 'row')}_stage92_{label}_{index:04d}"
    out["source_type"] = f"stage92_direct_failure_replay_{label}"
    out["split"] = "train"
    out["weight"] = float(weight)
    return out


def _anchor_row(row: dict[str, Any], *, label: str, index: int, weight_cap: float) -> dict[str, Any]:
    out = deepcopy(row)
    out["example_id"] = _stable_id("stage92_anchor", label, index, out.get("source_id"), out.get("encoder_text"), out.get("decoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_stage92_anchor_{label}_{index:05d}"
    out["source_type"] = f"stage92_anchor_{label}"
    out["split"] = "train"
    out["weight"] = min(max(float(out.get("weight") or 1.0), 1.0), float(weight_cap))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", required=True)
    parser.add_argument("--retrieval-manifest", default="")
    parser.add_argument("--failure-json", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--anchor-sample", type=int, default=18000)
    parser.add_argument("--retrieval-protect", type=int, default=2500)
    parser.add_argument("--failure-repeat", type=int, default=4)
    parser.add_argument("--failure-weight", type=float, default=84.0)
    parser.add_argument("--protect-task", action="append", default=[])
    parser.add_argument("--protect-per-task", type=int, default=0)
    parser.add_argument("--protect-weight", type=float, default=18.0)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    broad_manifest = _load_manifest(Path(args.broad_manifest).resolve())
    train_rows = list(_iter_jsonl(Path(broad_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(broad_manifest["eval_dataset_path"])))
    by_source = {str(row.get("source_id") or ""): row for row in train_rows + eval_rows}

    rows: list[dict[str, Any]] = []
    repair_count = 0
    for failure_path in [Path(path).resolve() for path in args.failure_json]:
        if not failure_path.exists():
            continue
        failures = json.loads(failure_path.read_text(encoding="utf-8")).get("failures", [])
        label = failure_path.parent.name
        for failure in failures:
            source_id = str(failure.get("source_id") or "")
            base = by_source.get(source_id)
            if base is None:
                continue
            for repeat in range(max(1, int(args.failure_repeat))):
                rows.append(
                    _repair_row(
                        base,
                        failure=failure,
                        label=label,
                        index=repair_count * max(1, int(args.failure_repeat)) + repeat,
                        weight=float(args.failure_weight),
                    )
                )
            repair_count += 1

    anchors = list(train_rows)
    rng.shuffle(anchors)
    for index, row in enumerate(anchors[: max(0, int(args.anchor_sample))]):
        rows.append(_anchor_row(row, label="broad", index=index, weight_cap=16.0))

    protected_task_counts: Counter[str] = Counter()
    protect_tasks = {str(task).strip() for task in args.protect_task if str(task).strip()}
    if protect_tasks and int(args.protect_per_task) > 0:
        by_task: dict[str, list[dict[str, Any]]] = {}
        for row in train_rows:
            task = str(row.get("task_type") or "")
            if task in protect_tasks:
                by_task.setdefault(task, []).append(row)
        for task in sorted(protect_tasks):
            candidates = list(by_task.get(task, []))
            rng.shuffle(candidates)
            for index, row in enumerate(candidates[: int(args.protect_per_task)]):
                protected = _anchor_row(row, label=f"protect_{task}", index=index, weight_cap=float(args.protect_weight))
                protected["weight"] = min(max(float(protected.get("weight") or 1.0), float(args.protect_weight)), float(args.protect_weight))
                rows.append(protected)
                protected_task_counts[task] += 1

    if args.retrieval_manifest:
        retrieval_manifest = _load_manifest(Path(args.retrieval_manifest).resolve())
        added = 0
        for row in _iter_jsonl(Path(retrieval_manifest["train_dataset_path"])):
            if added >= int(args.retrieval_protect):
                break
            if str(row.get("retrieval_query_text") or "").strip() and str(row.get("retrieval_doc_text") or "").strip():
                protected = _anchor_row(row, label="retrieval", index=added, weight_cap=1.0)
                protected["decoder_loss_weight"] = 0.0
                protected["retrieval_loss_weight"] = 1.0
                rows.append(protected)
                added += 1

    out_eval = []
    for index, row in enumerate(eval_rows[: min(len(eval_rows), 512)]):
        copied = deepcopy(row)
        copied["example_id"] = _stable_id("stage92_eval", index, copied.get("source_id"), copied.get("encoder_text"))
        copied["split"] = "eval"
        out_eval.append(copied)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, rows)
    _write_jsonl(eval_path, out_eval)
    manifest = {
        **broad_manifest,
        "artifact_kind": "agentkernel_lite_encdec_stage92_direct_failure_replay",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(out_eval),
        "failure_repair_examples": repair_count,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_stage92_direct_failure_replay",
        "protected_task_examples": dict(sorted(protected_task_counts.items())),
        "source_failure_json": [str(Path(path).resolve()) for path in args.failure_json],
        "total_examples": len(rows) + len(out_eval),
        "train_dataset_path": str(train_path),
        "train_examples": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
