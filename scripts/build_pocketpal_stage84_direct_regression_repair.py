#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = _read_jsonl(Path(manifest["train_dataset_path"]))
    rows.extend(_read_jsonl(Path(manifest["eval_dataset_path"])))
    return rows


def _split_row(row: dict[str, Any], eval_fraction: float) -> str:
    if str(row.get("split") or "") == "eval":
        return "eval"
    key = str(row.get("example_id") or _stable_id(row.get("encoder_text", ""), row.get("decoder_text", "")))
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < float(eval_fraction) else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage67_structured_copy_decoder_v172d_akv1/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--intent-anchor-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage80_intent_boundary_overfit_probe/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_stage84_direct_regression_repair"))
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--anchor-repeat-limit", type=int, default=7800)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    args = parser.parse_args()

    misses = {
        "stage25_extraction_00_003_repeat_001",
        "stage25_brainstorm_00_001_repeat_006",
        "stage25_plan_00_001_repeat_010",
        "stage25_risks_01_001_repeat_013",
    }
    sibling_prefixes = {
        "stage25_extraction_00_003",
        "stage25_brainstorm_00_001",
        "stage25_plan_00_001",
        "stage25_risks_01_001",
    }
    stage67_rows = _load_manifest_rows(Path(args.stage67_manifest).resolve())
    selected: list[dict[str, Any]] = []
    for row in stage67_rows:
        source_id = str(row.get("source_id") or "")
        task_type = str(row.get("task_type") or "")
        if source_id in misses or any(source_id.startswith(prefix) for prefix in sibling_prefixes):
            selected.append(row)
            continue
        if task_type in {"active_agent_extraction", "active_agent_brainstorm", "active_agent_plan", "active_agent_risks"}:
            # Add a small amount of neighboring task distribution so the repair is not
            # just four memorized rows.
            try:
                suffix = int(source_id.rsplit("_", 1)[-1])
            except ValueError:
                suffix = -1
            if 0 <= suffix < 3:
                selected.append(row)

    anchor_rows = _load_manifest_rows(Path(args.intent_anchor_manifest).resolve())[: int(args.anchor_repeat_limit)]
    out_rows: list[dict[str, Any]] = []
    for rep in range(max(1, int(args.repeat))):
        for row in selected:
            copied = deepcopy(row)
            copied["example_id"] = _stable_id("stage84", rep, copied.get("source_id"), copied.get("encoder_text"), copied.get("decoder_text"))
            copied["source_type"] = "stage84_direct_regression_repair"
            copied["weight"] = float(copied.get("weight", 1.0) or 1.0) * (2.0 if str(copied.get("source_id")) in misses else 1.3)
            copied["split"] = "train"
            out_rows.append(copied)
    for row in anchor_rows:
        copied = deepcopy(row)
        copied["source_type"] = str(copied.get("source_type") or "stage84_intent_anchor_replay")
        copied["weight"] = float(copied.get("weight", 1.0) or 1.0) * 1.1
        out_rows.append(copied)

    dedup: dict[str, dict[str, Any]] = {}
    for row in out_rows:
        dedup[str(row["example_id"])] = row
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in sorted(dedup.values(), key=lambda item: str(item.get("example_id", ""))):
        split = _split_row(row, float(args.eval_fraction))
        row["split"] = split
        if split == "eval":
            eval_rows.append(row)
        else:
            train_rows.append(row)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    intent_labels = json.loads(Path(args.stage67_manifest).resolve().read_text(encoding="utf-8")).get("intent_labels", {})
    manifest: dict[str, Any] = {
        "artifact_kind": "agentkernel_lite_encdec_stage84_direct_regression_repair",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": intent_labels,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_stage84_direct_regression_repair",
        "selected_regression_rows": len(selected),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
