#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _decision_content(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    decision = parsed.get("decision_packet", {}).get("decision")
    if not isinstance(decision, dict):
        decision = parsed.get("decision") if isinstance(parsed.get("decision"), dict) else parsed
    if not isinstance(decision, dict):
        return raw
    return str(decision.get("content", "") or "").strip() or raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _convert_rows(rows: list[dict[str, Any]], zero_bucket_repeats: int) -> list[dict[str, Any]]:
    zero_tasks = {
        "active_agent_json",
        "active_agent_plan",
        "active_agent_translation",
        "active_agent_brainstorm",
    }
    converted: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task_type", "") or "")
        if not task.startswith("active_agent_"):
            continue
        content = str(row.get("expected_content") or "").strip() or _decision_content(str(row.get("decoder_text", "")))
        if not content:
            continue
        repeats = max(1, int(zero_bucket_repeats)) if task in zero_tasks else 1
        for repeat in range(repeats):
            out = dict(row)
            out["decoder_text"] = content
            out["expected_content"] = content
            out["decoder_prefix"] = ""
            out["source_id"] = f"{row.get('source_id', 'row')}_content_r{repeat:02d}"
            out["dataset_stage"] = "stage62_direct_content"
            converted.append(out)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json",
    )
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage62_direct_content_curriculum")
    parser.add_argument("--zero-bucket-repeats", type=int, default=4)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    train_rows = _convert_rows(_read_jsonl(Path(source_manifest["train_dataset_path"])), int(args.zero_bucket_repeats))
    eval_rows = _convert_rows(_read_jsonl(Path(source_manifest["eval_dataset_path"])), 1)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        **source_manifest,
        "dataset_objective": "pocketpal_stage62_direct_content_curriculum",
        "source_manifest": str(source_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "stage62_zero_bucket_repeats": int(args.zero_bucket_repeats),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
    }
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "train": len(train_rows), "eval": len(eval_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
