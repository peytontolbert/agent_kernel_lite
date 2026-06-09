#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_pocketpal_stage21_active_agent_eval_curriculum import build_rows as build_stage21_rows
from build_pocketpal_stage23_raw_json_agent_dataset import build_rows as build_stage23_rows


def _is_eval_row(row: dict[str, Any]) -> bool:
    digest = hashlib.sha256(str(row["source_id"]).encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage24_combined_agent_protocol_v1")
    parser.add_argument("--stage21-repeats", type=int, default=80)
    parser.add_argument("--stage23-repeats", type=int, default=120)
    args = parser.parse_args()

    stage21_rows = [
        row
        for row in build_stage21_rows(args.stage21_repeats)
        if str(row.get("task_type", "")) != "active_agent_json"
    ]
    rows = [*stage21_rows, *build_stage23_rows(args.stage23_repeats)]
    train_rows = [row for row in rows if not _is_eval_row(row)]
    eval_rows = [row for row in rows if _is_eval_row(row)]
    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "pocketpal_stage24_combined_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_stage24_combined_agent_eval.jsonl"
    manifest_path = output_dir / "pocketpal_stage24_combined_agent_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage24_combined_agent_protocol",
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
