#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    metadata_path = path / "metadata.jsonl"
    if not metadata_path.exists():
        return []
    rows = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def copy_relative_file(source_root: Path, output_root: Path, relative_path: str) -> str:
    src = source_root / relative_path
    if not src.exists():
        raise FileNotFoundError(src)
    dst = output_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    return relative_path


def merge_targets(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(exist_ok=True)
    (output_dir / "targets").mkdir(exist_ok=True)

    existing_rows = read_rows(output_dir)
    seen = {row["target_id"] for row in existing_rows}
    merged_rows = list(existing_rows)
    added = 0
    skipped_missing = 0

    for source_text in args.sources:
        source_dir = Path(source_text)
        for row in read_rows(source_dir):
            target_id = row["target_id"]
            if target_id in seen:
                continue
            try:
                row = dict(row)
                row["target_path"] = copy_relative_file(source_dir, output_dir, row["target_path"])
                row["embedding_path"] = copy_relative_file(source_dir, output_dir, row["embedding_path"])
            except FileNotFoundError:
                skipped_missing += 1
                continue
            seen.add(target_id)
            merged_rows.append(row)
            added += 1

    tmp_metadata = output_dir / "metadata.jsonl.tmp"
    with tmp_metadata.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_metadata.replace(output_dir / "metadata.jsonl")

    manifest = {
        "artifact_kind": "agentkernel_lite_flux_flow_targets_merged",
        "rows": len(merged_rows),
        "sources": args.sources,
        "metadata": "metadata.jsonl",
        "embeddings": "embeddings",
        "targets": "targets",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(merged_rows), "added": added, "skipped_missing": skipped_missing}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FLUX flow target shards into a single trainable target directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()
    merge_targets(args)


if __name__ == "__main__":
    main()
