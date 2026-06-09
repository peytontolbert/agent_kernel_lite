#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pocketpal_structured_decode import json_to_content_tokens, json_to_structured_tokens


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _extract_source_text(text: str) -> str:
    match = re.search(r"<AK_SLOT_NAME>=SOURCE_TEXT\s+<AK_SLOT_VALUE>=(.*?)(?:\n|$)", str(text or ""), flags=re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"<AK_USER>\s*(.*?)(?:\nReturn compact JSON|\n?$)", str(text or ""), flags=re.S)
    return match.group(1).strip() if match else ""


def _convert_target(text: str, *, source_text: str, use_copy_source: bool, target_mode: str) -> str:
    if target_mode == "content":
        return json_to_content_tokens(text, source_text=source_text, use_copy_source=bool(use_copy_source))
    return json_to_structured_tokens(text, source_text=source_text, use_copy_source=bool(use_copy_source))


def _convert_row(row: dict[str, Any], *, use_copy_source: bool, target_mode: str) -> dict[str, Any]:
    converted = dict(row)
    original = str(converted.get("decoder_text", "") or "")
    source_text = _extract_source_text(str(converted.get("encoder_text", "") or ""))
    converted["json_decoder_text"] = original
    converted["decoder_text"] = _convert_target(
        original,
        source_text=source_text,
        use_copy_source=bool(use_copy_source),
        target_mode=target_mode,
    )
    negative = str(converted.get("negative_decoder_text", "") or "")
    if negative:
        converted["json_negative_decoder_text"] = negative
        converted["negative_decoder_text"] = _convert_target(
            negative,
            source_text=source_text,
            use_copy_source=bool(use_copy_source),
            target_mode=target_mode,
        )
    return converted


def build_dataset(source_manifest: Path, output_dir: Path, *, use_copy_source: bool, target_mode: str) -> dict[str, Any]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    train_count = _write_jsonl(
        train_path,
        (
            _convert_row(row, use_copy_source=bool(use_copy_source), target_mode=target_mode)
            for row in _iter_jsonl(Path(source["train_dataset_path"]))
        ),
    )
    eval_count = _write_jsonl(
        eval_path,
        (
            _convert_row(row, use_copy_source=bool(use_copy_source), target_mode=target_mode)
            for row in _iter_jsonl(Path(source["eval_dataset_path"]))
        ),
    )
    manifest = {
        **source,
        "artifact_kind": "agentkernel_lite_encdec_structured_decoder_dataset",
        "decoder_target_format": (
            "agentkernel_content_tokens_v1" if target_mode == "content" else "agentkernel_structured_tokens_v1"
        ),
        "decoder_copy_source_enabled": bool(use_copy_source),
        "decoder_target_mode": target_mode,
        "source_manifest_path": str(source_manifest.resolve()),
        "train_dataset_path": str(train_path.resolve()),
        "eval_dataset_path": str(eval_path.resolve()),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "manifest_path": str((output_dir / "agentkernel_lite_encdec_dataset_manifest.json").resolve()),
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-copy-source", type=int, choices=(0, 1), default=0)
    parser.add_argument("--target-mode", choices=("structured", "content"), default="structured")
    args = parser.parse_args()
    manifest = build_dataset(
        Path(args.source_manifest),
        Path(args.output_dir),
        use_copy_source=bool(args.use_copy_source),
        target_mode=str(args.target_mode),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
