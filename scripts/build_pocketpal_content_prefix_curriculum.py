#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from pocketpal_structured_decode import CONTENT, CONTENT_END, COPY_USER_SOURCE_1, END


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


def _extract_source_text(text: str) -> str:
    match = re.search(r"<AK_SLOT_NAME>=SOURCE_TEXT\s+<AK_SLOT_VALUE>=(.*?)(?:\n|$)", str(text or ""), flags=re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"<AK_USER>\s*(.*?)(?:\n(?:Return|<AK_|$)|$)", str(text or ""), flags=re.S)
    return match.group(1).strip() if match else ""


def _json_content(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return raw
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return raw
    if not isinstance(parsed, dict):
        return raw
    content = parsed.get("content", "")
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(content or "").strip()


def _read_content_tokens(text: str) -> str:
    raw = str(text or "").strip()
    match = re.search(rf"{re.escape(CONTENT)}\s*(.*?)\s*{re.escape(CONTENT_END)}", raw, flags=re.S)
    if match:
        return match.group(1).strip()
    if CONTENT in raw:
        tail = raw.split(CONTENT, 1)[1].strip()
        tail = tail.split(END, 1)[0].strip()
        return tail
    return ""


def _content_from_row(row: dict[str, Any], *, use_copy_source: bool) -> str:
    expected = str(row.get("expected_content") or "").strip()
    if expected:
        content = expected
    else:
        raw = str(row.get("decoder_text") or "")
        content = _read_content_tokens(raw) or _json_content(str(row.get("json_decoder_text") or raw))
    source_text = _extract_source_text(str(row.get("encoder_text") or ""))
    if bool(use_copy_source) and source_text and re.sub(r"\s+", " ", content).strip() == re.sub(r"\s+", " ", source_text).strip():
        return COPY_USER_SOURCE_1
    return content.strip()


def _anchor_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    out = deepcopy(row)
    out["split"] = split
    out.pop("decoder_train_prefix", None)
    return out


def _convert_row(row: dict[str, Any], *, split: str, use_copy_source: bool) -> dict[str, Any]:
    out = deepcopy(row)
    content = _content_from_row(out, use_copy_source=bool(use_copy_source))
    out["split"] = split
    out["decoder_train_prefix"] = CONTENT
    out["decoder_text"] = f"{content} {CONTENT_END} {END}".strip() if content else f"{CONTENT_END} {END}"
    out["content_prefix_decoder_text"] = out["decoder_text"]
    out["expected_content"] = content if content != COPY_USER_SOURCE_1 else str(out.get("expected_content") or "")
    out["source_type"] = f"{out.get('source_type', 'row')}_content_prefix"
    return out


def _convert_or_anchor(
    row: dict[str, Any],
    *,
    split: str,
    use_copy_source: bool,
    content_prefix_tasks: set[str],
) -> dict[str, Any]:
    if not content_prefix_tasks or str(row.get("task_type") or "") in content_prefix_tasks:
        return _convert_row(row, split=split, use_copy_source=bool(use_copy_source))
    return _anchor_row(row, split=split)


def build(source_manifest: Path, output_dir: Path, *, use_copy_source: bool, content_prefix_tasks: set[str]) -> dict[str, Any]:
    source = _load_manifest(source_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    train_count = _write_jsonl(
        train_path,
        (
            _convert_or_anchor(
                row,
                split="train",
                use_copy_source=bool(use_copy_source),
                content_prefix_tasks=content_prefix_tasks,
            )
            for row in _iter_jsonl(Path(source["train_dataset_path"]))
        ),
    )
    eval_count = _write_jsonl(
        eval_path,
        (
            _convert_or_anchor(
                row,
                split="eval",
                use_copy_source=bool(use_copy_source),
                content_prefix_tasks=content_prefix_tasks,
            )
            for row in _iter_jsonl(Path(source["eval_dataset_path"]))
        ),
    )
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        **source,
        "artifact_kind": "agentkernel_lite_content_prefix_curriculum",
        "decoder_target_mode": "content_prefix_masked",
        "decoder_train_prefix": CONTENT,
        "content_prefix_tasks": sorted(content_prefix_tasks),
        "decoder_copy_source_enabled": bool(use_copy_source),
        "source_manifest_path": str(source_manifest.resolve()),
        "train_dataset_path": str(train_path.resolve()),
        "eval_dataset_path": str(eval_path.resolve()),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "manifest_path": str(manifest_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-copy-source", type=int, choices=(0, 1), default=1)
    parser.add_argument("--content-prefix-task", action="append", default=[])
    args = parser.parse_args()
    tasks = {str(task).strip() for task in args.content_prefix_task if str(task).strip()}
    manifest = build(
        Path(args.source_manifest),
        Path(args.output_dir),
        use_copy_source=bool(args.use_copy_source),
        content_prefix_tasks=tasks,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
