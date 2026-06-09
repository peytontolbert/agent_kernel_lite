#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

from pocketpal_structured_decode import json_to_structured_tokens


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


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _content_spans(decoder_text: str, *, encoder_text: str = "", normalize_json_structured: bool = False) -> tuple[str, str, str] | None:
    text = str(decoder_text or "")
    if bool(normalize_json_structured) and "<AK_CONTENT>" not in text:
        text = json_to_structured_tokens(text, source_text=str(encoder_text or ""), use_copy_source=True)
    match = re.search(r"^(.*?<AK_CONTENT>\s*)(.*?)(\s*</AK_CONTENT>.*)$", text, flags=re.S)
    if not match:
        return None
    prefix, content, suffix = match.group(1), match.group(2), match.group(3)
    if not content.strip():
        return None
    return prefix, content, suffix


def _token_prefix(content: str, words: int) -> tuple[str, str] | None:
    pieces = re.findall(r"\S+\s*", str(content or ""), flags=re.S)
    if len(pieces) <= words:
        return None
    prefix = "".join(pieces[:words]).rstrip()
    suffix = "".join(pieces[words:]).lstrip()
    if not prefix or not suffix:
        return None
    return prefix, suffix


def _continuation_row(row: dict[str, Any], *, train_prefix: str, suffix: str, label: str, index: int, weight: float) -> dict[str, Any]:
    out = deepcopy(row)
    out["decoder_train_prefix"] = train_prefix.strip()
    out["decoder_text"] = suffix.strip()
    out["negative_decoder_text"] = ""
    out["negative_loss_weight"] = 0.0
    out["decoder_loss_weight"] = 1.0
    out["weight"] = float(weight)
    out["source_type"] = f"decoder_prefix_continuation_{label}"
    out["split"] = "train"
    out["example_id"] = _stable_id("prefix_continuation", label, index, row.get("source_id"), train_prefix, suffix)
    out["source_id"] = f"{row.get('source_id', 'row')}_prefix_continuation_{label}_{index:05d}"
    return out


def _anchor_row(row: dict[str, Any], *, index: int, weight_cap: float) -> dict[str, Any]:
    out = deepcopy(row)
    out.pop("decoder_train_prefix", None)
    out["example_id"] = _stable_id("prefix_anchor", index, out.get("source_id"), out.get("encoder_text"), out.get("decoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_prefix_anchor_{index:05d}"
    out["source_type"] = "decoder_prefix_anchor"
    out["split"] = "train"
    out["weight"] = min(max(float(out.get("weight") or 1.0), 1.0), float(weight_cap))
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    base_manifest_path = Path(args.base_manifest).expanduser().resolve()
    base_manifest = _load_manifest(base_manifest_path)
    train_rows = _read_jsonl(Path(base_manifest["train_dataset_path"]))
    eval_rows = _read_jsonl(Path(base_manifest["eval_dataset_path"]))
    task_filter = {item.strip() for item in str(args.task_filter or "").split(",") if item.strip()}

    candidates = [
        row for row in train_rows
        if (not task_filter or str(row.get("task_type") or "") in task_filter)
        and _content_spans(
            str(row.get("decoder_text") or ""),
            encoder_text=str(row.get("encoder_text") or ""),
            normalize_json_structured=bool(args.normalize_json_structured),
        ) is not None
    ]
    rng.shuffle(candidates)
    if int(args.balanced_per_task) > 0:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            grouped.setdefault(str(row.get("task_type") or "unknown"), []).append(row)
        balanced: list[dict[str, Any]] = []
        for task_rows in grouped.values():
            balanced.extend(task_rows[: int(args.balanced_per_task)])
        rng.shuffle(balanced)
        candidates = balanced[: max(0, int(args.continuation_rows))]
    else:
        candidates = candidates[: max(0, int(args.continuation_rows))]

    rows: list[dict[str, Any]] = []
    index = 0
    for row in candidates:
        spans = _content_spans(
            str(row.get("decoder_text") or ""),
            encoder_text=str(row.get("encoder_text") or ""),
            normalize_json_structured=bool(args.normalize_json_structured),
        )
        if spans is None:
            continue
        structure_prefix, content, close_suffix = spans
        rows.append(
            _continuation_row(
                row,
                train_prefix=structure_prefix,
                suffix=f"{content}{close_suffix}",
                label="structure_to_content",
                index=index,
                weight=float(args.structure_weight),
            )
        )
        index += 1
        for word_count in [int(item) for item in str(args.content_prefix_words).split(",") if item.strip()]:
            split = _token_prefix(content, word_count)
            if split is None:
                continue
            content_prefix, content_suffix = split
            rows.append(
                _continuation_row(
                    row,
                    train_prefix=f"{structure_prefix}{content_prefix}",
                    suffix=f"{content_suffix}{close_suffix}",
                    label=f"content_{word_count}",
                    index=index,
                    weight=float(args.content_weight),
                )
            )
            index += 1

    anchors = list(train_rows)
    rng.shuffle(anchors)
    rows.extend(_anchor_row(row, index=index + offset, weight_cap=float(args.anchor_weight_cap)) for offset, row in enumerate(anchors[: int(args.anchor_rows)]))
    rng.shuffle(rows)

    out_eval = []
    for eval_index, row in enumerate(eval_rows[: min(len(eval_rows), int(args.eval_rows))]):
        copied = deepcopy(row)
        copied.pop("decoder_train_prefix", None)
        copied["split"] = "eval"
        copied["example_id"] = _stable_id("prefix_eval", eval_index, copied.get("source_id"), copied.get("encoder_text"))
        out_eval.append(copied)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, rows)
    _write_jsonl(eval_path, out_eval)

    source_counts = Counter(str(row.get("source_type") or "unknown") for row in rows)
    task_counts = Counter(str(row.get("task_type") or "unknown") for row in rows)
    manifest = {
        **base_manifest,
        "artifact_kind": "agentkernel_lite_encdec_decoder_prefix_continuation_curriculum",
        "objective": "pocketpal_decoder_prefix_continuation",
        "source_manifest_path": str(base_manifest_path),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(rows),
        "eval_examples": len(out_eval),
        "total_examples": len(rows) + len(out_eval),
        "continuation_examples": index,
        "source_counts": dict(sorted(source_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
        "prefix_policy": {
            "task_filter": sorted(task_filter),
            "continuation_rows": int(args.continuation_rows),
            "balanced_per_task": int(args.balanced_per_task),
            "normalize_json_structured": bool(args.normalize_json_structured),
            "content_prefix_words": str(args.content_prefix_words),
            "structure_weight": float(args.structure_weight),
            "content_weight": float(args.content_weight),
            "anchor_rows": int(args.anchor_rows),
            "anchor_weight_cap": float(args.anchor_weight_cap),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--continuation-rows", type=int, default=12000)
    parser.add_argument("--balanced-per-task", type=int, default=0)
    parser.add_argument("--normalize-json-structured", type=int, default=0)
    parser.add_argument("--content-prefix-words", default="2,4")
    parser.add_argument("--structure-weight", type=float, default=8.0)
    parser.add_argument("--content-weight", type=float, default=5.0)
    parser.add_argument("--anchor-rows", type=int, default=24000)
    parser.add_argument("--anchor-weight-cap", type=float, default=10.0)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=104)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
