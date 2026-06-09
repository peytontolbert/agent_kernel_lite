#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONTENT_ONLY_INSTRUCTION = (
    "Return only the final content text for the selected action. "
    "Do not emit JSON, action names, or proposal metadata."
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _content_prompt(encoder_text: str, action: str) -> str:
    text = str(encoder_text or "").strip()
    lines = text.splitlines()
    if lines and "Return compact JSON with the correct action and content" in lines[-1]:
        lines = lines[:-1]
    lines.extend(
        [
            f"<AK_SELECTED_ACTION> {action}",
            CONTENT_ONLY_INSTRUCTION,
        ]
    )
    return "\n".join(lines).strip()


def _convert_row(
    row: dict[str, Any],
    *,
    split: str,
    index: int,
    negative_json_attractor: bool,
) -> dict[str, Any] | None:
    payload = _parse_payload(str(row.get("decoder_text") or ""))
    if not payload:
        return None
    action = str(payload.get("action") or row.get("action") or "respond").strip() or "respond"
    content = str(payload.get("content") or "").strip()
    if not content:
        return None
    negative_content = str(row.get("decoder_text") or "").strip() if negative_json_attractor else ""
    negative_payload = _parse_payload(str(row.get("negative_decoder_text") or ""))
    if negative_payload and not negative_json_attractor:
        negative_content = str(negative_payload.get("content") or "").strip()

    converted = dict(row)
    converted["example_id"] = str(row.get("example_id") or f"{split}_{index:08d}") + "_content"
    converted["split"] = split
    converted["source_type"] = str(row.get("source_type") or "") + "_content_decoder"
    converted["task_type"] = str(row.get("task_type") or "") + "_content_decoder"
    converted["action"] = action
    converted["encoder_text"] = _content_prompt(str(row.get("encoder_text") or ""), action)
    converted["decoder_text"] = content
    converted["negative_decoder_text"] = negative_content
    return converted


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_split(
    input_path: Path,
    output_path: Path,
    *,
    split: str,
    negative_json_attractor: bool,
) -> tuple[int, Counter[str], Counter[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    action_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(_iter_jsonl(input_path)):
            converted = _convert_row(
                row,
                split=split,
                index=index,
                negative_json_attractor=negative_json_attractor,
            )
            if converted is None:
                continue
            out.write(json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            action_counts[str(converted.get("action") or "")] += 1
            task_counts[str(converted.get("task_type") or "")] += 1
    return count, action_counts, task_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="pocketpal_content_decoder")
    parser.add_argument("--negative-json-attractor", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    input_manifest_path = Path(args.input_manifest).expanduser().resolve()
    input_manifest = _load_json(input_manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"

    train_count, train_actions, train_tasks = _write_split(
        Path(str(input_manifest["train_dataset_path"])).expanduser().resolve(),
        train_path,
        split="train",
        negative_json_attractor=bool(args.negative_json_attractor),
    )
    eval_count, eval_actions, eval_tasks = _write_split(
        Path(str(input_manifest["eval_dataset_path"])).expanduser().resolve(),
        eval_path,
        split="eval",
        negative_json_attractor=bool(args.negative_json_attractor),
    )
    action_counts = train_actions + eval_actions
    task_counts = train_tasks + eval_tasks
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": str(args.objective),
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "source_manifest_path": str(input_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "total_examples": int(train_count + eval_count),
        "target_action_counts": dict(sorted(action_counts.items())),
        "task_type_counts": dict(sorted(task_counts.items())),
        "content_only_instruction": CONTENT_ONLY_INSTRUCTION,
        "negative_json_attractor": bool(args.negative_json_attractor),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
