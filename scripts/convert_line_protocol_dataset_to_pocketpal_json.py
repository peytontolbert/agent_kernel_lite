#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact(value: object, limit: int = 1400) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())[:limit].strip()


def _parse_action_content(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    match = re.search(r"(?is)\bAction:\s*([a-z_]+)\s*Content:\s*(.*)\Z", raw)
    if match:
        return match.group(1).strip() or "respond", _compact(match.group(2))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "respond", _compact(raw)
    if not isinstance(payload, dict):
        return "respond", _compact(raw)
    action = str(payload.get("action") or "respond").strip() or "respond"
    content = str(payload.get("content") or payload.get("answer") or raw).strip()
    return action, _compact(content)


def _payload(action: str, content: str, task_type: str) -> str:
    action = action if action in {"respond", "ask_user", "extension_request", "save_memory", "gather_context"} else "respond"
    metadata: dict[str, Any] = {"task_type": task_type}
    return json.dumps({"action": action, "content": content, "proposal_metadata": metadata}, ensure_ascii=False, sort_keys=True)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _convert_split(source: Path, target: Path, *, split: str) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as out:
        for index, row in enumerate(_iter_jsonl(source)):
            action, content = _parse_action_content(str(row.get("decoder_text") or ""))
            if not content:
                continue
            converted = dict(row)
            task_type = str(row.get("task_type") or "line_protocol_teacher")
            converted["example_id"] = str(row.get("example_id") or row.get("source_id") or f"{split}_{index:08d}") + "_json"
            converted["source_type"] = str(row.get("source_type") or "line_protocol_teacher") + "_json"
            converted["source_id"] = str(row.get("source_id") or converted["example_id"])
            converted["split"] = split
            converted["task_type"] = task_type + "_json"
            converted["action"] = action
            converted["decoder_text"] = _payload(action, content, converted["task_type"])
            converted["negative_decoder_text"] = str(row.get("decoder_text") or "")
            converted["negative_loss_weight"] = 1.0
            converted["weight"] = float(row.get("weight") or 1.0)
            out.write(json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="line_protocol_teacher_json")
    args = parser.parse_args()

    input_manifest_path = Path(args.input_manifest).expanduser().resolve()
    input_manifest = _load_json(input_manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    train_count = _convert_split(Path(str(input_manifest["train_dataset_path"])), train_path, split="train")
    eval_count = _convert_split(Path(str(input_manifest["eval_dataset_path"])), eval_path, split="eval")
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": str(args.objective),
        "manifest_path": str(manifest_path),
        "source_manifest_path": str(input_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "total_examples": int(train_count + eval_count),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
