#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact(value: object, *, limit: int = 8000) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _decision_content(payload: dict[str, Any], raw: str) -> str:
    for key in ("content", "answer", "command", "code"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return raw.strip()


def _as_line_protocol(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "Action: respond\nContent:"
    if raw.lower().startswith("action:") or "\nAction:" in raw:
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return f"Action: respond\nContent: {raw}"
    if not isinstance(payload, dict):
        return f"Action: respond\nContent: {raw}"
    thought = _compact(payload.get("thought", payload.get("reasoning", "")), limit=700)
    action = _compact(payload.get("action", payload.get("type", "respond")), limit=80) or "respond"
    content = _decision_content(payload, raw)
    parts = []
    if thought:
        parts.append(f"Thought: {thought}")
    parts.append(f"Action: {action}")
    parts.append(f"Content: {content}")
    return "\n".join(parts)


def _as_plain_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict):
        return raw
    return _decision_content(payload, raw)


def _convert_decoder(text: str, decoder_format: str) -> str:
    if decoder_format == "json":
        return str(text or "")
    if decoder_format == "line":
        return _as_line_protocol(text)
    if decoder_format == "plain":
        return _as_plain_text(text)
    raise ValueError(f"unknown decoder format: {decoder_format}")


def _convert_jsonl(source: Path, target: Path, *, decoder_format: str) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as in_handle, target.open("w", encoding="utf-8") as out_handle:
        for line in in_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            clean = dict(row)
            clean["decoder_text"] = _convert_decoder(str(clean.get("decoder_text", "")), decoder_format)
            clean["decoder_format"] = decoder_format
            out_handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    source_manifest = _load_json(source_manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    decoder_format = str(args.decoder_format)
    train_examples = _convert_jsonl(
        Path(str(source_manifest["train_dataset_path"])),
        train_path,
        decoder_format=decoder_format,
    )
    eval_examples = _convert_jsonl(
        Path(str(source_manifest["eval_dataset_path"])),
        eval_path,
        decoder_format=decoder_format,
    )
    manifest = dict(source_manifest)
    manifest.update(
        {
            "manifest_path": str(manifest_path),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
            "source_dataset_manifest_path": str(source_manifest_path),
            "decoder_format": decoder_format,
            "train_examples": train_examples,
            "eval_examples": eval_examples,
            "total_examples": train_examples + eval_examples,
            "schema": {
                **dict(source_manifest.get("schema", {}) or {}),
                "decoder_text": f"{decoder_format} target for AgentKernel Lite decoder",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decoder-format", choices=("json", "line", "plain"), default="line")
    args = parser.parse_args()
    print(json.dumps(convert_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
