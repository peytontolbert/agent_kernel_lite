#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_device() -> str:
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda"
    return "cpu"


def _load_sampler(repo_root: Path):
    path = repo_root / "scripts" / "sample_agentkernel_lite_encdec.py"
    spec = importlib.util.spec_from_file_location("sample_agentkernel_lite_encdec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sampler script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _select_rows(rows: list[dict[str, Any]], *, max_examples: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or len(rows) <= max_examples:
        return rows
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        task_type = str(row.get("task_type", "unknown") or "unknown")
        if task_type not in groups:
            groups[task_type] = []
            order.append(task_type)
        groups[task_type].append(row)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < max_examples:
        emitted = False
        for task_type in order:
            group = groups[task_type]
            if index >= len(group):
                continue
            selected.append(group[index])
            emitted = True
            if len(selected) >= max_examples:
                break
        if not emitted:
            break
        index += 1
    return selected


def _parse_decision(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return _repair_decision(raw)
    return _repair_decision(raw)


def _extract_json_string_field(raw: str, key: str) -> str:
    pattern = re.search(rf'"{re.escape(key)}"\s*:', raw)
    if not pattern:
        return ""
    index = pattern.end()
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw) or raw[index] != '"':
        return ""
    index += 1
    value: list[str] = []
    escaped = False
    while index < len(raw):
        char = raw[index]
        if escaped:
            value.append("\n" if char == "n" else "\t" if char == "t" else char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return "".join(value)
        else:
            value.append(char)
        index += 1
    return ""


def _repair_decision(raw: str) -> dict[str, Any] | None:
    if '"action"' not in raw or '"content"' not in raw:
        return None
    action = _extract_json_string_field(raw, "action")
    content = _extract_json_string_field(raw, "content")
    if action not in {"respond", "ask_user", "extension_request", "save_memory"} or not content:
        return None
    return {"action": action, "content": content, "proposal_metadata": {"task_type": "repaired_decision"}}


def _runtime_normalized_action(parsed: dict[str, Any]) -> str:
    action = str(parsed.get("action", "") or "")
    if action != "respond":
        return action
    metadata = parsed.get("proposal_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task_type = str(metadata.get("task_type", "") or "")
    if task_type in {"memory_save", "converted_memory_save"}:
        return "save_memory"
    if task_type in {"ask_missing_slot", "converted_ask_user"}:
        return "ask_user"
    if task_type in {"extension_request", "converted_function_call"}:
        return "extension_request"
    return action


def _expected_task_type(row: dict[str, Any]) -> str:
    return str(row.get("task_type", "") or "")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    sampler = _load_sampler(repo_root)
    sampler._install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = sampler._load_manifest(bundle_dir)
    dataset_path = Path(str(args.dataset_path)).expanduser().resolve() if str(args.dataset_path).strip() else Path(
        str(manifest.get("dataset_manifest_path", ""))
    )
    if dataset_path.name.endswith("manifest.json"):
        dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset_path = Path(str(dataset_manifest["eval_dataset_path"]))
    rows = _select_rows(_iter_jsonl(dataset_path), max_examples=int(args.max_examples))

    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    total = 0
    json_valid = 0
    action_correct = 0
    runtime_action_correct = 0
    task_correct = 0
    by_task: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        total += 1
        expected_action = str(row.get("action", "") or "")
        expected_task = _expected_task_type(row)
        output = sampler._generate(
            model,
            tokenizer,
            str(row["encoder_text"]),
            decoder_prefix=str(args.decoder_prefix),
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        parsed = _parse_decision(output)
        bucket = by_task.setdefault(
            expected_task,
            {"total": 0, "json_valid": 0, "action_correct": 0, "runtime_action_correct": 0, "task_correct": 0},
        )
        bucket["total"] += 1
        if parsed is not None:
            json_valid += 1
            bucket["json_valid"] += 1
            predicted_action = str(parsed.get("action", "") or "")
            if predicted_action == expected_action:
                action_correct += 1
                bucket["action_correct"] += 1
            if _runtime_normalized_action(parsed) == expected_action:
                runtime_action_correct += 1
                bucket["runtime_action_correct"] += 1
            metadata = parsed.get("proposal_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            predicted_task = str(metadata.get("task_type", "") or "")
            if predicted_task == expected_task:
                task_correct += 1
                bucket["task_correct"] += 1
        if parsed is None or str((parsed or {}).get("action", "") or "") != expected_action:
            if len(failures) < int(args.max_failures):
                failures.append(
                    {
                        "source_id": row.get("source_id", ""),
                        "task_type": expected_task,
                        "expected_action": expected_action,
                        "output": output[:600],
                    }
                )
    def ratio(value: int) -> float:
        return float(value) / float(total or 1)

    return {
        "bundle_dir": str(bundle_dir),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "examples": int(total),
        "json_valid": int(json_valid),
        "json_valid_rate": ratio(json_valid),
        "action_correct": int(action_correct),
        "action_accuracy": ratio(action_correct),
        "runtime_action_correct": int(runtime_action_correct),
        "runtime_action_accuracy": ratio(runtime_action_correct),
        "task_type_correct": int(task_correct),
        "task_type_accuracy": ratio(task_correct),
        "by_task": by_task,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-path", default="", help="Eval JSONL path or dataset manifest path. Defaults to bundle dataset manifest eval split.")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-failures", type=int, default=8)
    parser.add_argument("--max-encoder-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--decoder-prefix", default="")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
