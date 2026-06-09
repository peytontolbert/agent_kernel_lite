#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
from typing import Any


WEAK_TASKS = {
    "active_agent_brainstorm",
    "active_agent_extraction",
    "active_agent_json",
    "active_agent_plan",
    "active_agent_translation",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _content_from_decision(text: str) -> str:
    try:
        parsed = json.loads(text)
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    decision = parsed.get("decision_packet", {}).get("decision")
    if isinstance(decision, dict):
        return str(decision.get("content", "") or "")
    return str(parsed.get("content", "") or "")


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    source_manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    shipped_eval = _load_module(repo_root / "scripts" / "evaluate_pocketpal_active_agent_shipped_path.py", "evaluate_pocketpal_active_agent_shipped_path")
    output_dir = Path(args.output_dir).expanduser().resolve()
    rng = random.Random(int(args.seed))

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for split_name, key in [("train", "train_dataset_path"), ("eval", "eval_dataset_path")]:
        for row in _iter_jsonl(Path(source_manifest[key])):
            task = str(row.get("task_type", "") or "")
            if task not in WEAK_TASKS:
                continue
            user_text = shipped_eval._user_text(row)
            fallback = shipped_eval._fallback(row, user_text)
            content = _content_from_decision(fallback)
            if not content.strip():
                continue
            out = dict(row)
            out["decoder_text"] = fallback
            out["expected_content"] = content
            out["weight"] = float(args.weight)
            out["source_id"] = f"stage53_operator_distill_{split_name}_{task}_{len(rows):06d}"
            out["curriculum_stage"] = "stage53_operator_distill"
            out["teacher_source"] = "shipped_path_operator"
            rows.append(out)
            counts[task] = counts.get(task, 0) + 1
            if len(rows) >= int(args.max_rows):
                break
        if len(rows) >= int(args.max_rows):
            break
    rng.shuffle(rows)
    eval_size = min(max(int(len(rows) * float(args.eval_ratio)), 1), max(len(rows) - 1, 1)) if len(rows) > 1 else 0
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "objective": "pocketpal_stage53_operator_distill_curriculum",
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(rows),
        "task_type_counts": counts,
        "teacher_source": "evaluate_pocketpal_active_agent_shipped_path._fallback",
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--source-manifest", default="tmp/pocketpal_v190_weak_task_focus_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage53_operator_distill_curriculum")
    parser.add_argument("--max-rows", type=int, default=24000)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
