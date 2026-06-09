#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _maybe_read_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    for name in ("agentkernel_lite_encdec_manifest.json", "manifest.json"):
        path = bundle_dir / name
        if path.exists():
            return _read_json(path)
    return {}


def _training_summary(bundle_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("training_summary")
    if isinstance(summary, dict):
        return summary
    path = bundle_dir / "training_summary.json"
    return _maybe_read_json(path)


def _capability_vector(eval_summary: dict[str, Any]) -> dict[str, Any]:
    by_task = eval_summary.get("by_task", {})
    vector: dict[str, Any] = {
        "passed": int(eval_summary.get("passed", 0) or 0),
        "examples": int(eval_summary.get("examples", 0) or 0),
        "pass_rate": float(eval_summary.get("pass_rate", 0.0) or 0.0),
        "mean_recall": float(eval_summary.get("mean_recall", 0.0) or 0.0),
        "zero_pass_tasks": [],
        "tasks": {},
    }
    if isinstance(by_task, dict):
        for name, values in sorted(by_task.items()):
            if not isinstance(values, dict):
                continue
            passed = int(values.get("passed", 0) or 0)
            total = int(values.get("total", 0) or 0)
            task = {
                "passed": passed,
                "total": total,
                "pass_rate": float(values.get("pass_rate", 0.0) or 0.0),
                "mean_recall": float(values.get("mean_recall", 0.0) or 0.0),
            }
            vector["tasks"][name] = task
            if total > 0 and passed == 0:
                vector["zero_pass_tasks"].append(name)
    return vector


def _failure_tags(eval_summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in eval_summary.get("failures", []) or []:
        if not isinstance(failure, dict):
            continue
        for tag in failure.get("failures", []) or []:
            key = str(tag).split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    eval_path = Path(args.eval_json).expanduser().resolve()
    eval_summary = _read_json(eval_path)
    bundle_dir = Path(args.bundle_dir or eval_summary.get("bundle_dir", "")).expanduser().resolve()
    manifest = _bundle_manifest(bundle_dir)
    training = _training_summary(bundle_dir, manifest)
    baseline = _maybe_read_json(Path(args.baseline_eval_json).expanduser().resolve()) if args.baseline_eval_json else {}
    capability = _capability_vector(eval_summary)
    baseline_capability = _capability_vector(baseline) if baseline else {}
    return {
        "run_id": str(args.run_id or bundle_dir.name),
        "phase": str(args.phase),
        "bundle_dir": str(bundle_dir),
        "eval_json": str(eval_path),
        "dataset_manifest": str(eval_summary.get("dataset_manifest", "")),
        "training_summary": training,
        "capability_vector": capability,
        "baseline": {
            "eval_json": str(Path(args.baseline_eval_json).expanduser().resolve()) if args.baseline_eval_json else "",
            "passed": baseline_capability.get("passed"),
            "delta_passed": (
                int(capability["passed"]) - int(baseline_capability.get("passed", 0))
                if baseline_capability
                else None
            ),
            "zero_pass_tasks": baseline_capability.get("zero_pass_tasks", []),
        },
        "failure_tags": _failure_tags(eval_summary),
        "decision": {
            "current_best_passed": int(args.current_best_passed),
            "promote": int(capability["passed"]) > int(args.current_best_passed),
            "keep_for_analysis": bool(args.keep_for_analysis),
            "notes": str(args.notes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument("--baseline-eval-json", default="")
    parser.add_argument("--ledger", default="artifacts/pocketpal_training_state_ledger.jsonl")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--phase", default="active_agent_promotion")
    parser.add_argument("--current-best-passed", type=int, default=65)
    parser.add_argument("--keep-for-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    record = build_record(args)
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
