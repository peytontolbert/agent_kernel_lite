#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_json(command: list[str], output_path: Path, *, allow_failure: bool = False) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"command failed with {completed.returncode}: {' '.join(command)}\n{completed.stdout}")
    if not output_path.exists():
        raise RuntimeError(f"command did not write expected JSON: {' '.join(command)}\n{completed.stdout}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["_returncode"] = int(completed.returncode)
    return payload


def _task_pass_rate(summary: dict[str, Any], task: str) -> float:
    bucket = (summary.get("by_task") or {}).get(task) or {}
    return float(bucket.get("pass_rate", 0.0) or 0.0)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_manifest = Path(args.dataset_manifest).expanduser().resolve()
    python = sys.executable

    direct_default_path = output_dir / "direct_default.json"
    direct_rep_path = output_dir / "direct_rep105_len120.json"
    shipped_path = output_dir / "shipped_path.json"
    head_assisted_path = output_dir / "head_assisted_hint_then_model.json"
    cert_path = output_dir / "certification_matrix.json"

    direct_default = _run_json(
        [
            python,
            str(repo_root / "scripts" / "evaluate_pocketpal_direct_agent_prompts.py"),
            "--repo-root",
            str(repo_root),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-manifest",
            str(dataset_manifest),
            "--output-json",
            str(direct_default_path),
            "--max-failures",
            str(args.max_failures),
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--device",
            str(args.device),
        ],
        direct_default_path,
    )
    direct_rep = _run_json(
        [
            python,
            str(repo_root / "scripts" / "evaluate_pocketpal_direct_agent_prompts.py"),
            "--repo-root",
            str(repo_root),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-manifest",
            str(dataset_manifest),
            "--output-json",
            str(direct_rep_path),
            "--max-failures",
            str(args.max_failures),
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--repetition-penalty",
            "1.05",
            "--max-new-tokens",
            "120",
            "--device",
            str(args.device),
        ],
        direct_rep_path,
    )
    shipped = _run_json(
        [
            python,
            str(repo_root / "scripts" / "evaluate_pocketpal_active_agent_shipped_path.py"),
            "--repo-root",
            str(repo_root),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-manifest",
            str(dataset_manifest),
            "--output-json",
            str(shipped_path),
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--device",
            str(args.device),
        ],
        shipped_path,
    )
    head_assisted = _run_json(
        [
            python,
            str(repo_root / "scripts" / "evaluate_pocketpal_head_assisted_agent.py"),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-manifest",
            str(dataset_manifest),
            "--intent-label-manifest",
            str(dataset_manifest),
            "--route-source",
            "hint_then_model",
            "--output-json",
            str(head_assisted_path),
            "--device",
            str(args.device),
            "--max-examples",
            str(args.head_assisted_examples),
            "--max-failures",
            str(args.max_failures),
        ],
        head_assisted_path,
    )
    certification = _run_json(
        [
            python,
            str(repo_root / "scripts" / "evaluate_pocketpal_agent_certification_matrix.py"),
            "--repo-root",
            str(repo_root),
            "--bundle-dir",
            str(bundle_dir),
            "--output-json",
            str(cert_path),
            "--device",
            str(args.device),
        ],
        cert_path,
        allow_failure=True,
    )

    best_direct = direct_rep if int(direct_rep.get("passed", 0) or 0) >= int(direct_default.get("passed", 0) or 0) else direct_default
    zero_pass_tasks = sorted(
        task
        for task, bucket in (best_direct.get("by_task") or {}).items()
        if int((bucket or {}).get("total", 0) or 0) > 0 and int((bucket or {}).get("passed", 0) or 0) == 0
    )
    weak_tasks = {
        task: _task_pass_rate(best_direct, task)
        for task in [
            "active_agent_rewrite",
            "active_agent_summary",
            "active_agent_extraction",
            "active_agent_json",
            "active_agent_plan",
            "active_agent_translation",
            "active_agent_action_items",
            "active_agent_checklist",
            "active_agent_risks",
            "active_agent_subject",
            "active_agent_brainstorm",
        ]
    }
    fallback_rate = float(shipped.get("fallback_count", 0) or 0) / float(shipped.get("examples", 1) or 1)
    checks = {
        "direct_best_pass_rate": float(best_direct.get("pass_rate", 0.0) or 0.0) >= float(args.min_direct_pass_rate),
        "direct_best_passed": int(best_direct.get("passed", 0) or 0) >= int(args.min_direct_passed),
        "direct_best_not_below_baseline": int(best_direct.get("passed", 0) or 0) >= int(args.baseline_direct_passed),
        "direct_zero_pass_tasks": len(zero_pass_tasks) <= int(args.max_zero_pass_tasks),
        "shipped_pass_rate": float(shipped.get("shipped_pass_rate", 0.0) or 0.0) >= float(args.min_shipped_pass_rate),
        "head_assisted_hint_route": float(head_assisted.get("pass_rate", 0.0) or 0.0)
        >= float(args.min_head_assisted_pass_rate),
        "fallback_rate": fallback_rate <= float(args.max_fallback_rate),
        "certification": bool(certification.get("ok", False)),
    }
    promoted = all(checks.values())
    ledger = {
        "bundle_dir": str(bundle_dir),
        "dataset_manifest": str(dataset_manifest),
        "promoted": promoted,
        "checks": checks,
        "thresholds": {
            "baseline_direct_passed": int(args.baseline_direct_passed),
            "min_direct_passed": int(args.min_direct_passed),
            "min_direct_pass_rate": float(args.min_direct_pass_rate),
            "max_zero_pass_tasks": int(args.max_zero_pass_tasks),
            "min_shipped_pass_rate": float(args.min_shipped_pass_rate),
            "min_head_assisted_pass_rate": float(args.min_head_assisted_pass_rate),
            "max_fallback_rate": float(args.max_fallback_rate),
        },
        "direct_default": {
            "passed": direct_default.get("passed"),
            "pass_rate": direct_default.get("pass_rate"),
            "mean_recall": direct_default.get("mean_recall"),
        },
        "direct_rep105_len120": {
            "passed": direct_rep.get("passed"),
            "pass_rate": direct_rep.get("pass_rate"),
            "mean_recall": direct_rep.get("mean_recall"),
        },
        "best_direct": {
            "passed": best_direct.get("passed"),
            "pass_rate": best_direct.get("pass_rate"),
            "mean_recall": best_direct.get("mean_recall"),
            "zero_pass_tasks": zero_pass_tasks,
            "weak_tasks": weak_tasks,
        },
        "shipped_path": {
            "raw_passed": shipped.get("raw_passed"),
            "raw_pass_rate": shipped.get("raw_pass_rate"),
            "fallback_count": shipped.get("fallback_count"),
            "fallback_rate": fallback_rate,
            "shipped_passed": shipped.get("shipped_passed"),
            "shipped_pass_rate": shipped.get("shipped_pass_rate"),
        },
        "head_assisted_hint_route": {
            "passed": head_assisted.get("passed"),
            "examples": head_assisted.get("examples"),
            "pass_rate": head_assisted.get("pass_rate"),
            "mean_recall": head_assisted.get("mean_recall"),
        },
        "certification": {
            "passed": certification.get("passed"),
            "total": certification.get("total"),
            "pass_rate": certification.get("pass_rate"),
            "ok": certification.get("ok"),
            "returncode": certification.get("_returncode"),
        },
        "artifacts": {
            "direct_default": str(direct_default_path),
            "direct_rep105_len120": str(direct_rep_path),
            "shipped_path": str(shipped_path),
            "head_assisted_hint_then_model": str(head_assisted_path),
            "certification_matrix": str(cert_path),
        },
    }
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-failures", type=int, default=80)
    parser.add_argument("--head-assisted-examples", type=int, default=300)
    parser.add_argument("--baseline-direct-passed", type=int, default=65)
    parser.add_argument("--min-direct-passed", type=int, default=0)
    parser.add_argument("--min-direct-pass-rate", type=float, default=0.42)
    parser.add_argument("--max-zero-pass-tasks", type=int, default=2)
    parser.add_argument("--min-shipped-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-head-assisted-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-fallback-rate", type=float, default=0.45)
    args = parser.parse_args()

    ledger = evaluate(args)
    text = json.dumps(ledger, indent=2, sort_keys=True)
    output_json = Path(args.output_json).expanduser().resolve() if str(args.output_json).strip() else Path(args.output_dir).expanduser().resolve() / "promotion_gate.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not ledger["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
