#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_eval_module(repo_root: Path):
    path = repo_root / "scripts" / "evaluate_pocketpal_direct_agent_prompts.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_direct_agent_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval_args(args: argparse.Namespace, bundle_dir: str, dataset_manifest: str) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(args.repo_root),
        bundle_dir=bundle_dir,
        dataset_manifest=dataset_manifest,
        device=str(args.device),
        max_encoder_tokens=int(args.max_encoder_tokens),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        decoder_prefix=str(args.decoder_prefix),
        decoder_prefix_by_task=[],
        keep_special_tokens=int(args.keep_special_tokens),
        forbid_ak_content_control=int(args.forbid_ak_content_control),
        max_examples=int(args.max_examples),
        max_failures=int(args.max_failures),
        output_json="",
    )


def _malformed_count(summary: dict[str, Any]) -> int:
    total = 0
    for failure in summary.get("failures", []):
        labels = failure.get("failures", [])
        if any(str(label).startswith("malformed") for label in labels):
            total += 1
    return total


def _compare_task_rates(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    baseline_tasks = baseline.get("by_task", {})
    candidate_tasks = candidate.get("by_task", {})
    for task in sorted(set(baseline_tasks) | set(candidate_tasks)):
        base_rate = float(baseline_tasks.get(task, {}).get("pass_rate", 0.0) or 0.0)
        cand_rate = float(candidate_tasks.get(task, {}).get("pass_rate", 0.0) or 0.0)
        out[task] = {
            "baseline_pass_rate": base_rate,
            "candidate_pass_rate": cand_rate,
            "delta": cand_rate - base_rate,
        }
    return out


def gate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    evaluator = _load_eval_module(repo_root)
    manifests = [str(Path(item).expanduser().resolve()) for item in args.dataset_manifest]
    datasets: list[dict[str, Any]] = []
    promote = True
    for manifest in manifests:
        baseline = evaluator.evaluate(_eval_args(args, str(args.baseline_bundle), manifest))
        candidate = evaluator.evaluate(_eval_args(args, str(args.candidate_bundle), manifest))
        task_deltas = _compare_task_rates(baseline, candidate)
        baseline_rate = float(baseline.get("pass_rate", 0.0) or 0.0)
        candidate_rate = float(candidate.get("pass_rate", 0.0) or 0.0)
        baseline_recall = float(baseline.get("mean_recall", 0.0) or 0.0)
        candidate_recall = float(candidate.get("mean_recall", 0.0) or 0.0)
        regression = baseline_rate - candidate_rate
        recall_regression = baseline_recall - candidate_recall
        malformed = _malformed_count(candidate)
        protected_regressions = {
            task: delta
            for task, delta in task_deltas.items()
            if delta["delta"] < -float(args.max_task_regression)
        }
        dataset_ok = (
            candidate_rate >= float(args.min_pass_rate)
            and regression <= float(args.max_overall_regression)
            and recall_regression <= float(args.max_mean_recall_regression)
            and malformed <= int(args.max_malformed_failures)
            and not protected_regressions
        )
        promote = promote and dataset_ok
        datasets.append(
            {
                "dataset_manifest": manifest,
                "baseline": {
                    "bundle_dir": baseline.get("bundle_dir"),
                    "examples": baseline.get("examples"),
                    "passed": baseline.get("passed"),
                    "pass_rate": baseline_rate,
                    "mean_recall": baseline_recall,
                },
                "candidate": {
                    "bundle_dir": candidate.get("bundle_dir"),
                    "examples": candidate.get("examples"),
                    "passed": candidate.get("passed"),
                    "pass_rate": candidate_rate,
                    "mean_recall": candidate_recall,
                    "malformed_failure_count": malformed,
                },
                "overall_pass_rate_delta": candidate_rate - baseline_rate,
                "mean_recall_delta": candidate_recall - baseline_recall,
                "task_deltas": task_deltas,
                "protected_regressions": protected_regressions,
                "promotion_allowed_for_dataset": dataset_ok,
                "candidate_failures": candidate.get("failures", []),
            }
        )
    return {
        "baseline_bundle": str(Path(args.baseline_bundle).expanduser().resolve()),
        "candidate_bundle": str(Path(args.candidate_bundle).expanduser().resolve()),
        "promotion_allowed": promote,
        "datasets": datasets,
        "thresholds": {
            "max_malformed_failures": int(args.max_malformed_failures),
            "max_mean_recall_regression": float(args.max_mean_recall_regression),
            "max_overall_regression": float(args.max_overall_regression),
            "max_task_regression": float(args.max_task_regression),
            "min_pass_rate": float(args.min_pass_rate),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--baseline-bundle", required=True)
    parser.add_argument("--candidate-bundle", required=True)
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.65)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--keep-special-tokens", type=int, choices=(0, 1), default=1)
    parser.add_argument("--forbid-ak-content-control", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-failures", type=int, default=12)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--max-overall-regression", type=float, default=0.02)
    parser.add_argument("--max-mean-recall-regression", type=float, default=0.0)
    parser.add_argument("--max-task-regression", type=float, default=0.05)
    parser.add_argument("--max-malformed-failures", type=int, default=0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = gate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not summary["promotion_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
