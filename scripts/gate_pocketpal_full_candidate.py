#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(repo_root: Path, script_name: str):
    path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _direct_args(args: argparse.Namespace, bundle_dir: str, dataset_manifest: str) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(args.repo_root),
        bundle_dir=bundle_dir,
        dataset_manifest=dataset_manifest,
        device=str(args.device),
        max_encoder_tokens=int(args.max_encoder_tokens),
        max_new_tokens=int(args.direct_max_new_tokens),
        temperature=0.0,
        top_p=0.65,
        repetition_penalty=1.0,
        decoder_prefix="",
        decoder_prefix_by_task=[],
        keep_special_tokens=1,
        forbid_ak_content_control=0,
        max_examples=int(args.direct_examples),
        max_failures=int(args.max_failures),
        output_json="",
    )


def _gate_args(args: argparse.Namespace, bundle_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(args.repo_root),
        bundle_dir=bundle_dir,
        device=str(args.device),
        max_encoder_tokens=int(args.max_encoder_tokens),
        max_new_tokens=220,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.0,
        decoder_prefix="",
        use_action_prefix=0,
        content_only_wrapper=0,
        allow_runtime_extension_fallback=0,
        include_experimental=0,
        output_json="",
    )


def _cert_args(args: argparse.Namespace, bundle_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=str(args.repo_root),
        bundle_dir=bundle_dir,
        device=str(args.device),
        max_encoder_tokens=int(args.max_encoder_tokens),
        max_new_tokens=220,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.0,
        decoder_prefix="",
        use_action_prefix=1,
        output_json="",
    )


def _run_json_script(repo_root: Path, script_name: str, argv: list[str]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="pocketpal_gate_", suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)
    try:
        cmd = [sys.executable, str(repo_root / "scripts" / script_name), *argv, "--output-json", str(output_path)]
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode not in (0, 1, 2):
            raise RuntimeError(f"{script_name} failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


def _malformed_count(summary: dict[str, Any]) -> int:
    count = 0
    for failure in summary.get("failures", []):
        labels = failure.get("failures", [])
        if any(str(label).startswith("malformed") for label in labels):
            count += 1
    return count


def _zero_pass_tasks(summary: dict[str, Any]) -> list[str]:
    out = []
    for task, bucket in sorted((summary.get("by_task") or {}).items()):
        if int(bucket.get("total") or 0) > 0 and int(bucket.get("passed") or 0) == 0:
            out.append(str(task))
    return out


def _protected_task_regressions(baseline: dict[str, Any], candidate: dict[str, Any], max_delta: float) -> dict[str, dict[str, float]]:
    regressions: dict[str, dict[str, float]] = {}
    base_tasks = baseline.get("by_task", {}) or {}
    cand_tasks = candidate.get("by_task", {}) or {}
    for task in sorted(set(base_tasks) | set(cand_tasks)):
        base_rate = float(base_tasks.get(task, {}).get("pass_rate", 0.0) or 0.0)
        cand_rate = float(cand_tasks.get(task, {}).get("pass_rate", 0.0) or 0.0)
        delta = cand_rate - base_rate
        if delta < -float(max_delta):
            regressions[task] = {
                "baseline_pass_rate": base_rate,
                "candidate_pass_rate": cand_rate,
                "delta": delta,
            }
    return regressions


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_digest(bundle_dir: str) -> dict[str, Any]:
    manifest_path = Path(bundle_dir) / "agentkernel_lite_encdec_manifest.json"
    manifest = _read_json(manifest_path)
    config = manifest.get("model_config") or {}
    return {
        "manifest_path": str(manifest_path),
        "retrieval_head_dim": config.get("retrieval_head_dim"),
        "agent_policy_heads": bool(config.get("agent_policy_heads")),
        "agent_intent_labels": config.get("agent_intent_labels"),
        "browser_bitnet_manifest_path": manifest.get("browser_bitnet_manifest_path"),
    }


def _webapp_digest(webapp_manifest: str, candidate_manifest: str) -> dict[str, Any]:
    if not str(webapp_manifest).strip():
        return {"checked": False}
    manifest_path = Path(webapp_manifest).expanduser().resolve()
    if not manifest_path.exists():
        return {
            "checked": True,
            "exists": False,
            "manifest_path": str(manifest_path),
            "matches_candidate": False,
            "has_policy_heads": False,
            "has_retrieval_head": False,
        }
    manifest = _read_json(manifest_path)
    model = manifest.get("model") or {}
    agent = manifest.get("agentkernel_lite") or {}
    source_manifest = str(agent.get("source_bundle_manifest_path") or "")
    return {
        "checked": True,
        "exists": True,
        "manifest_path": str(manifest_path),
        "source_bundle_manifest_path": source_manifest,
        "matches_candidate": Path(source_manifest).expanduser().resolve() == Path(candidate_manifest).expanduser().resolve()
        if source_manifest
        else False,
        "retrieval_head_dim": model.get("retrieval_head_dim"),
        "agent_policy_heads": bool(model.get("agent_policy_heads")),
        "agent_intent_labels": model.get("agent_intent_labels"),
        "has_policy_heads": bool(model.get("agent_policy_heads")),
        "has_retrieval_head": model.get("retrieval_head_dim") is not None,
    }


def gate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    direct_module = _load_module(repo_root, "evaluate_pocketpal_direct_agent_prompts.py")
    gates_module = _load_module(repo_root, "evaluate_pocketpal_agent_gates.py")
    cert_module = _load_module(repo_root, "evaluate_pocketpal_agent_certification_matrix.py")

    baseline_bundle = str(Path(args.baseline_bundle).expanduser().resolve())
    candidate_bundle = str(Path(args.candidate_bundle).expanduser().resolve())
    broad_manifest = str(Path(args.broad_manifest).expanduser().resolve())
    retrieval_eval = str(Path(args.retrieval_eval_jsonl).expanduser().resolve())
    candidate_arch = _bundle_digest(candidate_bundle)
    webapp = _webapp_digest(str(args.webapp_manifest), candidate_arch["manifest_path"])

    active_gates = gates_module.evaluate(_gate_args(args, candidate_bundle))
    certification = cert_module.evaluate(_cert_args(args, candidate_bundle))
    baseline_direct = direct_module.evaluate(_direct_args(args, baseline_bundle, broad_manifest))
    candidate_direct = direct_module.evaluate(_direct_args(args, candidate_bundle, broad_manifest))
    baseline_nll = _run_json_script(
        repo_root,
        "evaluate_pocketpal_decoder_nll.py",
        [
            "--bundle-dir",
            baseline_bundle,
            "--dataset-manifest",
            broad_manifest,
            "--split",
            "eval",
            "--device",
            str(args.device),
            "--limit",
            str(args.nll_examples),
            "--batch-size",
            str(args.nll_batch_size),
        ],
    )
    candidate_nll = _run_json_script(
        repo_root,
        "evaluate_pocketpal_decoder_nll.py",
        [
            "--bundle-dir",
            candidate_bundle,
            "--dataset-manifest",
            broad_manifest,
            "--split",
            "eval",
            "--device",
            str(args.device),
            "--limit",
            str(args.nll_examples),
            "--batch-size",
            str(args.nll_batch_size),
        ],
    )
    retrieval = _run_json_script(
        repo_root,
        "evaluate_pocketpal_retrieval_top1.py",
        [
            "--bundle-dir",
            candidate_bundle,
            "--eval-jsonl",
            retrieval_eval,
            "--device",
            str(args.device),
            "--max-examples",
            str(args.retrieval_examples),
        ],
    )
    intent = _run_json_script(
        repo_root,
        "evaluate_pocketpal_intent_head.py",
        [
            "--bundle-dir",
            candidate_bundle,
            "--dataset-manifest",
            broad_manifest,
            "--split",
            "eval",
            "--limit",
            str(args.intent_examples),
            "--device",
            str(args.device),
        ],
    )
    head_assisted = _run_json_script(
        repo_root,
        "evaluate_pocketpal_head_assisted_agent.py",
        [
            "--bundle-dir",
            candidate_bundle,
            "--dataset-manifest",
            broad_manifest,
            "--intent-label-manifest",
            broad_manifest,
            "--route-source",
            "hint_then_model",
            "--device",
            str(args.device),
            "--max-examples",
            str(args.head_assisted_examples),
            "--max-failures",
            str(args.max_failures),
        ],
    )

    baseline_rate = float(baseline_direct.get("pass_rate", 0.0) or 0.0)
    candidate_rate = float(candidate_direct.get("pass_rate", 0.0) or 0.0)
    baseline_recall = float(baseline_direct.get("mean_recall", 0.0) or 0.0)
    candidate_recall = float(candidate_direct.get("mean_recall", 0.0) or 0.0)
    malformed = _malformed_count(candidate_direct)
    zero_pass = _zero_pass_tasks(candidate_direct)
    task_regressions = _protected_task_regressions(baseline_direct, candidate_direct, float(args.max_task_regression))
    baseline_ppl = float((baseline_nll.get("overall") or {}).get("perplexity", float("inf")) or float("inf"))
    candidate_ppl = float((candidate_nll.get("overall") or {}).get("perplexity", float("inf")) or float("inf"))
    baseline_bpb = float((baseline_nll.get("overall") or {}).get("bits_per_byte", float("inf")) or float("inf"))
    candidate_bpb = float((candidate_nll.get("overall") or {}).get("bits_per_byte", float("inf")) or float("inf"))
    checks = {
        "active_gates": bool(active_gates.get("passed")),
        "certification": bool(certification.get("ok")),
        "direct_min_pass_rate": candidate_rate >= float(args.min_direct_pass_rate),
        "direct_not_regressed": (baseline_rate - candidate_rate) <= float(args.max_direct_regression),
        "direct_mean_recall_not_regressed": (baseline_recall - candidate_recall)
        <= float(args.max_direct_mean_recall_regression),
        "malformed_count": malformed <= int(args.max_malformed_failures),
        "zero_pass_tasks": len(zero_pass) <= int(args.max_zero_pass_tasks),
        "task_regressions": not task_regressions,
        "retrieval_accuracy": float(retrieval.get("accuracy", 0.0) or 0.0) >= float(args.min_retrieval_accuracy),
        "intent_accuracy": float(intent.get("accuracy", 0.0) or 0.0) >= float(args.min_intent_accuracy),
        "head_assisted_hint_route": float(head_assisted.get("pass_rate", 0.0) or 0.0)
        >= float(args.min_head_assisted_pass_rate),
        "decoder_perplexity_not_regressed": (candidate_ppl - baseline_ppl) <= float(args.max_decoder_ppl_regression),
        "decoder_bpb_not_regressed": (candidate_bpb - baseline_bpb) <= float(args.max_decoder_bpb_regression),
        "candidate_has_policy_heads": bool(candidate_arch.get("agent_policy_heads")),
        "candidate_has_retrieval_head": candidate_arch.get("retrieval_head_dim") is not None,
    }
    if bool(args.require_webapp_candidate):
        checks["webapp_matches_candidate"] = bool(webapp.get("matches_candidate"))
        checks["webapp_has_policy_heads"] = bool(webapp.get("has_policy_heads"))
        checks["webapp_has_retrieval_head"] = bool(webapp.get("has_retrieval_head"))
    promotion_allowed = all(checks.values())
    return {
        "baseline_bundle": baseline_bundle,
        "candidate_bundle": candidate_bundle,
        "promotion_allowed": promotion_allowed,
        "checks": checks,
        "thresholds": {
            "min_direct_pass_rate": float(args.min_direct_pass_rate),
            "max_direct_regression": float(args.max_direct_regression),
            "max_direct_mean_recall_regression": float(args.max_direct_mean_recall_regression),
            "max_task_regression": float(args.max_task_regression),
            "max_malformed_failures": int(args.max_malformed_failures),
            "max_zero_pass_tasks": int(args.max_zero_pass_tasks),
        "min_retrieval_accuracy": float(args.min_retrieval_accuracy),
        "min_intent_accuracy": float(args.min_intent_accuracy),
        "min_head_assisted_pass_rate": float(args.min_head_assisted_pass_rate),
        "max_decoder_ppl_regression": float(args.max_decoder_ppl_regression),
            "max_decoder_bpb_regression": float(args.max_decoder_bpb_regression),
        },
        "active_gates": {
            "passed": active_gates.get("passed"),
            "required_passed": active_gates.get("required_passed"),
            "required_total": active_gates.get("required_total"),
        },
        "certification": {
            "ok": certification.get("ok"),
            "passed": certification.get("passed"),
            "total": certification.get("total"),
            "by_regime": certification.get("by_regime"),
        },
        "architecture": {
            "candidate": candidate_arch,
            "webapp": webapp,
        },
        "direct": {
            "baseline_pass_rate": baseline_rate,
            "candidate_pass_rate": candidate_rate,
            "delta": candidate_rate - baseline_rate,
            "baseline_mean_recall": baseline_recall,
            "candidate_mean_recall": candidate_recall,
            "mean_recall_delta": candidate_recall - baseline_recall,
            "candidate_passed": candidate_direct.get("passed"),
            "examples": candidate_direct.get("examples"),
            "malformed_failure_count": malformed,
            "zero_pass_tasks": zero_pass,
            "task_regressions": task_regressions,
            "by_task": candidate_direct.get("by_task"),
            "sample_failures": candidate_direct.get("failures", [])[: int(args.max_failures)],
        },
        "retrieval": {
            "accuracy": retrieval.get("accuracy"),
            "top1": retrieval.get("top1"),
            "evaluated": retrieval.get("evaluated"),
            "mean_margin": retrieval.get("mean_margin"),
        },
        "intent": {
            "accuracy": intent.get("accuracy"),
            "correct": intent.get("correct"),
            "total": intent.get("total"),
            "sample_errors": intent.get("sample_errors", [])[:10],
        },
        "head_assisted_hint_route": {
            "passed": head_assisted.get("passed"),
            "examples": head_assisted.get("examples"),
            "pass_rate": head_assisted.get("pass_rate"),
            "mean_recall": head_assisted.get("mean_recall"),
            "failures": head_assisted.get("failures", [])[: int(args.max_failures)],
        },
        "decoder_nll": {
            "baseline": baseline_nll.get("overall"),
            "candidate": candidate_nll.get("overall"),
            "perplexity_delta": candidate_ppl - baseline_ppl,
            "bits_per_byte_delta": candidate_bpb - baseline_bpb,
            "candidate_by_task": candidate_nll.get("by_task"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--baseline-bundle", required=True)
    parser.add_argument("--candidate-bundle", required=True)
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-eval-jsonl", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_eval.jsonl")
    parser.add_argument("--webapp-manifest", default="web/models/pocketpal_controller_100m_bitnet_dev/manifest.json")
    parser.add_argument("--require-webapp-candidate", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--direct-max-new-tokens", type=int, default=220)
    parser.add_argument("--direct-examples", type=int, default=120)
    parser.add_argument("--retrieval-examples", type=int, default=200)
    parser.add_argument("--intent-examples", type=int, default=2000)
    parser.add_argument("--head-assisted-examples", type=int, default=300)
    parser.add_argument("--nll-examples", type=int, default=0)
    parser.add_argument("--nll-batch-size", type=int, default=8)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--min-direct-pass-rate", type=float, default=0.42)
    parser.add_argument("--max-direct-regression", type=float, default=0.0)
    parser.add_argument("--max-direct-mean-recall-regression", type=float, default=0.0)
    parser.add_argument("--max-task-regression", type=float, default=0.05)
    parser.add_argument("--max-malformed-failures", type=int, default=0)
    parser.add_argument("--max-zero-pass-tasks", type=int, default=3)
    parser.add_argument("--min-retrieval-accuracy", type=float, default=0.79)
    parser.add_argument("--min-intent-accuracy", type=float, default=0.88)
    parser.add_argument("--min-head-assisted-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-decoder-ppl-regression", type=float, default=0.0)
    parser.add_argument("--max-decoder-bpb-regression", type=float, default=0.0)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = gate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)
    if not summary["promotion_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
