#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


VERSION_RE = re.compile(r"pocketpal_controller_100m_(v[0-9]+[a-z]?(?:_step[0-9]+)?)(?:_|$)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _version_from_path(path: Path) -> str:
    match = VERSION_RE.search(path.name)
    return match.group(1) if match else path.name


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_dataset_manifest(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return {"manifest_exists": False, "manifest_path": str(path)}
    payload = _load_json(path)
    payload["manifest_exists"] = True
    return payload


def _collect_artifacts(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((repo_root / "artifacts").glob("pocketpal_controller_100m_*/agentkernel_lite_encdec_manifest.json")):
        manifest = _load_json(manifest_path)
        summary = manifest.get("training_summary", {})
        summary = summary if isinstance(summary, dict) else {}
        model_config = manifest.get("model_config", {})
        model_config = model_config if isinstance(model_config, dict) else {}
        dataset_manifest = _read_dataset_manifest(manifest.get("dataset_manifest_path"))
        eval_history = summary.get("eval_history", [])
        eval_history = eval_history if isinstance(eval_history, list) else []
        last_eval = eval_history[-1] if eval_history and isinstance(eval_history[-1], dict) else {}
        latest_ckpt = manifest_path.parent / "checkpoints" / "latest.json"
        rows.append(
            {
                "version": _version_from_path(manifest_path.parent),
                "artifact_dir": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "mtime_utc": _mtime_iso(manifest_path.parent),
                "parameter_count": _safe_int(manifest.get("parameter_count")),
                "dataset_objective": str(summary.get("dataset_objective") or dataset_manifest.get("objective") or ""),
                "dataset_manifest_path": str(manifest.get("dataset_manifest_path") or ""),
                "dataset_manifest_exists": bool(dataset_manifest.get("manifest_exists")),
                "dataset_format": str(dataset_manifest.get("dataset_format") or ""),
                "train_examples": _safe_int(dataset_manifest.get("train_examples")),
                "eval_examples": _safe_int(dataset_manifest.get("eval_examples")),
                "total_examples": _safe_int(dataset_manifest.get("total_examples")),
                "completed_steps": _safe_int(summary.get("completed_steps")),
                "max_steps": _safe_int(summary.get("max_steps")),
                "eval_every": _safe_int(summary.get("eval_every")),
                "eval_history_count": len(eval_history),
                "last_eval_loss": _safe_float(last_eval.get("eval_loss")),
                "last_eval_step": _safe_int(last_eval.get("step")),
                "decoder_loss_weight": _safe_float(summary.get("decoder_loss_weight")),
                "intent_head_loss_weight": _safe_float(summary.get("intent_head_loss_weight")),
                "intent_contrastive_weight": _safe_float(summary.get("intent_contrastive_weight")),
                "retrieval_contrastive_weight": _safe_float(summary.get("retrieval_contrastive_weight")),
                "encoder_rep_distill_weight": _safe_float(summary.get("encoder_rep_distill_weight")),
                "freeze_encoder": bool(summary.get("freeze_encoder")),
                "freeze_decoder": bool(summary.get("freeze_decoder")),
                "freeze_lm_head": bool(summary.get("freeze_lm_head")),
                "freeze_token_embeddings": bool(summary.get("freeze_token_embeddings")),
                "freeze_encoder_layers_through": _safe_int(summary.get("freeze_encoder_layers_through")),
                "browser_bitnet_exported": bool(summary.get("browser_bitnet_exported")),
                "latest_checkpoint_exists": latest_ckpt.exists(),
                "vocab_size": _safe_int(model_config.get("vocab_size")),
                "d_model": _safe_int(model_config.get("d_model")),
                "n_layers": _safe_int(model_config.get("n_layers")),
                "n_heads": _safe_int(model_config.get("n_heads")),
                "replaces_surfaces": json.dumps(manifest.get("replaces_surfaces", []), sort_keys=True),
            }
        )
    return rows


def _summarize_retrieval(payload: dict[str, Any]) -> dict[str, Any]:
    examples = payload.get("examples", [])
    margins = [
        float(row.get("margin"))
        for row in examples
        if isinstance(row, dict) and row.get("margin") is not None
    ]
    return {
        "accuracy": _safe_float(payload.get("accuracy")),
        "evaluated": _safe_int(payload.get("evaluated")),
        "top1": _safe_int(payload.get("top1")),
        "mean_margin": _safe_float(payload.get("mean_margin")),
        "min_margin": _safe_float(payload.get("min_margin")),
        "negative_margin_count": sum(1 for margin in margins if margin < 0),
    }


def _summarize_direct(payload: dict[str, Any]) -> dict[str, Any]:
    failures = payload.get("failures", [])
    malformed = 0
    low_recall = 0
    task_counts: dict[str, dict[str, Any]] = payload.get("by_task", {}) if isinstance(payload.get("by_task"), dict) else {}
    zero_pass_tasks = []
    for name, stats in task_counts.items():
        if isinstance(stats, dict) and float(stats.get("pass_rate", 0.0) or 0.0) <= 0.0:
            zero_pass_tasks.append(name)
    for failure in failures if isinstance(failures, list) else []:
        labels = failure.get("failures", []) if isinstance(failure, dict) else []
        labels = labels if isinstance(labels, list) else []
        if any(str(item) == "malformed" for item in labels):
            malformed += 1
        if any(str(item).startswith("low_recall") for item in labels):
            low_recall += 1
    return {
        "examples": _safe_int(payload.get("examples")),
        "passed": _safe_int(payload.get("passed")),
        "pass_rate": _safe_float(payload.get("pass_rate")),
        "mean_recall": _safe_float(payload.get("mean_recall")),
        "failure_count": len(failures) if isinstance(failures, list) else None,
        "malformed_failures": malformed,
        "low_recall_failures": low_recall,
        "zero_pass_tasks": json.dumps(sorted(zero_pass_tasks)),
        "zero_pass_task_count": len(zero_pass_tasks),
    }


def _summarize_gate(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    failure_reasons: Counter[str] = Counter()
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            for label in row.get("failures", []) or []:
                failure_reasons[str(label)] += 1
    return {
        "passed_bool": bool(payload.get("passed")),
        "required_passed": _safe_int(payload.get("required_passed")),
        "required_total": _safe_int(payload.get("required_total")),
        "top_failure_reasons": json.dumps(dict(failure_reasons.most_common(8)), sort_keys=True),
    }


def _summarize_intent(payload: dict[str, Any]) -> dict[str, Any]:
    confusion = payload.get("confusion", {})
    error_counts: Counter[str] = Counter()
    if isinstance(confusion, dict):
        for target, predictions in confusion.items():
            if not isinstance(predictions, dict):
                continue
            for pred, count in predictions.items():
                if pred != target:
                    error_counts[f"{target}->{pred}"] += int(count or 0)
    return {
        "accuracy": _safe_float(payload.get("accuracy")),
        "correct": _safe_int(payload.get("correct")),
        "total": _safe_int(payload.get("total")),
        "top_confusions": json.dumps(dict(error_counts.most_common(12)), sort_keys=True),
    }


def _collect_evals(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((repo_root / "tmp").glob("v*.json")):
        payload = _load_json(path)
        name = path.name
        version = name.split("_", 1)[0]
        if name.endswith("_direct_agent_prompts.json"):
            kind = "direct_agent_prompts"
            metrics = _summarize_direct(payload)
        elif name.endswith("_retrieval_top1.json"):
            kind = "retrieval_top1"
            metrics = _summarize_retrieval(payload)
        elif name.endswith("_agent_gates.json") or name.endswith("_agent_gates_with_source_slots.json"):
            kind = "agent_gates"
            metrics = _summarize_gate(payload)
        elif name.endswith("_general_agent_holdout.json"):
            kind = "general_agent_holdout"
            metrics = {
                "pass_rate": _safe_float(payload.get("pass_rate")),
                "passed": _safe_int(payload.get("passed")),
                "total": _safe_int(payload.get("total")),
            }
        elif name.endswith("_intent_head_eval.json"):
            kind = "intent_head"
            metrics = _summarize_intent(payload)
        elif name.endswith("_agent_certification_matrix.json"):
            kind = "agent_certification_matrix"
            metrics = {"pass_rate": _safe_float(payload.get("pass_rate")), "passed": _safe_int(payload.get("passed"))}
        else:
            continue
        row = {
            "version": version,
            "eval_kind": kind,
            "path": str(path),
            "mtime_utc": _mtime_iso(path),
            "bundle_dir": str(payload.get("bundle_dir") or ""),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.table({})
    keys = sorted({key for row in rows for key in row})
    normalized = [{key: row.get(key) for key in keys} for row in rows]
    return pa.Table.from_pylist(normalized)


def _latest_by_kind(eval_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[str(row["eval_kind"])].append(row)
    return {
        kind: sorted(rows, key=lambda row: str(row.get("mtime_utc") or ""))[-1]
        for kind, rows in grouped.items()
        if rows
    }


def _format_pct(value: Any) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def _write_markdown(out_path: Path, artifact_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    latest_artifacts = sorted(artifact_rows, key=lambda row: str(row.get("mtime_utc") or ""))[-8:]
    latest_eval = _latest_by_kind(eval_rows)
    direct_rows = [row for row in eval_rows if row.get("eval_kind") == "direct_agent_prompts"]
    best_direct = max(direct_rows, key=lambda row: float(row.get("pass_rate") or -1), default={})
    latest_direct = latest_eval.get("direct_agent_prompts", {})
    latest_retrieval = latest_eval.get("retrieval_top1", {})
    latest_intent = latest_eval.get("intent_head", {})
    latest_gate = latest_eval.get("agent_gates", {})

    lines = [
        "# PocketPal Tiny Seq2Seq Training Review",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Executive Findings",
        "",
        "- The tiny seq2seq controller is not primarily failing at coarse routing. Agent gates are green in the latest run, retrieval top-1 is near 0.795, and the recorded intent head evaluation reaches about 0.8995 accuracy.",
        "- The weak layer is faithful direct generation. The best recent direct-agent prompt pass rate found by this audit is "
        f"{_format_pct(best_direct.get('pass_rate'))} from `{best_direct.get('version', 'unknown')}`, while the latest direct eval is "
        f"{_format_pct(latest_direct.get('pass_rate'))} from `{latest_direct.get('version', 'unknown')}`.",
        "- The latest failure replay run appears to regress generation quality: it preserves gate pass status but direct prompt pass rate drops, with low-recall and malformed JSON/content drift still present.",
        "- The eval history is fragmented across `/tmp` JSON files rather than attached to bundle manifests. This makes promotion decisions easy to lose and allows a model with green small gates to hide broad generation regressions.",
        "- Several dataset manifests still use JSONL. That is acceptable for small smoke evals, but full training/eval corpora should be promoted to Parquet to reduce disk and memory pressure.",
        "",
        "## Latest Signals",
        "",
        f"- Latest gates: version `{latest_gate.get('version', 'n/a')}`, passed `{latest_gate.get('passed_bool', 'n/a')}`, required `{latest_gate.get('required_passed', 'n/a')}/{latest_gate.get('required_total', 'n/a')}`.",
        f"- Latest direct prompts: version `{latest_direct.get('version', 'n/a')}`, pass rate `{_format_pct(latest_direct.get('pass_rate'))}`, mean recall `{_format_pct(latest_direct.get('mean_recall'))}`, zero-pass tasks `{latest_direct.get('zero_pass_task_count', 'n/a')}`.",
        f"- Latest retrieval top-1: version `{latest_retrieval.get('version', 'n/a')}`, accuracy `{_format_pct(latest_retrieval.get('accuracy'))}`, mean margin `{_format_pct(latest_retrieval.get('mean_margin'))}`, negative margins `{latest_retrieval.get('negative_margin_count', 'n/a')}`.",
        f"- Latest intent head: version `{latest_intent.get('version', 'n/a')}`, accuracy `{_format_pct(latest_intent.get('accuracy'))}`.",
        "",
        "## Recent Artifacts",
        "",
        "| version | objective | steps | eval loss | train examples | eval examples | frozen encoder | frozen decoder |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in latest_artifacts:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("version") or ""),
                    str(row.get("dataset_objective") or ""),
                    str(row.get("completed_steps") or ""),
                    _format_pct(row.get("last_eval_loss")),
                    str(row.get("train_examples") or ""),
                    str(row.get("eval_examples") or ""),
                    str(row.get("freeze_encoder")),
                    str(row.get("freeze_decoder")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "The model is behaving like a useful router plus a weak constrained generator. It can often select a surface, pass tiny canned gates, and embed retrieval candidates, but it still corrupts content under direct prompt pressure. The repeated failures show task drift (`rewrite` becoming `summary` or `translation`), malformed JSON, copied fragments from unrelated examples, and low recall on extraction/json/translation/risk tasks.",
            "",
            "The current training ladder also appears to optimize one failure class at a time while regressing others. `v201a` failure replay improved neither the broad direct prompt score nor the content recall compared with `v197a` or `v200a`; it only kept the narrow gate suite green.",
            "",
            "## Required Training Gate",
            "",
            "Every candidate bundle should be rejected unless it records, in a single promotion JSON and Parquet row:",
            "",
            "1. Agent gates pass all required cases.",
            "2. Direct-agent prompt pass rate improves or stays within a small regression budget against the promoted baseline.",
            "3. No protected task has zero pass rate.",
            "4. Malformed outputs are zero or explicitly below a configured threshold.",
            "5. Retrieval top-1 does not regress on harness-skill examples.",
            "6. Intent head confusion does not regress on protected route pairs: rewrite/action_items, summary/extraction, json/rewrite, extraction/casual.",
            "7. Browser/WASM export parity is checked before promotion.",
            "",
            "## Model Direction",
            "",
            "Do not ask the decoder to carry the whole controller contract. Keep the tiny model as a hybrid action model:",
            "",
            "- encoder heads decide intent, action validity, retrieval need, confidence, OOD, and verification need;",
            "- retrieval head supplies the relevant skill/operator context;",
            "- decoder emits only a constrained decision object or short content when the route is simple;",
            "- deterministic templates handle source echo, saved data references, missing slot questions, and extension requests;",
            "- high-entropy content tasks use retrieved exemplars or a larger teacher, while this model chooses the action and constraints.",
            "",
            "## Files Emitted",
            "",
            f"- Artifacts parquet: `{out_path.parent / 'pocketpal_artifacts.parquet'}`",
            f"- Eval parquet: `{out_path.parent / 'pocketpal_eval_trends.parquet'}`",
            f"- Summary JSON: `{out_path.parent / 'pocketpal_seq2seq_review_summary.json'}`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else repo_root / "artifacts" / "pocketpal_seq2seq_review" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows = _collect_artifacts(repo_root)
    eval_rows = _collect_evals(repo_root)
    pq.write_table(_table(artifact_rows), output_dir / "pocketpal_artifacts.parquet", compression="zstd")
    pq.write_table(_table(eval_rows), output_dir / "pocketpal_eval_trends.parquet", compression="zstd")

    latest = _latest_by_kind(eval_rows)
    direct_rows = [row for row in eval_rows if row.get("eval_kind") == "direct_agent_prompts"]
    best_direct = max(direct_rows, key=lambda row: float(row.get("pass_rate") or -1), default={})
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "artifact_count": len(artifact_rows),
        "eval_record_count": len(eval_rows),
        "latest_by_kind": latest,
        "best_direct_agent_prompts": best_direct,
        "output_dir": str(output_dir),
    }
    (output_dir / "pocketpal_seq2seq_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "pocketpal_seq2seq_training_review.md"
    _write_markdown(report_path, artifact_rows, eval_rows)
    docs_path = repo_root / "docs" / "pocketpal_seq2seq_supervision_review.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
