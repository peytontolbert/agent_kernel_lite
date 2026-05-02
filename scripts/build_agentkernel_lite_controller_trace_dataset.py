#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SPECIAL_TOKENS = [
    "<AK_USER>",
    "<AK_LOOP>",
    "<AK_STATE>",
    "<AK_CONTEXT>",
    "<AK_HISTORY>",
    "<AK_ACTION_SPACE_CODE>",
    "<AK_ACTION_SPACE_ARTIFACT>",
    "<AK_ACTION_SPACE_RETRIEVAL>",
    "<AK_ACTION_SPACE_RESPOND>",
    "<AK_RETRIEVE>",
    "<AK_NO_RETRIEVAL>",
    "<AK_RET_CODE>",
    "<AK_RET_MEMORY>",
    "<AK_RET_EXACT>",
    "<AK_RET_SEMANTIC>",
    "<AK_VERIFY>",
    "<AK_CONF_HIGH>",
    "<AK_CONF_MEDIUM>",
    "<AK_CONF_LOW>",
    "<AK_OOD>",
    "<AK_ARTIFACT_REPAIR>",
    "<AK_SOURCE_INSPECT>",
    "<AK_PATCH_BUILD>",
    "<AK_RESPOND>",
    "<AK_SAFE_STOP>",
]


def _compact(value: object, *, limit: int = 1600) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text[:limit].rstrip()


def _one_line(value: object, *, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit].rstrip()


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _report_index(report_roots: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for root in report_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            payload = _read_json(path)
            task_id = str(payload.get("task_id", "")).strip()
            if not task_id:
                continue
            current = index.get(task_id)
            if current is None or str(payload.get("generated_at", "")) >= str(current.get("generated_at", "")):
                index[task_id] = payload
    return index


def _source_context_excerpt(checkpoint: dict[str, Any], *, limit: int = 1400) -> str:
    contract = checkpoint.get("task_contract") if isinstance(checkpoint.get("task_contract"), dict) else {}
    prompt = _compact(contract.get("prompt", ""), limit=limit)
    if prompt:
        return prompt
    return _compact(checkpoint.get("active_subgoal", ""), limit=limit)


def _step_result_summary(step: dict[str, Any]) -> str:
    result = step.get("command_result") if isinstance(step.get("command_result"), dict) else {}
    if not result:
        return ""
    pieces = [f"exit={result.get('exit_code', '')}"]
    stdout = _one_line(result.get("stdout", ""), limit=180)
    stderr = _one_line(result.get("stderr", ""), limit=180)
    if stdout:
        pieces.append(f"stdout={stdout}")
    if stderr:
        pieces.append(f"stderr={stderr}")
    return " ".join(piece for piece in pieces if piece)


def _retrieval_tokens(step: dict[str, Any], content: str) -> tuple[str, str, float, float]:
    lowered = content.lower()
    if bool(step.get("retrieval_influenced")) or "source_lines/" in lowered or "source_context/" in lowered:
        return "<AK_RETRIEVE> <AK_RET_CODE> <AK_RET_EXACT>", "code", 1.0, 0.75
    if content.startswith(("cat ", "sed ", "head ", "tail ", "grep ")):
        return "<AK_RETRIEVE> <AK_RET_CODE> <AK_RET_EXACT>", "code", 1.0, 0.6
    return "<AK_NO_RETRIEVAL>", "", 0.0, 0.0


def _action_space(action: str, content: str, decision_source: str) -> str:
    if action == "respond":
        return "<AK_ACTION_SPACE_RESPOND>"
    if content.startswith(("cat ", "sed ", "head ", "tail ", "grep ")):
        return "<AK_ACTION_SPACE_RETRIEVAL>"
    if "artifact" in decision_source or "patch_builder" in content or "swe_patch_builder" in content:
        return "<AK_ACTION_SPACE_ARTIFACT>"
    return "<AK_ACTION_SPACE_CODE>"


def _cognitive_tokens(
    *,
    action: str,
    content: str,
    decision_source: str,
    verification_passed: bool | None,
    failure_mode: str,
) -> str:
    tokens: list[str] = []
    tokens.append(_action_space(action, content, decision_source))
    if content.startswith(("cat ", "sed ", "head ", "tail ", "grep ")):
        tokens.append("<AK_SOURCE_INSPECT>")
    elif "artifact" in decision_source or "patch_builder" in content or "swe_patch_builder" in content:
        tokens.append("<AK_ARTIFACT_REPAIR>")
    if "patch_builder" in content or "swe_patch_builder" in content:
        tokens.append("<AK_PATCH_BUILD>")
    if action == "respond":
        tokens.append("<AK_RESPOND>")
    if verification_passed is False:
        tokens.append("<AK_VERIFY>")
        tokens.append("<AK_CONF_LOW>")
    elif verification_passed is True:
        tokens.append("<AK_CONF_HIGH>")
    else:
        tokens.append("<AK_CONF_MEDIUM>")
    if failure_mode and failure_mode not in {"artifact_contract_success", "not_artifact_contract"}:
        tokens.append("<AK_OOD>")
    return " ".join(dict.fromkeys(tokens))


def _decoder_text(
    *,
    action: str,
    content: str,
    cognitive_tokens: str,
    failure_mode: str,
) -> str:
    lines = [cognitive_tokens, f"Action: {action}"]
    if failure_mode:
        lines.append(f"Artifact-Failure-Mode: {failure_mode}")
    lines.append(f"Content: {_compact(content, limit=900)}")
    return "\n".join(lines)


def _encoder_text(
    *,
    checkpoint: dict[str, Any],
    report: dict[str, Any],
    history_prefix: list[dict[str, Any]],
    failure_mode: str,
    retrieval_tokens: str,
) -> str:
    task_id = str(checkpoint.get("task_id", "")).strip()
    outcome = str(report.get("outcome", checkpoint.get("status", ""))).strip()
    subgoal = _one_line(checkpoint.get("active_subgoal", ""), limit=220)
    prompt = _source_context_excerpt(checkpoint, limit=1400)
    history_lines: list[str] = []
    for prior in history_prefix[-4:]:
        action = str(prior.get("action", "")).strip()
        content = _one_line(prior.get("content", ""), limit=220)
        result = _step_result_summary(prior)
        history_lines.append(f"- {action}: {content}{' -> ' + result if result else ''}")
    return "\n".join(
        [
            f"<AK_USER> task_id={task_id}",
            f"<AK_LOOP> <AK_STATE> outcome={outcome or 'unknown'} failure_mode={failure_mode or 'none'}",
            retrieval_tokens,
            f"subgoal={subgoal}" if subgoal else "subgoal=none",
            "<AK_CONTEXT>",
            prompt,
            "<AK_HISTORY>",
            "\n".join(history_lines) if history_lines else "none",
        ]
    ).strip()


def _row_from_step(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    report: dict[str, Any],
    step: dict[str, Any],
    history_prefix: list[dict[str, Any]],
    eval_fraction: float,
) -> dict[str, Any] | None:
    action = str(step.get("action", "")).strip()
    content = str(step.get("content", "")).strip()
    if not action or not content:
        return None
    task_id = str(checkpoint.get("task_id", "")).strip() or checkpoint_path.stem
    decision_source = str(step.get("decision_source", "")).strip()
    artifact_failure = report.get("artifact_contract_failure") if isinstance(report.get("artifact_contract_failure"), dict) else {}
    failure_mode = str(artifact_failure.get("mode", "")).strip()
    verification_passed = step.get("verification_passed")
    if verification_passed is None:
        result = step.get("command_result") if isinstance(step.get("command_result"), dict) else {}
        verification_passed = result.get("exit_code") == 0 if result else None
    retrieval_tokens, retrieval_namespace, retrieval_weight, retrieval_coverage = _retrieval_tokens(step, content)
    cognitive_tokens = _cognitive_tokens(
        action=action,
        content=content,
        decision_source=decision_source,
        verification_passed=verification_passed if isinstance(verification_passed, bool) else None,
        failure_mode=failure_mode,
    )
    source_id = f"{checkpoint_path}:{step.get('index', len(history_prefix) + 1)}"
    split = _hash_split(source_id, eval_fraction)
    confidence = 0.85 if verification_passed is True else 0.25 if verification_passed is False else 0.55
    needs_verification = 0.85 if action == "code_execute" else 0.35
    return {
        "example_id": hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:20],
        "split": split,
        "source_type": "agentkernel_controller_trace",
        "source_id": source_id,
        "task_type": "controller_action_policy",
        "encoder_text": _encoder_text(
            checkpoint=checkpoint,
            report=report,
            history_prefix=history_prefix,
            failure_mode=failure_mode,
            retrieval_tokens=retrieval_tokens,
        ),
        "decoder_text": _decoder_text(
            action=action,
            content=content,
            cognitive_tokens=cognitive_tokens,
            failure_mode=failure_mode,
        ),
        "action": action,
        "weight": 1.0 if verification_passed is True else 0.7,
        "retrieval_query_text": _one_line(content, limit=240) if retrieval_weight else "",
        "retrieval_doc_text": _compact(_source_context_excerpt(checkpoint, limit=900), limit=900) if retrieval_weight else "",
        "retrieval_loss_weight": retrieval_weight,
        "query_confidence_target": confidence,
        "retrieval_coverage_target": retrieval_coverage,
        "ood_query_target": 0.15 if retrieval_namespace else 0.35,
        "ood_evidence_target": 0.2 if verification_passed is True else 0.65 if verification_passed is False else 0.45,
        "answer_confidence_target": confidence,
        "needs_verification_target": needs_verification,
        "paper_action_validity_target": 0.0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    report_roots = [Path(item).expanduser().resolve() for item in args.report_root]
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_by_task = _report_index(report_roots)
    rows: list[dict[str, Any]] = []
    checkpoint_paths = [
        path
        for path in sorted(checkpoint_root.rglob("*.json"))
        if not path.name.endswith(".progress.json")
    ]
    if args.max_checkpoints > 0:
        checkpoint_paths = checkpoint_paths[: int(args.max_checkpoints)]
    for checkpoint_path in checkpoint_paths:
        checkpoint = _read_json(checkpoint_path)
        raw_history = checkpoint.get("history", [])
        if not isinstance(raw_history, list):
            continue
        history = [item for item in raw_history if isinstance(item, dict)]
        if not history:
            continue
        task_id = str(checkpoint.get("task_id", "")).strip()
        report = report_by_task.get(task_id, {})
        for index, step in enumerate(history):
            row = _row_from_step(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                report=report,
                step=step,
                history_prefix=history[:index],
                eval_fraction=float(args.eval_fraction),
            )
            if row is not None:
                rows.append(row)
            if args.max_examples > 0 and len(rows) >= int(args.max_examples):
                break
        if args.max_examples > 0 and len(rows) >= int(args.max_examples):
            break
    train_rows = [row for row in rows if row.get("split") != "eval"]
    eval_rows = [row for row in rows if row.get("split") == "eval"]
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    source_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        source_counts[str(row.get("source_type", ""))] = source_counts.get(str(row.get("source_type", "")), 0) + 1
        action_counts[str(row.get("action", ""))] = action_counts.get(str(row.get("action", "")), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "agentkernel_controller_trace_policy",
        "dataset_format": "jsonl",
        "decoder_format": "line",
        "agentkernel_special_tokens": SPECIAL_TOKENS,
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "checkpoint_root": str(checkpoint_root),
        "report_roots": [str(path) for path in report_roots],
        "schema": {
            "encoder_text": "AgentKernel controller state, context, retrieval/control tokens, and recent history",
            "decoder_text": "cognitive/action-space tokens followed by line protocol action/content target",
            "retrieval_query_text": "optional retrieval query supervision from source-inspection actions",
            "policy_targets": "confidence, OOD, retrieval coverage, and verification-need float targets",
        },
    }
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="benchmarks")
    parser.add_argument("--report-root", action="append", default=[])
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/controller_trace_dataset")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    args = parser.parse_args()
    if not args.report_root:
        args.report_root = ["benchmarks"]
    print(json.dumps(build_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
