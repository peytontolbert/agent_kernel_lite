#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


AK_CHAT = "<AK_CHAT>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_PROFILE = "<AK_PROFILE>"
AK_RESPOND = "<AK_RESPOND>"
AK_SLOT = "<AK_SLOT>"
AK_SLOT_NAME = "<AK_SLOT_NAME>"
AK_SLOT_VALUE = "<AK_SLOT_VALUE>"
AK_USER = "<AK_USER>"


def _compact(value: object, *, limit: int = 5000) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].rstrip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _decision(action: str, content: object, metadata: dict[str, Any]) -> str:
    return json.dumps(
        {
            "action": action,
            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True),
            "proposal_metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _row(
    *,
    source_id: str,
    task_type: str,
    encoder_text: str,
    action: str,
    content: object,
    metadata: dict[str, Any],
    weight: float,
    intent_label: str,
    negative: str = "",
) -> dict[str, Any]:
    decoder_text = _decision(action, content, {"task_type": task_type, **metadata})
    example_id = hashlib.sha256(f"{source_id}\n{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
    out = {
        "example_id": example_id,
        "source_type": "pocketpal_stage51_local_agentkernel_trace_curriculum",
        "source_id": source_id,
        "task_type": task_type,
        "encoder_text": encoder_text,
        "decoder_text": decoder_text,
        "action": action,
        "source_action": action,
        "intent_label": intent_label,
        "retrieval_query_text": "",
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "weight": float(weight),
    }
    if negative:
        out["negative_decoder_text"] = negative
        out["negative_loss_weight"] = 1.0
    return out


def _base_context() -> str:
    return (
        f"{AK_CHAT} {AK_RESPOND} PocketPal local controller trace example.\n"
        f"{AK_PROFILE} Local-first private assistant. Follow the active agent contract, use the current source text, "
        "and choose a compact structured action.\n"
    )


def _episode_examples(path: Path, *, max_steps: int, weight: float) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return []
    prompt = _compact(payload.get("prompt"), limit=1400)
    if not prompt:
        return []
    task_id = _compact(payload.get("task_id") or path.stem, limit=160)
    contract = payload.get("task_contract") if isinstance(payload.get("task_contract"), dict) else {}
    metadata = payload.get("task_metadata") if isinstance(payload.get("task_metadata"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    plan = [str(item) for item in (payload.get("plan") or []) if str(item).strip()][:6]
    expected_files = [str(item) for item in (contract.get("expected_files") or []) if str(item).strip()][:8]
    forbidden_files = [str(item) for item in (contract.get("forbidden_files") or []) if str(item).strip()][:6]
    suggested_commands = [str(item) for item in (contract.get("suggested_commands") or []) if str(item).strip()][:4]
    success = bool(payload.get("success"))
    family = _compact(metadata.get("benchmark_family") or contract.get("metadata", {}).get("benchmark_family"), limit=80)
    difficulty = _compact(metadata.get("difficulty"), limit=80)
    task_summary = {
        "task_id": task_id,
        "family": family,
        "difficulty": difficulty,
        "expected_files": expected_files,
        "forbidden_files": forbidden_files,
        "success": success,
    }
    contract_block = json.dumps(task_summary, ensure_ascii=False, sort_keys=True)
    rows: list[dict[str, Any]] = []

    encoder = (
        f"{_base_context()}"
        "<AK_AGENT_ACTIVE>\n"
        "Agent name: Planner\n"
        "Agent instruction: Turn the user goal into a short executable plan. Do not invent requirements.\n"
        "Retrieval policy: use_current_context_first\n"
        "Tool policy: ask_before_extensions\n"
        "Action policy: respond_or_ask\n"
        "</AK_AGENT_ACTIVE>\n"
        f"{AK_CONTEXT} Task contract: {contract_block}\n"
        f"{AK_CONTEXT} Suggested command examples: {json.dumps(suggested_commands, ensure_ascii=False)}\n"
        f"{AK_USER} {prompt}\n"
        "Return compact JSON with action=respond and the plan content."
    )
    plan_content = "\n".join(f"- {item}" for item in (plan or ["identify required artifacts", "perform the smallest safe action", "verify against the contract"]))
    rows.append(
        _row(
            source_id=f"{task_id}:plan",
            task_type="active_agent_plan",
            encoder_text=encoder,
            action="respond",
            content=plan_content,
            metadata={"family": family, "source_path": str(path)},
            weight=weight,
            intent_label="plan",
            negative=_decision("respond", "I can help with that. What would you like me to do?", {"task_type": "wrong_friendly_chat"}),
        )
    )

    verify_content = {
        "expected_files": expected_files,
        "forbidden_files": forbidden_files,
        "success_command": _compact(contract.get("success_command"), limit=700),
        "needs_verification": True,
    }
    encoder = (
        f"{_base_context()}"
        "<AK_AGENT_ACTIVE>\n"
        "Agent name: Verifier\n"
        "Agent instruction: Extract the verification checklist from the current task contract.\n"
        "Retrieval policy: current_context_only\n"
        "Tool policy: no_tools\n"
        "Action policy: respond_or_ask\n"
        "</AK_AGENT_ACTIVE>\n"
        f"{AK_CONTEXT} Task contract: {json.dumps(contract, ensure_ascii=False, sort_keys=True)[:3500]}\n"
        f"{AK_USER} What should be checked before saying this task is done?\n"
        "Return compact JSON with action=respond and checklist content."
    )
    rows.append(
        _row(
            source_id=f"{task_id}:verify",
            task_type="active_agent_extract",
            encoder_text=encoder,
            action="respond",
            content=verify_content,
            metadata={"family": family, "source_path": str(path)},
            weight=weight * 1.1,
            intent_label="extract",
            negative=_decision("respond", {"expected_files": [], "needs_verification": False}, {"task_type": "ignored_contract"}),
        )
    )

    steps = [step for step in (payload.get("steps") or []) if isinstance(step, dict)]
    for index, step in enumerate(steps[:max_steps], start=1):
        action = _compact(step.get("action"), limit=80)
        content = _compact(step.get("content") or step.get("command_result", {}).get("command"), limit=1200)
        decision_source = _compact(step.get("decision_source"), limit=80)
        failure_origin = _compact(step.get("failure_origin"), limit=120)
        progress = step.get("latent_state_summary", {}).get("completion_ratio")
        if not content:
            continue
        step_context = {
            "task_id": task_id,
            "step": index,
            "prior_action": action,
            "decision_source": decision_source,
            "failure_origin": failure_origin,
            "completion_ratio": progress,
        }
        encoder = (
            f"{_base_context()}"
            "<AK_AGENT_ACTIVE>\n"
            "Agent name: Action Auditor\n"
            "Agent instruction: Summarize the next safe action and whether verification is required.\n"
            "Retrieval policy: use_current_context_first\n"
            "Tool policy: ask_before_extensions\n"
            "Action policy: respond_or_ask\n"
            "</AK_AGENT_ACTIVE>\n"
            f"{AK_CONTEXT} Task contract summary: {contract_block}\n"
            f"{AK_CONTEXT} Observed step: {json.dumps(step_context, ensure_ascii=False, sort_keys=True)}\n"
            f"{AK_SLOT} {AK_SLOT_NAME}=CURRENT_ACTION {AK_SLOT_VALUE}={content}\n"
            f"{AK_USER} Review this action for the task.\n"
            "Return compact JSON with action=respond and concise audit content."
        )
        audit = {
            "action": action or "unknown",
            "content": content,
            "requires_verification": True,
            "safe_if_contract_aligned": not bool(step.get("command_governance", {}).get("blocked")),
        }
        rows.append(
            _row(
                source_id=f"{task_id}:step:{index}",
                task_type="active_agent_summarize",
                encoder_text=encoder,
                action="respond",
                content=audit,
                metadata={"family": family, "source_path": str(path), "decision_source": decision_source},
                weight=weight * 0.8,
                intent_label="summarize",
                negative=_decision("respond", "Looks good.", {"task_type": "underverified_action"}),
            )
        )

    if not success:
        failure_signals = summary.get("failure_signals") or summary.get("failure_types") or []
        encoder = (
            f"{_base_context()}"
            "<AK_AGENT_ACTIVE>\n"
            "Agent name: Recovery Planner\n"
            "Agent instruction: Given a failed task trace, identify the likely failure and next recovery step.\n"
            "Retrieval policy: use_failure_memory\n"
            "Tool policy: ask_before_extensions\n"
            "Action policy: respond_or_ask\n"
            "</AK_AGENT_ACTIVE>\n"
            f"{AK_CONTEXT} Failed task summary: {json.dumps(summary, ensure_ascii=False, sort_keys=True)[:3000]}\n"
            f"{AK_CONTEXT} Task contract: {contract_block}\n"
            f"{AK_USER} The attempt failed. What should the agent do next?\n"
            "Return compact JSON with action=respond and recovery content."
        )
        recovery = {
            "failure_signals": failure_signals,
            "next_step": "compare current state against the task contract, repair only missing or wrong artifacts, then verify",
            "avoid": "do not declare success without contract evidence",
        }
        rows.append(
            _row(
                source_id=f"{task_id}:recovery",
                task_type="active_agent_plan",
                encoder_text=encoder,
                action="respond",
                content=recovery,
                metadata={"family": family, "source_path": str(path)},
                weight=weight * 1.2,
                intent_label="plan",
                negative=_decision("respond", "The task is complete.", {"task_type": "false_success"}),
            )
        )
    return rows


def _strategy_examples(path: Path, *, max_rows: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        lesson = _compact(node.get("analysis_lesson"), limit=1000)
        motivation = _compact(node.get("motivation"), limit=800)
        subsystem = _compact(node.get("subsystem"), limit=80)
        retention_state = _compact(node.get("retention_state"), limit=80)
        reuse = [_compact(item, limit=220) for item in (node.get("reuse_conditions") or []) if str(item).strip()][:4]
        avoid = [_compact(item, limit=220) for item in (node.get("avoid_conditions") or []) if str(item).strip()][:4]
        if not (lesson or reuse or avoid):
            continue
        context = {
            "subsystem": subsystem,
            "retention_state": retention_state,
            "motivation": motivation,
            "lesson": lesson,
            "reuse": reuse,
            "avoid": avoid,
        }
        encoder = (
            f"{_base_context()}"
            "<AK_AGENT_ACTIVE>\n"
            "Agent name: Memory User\n"
            "Agent instruction: Use strategy memory to choose whether to reuse, avoid, or verify a prior approach.\n"
            "Retrieval policy: use_memory_when_relevant\n"
            "Tool policy: ask_before_extensions\n"
            "Action policy: respond_or_ask\n"
            "</AK_AGENT_ACTIVE>\n"
            f"{AK_CONTEXT} Strategy memory: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
            f"{AK_USER} Should this prior approach be reused for the current task?\n"
            "Return compact JSON with action=respond and a decision."
        )
        decision = {
            "reuse": retention_state in {"retained", "accepted"},
            "verify_before_reuse": True,
            "subsystem": subsystem,
            "avoid_conditions": avoid,
            "reason": lesson or motivation,
        }
        rows.append(
            _row(
                source_id=f"{path.parent.parent.name}:{path.stem}:{index}",
                task_type="active_agent_memory_use",
                encoder_text=encoder,
                action="respond",
                content=decision,
                metadata={"source_path": str(path), "subsystem": subsystem},
                weight=weight,
                intent_label="memory",
                negative=_decision("respond", {"reuse": True, "verify_before_reuse": False}, {"task_type": "unsafe_memory_reuse"}),
            )
        )
    return rows


def _hash_split(example_id: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, eval_fraction)) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="tmp/pocketpal_v208e_failure_replay_from_v208d/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--agentkernel-root", default="/data/agentkernel/var")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage51_local_agentkernel_trace_curriculum")
    parser.add_argument("--max-episodes", type=int, default=160)
    parser.add_argument("--max-steps-per-episode", type=int, default=3)
    parser.add_argument("--max-strategy-rows", type=int, default=600)
    parser.add_argument("--local-weight", type=float, default=4.0)
    parser.add_argument("--eval-fraction", type=float, default=0.04)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    base_manifest = _load_json(Path(args.base_manifest).resolve())
    train_rows = list(_iter_jsonl(Path(base_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(base_manifest["eval_dataset_path"])))
    local_rows: list[dict[str, Any]] = []
    root = Path(args.agentkernel_root).resolve()
    episode_paths = sorted(root.glob("self_improve_*/episodes/*.json"))[: int(args.max_episodes)]
    for path in episode_paths:
        try:
            local_rows.extend(_episode_examples(path, max_steps=int(args.max_steps_per_episode), weight=float(args.local_weight)))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    strategy_paths = sorted(root.glob("self_improve_*/strategy_memory/nodes.jsonl"))
    per_strategy = max(1, int(args.max_strategy_rows) // max(1, len(strategy_paths)))
    for path in strategy_paths:
        try:
            local_rows.extend(_strategy_examples(path, max_rows=per_strategy, weight=float(args.local_weight) * 0.8))
        except OSError:
            continue

    dedup: dict[str, dict[str, Any]] = {}
    for row in [*train_rows, *eval_rows, *local_rows]:
        key = str(row.get("example_id") or hashlib.sha256(f"{row.get('encoder_text','')}-->{row.get('decoder_text','')}".encode("utf-8")).hexdigest())
        row["example_id"] = key
        dedup[key] = row
    combined_train: list[dict[str, Any]] = []
    combined_eval: list[dict[str, Any]] = []
    local_count = 0
    for row in sorted(dedup.values(), key=lambda item: str(item.get("example_id", ""))):
        if row.get("source_type") == "pocketpal_stage51_local_agentkernel_trace_curriculum":
            local_count += 1
            split = _hash_split(str(row["example_id"]), float(args.eval_fraction))
            row["split"] = split
        else:
            split = str(row.get("split") or "train")
        if split == "eval":
            combined_eval.append(row)
        else:
            combined_train.append(row)

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, combined_train)
    _write_jsonl(eval_path, combined_eval)
    task_counts: dict[str, int] = {}
    for row in [*combined_train, *combined_eval]:
        task = str(row.get("task_type") or "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage51_local_agentkernel_trace_curriculum",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(combined_train) + len(combined_eval),
        "train_examples": len(combined_train),
        "eval_examples": len(combined_eval),
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "local_examples": local_count,
        "episode_paths": len(episode_paths),
        "strategy_paths": len(strategy_paths),
        "task_type_counts": dict(sorted(task_counts.items())),
        "source_counts": {
            "base_replay": len(train_rows) + len(eval_rows),
            "pocketpal_stage51_local_agentkernel_trace_curriculum": local_count,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
