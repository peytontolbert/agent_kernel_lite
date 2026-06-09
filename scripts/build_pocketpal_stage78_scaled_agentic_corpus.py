#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

INTENT_LABELS = {
    "plan": 0,
    "action_items": 1,
    "rewrite": 2,
    "translation": 3,
    "web_search": 4,
    "casual": 5,
    "source_echo": 6,
    "saved_data": 7,
    "ask_user": 8,
    "summary": 9,
    "title": 10,
    "checklist": 11,
    "risks": 12,
    "json": 13,
    "ranking": 14,
    "extraction": 15,
    "subject": 16,
    "brainstorm": 17,
}


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _compact(value: object, *, limit: int = 1200) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _ak_decision(task_type: str, content: str, *, action: str = "respond") -> str:
    return (
        f"<AK_STRUCTURED> <AK_ACTION_{action.upper()}> <AK_TASK_TYPE> {task_type} "
        f"<AK_CONTENT> {_compact(content, limit=900)} </AK_CONTENT> <AK_END>"
    )


def _json_decision(task_type: str, content: str, *, action: str = "respond") -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _prompt(*, agent_name: str, instruction: str, intent: str, task_type: str, user: str, context: str = "") -> str:
    return (
        "<AK_CHAT> <AK_RESPOND> PocketPal scaled agentic corpus example.\n"
        "<AK_AGENT_ACTIVE>\n"
        f"Agent name: {agent_name}\n"
        f"Agent instruction: {instruction}\n"
        "Retrieval policy: auto\n"
        "Tool policy: ask_before_extensions\n"
        "Action policy: respond_or_ask\n"
        "The active agent instruction is the primary task contract for this turn.\n"
        "</AK_AGENT_ACTIVE>\n"
        f"<AK_TASK_HINT> intent={intent} task={task_type} source_text_required=true\n"
        "<AK_CONTEXT> Saved user data: none\n"
        f"{('<AK_CONTEXT> ' + context + chr(10)) if context else ''}"
        f"<AK_USER> {user}\n"
        "Return AK structured tokens for the active agent decision."
    )


def _base_row(
    *,
    source_id: str,
    source_type: str,
    task_type: str,
    intent: str,
    encoder_text: str,
    content: str,
    weight: float,
    retrieval_query_text: str = "",
    retrieval_doc_text: str = "",
    state_text: str = "",
    split: str = "train",
) -> dict[str, Any] | None:
    content = _compact(content, limit=1000)
    if not content or intent not in INTENT_LABELS:
        return None
    decoder_text = _ak_decision(task_type, content)
    return {
        "action": "respond",
        "decoder_loss_weight": 1.0,
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": _stable_id("stage78", source_id, encoder_text, decoder_text),
        "expected_content": content,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS[intent],
        "json_decoder_text": _json_decision(task_type, content),
        "negative_decoder_text": None,
        "negative_loss_weight": None,
        "retrieval_doc_text": retrieval_doc_text,
        "retrieval_loss_weight": 1.0 if retrieval_query_text and retrieval_doc_text else 0.0,
        "retrieval_query_text": retrieval_query_text,
        "source_id": source_id,
        "source_type": source_type,
        "split": split,
        "state_text": state_text,
        "task_type": task_type,
        "weight": float(weight),
    }


def _messages_to_rows(path: Path, *, max_rows: int, repeat: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        system = ""
        user = ""
        assistant = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = _compact(message.get("content"), limit=1200)
            if role == "system" and not system:
                system = content
            elif role == "user" and not user:
                user = content
            elif role == "assistant" and not assistant:
                assistant = content
        noisy = assistant.lower()
        if not user or not assistant or len(assistant) < 8:
            continue
        if any(
            marker in noisy
            for marker in (
                "vllm request failed",
                "inference_failure",
                "traceback",
                "connection refused",
                "error after",
            )
        ):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source = _compact(metadata.get("source_id") or path.name, limit=220)
        context = _compact(system or "local agentkernel supervised trace", limit=500)
        encoder = _prompt(
            agent_name="Local Knowledge Agent",
            instruction="Answer or continue from the provided local agent-kernel trace. Keep the response concise and grounded.",
            intent="source_echo",
            task_type="active_agent_source_echo",
            user=user,
            context=f"Trace source: {source}. System context: {context}",
        )
        for rep in range(repeat):
            made = _base_row(
                source_id=f"{source}:qwen:{index:06d}:{rep:02d}",
                source_type="stage78_qwen_adapter_trace",
                task_type="active_agent_source_echo",
                intent="source_echo",
                encoder_text=encoder,
                content=assistant,
                weight=weight,
                retrieval_query_text=user,
                retrieval_doc_text=assistant,
                state_text=f"source={source} difficulty={row.get('difficulty','')}",
            )
            if made:
                rows.append(made)
    return rows


def _universal_rows(path: Path, *, max_rows: int, repeat: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        prompt = _compact(row.get("prompt"), limit=900)
        target = _compact(row.get("target"), limit=900)
        if not prompt or not target:
            continue
        source = _compact(row.get("source_id") or path.name, limit=220)
        encoder = _prompt(
            agent_name="Continuation Agent",
            instruction="Continue the local technical note using the current context. Do not add unrelated facts.",
            intent="source_echo",
            task_type="active_agent_source_echo",
            user=prompt,
            context=f"Source type: {_compact(row.get('source_type'), limit=80)}",
        )
        for rep in range(repeat):
            made = _base_row(
                source_id=f"{source}:universal:{index:06d}:{rep:02d}",
                source_type="stage78_universal_decoder_trace",
                task_type="active_agent_source_echo",
                intent="source_echo",
                encoder_text=encoder,
                content=target,
                weight=weight,
                retrieval_query_text=prompt,
                retrieval_doc_text=target,
                state_text=f"source={source}",
            )
            if made:
                rows.append(made)
    return rows


def _supervised_rows(path: Path, *, max_rows: int, repeat: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        prompt = _compact(row.get("prompt"), limit=1000)
        if not prompt:
            continue
        expected = [str(item) for item in (row.get("expected_files") or []) if str(item).strip()][:8]
        forbidden = [str(item) for item in (row.get("forbidden_files") or []) if str(item).strip()][:6]
        commands = [str(item) for item in (row.get("suggested_commands") or []) if str(item).strip()][:4]
        contract = json.dumps(
            {
                "difficulty": row.get("difficulty", ""),
                "family": row.get("benchmark_family", ""),
                "expected_files": expected,
                "forbidden_files": forbidden,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        plan_content = "\n".join(
            [
                "1. Read the task contract and preserve required existing files.",
                "2. Create or update only the expected artifacts.",
                "3. Verify expected files exist and forbidden files were not created.",
            ]
        )
        checklist = "\n".join([f"- expected: {item}" for item in expected] + [f"- forbidden: {item}" for item in forbidden])
        if commands:
            checklist += "\n- suggested command pattern: " + _compact(commands[0], limit=220)
        for intent, task_type, agent, instruction, user, content, scale in [
            (
                "plan",
                "active_agent_plan",
                "Task Planner",
                "Turn the task into a short safe execution plan.",
                prompt,
                plan_content,
                1.0,
            ),
            (
                "checklist",
                "active_agent_checklist",
                "Task Verifier",
                "Extract the verification checklist from the task contract.",
                "What should be checked before this task is complete?",
                checklist or "Verify the required task artifacts and avoid forbidden changes.",
                1.1,
            ),
        ]:
            encoder = _prompt(
                agent_name=agent,
                instruction=instruction,
                intent=intent,
                task_type=task_type,
                user=user,
                context=f"Task contract: {contract}",
            )
            for rep in range(repeat):
                made = _base_row(
                    source_id=f"{row.get('task_id','task')}:{intent}:{index:05d}:{rep:02d}",
                    source_type="stage78_agentkernel_supervised_task",
                    task_type=task_type,
                    intent=intent,
                    encoder_text=encoder,
                    content=content,
                    weight=weight * scale,
                    retrieval_query_text=prompt,
                    retrieval_doc_text=contract,
                    state_text=contract,
                )
                if made:
                    rows.append(made)
    return rows


def _policy_rows(path: Path, *, max_rows: int, repeat: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        content = _compact(row.get("content"), limit=700)
        if not content:
            continue
        failures = [str(item) for item in (row.get("failure_signals") or []) if str(item).strip()][:5]
        expected = [str(item) for item in (row.get("expected_files") or []) if str(item).strip()][:6]
        state = json.dumps(
            {
                "action": row.get("action", ""),
                "difficulty": row.get("difficulty", ""),
                "episode_success": row.get("episode_success", False),
                "expected_files": expected,
                "failure_signals": failures,
                "progress_delta": row.get("progress_delta", 0.0),
                "step_passed": row.get("step_passed", False),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        answer = (
            "Review the proposed action, verify it against expected files, and continue only if it advances the contract. "
            f"Current action: {content}"
        )
        encoder = _prompt(
            agent_name="Action Auditor",
            instruction="Assess whether a proposed local action advances the task and identify required verification.",
            intent="action_items",
            task_type="active_agent_action_items",
            user="What should the agent do next with this proposed action?",
            context=f"Policy step state: {state}",
        )
        for rep in range(repeat):
            made = _base_row(
                source_id=f"{row.get('task_id','policy')}:{index:05d}:{rep:02d}",
                source_type="stage78_agentkernel_policy_step",
                task_type="active_agent_action_items",
                intent="action_items",
                encoder_text=encoder,
                content=answer,
                weight=weight,
                retrieval_query_text=content,
                retrieval_doc_text=state,
                state_text=state,
            )
            if made:
                rows.append(made)
    return rows


def _strategy_rows(path: Path, *, max_rows: int, repeat: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        lesson = _compact(node.get("analysis_lesson"), limit=700)
        motivation = _compact(node.get("motivation"), limit=500)
        avoid = [_compact(item, limit=180) for item in (node.get("avoid_conditions") or []) if str(item).strip()][:4]
        reuse = [_compact(item, limit=180) for item in (node.get("reuse_conditions") or []) if str(item).strip()][:4]
        if not (lesson or avoid or reuse):
            continue
        state = json.dumps(
            {
                "avoid": avoid,
                "lesson": lesson,
                "motivation": motivation,
                "retention_state": node.get("retention_state", ""),
                "reuse": reuse,
                "subsystem": node.get("subsystem", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        content = "\n".join(
            [
                "Risks to watch before reusing this strategy:",
                *[f"- {item}" for item in avoid],
                f"- verify retention state before reuse: {node.get('retention_state','unknown')}",
            ]
        )
        encoder = _prompt(
            agent_name="Strategy Risk Agent",
            instruction="Summarize strategy-memory risks and when to avoid repeating a failed approach.",
            intent="risks",
            task_type="active_agent_risks",
            user="What risks should the agent consider before reusing this approach?",
            context=f"Strategy memory: {state}",
        )
        for rep in range(repeat):
            made = _base_row(
                source_id=f"{path.parent.parent.name}:strategy:{index:05d}:{rep:02d}",
                source_type="stage78_strategy_memory",
                task_type="active_agent_risks",
                intent="risks",
                encoder_text=encoder,
                content=content,
                weight=weight,
                retrieval_query_text=motivation or lesson,
                retrieval_doc_text=state,
                state_text=state,
            )
            if made:
                rows.append(made)
    return rows


def _split_row(row: dict[str, Any], eval_fraction: float) -> str:
    if str(row.get("split") or "") == "eval":
        return "eval"
    bucket = int(str(row["example_id"])[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < float(eval_fraction) else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage74_intent_boundary_curriculum_v287/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--agentkernel-var", default="/data/agentkernel/var")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_stage78_scaled_agentic_corpus"))
    parser.add_argument("--max-qwen-rows", type=int, default=9000)
    parser.add_argument("--max-universal-rows", type=int, default=2400)
    parser.add_argument("--max-supervised-rows", type=int, default=1200)
    parser.add_argument("--max-policy-rows", type=int, default=600)
    parser.add_argument("--max-strategy-rows", type=int, default=600)
    parser.add_argument("--qwen-repeat", type=int, default=3)
    parser.add_argument("--universal-repeat", type=int, default=3)
    parser.add_argument("--task-repeat", type=int, default=5)
    parser.add_argument("--strategy-repeat", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=7801)
    args = parser.parse_args()

    random.seed(int(args.seed))
    base_manifest = _read_json(Path(args.base_manifest).resolve())
    train_rows = _read_jsonl(Path(base_manifest["train_dataset_path"]))
    eval_rows = _read_jsonl(Path(base_manifest["eval_dataset_path"]))
    root = Path(args.agentkernel_var).resolve()

    additions: list[dict[str, Any]] = []
    per_qwen = max(1, int(args.max_qwen_rows))
    qwen_paths = sorted(root.glob("**/qwen_adapter_artifact/dataset/qwen_sft_train.jsonl"))
    per_file_qwen = max(1, per_qwen // max(1, len(qwen_paths)))
    for path in qwen_paths:
        additions.extend(_messages_to_rows(path, max_rows=per_file_qwen, repeat=int(args.qwen_repeat), weight=2.4))

    universal_paths = sorted(root.glob("**/universal_dataset/**/*decoder_train.jsonl"))
    per_file_universal = max(1, int(args.max_universal_rows) // max(1, len(universal_paths)))
    for path in universal_paths:
        additions.extend(_universal_rows(path, max_rows=per_file_universal, repeat=int(args.universal_repeat), weight=2.2))

    supervised_paths = sorted(root.glob("**/retained/tolbert_model/store/dataset/*/supervised_examples.jsonl"))
    per_file_supervised = max(1, int(args.max_supervised_rows) // max(1, len(supervised_paths)))
    for path in supervised_paths:
        additions.extend(_supervised_rows(path, max_rows=per_file_supervised, repeat=int(args.task_repeat), weight=3.4))

    policy_paths = sorted(root.glob("**/retained/tolbert_model/store/dataset/*/policy_examples.jsonl"))
    per_file_policy = max(1, int(args.max_policy_rows) // max(1, len(policy_paths)))
    for path in policy_paths:
        additions.extend(_policy_rows(path, max_rows=per_file_policy, repeat=int(args.task_repeat), weight=2.8))

    strategy_paths = sorted(root.glob("self_improve_*/strategy_memory/nodes.jsonl"))
    per_file_strategy = max(1, int(args.max_strategy_rows) // max(1, len(strategy_paths)))
    for path in strategy_paths:
        additions.extend(_strategy_rows(path, max_rows=per_file_strategy, repeat=int(args.strategy_repeat), weight=2.6))

    dedup: dict[str, dict[str, Any]] = {}
    for row in [*train_rows, *eval_rows]:
        copied = deepcopy(row)
        key = str(copied.get("example_id") or _stable_id(str(copied.get("encoder_text", "")), str(copied.get("decoder_text", ""))))
        copied["example_id"] = key
        dedup[key] = copied
    base_count = len(dedup)
    for row in additions:
        dedup[str(row["example_id"])] = row

    combined_train: list[dict[str, Any]] = []
    combined_eval: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    task_type_counts: dict[str, int] = {}
    for row in sorted(dedup.values(), key=lambda item: str(item.get("example_id", ""))):
        source = str(row.get("source_type") or "unknown")
        task = str(row.get("task_type") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        task_type_counts[task] = task_type_counts.get(task, 0) + 1
        split = _split_row(row, float(args.eval_fraction))
        row["split"] = split
        if split == "eval":
            combined_eval.append(row)
        else:
            combined_train.append(row)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, combined_train)
    _write_jsonl(eval_path, combined_eval)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_stage78_scaled_agentic_corpus",
        "base_examples": base_count,
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(combined_eval),
        "intent_labels": INTENT_LABELS,
        "local_additions_before_dedup": len(additions),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_stage78_scaled_agentic_corpus",
        "source_counts": dict(sorted(source_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "total_examples": len(combined_train) + len(combined_eval),
        "train_dataset_path": str(train_path),
        "train_examples": len(combined_train),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["manifest_path", "train_examples", "eval_examples", "local_additions_before_dedup"]}, indent=2))


if __name__ == "__main__":
    main()
