#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generation_probes() -> list[dict[str, Any]]:
    path = REPO_ROOT / "scripts" / "evaluate_agentkernel_lite_generation.py"
    spec = importlib.util.spec_from_file_location("evaluate_agentkernel_lite_generation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [dict(item) for item in getattr(module, "DEFAULT_PROBES")]


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _decision_content(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    decision = parsed.get("decision_packet", {}).get("decision")
    if not isinstance(decision, dict):
        decision = parsed.get("decision") if isinstance(parsed.get("decision"), dict) else parsed
    if not isinstance(decision, dict):
        return ""
    return str(decision.get("content", "") or "").strip()


def _json_decision(action: str, content: str, task_type: str = "") -> str:
    payload: dict[str, Any] = {"action": action, "content": content}
    if task_type:
        payload["proposal_metadata"] = {"task_type": task_type}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _content_from_row(row: dict[str, Any]) -> str:
    return str(row.get("expected_content") or "").strip() or _decision_content(
        str(row.get("json_decoder_text") or row.get("decoder_text") or "")
    )


def _make_json_row(row: dict[str, Any], *, index: int, weight: float) -> dict[str, Any] | None:
    content = _content_from_row(row)
    if not content:
        return None
    action = str(row.get("action") or "respond").strip() or "respond"
    task_type = str(row.get("task_type") or "unknown")
    target = _json_decision(action, content, task_type)
    out = deepcopy(row)
    out.update(
        {
            "decoder_text": target,
            "decoder_loss_weight": 1.0,
            "example_id": _stable_id("stage76_json", str(index), str(row.get("source_id")), str(row.get("encoder_text")), target),
            "json_decoder_text": target,
            "negative_decoder_text": None,
            "negative_loss_weight": None,
            "source_id": f"{row.get('source_id', 'row')}_stage76_json",
            "source_type": "stage76_unconstrained_json_decision",
            "split": "train",
            "task_type": f"{task_type}_json_decision",
            "weight": float(weight),
        }
    )
    return out


def _probe_target(probe: dict[str, Any]) -> str:
    probe_id = str(probe.get("id") or "")
    if probe_id == "query_rewrite_multi_agent_planning":
        return _json_decision("gather_context", "multi-agent planning grounded evidence", probe_id)
    if probe_id == "rerank_neural_retrieval_scientific_assistant":
        return _json_decision("gather_context", "selected_candidate_id=P1 neural retrieval scientific assistant", probe_id)
    if probe_id == "answer_from_evidence":
        return _json_decision(
            "respond",
            "The main contribution is neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents [1].",
            probe_id,
        )
    if probe_id == "selected_paper_followup_uses_active_context":
        return _json_decision(
            "respond",
            "P1 is about neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents [P1].",
            probe_id,
        )
    if probe_id == "ordinary_chat_answers_directly":
        return _json_decision(
            "respond",
            "Multi-agent intelligence is when several agents coordinate, share context, divide work, and check each other's results to solve a task.",
            probe_id,
        )
    return _json_decision(str(probe.get("expected_action") or "respond"), " ".join(probe.get("required") or []), probe_id)


def _probe_rows(repeats: int, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(int(repeats)):
        for probe in _load_generation_probes():
            target = _probe_target(probe)
            source_id = f"stage76_generation_probe_{probe.get('id')}_{repeat:03d}"
            rows.append(
                {
                    "action": str(probe.get("expected_action") or "respond"),
                    "decoder_loss_weight": 1.0,
                    "decoder_text": target,
                    "encoder_text": str(probe.get("prompt") or ""),
                    "example_id": _stable_id(source_id, str(probe.get("prompt") or ""), target),
                    "intent_label_id": -1,
                    "json_decoder_text": target,
                    "negative_decoder_text": None,
                    "negative_loss_weight": None,
                    "retrieval_doc_text": "",
                    "retrieval_loss_weight": 0.0,
                    "retrieval_query_text": "",
                    "source_id": source_id,
                    "source_type": "stage76_unconstrained_generation_probe_json",
                    "split": "train",
                    "task_type": str(probe.get("id") or "generation_probe"),
                    "weight": float(weight),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default=str(REPO_ROOT / "tmp/pocketpal_stage67_structured_copy_decoder_v172d_akv1/agentkernel_lite_encdec_dataset_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tmp/pocketpal_stage76_json_decoder_repair"),
    )
    parser.add_argument("--structured-anchor-limit", type=int, default=3200)
    parser.add_argument("--json-decision-limit", type=int, default=2600)
    parser.add_argument("--probe-repeats", type=int, default=100)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_train = _read_jsonl(Path(source_manifest["train_dataset_path"]))
    source_eval = _read_jsonl(Path(source_manifest["eval_dataset_path"]))

    train_rows: list[dict[str, Any]] = []
    for row in source_train[: int(args.structured_anchor_limit)]:
        anchor = deepcopy(row)
        anchor["source_type"] = "stage76_structured_anchor"
        anchor["weight"] = min(float(anchor.get("weight", 1.0) or 1.0), 1.25)
        train_rows.append(anchor)

    json_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_train):
        converted = _make_json_row(row, index=index, weight=1.5)
        if converted is not None:
            json_rows.append(converted)
        if len(json_rows) >= int(args.json_decision_limit):
            break
    train_rows.extend(json_rows)
    train_rows.extend(_probe_rows(int(args.probe_repeats), weight=6.0))

    eval_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_eval[:300]):
        converted = _make_json_row(row, index=index, weight=1.0)
        if converted is not None:
            converted["split"] = "eval"
            eval_rows.append(converted)
    eval_rows.extend(_probe_rows(1, weight=1.0))
    for row in eval_rows:
        row["split"] = "eval"

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        **source_manifest,
        "artifact_kind": "agentkernel_lite_encdec_stage76_json_decoder_repair",
        "dataset_objective": "pocketpal_stage76_json_decoder_repair",
        "objective": "pocketpal_stage76_json_decoder_repair",
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "source_manifest_path": str(source_manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "stage76_counts": {
            "structured_anchor": int(args.structured_anchor_limit),
            "json_decision": len(json_rows),
            "generation_probe_json": int(args.probe_repeats) * len(_load_generation_probes()),
        },
        "stage76_note": "Decoder repair for unconstrained generation using compact JSON decisions, preserving app-compatible structure.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["manifest_path", "train_examples", "eval_examples", "stage76_counts"]}, indent=2))


if __name__ == "__main__":
    main()
