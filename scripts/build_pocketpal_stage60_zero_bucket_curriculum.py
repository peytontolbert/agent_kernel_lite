#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ZERO_BUCKETS = {
    "active_agent_json": 9000,
    "active_agent_plan": 7000,
    "active_agent_translation": 9000,
    "active_agent_extraction": 12000,
    "active_agent_brainstorm": 5000,
}

ANCHOR_BUCKETS = {
    "active_agent_action_items": 2500,
    "active_agent_checklist": 2500,
    "active_agent_rewrite": 3000,
    "active_agent_risks": 2500,
    "active_agent_subject": 2500,
    "active_agent_summary": 3000,
}

NEGATIVE_STALE = {
    "active_agent_json": '{"action":"respond","content":"Security Review and Launch Blocker","proposal_metadata":{"task_type":"active_agent_json"}}',
    "active_agent_plan": '{"action":"respond","content":"- Send the launch memo by Monday\\n- Review the launch memo\\n- Resolve legal approval","proposal_metadata":{"task_type":"active_agent_plan"}}',
    "active_agent_translation": '{"action":"respond","content":"La reunion se ha cambiado al viernes.","proposal_metadata":{"task_type":"active_agent_translation"}}',
    "active_agent_extraction": '{"action":"respond","content":"- Owner: Devon\\n- Reviewer: Harper\\n- Object: launch memo\\n- Date: Monday\\n- Blocker: legal approval","proposal_metadata":{"task_type":"active_agent_extraction"}}',
    "active_agent_brainstorm": '{"action":"respond","content":"1. Send the launch memo\\n2. Review security approval\\n3. Resolve QA signoff","proposal_metadata":{"task_type":"active_agent_brainstorm"}}',
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _decision_content(text: str) -> str:
    try:
        parsed = json.loads(str(text or ""))
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    decision = parsed.get("decision_packet", {}).get("decision")
    if isinstance(decision, dict):
        parsed = decision
    return str(parsed.get("content", "") or "").strip()


def _sample_by_task(rows_by_task: dict[str, list[dict[str, Any]]], task_counts: dict[str, int], rng: random.Random) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for task, count in task_counts.items():
        rows = list(rows_by_task.get(task, []))
        if not rows:
            continue
        rng.shuffle(rows)
        if len(rows) >= count:
            picked = rows[:count]
        else:
            picked = [rows[i % len(rows)] for i in range(count)]
        sampled.extend(picked)
    return sampled


def _prepare(row: dict[str, Any], *, emphasis: str) -> dict[str, Any]:
    out = dict(row)
    task = str(out.get("task_type", "") or "")
    content = str(out.get("expected_content") or "").strip()
    if not content:
        content = _decision_content(str(out.get("decoder_text", "") or ""))
    if content:
        out["expected_content"] = content
        out["state_text"] = content
    if task in ZERO_BUCKETS:
        out["weight"] = max(float(out.get("weight", 1.0) or 1.0), 8.0)
        out["negative_decoder_text"] = str(out.get("negative_decoder_text") or NEGATIVE_STALE.get(task) or "")
        out["negative_loss_weight"] = max(float(out.get("negative_loss_weight", 0.0) or 0.0), 0.8)
        out["source_type"] = "pocketpal_stage60_zero_bucket_curriculum"
    else:
        out["weight"] = max(float(out.get("weight", 1.0) or 1.0), 2.0)
        out["source_type"] = "pocketpal_stage60_anchor_curriculum"
    out["source_id"] = f"stage60_{emphasis}_{out.get('source_id', out.get('example_id', 'row'))}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage60_zero_bucket_curriculum")
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    base_manifest = json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    train_path = Path(base_manifest["train_dataset_path"])
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wanted = set(ZERO_BUCKETS) | set(ANCHOR_BUCKETS)
    for row in _iter_jsonl(train_path):
        task = str(row.get("task_type", "") or "")
        if task in wanted:
            rows_by_task[task].append(row)

    rows = []
    rows.extend(_prepare(row, emphasis="zero") for row in _sample_by_task(rows_by_task, ZERO_BUCKETS, rng))
    rows.extend(_prepare(row, emphasis="anchor") for row in _sample_by_task(rows_by_task, ANCHOR_BUCKETS, rng))
    rng.shuffle(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_count = max(1, int(len(rows) * float(args.eval_ratio)))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    train_out = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_out = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    with train_out.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with eval_out.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    task_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        task_counts[str(row.get("task_type", "unknown") or "unknown")] += 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_stage60_zero_bucket_curriculum",
        "source_manifest_path": str(Path(args.base_manifest).resolve()),
        "manifest_path": str((output_dir / "agentkernel_lite_encdec_dataset_manifest.json").resolve()),
        "train_dataset_path": str(train_out.resolve()),
        "eval_dataset_path": str(eval_out.resolve()),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(rows),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
