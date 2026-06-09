#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


WEAK_TASKS = {
    "active_agent_extraction",
    "active_agent_json",
    "active_agent_translation",
}
SUPPORT_TASKS = {
    "active_agent_action_items",
    "active_agent_checklist",
    "active_agent_risks",
    "active_agent_summary",
    "active_agent_subject",
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash_float(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _normalize(row: dict[str, Any], suffix: str, floor: float, cap: float, source: str) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = f"{row.get('example_id') or row.get('source_id')}_{suffix}"
    out["source_type"] = f"{row.get('source_type', 'unknown')}_{source}"
    out["weight"] = min(max(float(row.get("weight") or 1.0), floor), cap)
    return out


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--failure-json", default="tmp/v191b_direct_agent_prompts.json")
    parser.add_argument("--slot-copy-manifest", default="tmp/pocketpal_v186_slot_copy_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--v191-manifest", default="tmp/pocketpal_v191_action_summary_risks_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-weak-train", type=int, default=26000)
    parser.add_argument("--max-support-train", type=int, default=12000)
    parser.add_argument("--max-weak-eval", type=int, default=4000)
    parser.add_argument("--slot-repeat", type=int, default=3)
    parser.add_argument("--rewrite-repeat", type=int, default=6)
    parser.add_argument("--failure-repeat", type=int, default=18)
    parser.add_argument("--retrieval-protect", type=int, default=3000)
    args = parser.parse_args()

    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    slot_manifest = json.loads(Path(args.slot_copy_manifest).read_text(encoding="utf-8"))
    rewrite_manifest = json.loads(Path(args.rewrite_slot_manifest).read_text(encoding="utf-8"))
    v191_manifest = json.loads(Path(args.v191_manifest).read_text(encoding="utf-8"))
    retrieval_manifest = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
    failure_report = json.loads(Path(args.failure_json).read_text(encoding="utf-8"))
    bad_by_source = {str(item.get("source_id")): str(item.get("output") or "") for item in failure_report.get("failures", [])}

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    broad_train = list(_iter_jsonl(Path(broad_manifest["train_dataset_path"])))
    broad_eval = list(_iter_jsonl(Path(broad_manifest["eval_dataset_path"])))

    by_source: dict[str, dict[str, Any]] = {}
    for row in broad_train + broad_eval:
        source_id = str(row.get("source_id") or row.get("example_id") or "")
        if source_id and source_id not in by_source:
            by_source[source_id] = row

    for repeat in range(int(args.failure_repeat)):
        for source_id, bad_output in sorted(bad_by_source.items()):
            row = by_source.get(source_id)
            if row is None:
                continue
            out = _normalize(row, f"v192_fail_{repeat:02d}", 35.0, 58.0, "v192_failure_negative")
            if bad_output.strip():
                out["negative_decoder_text"] = bad_output
                out["negative_loss_weight"] = 0.7
            train_rows.append(out)

    weak = [row for row in broad_train if str(row.get("task_type") or "") in WEAK_TASKS]
    weak.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
    for index, row in enumerate(weak[: int(args.max_weak_train)]):
        train_rows.append(_normalize(row, f"v192_weak_{index:06d}", 9.0, 22.0, "v192_weak_task"))

    support = [row for row in broad_train if str(row.get("task_type") or "") in SUPPORT_TASKS]
    support.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
    for index, row in enumerate(support[: int(args.max_support_train)]):
        train_rows.append(_normalize(row, f"v192_support_{index:06d}", 5.0, 14.0, "v192_support_task"))

    weak_eval = [row for row in broad_eval if str(row.get("task_type") or "") in (WEAK_TASKS | SUPPORT_TASKS)]
    weak_eval.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
    for index, row in enumerate(weak_eval[: int(args.max_weak_eval)]):
        eval_rows.append(_normalize(row, f"v192_eval_{index:06d}", 5.0, 18.0, "v192_eval"))

    for manifest, label, repeat_count, floor, cap in [
        (slot_manifest, "slot", int(args.slot_repeat), 14.0, 38.0),
        (rewrite_manifest, "rewrite", int(args.rewrite_repeat), 18.0, 46.0),
        (v191_manifest, "v191", 1, 6.0, 18.0),
    ]:
        for repeat in range(repeat_count):
            for row in _iter_jsonl(Path(manifest["train_dataset_path"])):
                train_rows.append(_normalize(row, f"v192_{label}_{repeat:02d}", floor, cap, f"v192_{label}_protect"))
            for row in _iter_jsonl(Path(manifest["eval_dataset_path"])):
                eval_rows.append(_normalize(row, f"v192_{label}_{repeat:02d}", floor, cap, f"v192_{label}_protect"))

    added = 0
    for row in _iter_jsonl(Path(retrieval_manifest["train_dataset_path"])):
        if added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            out = dict(row)
            out["example_id"] = f"{row.get('example_id')}_v192_retrieval"
            out["source_type"] = "v182_retrieval_protection_v192"
            out["weight"] = 0.0
            train_rows.append(out)
            added += 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, train_rows)
    _write(eval_path, eval_rows)
    all_rows = train_rows + eval_rows
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "failure_sources": len(bad_by_source),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v192_failure_negative_repair",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
