#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


WEAK_TASKS = {"active_agent_extraction", "active_agent_json", "active_agent_translation"}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash_float(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _copy(row: dict[str, Any], suffix: str, weight_floor: float, weight_cap: float, source: str) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = f"{row.get('example_id') or row.get('source_id')}_{suffix}"
    out["source_type"] = f"{row.get('source_type', 'unknown')}_{source}"
    out["weight"] = min(max(float(row.get("weight") or 1.0), weight_floor), weight_cap)
    return out


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--failure-json", default="tmp/v192b_direct_agent_prompts.json")
    parser.add_argument("--slot-copy-manifest", default="tmp/pocketpal_v186_slot_copy_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--protocol-manifest", default="tmp/pocketpal_v193_protocol_cleanup/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weak-repeat", type=int, default=3)
    parser.add_argument("--failure-repeat", type=int, default=36)
    parser.add_argument("--max-weak-train", type=int, default=32000)
    parser.add_argument("--max-weak-eval", type=int, default=5000)
    parser.add_argument("--retrieval-protect", type=int, default=3000)
    args = parser.parse_args()

    broad = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    slot = json.loads(Path(args.slot_copy_manifest).read_text(encoding="utf-8"))
    rewrite = json.loads(Path(args.rewrite_slot_manifest).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol_manifest).read_text(encoding="utf-8"))
    retrieval = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
    failure_report = json.loads(Path(args.failure_json).read_text(encoding="utf-8"))
    bad_by_source = {str(item.get("source_id")): str(item.get("output") or "") for item in failure_report.get("failures", [])}

    broad_train = list(_iter_jsonl(Path(broad["train_dataset_path"])))
    broad_eval = list(_iter_jsonl(Path(broad["eval_dataset_path"])))
    by_source: dict[str, dict[str, Any]] = {}
    for row in broad_train + broad_eval:
        source_id = str(row.get("source_id") or row.get("example_id") or "")
        if source_id:
            by_source.setdefault(source_id, row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    weak_train = [row for row in broad_train if str(row.get("task_type") or "") in WEAK_TASKS]
    weak_train.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
    weak_eval = [row for row in broad_eval if str(row.get("task_type") or "") in WEAK_TASKS]
    weak_eval.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))

    for repeat in range(int(args.weak_repeat)):
        for index, row in enumerate(weak_train[: int(args.max_weak_train)]):
            train_rows.append(_copy(row, f"v196_weak_{repeat:02d}_{index:06d}", 30.0, 68.0, "v196_weak_skill_drill"))
    for index, row in enumerate(weak_eval[: int(args.max_weak_eval)]):
        eval_rows.append(_copy(row, f"v196_eval_{index:06d}", 20.0, 50.0, "v196_weak_skill_eval"))

    for repeat in range(int(args.failure_repeat)):
        for source_id, bad_output in sorted(bad_by_source.items()):
            row = by_source.get(source_id)
            if row is None or str(row.get("task_type") or "") not in WEAK_TASKS:
                continue
            out = _copy(row, f"v196_fail_{repeat:02d}", 70.0, 95.0, "v196_direct_failure")
            if bad_output.strip():
                out["negative_decoder_text"] = bad_output
                out["negative_loss_weight"] = 0.85
            train_rows.append(out)

    for manifest, label, repeat_count, floor, cap in [
        (slot, "slot", 2, 14.0, 38.0),
        (rewrite, "rewrite", 5, 18.0, 46.0),
        (protocol, "protocol", 1, 20.0, 60.0),
    ]:
        for repeat in range(repeat_count):
            for path_key, dest in [("train_dataset_path", train_rows), ("eval_dataset_path", eval_rows)]:
                for row in _iter_jsonl(Path(manifest[path_key])):
                    dest.append(_copy(row, f"v196_{label}_{repeat:02d}", floor, cap, f"v196_{label}_protect"))

    added = 0
    for row in _iter_jsonl(Path(retrieval["train_dataset_path"])):
        if added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            out = dict(row)
            out["example_id"] = f"{row.get('example_id')}_v196_retrieval"
            out["source_type"] = "v182_retrieval_protection_v196"
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
        "objective": "pocketpal_v196_weak_skill_drill",
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
