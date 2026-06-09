#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


WEAK_TASKS = {
    "active_agent_action_items",
    "active_agent_brainstorm",
    "active_agent_checklist",
    "active_agent_extraction",
    "active_agent_json",
    "active_agent_plan",
    "active_agent_risks",
    "active_agent_summary",
    "active_agent_translation",
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash_float(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


def _normalize(row: dict[str, Any], suffix: str, floor: float, cap: float, source: str) -> dict[str, Any]:
    out = dict(row)
    task = str(out.get("task_type") or "")
    out["example_id"] = f"{row.get('example_id') or row.get('source_id')}_{suffix}"
    out["source_type"] = f"{row.get('source_type', 'unknown')}_{source}"
    out["weight"] = min(max(float(row.get("weight") or 1.0), floor), cap)
    if not str(out.get("negative_decoder_text") or "").strip():
        wrong = ""
        if task in {"active_agent_extraction", "active_agent_action_items", "active_agent_checklist"}:
            wrong = _payload("respond", "Hello, I hope you are well.", "negative_greeting_attractor")
        elif task in {"active_agent_json", "active_agent_translation"}:
            wrong = _payload("respond", "Source text: [[SOURCE_TEXT]]", "negative_source_copy_attractor")
        elif task in {"active_agent_risks", "active_agent_summary"}:
            wrong = _payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "negative_rewrite_slot_attractor")
        if wrong:
            out["negative_decoder_text"] = wrong
            out["negative_loss_weight"] = 0.5
    return out


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--slot-copy-manifest", default="tmp/pocketpal_v186_slot_copy_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-focus-train", type=int, default=55000)
    parser.add_argument("--max-focus-eval", type=int, default=6000)
    parser.add_argument("--slot-repeat", type=int, default=2)
    parser.add_argument("--rewrite-repeat", type=int, default=5)
    parser.add_argument("--retrieval-protect", type=int, default=2500)
    args = parser.parse_args()

    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    slot_manifest = json.loads(Path(args.slot_copy_manifest).read_text(encoding="utf-8"))
    rewrite_manifest = json.loads(Path(args.rewrite_slot_manifest).read_text(encoding="utf-8"))
    retrieval_manifest = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for path_key, limit, dest, split in [
        ("train_dataset_path", int(args.max_focus_train), train_rows, "train"),
        ("eval_dataset_path", int(args.max_focus_eval), eval_rows, "eval"),
    ]:
        rows = [row for row in _iter_jsonl(Path(broad_manifest[path_key])) if str(row.get("task_type") or "") in WEAK_TASKS]
        rows.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
        for index, row in enumerate(rows[:limit]):
            dest.append(_normalize(row, f"v190_focus_{split}_{index:06d}", 8.0, 22.0, "v190_weak_focus"))

    for repeat in range(int(args.slot_repeat)):
        for row in _iter_jsonl(Path(slot_manifest["train_dataset_path"])):
            train_rows.append(_normalize(row, f"v190_slot_{repeat:02d}", 14.0, 38.0, "v190_slot_protect"))
        for row in _iter_jsonl(Path(slot_manifest["eval_dataset_path"])):
            eval_rows.append(_normalize(row, f"v190_slot_{repeat:02d}", 14.0, 38.0, "v190_slot_protect"))

    for repeat in range(int(args.rewrite_repeat)):
        for row in _iter_jsonl(Path(rewrite_manifest["train_dataset_path"])):
            train_rows.append(_normalize(row, f"v190_rewrite_{repeat:02d}", 18.0, 44.0, "v190_rewrite_protect"))
        for row in _iter_jsonl(Path(rewrite_manifest["eval_dataset_path"])):
            eval_rows.append(_normalize(row, f"v190_rewrite_{repeat:02d}", 18.0, 44.0, "v190_rewrite_protect"))

    added = 0
    for row in _iter_jsonl(Path(retrieval_manifest["train_dataset_path"])):
        if added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            out = dict(row)
            out["example_id"] = f"{row.get('example_id')}_v190_retrieval"
            out["source_type"] = "v182_retrieval_protection_v190"
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
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v190_weak_task_focus_mix",
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
