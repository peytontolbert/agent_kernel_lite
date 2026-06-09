#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FOCUS_PREFIXES = (
    "active_agent_action_items",
    "active_agent_brainstorm",
    "active_agent_checklist",
    "active_agent_extraction",
    "active_agent_json",
    "active_agent_plan",
    "active_agent_rewrite",
    "active_agent_risks",
    "active_agent_subject",
    "active_agent_summary",
    "active_agent_translation",
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash_float(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


def _normalize_row(row: dict[str, Any], suffix: str, weight_floor: float, weight_cap: float) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = f"{row.get('example_id') or row.get('source_id')}_{suffix}"
    out["source_type"] = f"{row.get('source_type', 'unknown')}_v183_broad_repair"
    out["weight"] = min(max(float(row.get("weight") or 1.0), weight_floor), weight_cap)
    if not str(out.get("negative_decoder_text") or "").strip():
        task = str(out.get("task_type") or "")
        wrong = ""
        if task.endswith("rewrite"):
            wrong = _payload("respond", "Hello, I hope you are well.", "negative_greeting_attractor")
        elif task.endswith("summary"):
            wrong = _payload("respond", "- Greeting: Hi, how are you?", "negative_summary_attractor")
        elif task.endswith("extraction") or task.endswith("checklist") or task.endswith("action_items"):
            wrong = _payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "negative_slot_template_attractor")
        if wrong:
            out["negative_decoder_text"] = wrong
            out["negative_loss_weight"] = 0.4
    return out


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--gate-manifest", default="tmp/pocketpal_v181_greeting_slot_boundary/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-broad-train", type=int, default=70000)
    parser.add_argument("--max-broad-eval", type=int, default=6000)
    parser.add_argument("--gate-repeat", type=int, default=8)
    parser.add_argument("--retrieval-protect", type=int, default=4000)
    args = parser.parse_args()

    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    gate_manifest = json.loads(Path(args.gate_manifest).read_text(encoding="utf-8"))
    retrieval_manifest = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for split_name, path_key, limit, dest in [
        ("train", "train_dataset_path", int(args.max_broad_train), train_rows),
        ("eval", "eval_dataset_path", int(args.max_broad_eval), eval_rows),
    ]:
        selected = []
        for row in _iter_jsonl(Path(broad_manifest[path_key])):
            task = str(row.get("task_type") or "")
            if task.startswith(FOCUS_PREFIXES):
                selected.append(row)
        selected.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
        for index, row in enumerate(selected[:limit]):
            dest.append(_normalize_row(row, f"v183_{split_name}_{index:06d}", weight_floor=6.0, weight_cap=18.0))

    for repeat in range(int(args.gate_repeat)):
        for row in _iter_jsonl(Path(gate_manifest["train_dataset_path"])):
            train_rows.append(_normalize_row(row, f"v183_gate_{repeat:02d}", weight_floor=18.0, weight_cap=36.0))
        for row in _iter_jsonl(Path(gate_manifest["eval_dataset_path"])):
            eval_rows.append(_normalize_row(row, f"v183_gate_{repeat:02d}", weight_floor=18.0, weight_cap=36.0))

    retrieval_added = 0
    for row in _iter_jsonl(Path(retrieval_manifest["train_dataset_path"])):
        if retrieval_added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            protected = dict(row)
            protected["example_id"] = f"{row.get('example_id')}_v183_retrieval"
            protected["source_type"] = "v182_retrieval_protection"
            protected["weight"] = 0.0
            train_rows.append(protected)
            retrieval_added += 1

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
        "objective": "pocketpal_v183_broad_agent_repair_mix",
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
