#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _content_from_decoder(decoder_text: str) -> str:
    try:
        parsed = json.loads(decoder_text)
    except Exception:
        return ""
    decision = parsed.get("decision_packet", {}).get("decision") if isinstance(parsed, dict) else None
    if not decision and isinstance(parsed, dict):
        decision = parsed.get("decision") or parsed
    return str((decision or {}).get("content") or "")


def _normalize(row: dict[str, Any], *, suffix: str, weight: float | None = None, negative: str | None = None) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = f"{out.get('example_id')}_{suffix}"
    out["source_id"] = f"{out.get('source_id')}_{suffix}"
    out["source_type"] = f"{out.get('source_type', 'unknown')}_{suffix}"
    if weight is not None:
        out["weight"] = float(weight)
    if negative:
        out["negative_decoder_text"] = negative
        out["negative_loss_weight"] = 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--failure-json", default="tmp/v200a_direct_agent_prompts.json")
    parser.add_argument("--protocol-manifest", default="tmp/pocketpal_v193_protocol_cleanup/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--broad-sample", type=int, default=12000)
    parser.add_argument("--retrieval-protect", type=int, default=1800)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    broad_eval_by_source = {str(row.get("source_id") or ""): row for row in _iter_jsonl(Path(broad_manifest["eval_dataset_path"]))}
    failures = json.loads(Path(args.failure_json).read_text(encoding="utf-8")).get("failures", [])
    for index, failure in enumerate(failures):
        source_id = str(failure.get("source_id") or "")
        row = broad_eval_by_source.get(source_id)
        if not row:
            continue
        expected = str(failure.get("expected") or _content_from_decoder(str(row.get("decoder_text") or "")))
        if expected and expected not in str(row.get("decoder_text") or ""):
            # Keep the original action/metadata, but replace content when the eval expected_content is more direct.
            try:
                packet = json.loads(row["decoder_text"])
                packet["content"] = expected
                row = dict(row)
                row["decoder_text"] = json.dumps(packet, ensure_ascii=False, sort_keys=True)
            except Exception:
                pass
        negative = str(failure.get("output") or "")
        repaired = _normalize(row, suffix=f"v201_failure_{index:03d}", weight=70.0, negative=negative)
        rows.append(repaired)
        eval_rows.append(_normalize(row, suffix=f"v201_eval_failure_{index:03d}", weight=70.0, negative=negative))

    broad_train = list(_iter_jsonl(Path(broad_manifest["train_dataset_path"])))
    rng.shuffle(broad_train)
    for index, row in enumerate(broad_train[: int(args.broad_sample)]):
        rows.append(_normalize(row, suffix=f"v201_broad_{index:05d}", weight=min(max(float(row.get("weight") or 1.0), 2.0), 18.0)))

    for manifest_arg, label, repeat_count, weight in [
        (args.protocol_manifest, "protocol", 1, 28.0),
        (args.rewrite_slot_manifest, "rewrite", 4, 28.0),
    ]:
        manifest = json.loads(Path(manifest_arg).read_text(encoding="utf-8"))
        for repeat in range(repeat_count):
            for key, dest in [("train_dataset_path", rows), ("eval_dataset_path", eval_rows)]:
                for row in _iter_jsonl(Path(manifest[key])):
                    dest.append(_normalize(row, suffix=f"v201_{label}_{repeat:02d}", weight=weight))

    retrieval = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
    added = 0
    for row in _iter_jsonl(Path(retrieval["train_dataset_path"])):
        if added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            rows.append(_normalize(row, suffix="v201_retrieval", weight=0.0))
            added += 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, rows)
    _write(eval_path, eval_rows)
    all_rows = rows + eval_rows
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v201_failure_replay_from_v200a_eval",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
