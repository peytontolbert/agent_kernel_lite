#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def _stable_id(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % 100000


def _state_doc(row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type", "") or "")
    intent = str(row.get("intent_label", "") or "")
    expected = str(row.get("expected_content", "") or "").strip()
    if not expected:
        decoder = str(row.get("decoder_text", "") or "")
        try:
            expected = str((json.loads(decoder).get("content", "") if decoder else "") or "").strip()
        except json.JSONDecodeError:
            expected = decoder.strip()
    lines = [f"intent={intent}", f"task_type={task_type}"]
    for line in expected.splitlines():
        text = line.strip()
        if not text:
            continue
        if ":" in text:
            key, value = text.lstrip("- ").split(":", 1)
            lines.append(f"{key.strip().lower().replace(' ', '_')}={value.strip()}")
        else:
            lines.append(f"output={text}")
    return "; ".join(lines)


def _augment(row: dict[str, Any], *, weight: float, retrieval_weight: float) -> dict[str, Any]:
    out = dict(row)
    doc = _state_doc(row)
    out["retrieval_query_text"] = str(row.get("encoder_text", "") or "")
    out["retrieval_doc_text"] = doc
    out["retrieval_loss_weight"] = float(retrieval_weight)
    out["weight"] = float(weight)
    out["source_type"] = "pocketpal_stage55_state_retrieval_curriculum"
    out["source_id"] = f"stage55_{row.get('source_id', row.get('example_id', 'row'))}"
    out["contrastive_label_id"] = int(row.get("contrastive_label_id", -1) if row.get("contrastive_label_id", "") != "" else _stable_id(doc))
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default="tmp/pocketpal_stage54_world_compression_curriculum/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage55_state_retrieval_curriculum")
    parser.add_argument("--weight", type=float, default=7.0)
    parser.add_argument("--retrieval-weight", type=float, default=2.5)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()

    manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    rows = [_augment(row, weight=float(args.weight), retrieval_weight=float(args.retrieval_weight)) for row in _read_jsonl(Path(manifest["train_dataset_path"]))]
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    eval_count = max(1, int(len(rows) * float(args.eval_ratio)))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]

    output_dir = Path(args.output_dir)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    output = {
        "objective": "pocketpal_stage55_state_retrieval_curriculum",
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "train_dataset_path": str(train_path.resolve()),
        "eval_dataset_path": str(eval_path.resolve()),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "total_examples": len(rows),
        "state_retrieval": True,
        "retrieval_doc": "canonical latent state string",
    }
    manifest_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**output, "manifest_path": str(manifest_path.resolve())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
