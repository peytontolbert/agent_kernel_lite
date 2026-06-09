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


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _with_negatives(rows: list[dict[str, Any]], *, negatives: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    docs_by_task: dict[str, list[tuple[str, str]]] = {}
    docs_all: list[tuple[str, str]] = []
    for row in rows:
        doc = str(row.get("retrieval_doc_text", "") or "").strip()
        query = str(row.get("retrieval_query_text", "") or "").strip()
        if not doc or not query:
            continue
        key = str(row.get("example_id") or row.get("source_id") or _stable_id(query, doc))
        task = str(row.get("task_type") or "unknown")
        docs_by_task.setdefault(task, []).append((key, doc))
        docs_all.append((key, doc))
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        copied = deepcopy(row)
        doc = str(copied.get("retrieval_doc_text", "") or "").strip()
        query = str(copied.get("retrieval_query_text", "") or "").strip()
        if doc and query:
            key = str(copied.get("example_id") or copied.get("source_id") or _stable_id(query, doc))
            task = str(copied.get("task_type") or "unknown")
            candidates = [item for item in docs_by_task.get(task, []) if item[0] != key and item[1] != doc]
            if len(candidates) < negatives:
                candidates.extend(item for item in docs_all if item[0] != key and item[1] != doc)
            rng.shuffle(candidates)
            selected: list[str] = []
            seen = {doc}
            for _, candidate_doc in candidates:
                if candidate_doc in seen:
                    continue
                selected.append(candidate_doc)
                seen.add(candidate_doc)
                if len(selected) >= negatives:
                    break
            copied["retrieval_negative_doc_texts"] = json.dumps(selected, ensure_ascii=False)
            copied["retrieval_loss_weight"] = max(float(copied.get("retrieval_loss_weight", 1.0) or 1.0), 1.0)
        out.append(copied)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage82_broad_intelligence_corpus/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_stage86_retrieval_hard_negative_corpus"))
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8601)
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest).resolve()
    source_manifest = _load_manifest(source_manifest_path)
    train_rows = _load_rows(Path(source_manifest["train_dataset_path"]))
    eval_rows = _load_rows(Path(source_manifest["eval_dataset_path"]))
    train_rows = _with_negatives(train_rows, negatives=int(args.negatives), seed=int(args.seed))
    eval_rows = _with_negatives(eval_rows, negatives=int(args.negatives), seed=int(args.seed) + 17)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "artifact_kind": "agentkernel_lite_encdec_stage86_retrieval_hard_negative_corpus",
            "base_manifest": str(source_manifest_path),
            "eval_dataset_path": str(eval_path),
            "eval_examples": len(eval_rows),
            "hard_negatives_per_retrieval_row": int(args.negatives),
            "manifest_path": str(manifest_path),
            "objective": "pocketpal_stage86_retrieval_hard_negative_corpus",
            "train_dataset_path": str(train_path),
            "train_examples": len(train_rows),
            "total_examples": len(train_rows) + len(eval_rows),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["manifest_path", "train_examples", "eval_examples", "hard_negatives_per_retrieval_row"]}, indent=2))


if __name__ == "__main__":
    main()
