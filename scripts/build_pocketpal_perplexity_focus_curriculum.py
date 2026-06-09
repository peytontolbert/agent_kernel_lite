#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


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


def _copy_row(row: dict[str, Any], *, label: str, index: int, weight: float) -> dict[str, Any]:
    out = deepcopy(row)
    out["example_id"] = _stable_id("ppl_focus", label, index, out.get("source_id"), out.get("encoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_ppl_focus_{label}_{index:05d}"
    out["source_type"] = f"ppl_focus_{label}_{out.get('source_type', 'unknown')}"
    out["split"] = "train"
    out["weight"] = float(weight)
    out["decoder_loss_weight"] = 1.0
    return out


def _is_hard_row(row: dict[str, Any], hard_tasks: set[str], hard_sources: tuple[str, ...]) -> bool:
    task_type = str(row.get("task_type") or "")
    source_type = str(row.get("source_type") or row.get("source_dataset") or "")
    if task_type in hard_tasks:
        return True
    return any(item and item in source_type for item in hard_sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", required=True)
    parser.add_argument("--retrieval-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--anchor-sample", type=int, default=26000)
    parser.add_argument("--hard-sample", type=int, default=14000)
    parser.add_argument("--retrieval-protect", type=int, default=2500)
    parser.add_argument("--hard-weight", type=float, default=10.0)
    parser.add_argument("--anchor-weight-cap", type=float, default=8.0)
    parser.add_argument(
        "--hard-task",
        action="append",
        default=["active_agent_summary", "active_agent_plan", "runtime_web_search_request"],
    )
    parser.add_argument(
        "--hard-source-contains",
        action="append",
        default=["stage82_openclaw_hermes_skill", "stage82_paper_repo_alignment", "stage82_repo_concept"],
    )
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    broad_manifest = json.loads(Path(args.broad_manifest).read_text(encoding="utf-8"))
    train_rows = list(_iter_jsonl(Path(broad_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(broad_manifest["eval_dataset_path"])))
    hard_tasks = {str(item) for item in args.hard_task}
    hard_sources = tuple(str(item) for item in args.hard_source_contains)

    hard_rows = [row for row in train_rows if _is_hard_row(row, hard_tasks, hard_sources)]
    anchor_rows = [row for row in train_rows if not _is_hard_row(row, hard_tasks, hard_sources)]
    rng.shuffle(hard_rows)
    rng.shuffle(anchor_rows)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(hard_rows[: max(0, int(args.hard_sample))]):
        task_type = str(row.get("task_type") or "unknown").replace("active_agent_", "")
        rows.append(_copy_row(row, label=f"hard_{task_type}", index=index, weight=float(args.hard_weight)))
    for index, row in enumerate(anchor_rows[: max(0, int(args.anchor_sample))]):
        weight = min(max(float(row.get("weight") or 1.0), 1.0), float(args.anchor_weight_cap))
        rows.append(_copy_row(row, label="anchor", index=index, weight=weight))

    retrieval_added = 0
    if args.retrieval_manifest:
        retrieval_manifest = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
        for row in _iter_jsonl(Path(retrieval_manifest["train_dataset_path"])):
            if retrieval_added >= int(args.retrieval_protect):
                break
            if str(row.get("retrieval_query_text") or "").strip() and str(row.get("retrieval_doc_text") or "").strip():
                protected = _copy_row(row, label="retrieval", index=retrieval_added, weight=1.0)
                protected["decoder_loss_weight"] = 0.0
                protected["retrieval_loss_weight"] = 1.0
                rows.append(protected)
                retrieval_added += 1

    rng.shuffle(rows)
    out_eval = []
    for index, row in enumerate(eval_rows):
        copied = deepcopy(row)
        copied["example_id"] = _stable_id("ppl_focus_eval", index, copied.get("source_id"), copied.get("encoder_text"))
        copied["split"] = "eval"
        out_eval.append(copied)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, rows)
    _write_jsonl(eval_path, out_eval)
    manifest = {
        **broad_manifest,
        "artifact_kind": "agentkernel_lite_encdec_perplexity_focus_curriculum",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(out_eval),
        "hard_examples_available": len(hard_rows),
        "hard_examples_used": min(len(hard_rows), max(0, int(args.hard_sample))),
        "hard_source_contains": list(hard_sources),
        "hard_tasks": sorted(hard_tasks),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_perplexity_focus_curriculum",
        "retrieval_protect_examples": retrieval_added,
        "total_examples": len(rows) + len(out_eval),
        "train_dataset_path": str(train_path),
        "train_examples": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
