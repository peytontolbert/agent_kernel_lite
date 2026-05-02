#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def compact(value: object, limit: int = 900) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())[:limit].strip()


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text or "", flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return compact(match.group(1), 700) if match else ""


def title_for_id(text: str, evidence_id: str) -> str:
    pattern = (
        rf"<AK_EVIDENCE>\s+<AK_EVIDENCE_ID>\s+{re.escape(evidence_id)}"
        rf".*?<AK_TITLE>\s+(.+?)(?=\s+<AK_PAPER_ID>|\n<AK_PAPER_ID>|\Z)"
    )
    return first_match(pattern, text)


def abstract_for_id(text: str, evidence_id: str) -> str:
    pattern = (
        rf"<AK_EVIDENCE>\s+<AK_EVIDENCE_ID>\s+{re.escape(evidence_id)}"
        rf".*?<AK_ABSTRACT>\s+(.+?)(?=\n\s*<AK_EVIDENCE>|\Z)"
    )
    return first_match(pattern, text)


def sentence(text: str) -> str:
    value = compact(text, 700)
    parts = re.findall(r"[^.!?]+[.!?]", value)
    return compact(parts[0] if parts else value, 420).rstrip(".") + "."


def line(content: str) -> str:
    return f"Action: respond\nContent: {compact(content, 1200)}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def repair_row(row: dict[str, Any]) -> dict[str, Any] | None:
    task_type = str(row.get("task_type", ""))
    encoder = str(row.get("encoder_text", ""))
    title = title_for_id(encoder, "P1")
    abstract = abstract_for_id(encoder, "P1")
    if not title or not abstract:
        return None
    summary = sentence(abstract)
    out = dict(row)
    out["source_type"] = "agentkernel_lite_copy_grounded_repair"
    out["source_id"] = f"copy_grounded:{row.get('source_id', '')}"
    out["weight"] = 8.0
    if task_type in {"selected_paper_followup_with_history", "selected_paper_followup_no_new_retrieval"}:
        out["task_type"] = "copy_grounded_selected_context_answer"
        out["decoder_text"] = line(
            f"I am using the active selected paper [P1], {title}, not retrieving new papers. "
            f"The paper focuses on this point: {summary} "
            "In more detail, read it by separating the problem it studies, the mechanism or method it proposes, "
            "the evidence it gives, and the limitations that would need follow-up."
        )
        return out
    if task_type in {"grounded_recommendation_with_reading_notes", "think_mode_synthesis_across_evidence", "deep_research_evidence_audit"}:
        out["task_type"] = "copy_grounded_retrieved_evidence_answer"
        out["decoder_text"] = line(
            f"The best-supported match in the retrieved evidence is [1], {title}. "
            f"It is the strongest answer because the evidence says: {summary} "
            "I would use the other retrieved papers only as comparisons unless they directly match the user's topic."
        )
        return out
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    source_manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_source = Path(source_manifest["train_dataset_path"])
    eval_source = Path(source_manifest["eval_dataset_path"])
    train_rows = [row for row in (repair_row(row) for row in load_jsonl(train_source)) if row is not None]
    eval_rows = [row for row in (repair_row(row) for row in load_jsonl(eval_source)) if row is not None]
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]
        eval_rows = eval_rows[: max(1, args.max_rows // 20)]

    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    task_counts: dict[str, int] = {}
    for row in [*train_rows, *eval_rows]:
        task = str(row.get("task_type", ""))
        task_counts[task] = task_counts.get(task, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "copy_grounded_research_assistant_repair",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "task_type_counts": dict(sorted(task_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
