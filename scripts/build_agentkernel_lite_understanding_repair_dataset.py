#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_agentkernel_lite_paper_understanding import PAPER_CASES, _build_prompt


def line(content: str) -> str:
    return "Action: respond\nContent: " + " ".join(content.split())


def answer_for(case: dict[str, Any], question_variant: str) -> str:
    title = str(case["title"])
    abstract = str(case["abstract"])
    if "takeaway" in question_variant.lower():
        return line(
            f"The main takeaway from the active selected paper [P1], {title}, is that {abstract} "
            "I would read it by tracking the problem, the method, the evidence, and the limitation it leaves open."
        )
    if "method" in question_variant.lower() or "why" in question_variant.lower():
        return line(
            f"The active selected paper [P1], {title}, matters because it studies this specific issue: {abstract} "
            "Its method should be understood through the mechanism described in the evidence, not through unrelated papers."
        )
    return line(
        f"I am answering from the active selected paper [P1], {title}. "
        f"In detail, the paper says: {abstract} "
        "So the grounded explanation should preserve those entities and claims, then separate the problem, method, evidence, and limitations."
    )


def build_rows(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = [
        "explain this paper in detail",
        "tell me more about this paper please",
        "what are the main takeaways from this paper?",
        "summarize the method and why it matters",
        "explain the active paper without searching for more papers",
    ]
    for repeat in range(max(1, int(repeats))):
        for case in PAPER_CASES:
            for variant in variants:
                prompt_case = dict(case)
                prompt_case["question"] = variant
                rows.append(
                    {
                        "source_type": "agentkernel_lite_understanding_repair",
                        "source_id": f"{case['id']}:{repeat}:{variant}",
                        "task_type": "active_paper_understanding_repair",
                        "encoder_text": _build_prompt(prompt_case),
                        "decoder_text": answer_for(prompt_case, variant),
                        "action": "respond",
                        "weight": 12.0,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--eval-repeats", type=int, default=2)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = build_rows(int(args.repeats))
    eval_rows = build_rows(int(args.eval_repeats))
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    for path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "active_paper_understanding_repair",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "task_type_counts": {"active_paper_understanding_repair": len(train_rows) + len(eval_rows)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
