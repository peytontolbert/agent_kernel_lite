#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


VARIANT_RE = re.compile(
    r"\breference variant\s+\d+\b|\balternate real-world\b|\bclose full-object\b|\bmedium full-object\b",
    re.IGNORECASE,
)


def fixed_ref(row: dict[str, Any]) -> str:
    for key in ("teacher_ref", "image_ref", "image_path", "path", "real_ref", "source_ref"):
        if row.get(key):
            return str(row[key])
    return ""


def score_ref_row(row: dict[str, Any]) -> float:
    prompt = str(row.get("prompt", "")).lower()
    score = 0.0
    if row.get("dedup_variant_index") is None:
        score += 100.0
    if not VARIANT_RE.search(prompt):
        score += 80.0
    if "preserve the real target object structure" in prompt:
        score += 60.0
    if "physically plausible" in prompt:
        score += 40.0
    if "plain white background" in prompt:
        score += 20.0
    if "single object only" in prompt or "single main object visible" in prompt:
        score += 15.0
    if "reference variant" in prompt:
        score -= 100.0
    if str(row.get("variant_axis", "")).startswith("dedup"):
        score -= 20.0
    score -= min(len(prompt), 500) * 0.001
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove contradictory prompt variants for fixed teacher_ref/image targets."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")

    rows = []
    for line_index, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        row["_source_line"] = line_index
        rows.append(row)

    kept_no_ref = []
    chosen_by_ref: dict[str, dict[str, Any]] = {}
    dropped = []
    for row in rows:
        ref = fixed_ref(row)
        if not ref:
            kept_no_ref.append(row)
            continue
        previous = chosen_by_ref.get(ref)
        if previous is None:
            chosen_by_ref[ref] = row
        elif score_ref_row(row) > score_ref_row(previous):
            dropped.append(previous)
            chosen_by_ref[ref] = row
        else:
            dropped.append(row)

    kept = [*kept_no_ref, *chosen_by_ref.values()]
    kept.sort(key=lambda row: int(row.get("_source_line", 0)))

    for row in kept:
        row.pop("_source_line", None)
        if fixed_ref(row):
            row.pop("dedup_source_prompt", None)
            row.pop("dedup_variant_index", None)
            stage = str(row.get("curriculum_stage", ""))
            row["curriculum_stage"] = (stage + "|refsafe_one_prompt_per_teacher_ref").strip("|")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_sets_by_ref: dict[str, set[str]] = {}
    for row in kept:
        ref = fixed_ref(row)
        if ref:
            prompt_sets_by_ref.setdefault(ref, set()).add(str(row.get("prompt", "")))
    conflicts = {ref: prompts for ref, prompts in prompt_sets_by_ref.items() if len(prompts) > 1}
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(kept),
        "dropped_fixed_ref_duplicate_rows": len(dropped),
        "fixed_teacher_refs": len(prompt_sets_by_ref),
        "fixed_ref_multi_prompt_violations": len(conflicts),
        "domains": dict(Counter(str(row.get("domain", "")) for row in kept)),
    }
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if conflicts:
        raise SystemExit("fixed_ref_multi_prompt_violations remained after sanitization")


if __name__ == "__main__":
    main()
