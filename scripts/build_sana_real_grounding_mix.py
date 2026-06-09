#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(json.loads(line))
    return rows


def label_for(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("class_label") or row.get("object_id") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative Sana prompt mix with bounded real-image grounding rows."
    )
    parser.add_argument("--synthetic-jsonl", required=True)
    parser.add_argument("--real-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--labels", default="bicycle,bus,car,dog,horse,motorcycle,truck")
    parser.add_argument("--synthetic-per-real", type=int, default=16)
    parser.add_argument("--max-real-rows", type=int, default=64)
    parser.add_argument("--max-real-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument(
        "--preserve-synthetic-order",
        action="store_true",
        help="Keep synthetic rows in input order instead of shuffling before interleaving real rows.",
    )
    parser.add_argument(
        "--warmup-synthetic-rows",
        type=int,
        default=0,
        help="Emit this many synthetic rows before the first real row.",
    )
    parser.add_argument(
        "--real-domain",
        default="real_object_crop_target_safe",
        help="Domain value to assign to accepted real rows so trainer real-domain scales apply.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_rows = read_jsonl(Path(args.synthetic_jsonl))
    real_rows = read_jsonl(Path(args.real_jsonl))
    allowed = {item.strip() for item in args.labels.split(",") if item.strip()}

    real_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real_rows:
        label = label_for(row)
        if label not in allowed:
            continue
        candidate = dict(row)
        candidate["label"] = label
        candidate["class_label"] = label
        candidate["domain"] = args.real_domain
        candidate["curriculum_stage"] = "safe_real_crop_grounding"
        candidate["real_grounding_mode"] = "decoded_near_clean_grounding"
        real_by_label[label].append(candidate)

    selected_real: list[dict[str, Any]] = []
    labels = sorted(real_by_label)
    while labels and len(selected_real) < args.max_real_rows:
        progressed = False
        for label in labels:
            bucket = real_by_label[label]
            if not bucket:
                continue
            current_for_label = sum(1 for row in selected_real if label_for(row) == label)
            if current_for_label >= args.max_real_per_label:
                continue
            selected_real.append(bucket.pop(rng.randrange(len(bucket))))
            progressed = True
            if len(selected_real) >= args.max_real_rows:
                break
        if not progressed:
            break

    synthetic_count = min(len(synthetic_rows), max(0, len(selected_real) * args.synthetic_per_real))
    selected_synthetic = list(synthetic_rows)
    if not args.preserve_synthetic_order:
        rng.shuffle(selected_synthetic)
    selected_synthetic = selected_synthetic[:synthetic_count]

    mixed: list[dict[str, Any]] = []
    synth_index = 0
    warmup = min(max(0, args.warmup_synthetic_rows), len(selected_synthetic))
    mixed.extend(selected_synthetic[:warmup])
    synth_index = warmup
    for real in selected_real:
        for _ in range(args.synthetic_per_real):
            if synth_index < len(selected_synthetic):
                mixed.append(selected_synthetic[synth_index])
                synth_index += 1
        mixed.append(real)
    mixed.extend(selected_synthetic[synth_index:])

    with output_path.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "rows": len(mixed),
                "synthetic_rows": len(selected_synthetic),
                "real_rows": len(selected_real),
                "synthetic_per_real": args.synthetic_per_real,
                "real_by_label": Counter(label_for(row) for row in selected_real),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
