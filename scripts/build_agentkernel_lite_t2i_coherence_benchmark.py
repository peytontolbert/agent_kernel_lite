#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


TASKS: dict[str, list[dict[str, object]]] = {
    "single_object": [
        {"prompt": "a clear realistic photo of a violin on a wooden table", "objects": ["violin"]},
        {"prompt": "a clear realistic photo of a microscope on a lab bench", "objects": ["microscope"]},
        {"prompt": "a clear realistic photo of an accordion on a plain background", "objects": ["accordion"]},
        {"prompt": "a clear realistic photo of a tennis racket on a court", "objects": ["tennis racket"]},
    ],
    "animal": [
        {"prompt": "a golden retriever sitting in grass, realistic photo", "objects": ["golden retriever"]},
        {"prompt": "an orangutan in a park, realistic photo", "objects": ["orangutan"]},
        {"prompt": "a goldfish swimming in a glass bowl, realistic photo", "objects": ["goldfish"]},
        {"prompt": "a horse standing beside a wooden fence, realistic photo", "objects": ["horse"]},
    ],
    "vehicle": [
        {"prompt": "a yellow school bus parked on a street, realistic photo", "objects": ["school bus"]},
        {"prompt": "a small airplane on a runway, realistic photo", "objects": ["airplane"]},
        {"prompt": "a red bicycle leaning against a wall, realistic photo", "objects": ["bicycle"]},
        {"prompt": "a blue pickup truck in a driveway, realistic photo", "objects": ["pickup truck"]},
    ],
    "food": [
        {"prompt": "a bowl of ramen on a restaurant table, realistic photo", "objects": ["ramen"]},
        {"prompt": "a red apple on a white plate, realistic photo", "objects": ["apple"]},
        {"prompt": "a cup of coffee with heart-shaped latte art, realistic photo", "objects": ["coffee"]},
        {"prompt": "a slice of pizza on a wooden board, realistic photo", "objects": ["pizza"]},
    ],
    "attribute_binding": [
        {"prompt": "a red backpack beside a blue water bottle, realistic photo", "objects": ["red backpack", "blue water bottle"]},
        {"prompt": "a green chair next to a yellow lamp, realistic photo", "objects": ["green chair", "yellow lamp"]},
        {"prompt": "a black camera on top of a white book, realistic photo", "objects": ["black camera", "white book"]},
        {"prompt": "a silver laptop beside an orange mug, realistic photo", "objects": ["silver laptop", "orange mug"]},
    ],
    "spatial_relation": [
        {"prompt": "a spoon inside a ceramic bowl, realistic photo", "objects": ["spoon", "bowl"]},
        {"prompt": "a toy car under a wooden chair, realistic photo", "objects": ["toy car", "chair"]},
        {"prompt": "a houseplant beside a window, realistic photo", "objects": ["houseplant", "window"]},
        {"prompt": "a pair of shoes on a staircase, realistic photo", "objects": ["shoes", "staircase"]},
    ],
    "scene": [
        {"prompt": "a quiet kitchen with a fruit bowl on the counter, realistic photo", "objects": ["kitchen", "fruit bowl"]},
        {"prompt": "a city street after rain with reflections on the road, realistic photo", "objects": ["city street", "rain reflections"]},
        {"prompt": "a forest trail in soft morning light, realistic photo", "objects": ["forest trail"]},
        {"prompt": "a bedroom with a neatly made bed and a bedside lamp, realistic photo", "objects": ["bedroom", "bedside lamp"]},
    ],
    "rare_object": [
        {"prompt": "a realistic photo of an abacus on a desk", "objects": ["abacus"]},
        {"prompt": "a realistic photo of a sextant on a nautical map", "objects": ["sextant"]},
        {"prompt": "a realistic photo of a rickshaw parked near a sidewalk", "objects": ["rickshaw"]},
        {"prompt": "a realistic photo of a metronome on a piano", "objects": ["metronome"]},
    ],
}


RUBRIC = {
    "prompt_compliance": "Does the image show the requested subject, attributes, and relation?",
    "visual_coherence": "Is the image globally coherent rather than noisy, melted, or structurally broken?",
    "physical_detail_fidelity": "Are object shapes, edges, proportions, and local details plausible?",
    "photographic_quality": "Does it look like a natural 512px photo rather than a low-quality painting or artifact?",
}


def build(args: argparse.Namespace) -> None:
    rows = []
    for task, items in TASKS.items():
        for index, item in enumerate(items):
            rows.append(
                {
                    "id": f"{task}_{index:03d}",
                    "task": task,
                    "prompt": item["prompt"],
                    "expected_objects": item["objects"],
                    "rubric": RUBRIC,
                    "score_scale": "1-5",
                }
            )
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_lite_t2i_coherence_benchmark",
                "rows": len(rows),
                "tasks": sorted(TASKS),
                "rubric": RUBRIC,
                "source_inspiration": "REDEdit-Bench category-wise prompt compliance and visual fidelity scoring",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed T2I coherence benchmark for Agent Kernel Lite image checkpoints.")
    parser.add_argument("--output", default="data/vision/prompts/agentkernel_lite_t2i_coherence_bench_v0.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
