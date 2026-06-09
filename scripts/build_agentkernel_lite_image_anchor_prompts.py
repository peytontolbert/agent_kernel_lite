#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBJECTS = [
    "golden retriever sitting on a kitchen floor",
    "orange cat looking out a sunny window",
    "red fox walking through snow",
    "green tree frog on a wet leaf",
    "brown horse standing in a field",
    "small parrot on a wooden perch",
    "black bicycle leaning against a brick wall",
    "passenger airplane above snowy mountains",
    "blue pickup truck at a desert gas station",
    "red sports car on a mountain road",
    "yellow school bus parked near a curb",
    "white sailboat on a calm lake",
    "wooden violin on a velvet chair",
    "silver trumpet on a black studio table",
    "acoustic guitar beside a window",
    "upright piano in a quiet living room",
    "glass teapot with amber tea",
    "red apple on a plain ceramic plate",
    "bowl of ramen on a wooden table",
    "fresh croissant on a linen napkin",
    "ceramic vase filled with wildflowers",
    "blue running shoe on a plain background",
    "wristwatch on a marble surface",
    "open book beside a cup of coffee",
    "portrait of a baker holding fresh bread",
    "portrait of a mechanic in a garage",
    "portrait of a violinist holding a bow",
    "child wearing a yellow raincoat",
    "modern library interior with people reading",
    "small bedroom with morning light",
    "cozy kitchen with plants on the counter",
    "city street market at night",
    "steam locomotive crossing a stone bridge",
    "wooden cabin beside a pine forest",
    "lighthouse on a rocky coast",
    "green tractor in a farm field",
    "red umbrella on a rainy sidewalk",
    "white ceramic mug on a desk",
    "pair of glasses on an open notebook",
    "basket of oranges at a grocery stall",
]

SETTINGS = [
    "realistic photo",
    "soft daylight photo",
    "studio product photo",
    "cinematic natural light photo",
    "clear documentary photo",
    "high quality mobile photo",
]

COMPOSITIONS = [
    "centered subject, uncluttered background",
    "three quarter view, natural shadows",
    "close-up view, sharp subject",
    "wide view, readable scene layout",
    "simple composition, coherent object shape",
]


def build_prompts(limit: int) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for subject in SUBJECTS:
        for setting in SETTINGS:
            for composition in COMPOSITIONS:
                prompt = f"{setting} of a {subject}, {composition}"
                key = prompt.lower()
                if key in seen:
                    continue
                seen.add(key)
                prompts.append(prompt)
                if len(prompts) >= limit:
                    return prompts
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simple coherence anchor prompts for image distillation.")
    parser.add_argument("--output", default="data/vision/prompts/general_coherence_anchor_256_v0.jsonl")
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(args.limit)
    with output.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts):
            handle.write(json.dumps({"prompt": prompt, "source": "coherence_anchor_v0", "index": index}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "prompts": len(prompts)}, indent=2))


if __name__ == "__main__":
    main()
