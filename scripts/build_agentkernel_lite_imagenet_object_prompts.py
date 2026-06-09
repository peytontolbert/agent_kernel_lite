#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re


PHOTOGRAPHIC_TEMPLATES = (
    "a clear realistic photo of {article} {label}",
    "a centered studio photograph of {article} {label} on a plain background",
    "a detailed close-up photo of {article} {label}",
    "a natural daylight photograph of {article} {label}",
    "a sharp documentary photo of {article} {label} in its normal environment",
    "a realistic 512px image of a single {label}, full object visible",
    "a high quality product-style photograph of {article} {label}",
    "a simple composition showing one {label} clearly",
    "a realistic photo of {article} {label} with accurate shape and proportions",
    "a clean reference photograph of {article} {label}, no text, no watermark",
    "{article} {label} on a table, realistic lighting, clear silhouette",
    "{article} {label} outdoors, realistic photograph, clear subject",
)


SCENE_TEMPLATES = (
    "a realistic photo of {article} {label} in a kitchen",
    "a realistic photo of {article} {label} in a living room",
    "a realistic photo of {article} {label} on a city street",
    "a realistic photo of {article} {label} in a park",
    "a realistic photo of {article} {label} near a window",
    "a realistic photo of {article} {label} on a wooden surface",
    "a realistic photo of {article} {label} in soft morning light",
    "a realistic photo of {article} {label} with shallow depth of field",
)


BLOCKED_LABEL_TERMS = {
    "cock",
}


def normalize_label(label: str) -> str:
    label = label.split(",", 1)[0]
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip().lower()
    return label


def load_imagenet_categories() -> list[str]:
    from torchvision.models import ResNet50_Weights

    categories = ResNet50_Weights.DEFAULT.meta.get("categories") or []
    labels = []
    seen = set()
    for category in categories:
        label = normalize_label(str(category))
        if not label or label in seen or label in BLOCKED_LABEL_TERMS:
            continue
        seen.add(label)
        labels.append(label)
    if len(labels) < 900:
        raise RuntimeError(f"expected ImageNet category list, got {len(labels)} labels")
    return labels


def article(label: str) -> str:
    return "an" if label[:1] in {"a", "e", "i", "o", "u"} else "a"


def render(template: str, label: str) -> str:
    text = template.format(label=label, article=article(label))
    return text


def build_prompts(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    labels = load_imagenet_categories()
    templates = list(PHOTOGRAPHIC_TEMPLATES)
    if args.include_scenes:
        templates.extend(SCENE_TEMPLATES)
    records = []
    for label_index, label in enumerate(labels):
        selected = templates[:]
        rng.shuffle(selected)
        for template_index, template in enumerate(selected[: args.templates_per_class]):
            records.append(
                {
                    "prompt": render(template, label),
                    "source_dataset": "torchvision_imagenet_1k_categories",
                    "source_name": "imagenet_object_templates",
                    "source_index": label_index,
                    "label": label,
                    "template_index": template_index,
                    "filter": "object_photo_curriculum_v0",
                }
            )
    rng.shuffle(records)
    if args.limit > 0:
        records = records[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "artifact_kind": "agentkernel_lite_imagenet_object_prompt_corpus",
        "rows": len(records),
        "labels": len(labels),
        "templates_per_class": int(args.templates_per_class),
        "include_scenes": bool(args.include_scenes),
        "seed": int(args.seed),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(records), "labels": len(labels)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ImageNet object prompt templates for FLUX student distillation.")
    parser.add_argument("--output", default="data/vision/prompts/imagenet_object_photo_12k_v0.jsonl")
    parser.add_argument("--templates-per-class", type=int, default=12)
    parser.add_argument("--include-scenes", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()
    build_prompts(args)


if __name__ == "__main__":
    main()
