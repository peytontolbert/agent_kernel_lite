#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Iterable


ANIMAL_TERMS = {
    "dog", "cat", "horse", "bear", "wolf", "fox", "lion", "tiger", "leopard", "cheetah",
    "zebra", "deer", "cow", "ox", "sheep", "goat", "pig", "rabbit", "hare", "squirrel",
    "monkey", "ape", "gorilla", "orangutan", "chimpanzee", "macaque", "lemur", "baboon",
    "elephant", "giraffe", "camel", "kangaroo", "koala", "panda", "otter", "beaver", "raccoon", "skunk", "hamster",
    "mouse", "rat", "bird", "eagle", "hawk", "owl", "duck", "goose", "swan", "chicken",
    "penguin", "fish", "shark", "whale", "dolphin", "snake", "lizard", "frog", "turtle",
    "cock", "hen", "ostrich", "brambling", "goldfinch", "finch", "junco", "bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee", "ouzel", "kite", "vulture",
    "grouse", "quail", "partridge", "peacock", "lorikeet", "hummingbird", "jacamar",
    "toucan", "drake", "merganser", "pelican", "albatross", "stork", "flamingo",
    "heron", "egret", "bittern", "crane", "limpkin", "gallinule", "coot", "bustard",
    "sandpiper", "redshank", "dowitcher", "oystercatcher", "cormorant", "kingfisher",
    "hammerhead", "ray", "stingray", "eel", "salmon", "trout", "sturgeon", "gar",
    "lionfish", "puffer", "pufferfish", "tench", "goldfish",
    "newt", "salamander", "gecko", "iguana", "chameleon", "crocodile", "alligator",
    "scorpion", "spider", "tarantula", "tick", "centipede", "beetle", "butterfly",
    "moth", "dragonfly", "damselfly", "grasshopper", "cricket", "mantis", "bee", "ant",
    "snail", "slug", "conch", "starfish", "urchin", "crab", "lobster", "crayfish",
}
VEHICLE_TERMS = {
    "car", "truck", "bus", "van", "motorcycle", "bicycle", "tractor", "train", "airplane",
    "aeroplane", "boat", "ship", "scooter", "wagon", "ambulance", "fire engine", "taxi",
    "limousine", "jeep", "snowmobile", "forklift", "crane",
}
FURNITURE_TERMS = {"chair", "sofa", "couch", "table", "desk", "bed", "bench", "cabinet", "stool", "wardrobe"}
TOOL_TERMS = {"hammer", "drill", "saw", "wrench", "screwdriver", "knife", "scissors", "axe", "shovel", "rake"}
FOOD_TERMS = {"apple", "orange", "banana", "lemon", "strawberry", "pineapple", "pizza", "burger", "bread", "cake"}
DEFAULT_EXCLUDE_TERMS = {
    "website", "web site", "screen", "monitor", "menu", "book jacket", "comic book", "envelope",
    "street sign", "traffic light", "traffic signal", "scoreboard", "barbershop", "restaurant",
    "grocery store", "library", "palace", "castle", "church", "monastery", "prison", "school",
    "theater", "lakeside", "seashore", "valley", "alp", "cliff", "promontory", "geyser",
    "volcano", "coral reef", "sandbar", "planetarium", "maze", "velvet", "wool", "jersey",
    "maillot", "suit", "abaya", "cloak", "gown", "kimono", "sarong", "brassiere", "bikini",
    "spider web", "hen of the woods",
}


GLOBAL_NEGATIVE = "no duplicate object, no extra object, no text, no watermark, no cropped body"


def normalize_label(label: str) -> str:
    label = label.split(",", 1)[0]
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip().lower()
    return label


def article(label: str) -> str:
    return "an" if label[:1] in {"a", "e", "i", "o", "u"} else "a"


def load_imagenet_labels() -> list[str]:
    from torchvision.models import ResNet50_Weights

    categories = ResNet50_Weights.DEFAULT.meta.get("categories") or []
    labels: list[str] = []
    seen: set[str] = set()
    for category in categories:
        label = normalize_label(str(category))
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if len(labels) < 900:
        raise RuntimeError(f"expected ImageNet 1k labels, found {len(labels)}")
    return labels


def load_wordnet_object_labels(limit: int) -> list[str]:
    from nltk.corpus import wordnet as wn

    physical = wn.synset("physical_entity.n.01")
    labels: list[str] = []
    seen: set[str] = set()
    for synset in wn.all_synsets("n"):
        if physical not in {node for path in synset.hypernym_paths() for node in path}:
            continue
        lemmas = [normalize_label(lemma.name()) for lemma in synset.lemmas()]
        for label in lemmas:
            if not label or label in seen:
                continue
            if len(label) < 3 or len(label) > 36:
                continue
            if any(char.isdigit() for char in label):
                continue
            words = label.split()
            if len(words) > 4:
                continue
            if any(len(word) == 1 for word in words):
                continue
            seen.add(label)
            labels.append(label)
            break
        if len(labels) >= limit > 0:
            break
    if limit > 0 and len(labels) < limit:
        raise RuntimeError(f"requested {limit} WordNet object labels, found {len(labels)}")
    return labels


def load_labels_from_jsonl(path: Path, limit: int) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            label = normalize_label(str(record.get("label") or ""))
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
            if limit > 0 and len(labels) >= limit:
                break
    if limit > 0 and len(labels) < limit:
        raise RuntimeError(f"requested {limit} labels from {path}, found {len(labels)}")
    return labels


def object_family(label: str) -> str:
    padded = f" {label} "
    # Match complete label tokens only. Suffix matching misclassifies labels like
    # "colobus" as "bus" and "crane bird" as construction equipment.
    if any(f" {term} " in padded for term in ANIMAL_TERMS):
        return "animal"
    if any(f" {term} " in padded for term in VEHICLE_TERMS):
        return "vehicle"
    if any(f" {term} " in padded for term in FURNITURE_TERMS):
        return "furniture"
    if any(f" {term} " in padded for term in TOOL_TERMS):
        return "tool"
    if any(f" {term} " in padded for term in FOOD_TERMS):
        return "food"
    return "object"


def expected_parts(label: str, family: str) -> list[str]:
    if family == "vehicle":
        if "bicycle" in label:
            return ["two wheels", "frame", "handlebar", "saddle"]
        if "motorcycle" in label:
            return ["two wheels", "handlebars", "seat", "front fork", "rear wheel"]
        if any(term in label for term in ("bus", "truck", "van", "car")):
            return ["main body", "front", "rear", "windows", "wheels"]
        return ["main body", "front", "rear", "wheels"]
    if family == "animal":
        if any(term in label for term in (
            "bird", "cock", "hen", "ostrich", "finch", "junco", "bunting", "robin", "bulbul",
            "jay", "magpie", "chickadee", "eagle", "hawk", "kite", "vulture", "owl", "duck",
            "drake", "goose", "swan", "penguin", "peacock", "lorikeet", "hummingbird",
            "toucan", "pelican", "albatross", "stork", "flamingo", "heron", "egret",
            "crane", "sandpiper", "kingfisher", "cormorant",
        )):
            return ["head", "body", "wings", "beak", "legs"]
        if any(term in label for term in ("fish", "tench", "shark", "hammerhead", "ray", "stingray", "eel", "salmon", "trout", "whale", "dolphin")):
            return ["head", "body", "tail", "fins"]
        if any(term in label for term in ("snake",)):
            return ["head", "continuous body", "tail"]
        return ["head", "body", "legs", "tail"]
    if family == "furniture":
        return ["main surface", "supports", "back or frame"]
    if family == "tool":
        return ["handle", "working end", "single connected object"]
    if family == "food":
        return ["single item", "clear outline", "natural surface texture"]
    return ["single object", "clear outline", "main body"]


def base_constraints(family: str) -> str:
    if family == "vehicle":
        return "physically plausible vehicle structure, all support parts visible"
    if family == "animal":
        return "physically plausible anatomy, all main body parts connected"
    if family == "furniture":
        return "physically plausible furniture structure, stable supports"
    return "physically plausible structure, single connected object"


def variants_for(label: str, family: str) -> list[tuple[str, str, str, str]]:
    a = article(label)
    constraints = base_constraints(family)
    base = (
        "realistic studio product photo of exactly one {label}, fully visible and centered, "
        "plain white background, single object only, {negative}, {constraints}"
    )
    rows = [
        ("scaffold", "canonical_front", base),
        ("scaffold", "canonical_side", base.replace("centered", "centered, side view")),
        ("scaffold", "small_centered", base.replace("centered", "small centered with wide margin")),
        ("scaffold", "large_centered", base.replace("centered", "large centered with full object still visible")),
        ("scaffold", "left_position", base.replace("centered", "slightly left of center with full object visible")),
        ("scaffold", "right_position", base.replace("centered", "slightly right of center with full object visible")),
        ("scaffold", "soft_light", base.replace("plain white background", "plain white background, soft studio lighting")),
        ("scaffold", "catalog_light", base.replace("plain white background", "plain white background, even catalog lighting")),
        ("perspective", "front_three_quarter", base.replace("centered", "centered, front three-quarter view")),
        ("perspective", "rear_three_quarter", base.replace("centered", "centered, rear three-quarter view")),
        ("perspective", "slightly_above", base.replace("centered", "centered, viewed slightly from above")),
        ("perspective", "slightly_below", base.replace("centered", "centered, viewed slightly from below")),
        ("appearance", "dark_variant", base.replace("exactly one", "exactly one dark colored")),
        ("appearance", "light_variant", base.replace("exactly one", "exactly one light colored")),
        ("appearance", "matte_variant", base.replace("realistic studio product photo", "matte realistic studio product photo")),
        ("appearance", "glossy_variant", base.replace("realistic studio product photo", "glossy realistic studio product photo")),
        ("simple_scene", "floor_shadow", base.replace("plain white background", "plain white background with a faint ground shadow")),
        ("simple_scene", "tabletop_or_floor", "realistic photo of exactly one {label}, fully visible, simple clean surface, uncluttered background, single object only, {negative}, {constraints}"),
        ("hard_structure", "clear_separation", base.replace("centered", "centered, clear silhouette with all main parts separated")),
        ("eval", "heldout_catalog", base.replace("realistic studio product photo", "held-out clean catalog photo")),
    ]
    return [(axis, variant, template.format(label=label, article=a, negative=GLOBAL_NEGATIVE, constraints=constraints), constraints) for axis, variant, template in rows]


def build(args: argparse.Namespace) -> None:
    if args.source == "imagenet":
        labels = load_imagenet_labels()
        source_dataset = "torchvision_imagenet_1k_categories"
    elif args.source == "wordnet":
        labels = load_wordnet_object_labels(args.objects)
        source_dataset = "wordnet_physical_entity_nouns"
    elif args.source == "jsonl":
        labels = load_labels_from_jsonl(Path(args.source_jsonl), args.objects)
        source_dataset = f"jsonl_labels:{args.source_jsonl}"
    else:
        raise ValueError(f"unknown source: {args.source}")

    rng = random.Random(args.seed)
    if args.shuffle_objects:
        rng.shuffle(labels)

    include_families = {
        family.strip().lower()
        for family in str(args.include_families).split(",")
        if family.strip()
    }
    exclude_terms = set(DEFAULT_EXCLUDE_TERMS)
    exclude_terms.update(
        normalize_label(term)
        for term in str(args.exclude_terms).split(",")
        if term.strip()
    )
    if include_families or exclude_terms:
        filtered_labels = []
        for label in labels:
            family = object_family(label)
            if include_families and family not in include_families:
                continue
            if any(term == label or f" {term} " in f" {label} " for term in exclude_terms):
                continue
            filtered_labels.append(label)
            if args.objects > 0 and len(filtered_labels) >= args.objects:
                break
        labels = filtered_labels
        if args.objects > 0 and len(labels) < args.objects:
            print(f"warning: requested {args.objects} filtered labels, found {len(labels)}")

    if args.objects > 0:
        labels = labels[: args.objects]

    records = []
    for object_index, label in enumerate(labels):
        family = object_family(label)
        parts = expected_parts(label, family)
        object_id = f"{family}.{re.sub(r'[^a-z0-9]+', '_', label).strip('_')}"
        variants = variants_for(label, family)[: args.transforms_per_object]
        for variant_index, (axis, variant, prompt, _constraints) in enumerate(variants):
            records.append(
                {
                    "object_id": object_id,
                    "label": label,
                    "object_family": family,
                    "variant_axis": axis,
                    "variant": variant,
                    "prompt": prompt,
                    "expected_parts": parts,
                    "negative_constraints": [GLOBAL_NEGATIVE, "single object only", "fully visible"],
                    "seed_offset": object_index * 1000 + variant_index,
                    "source_dataset": source_dataset,
                    "source_name": "sana_object_manifold_prompts",
                    "source_index": object_index,
                    "curriculum_stage": args.curriculum_stage,
                    "domain": "synthetic_single_object_manifold",
                    "risk_tags": [family, axis],
                }
            )
    if args.interleave_variants:
        records.sort(key=lambda item: (int(item["seed_offset"]) % 1000, int(item["source_index"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "artifact_kind": "sana_object_manifold_prompt_corpus",
        "source": args.source,
        "objects": len(labels),
        "rows": len(records),
        "transforms_per_object": int(args.transforms_per_object),
        "curriculum_stage": args.curriculum_stage,
        "interleave_variants": bool(args.interleave_variants),
        "seed": int(args.seed),
        "output": str(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured multi-transform object manifold prompts for SANA student distillation.")
    parser.add_argument("--output", default="data/vision/prompts/sana_imagenet1k_object_manifold_20x_v0.jsonl")
    parser.add_argument("--objects", type=int, default=1000)
    parser.add_argument("--transforms-per-object", type=int, default=20)
    parser.add_argument("--curriculum-stage", default="imagenet1k_object_manifold_20x_v0")
    parser.add_argument("--source", choices=("imagenet", "wordnet", "jsonl"), default="imagenet")
    parser.add_argument("--source-jsonl", default="data/vision/prompts/sana_wordnet10k_object_manifold_scaffold8_interleaved_v0.jsonl")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--shuffle-objects", action="store_true")
    parser.add_argument("--interleave-variants", action="store_true")
    parser.add_argument(
        "--include-families",
        default="",
        help="Comma-separated families to keep, e.g. animal,vehicle,furniture,tool,food. Empty keeps all families.",
    )
    parser.add_argument(
        "--exclude-terms",
        default="",
        help="Comma-separated normalized labels to exclude in addition to the built-in ambiguous label blocklist.",
    )
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
