#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VEHICLE_PARTS = {
    "bicycle": ["two wheels", "frame", "handlebar", "saddle", "pedals"],
    "motorcycle": ["two wheels", "front fork", "handlebars", "seat", "rear wheel"],
    "bus": ["long rectangular body", "side windows", "doors", "two axles", "black tires"],
    "truck": ["cab", "cargo body", "front", "rear", "wheels"],
    "car": ["main body", "windshield", "side windows", "front", "rear", "wheels"],
    "van": ["boxy body", "windshield", "side windows", "sliding door", "wheels"],
}

TRANSFORMS = [
    ("view", "left_side_view", "left side view, full side profile visible"),
    ("view", "right_side_view", "right side view, full side profile visible"),
    ("view", "front_three_quarter_view", "front three-quarter view, front and side visible"),
    ("view", "rear_three_quarter_view", "rear three-quarter view, rear and side visible"),
    ("view", "front_view_simple", "front view, symmetric front visible, no duplicated side body"),
    ("view", "rear_view_simple", "rear view, symmetric rear visible, no duplicated side body"),
    ("scale", "small_centered", "small centered object with wide margin"),
    ("scale", "large_centered", "large centered object with full object still visible"),
    ("position", "slightly_left", "slightly left of center with full object visible"),
    ("position", "slightly_right", "slightly right of center with full object visible"),
    ("orientation", "slight_turn_toward_viewer", "slightly turned toward the viewer, coherent single body"),
    ("orientation", "slight_turn_away", "slightly turned away from the viewer, coherent single body"),
    ("lighting", "soft_studio_light", "soft studio lighting with a faint ground shadow"),
    ("lighting", "catalog_even_light", "even catalog lighting with clean shadow"),
    ("appearance", "matte_surface", "matte realistic surface, object geometry unchanged"),
    ("appearance", "glossy_surface", "glossy realistic surface, object geometry unchanged"),
    ("background", "plain_white_floor_shadow", "plain white background with a subtle floor shadow"),
    ("background", "simple_gray_floor", "simple light gray studio floor, uncluttered background"),
    ("structure", "clear_support_parts", "clear separated support parts, wheels attached in correct places"),
    ("eval", "heldout_clean_catalog", "clean catalog photo, fully visible, physically plausible geometry"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip().lower()


def label_for(row: dict[str, Any]) -> str:
    return normalize(str(row.get("label") or row.get("class_label") or row.get("object_label") or "object"))


def color_for(row: dict[str, Any]) -> str:
    color = normalize(str(row.get("clip_color") or ""))
    return color if color else "realistic colored"


def parts_for(label: str) -> list[str]:
    return VEHICLE_PARTS.get(label, ["single connected object", "clear outline", "physically plausible parts"])


def prompt_for(label: str, color: str, transform_text: str, parts: list[str], include_scene: bool) -> str:
    part_text = ", ".join(parts)
    background = "plain white background"
    if include_scene:
        background = "simple clean studio setting"
    return (
        f"realistic photo of exactly one {color} {label}, {transform_text}, fully visible and centered, "
        f"{background}, single object only, no duplicate object, no extra wheels, no extra body parts, "
        f"no cropped object, physically plausible structure, {part_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand real cutout-derived object rows into controlled teacher-generation transform prompts."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--transforms-per-object", type=int, default=20)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--labels", default="")
    parser.add_argument("--curriculum-stage", default="realderived_object_transform_20x_v0")
    parser.add_argument("--seed-base", type=int, default=20269000)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    allowed = {normalize(item) for item in args.labels.split(",") if item.strip()}
    if allowed:
        rows = [row for row in rows if label_for(row) in allowed]
    if args.max_objects > 0:
        rows = rows[: args.max_objects]

    records: list[dict[str, Any]] = []
    transforms = TRANSFORMS[: max(1, args.transforms_per_object)]
    for object_index, row in enumerate(rows):
        label = label_for(row)
        color = color_for(row)
        parts = parts_for(label)
        object_key = str(row.get("source_image_id") or row.get("seed") or object_index)
        for transform_index, (axis, variant, transform_text) in enumerate(transforms):
            records.append(
                {
                    "object_id": f"realderived.{label}.{object_key}",
                    "label": label,
                    "class_label": label,
                    "object_family": "vehicle" if label in VEHICLE_PARTS else "object",
                    "variant_axis": axis,
                    "variant": f"{color}_{variant}",
                    "prompt": prompt_for(
                        label=label,
                        color=color,
                        transform_text=transform_text,
                        parts=parts,
                        include_scene=axis == "background",
                    ),
                    "expected_parts": parts,
                    "negative_constraints": [
                        "single object only",
                        "fully visible",
                        "no duplicate object",
                        "no extra wheels",
                        "no cropped object",
                    ],
                    "seed": args.seed_base + object_index * 100 + transform_index,
                    "seed_offset": object_index * 1000 + transform_index,
                    "source_dataset": row.get("source_dataset", "real_cutout_derived"),
                    "source_real_ref": row.get("teacher_ref") or row.get("source_real_ref"),
                    "source_image_id": row.get("source_image_id"),
                    "source_object_index": row.get("source_object_index"),
                    "clip_color": color,
                    "clip_view": row.get("clip_view"),
                    "curriculum_stage": args.curriculum_stage,
                    "domain": "synthetic_teacher_from_real_object_transform",
                    "risk_tags": ["realderived", "topology", axis],
                }
            )

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "artifact_kind": "realderived_object_transform_prompt_corpus",
        "input_jsonl": args.input_jsonl,
        "output_jsonl": str(output),
        "objects": len(rows),
        "rows": len(records),
        "transforms_per_object": len(transforms),
        "labels": Counter(label_for(row) for row in rows),
        "curriculum_stage": args.curriculum_stage,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
