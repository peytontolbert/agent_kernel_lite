#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


PART_HINTS = {
    "bicycle": "two round wheels, thin frame, handlebar, pedals and seat",
    "motorcycle": "two wheels, fork, handlebar, seat and engine",
    "bus": "rectangular body, windows, windshield, doors and wheels",
    "car": "car body, windshield, side windows, headlights and wheels",
    "truck": "truck cab, cargo body, windshield and wheels",
    "dog": "one head, one body, four separate legs and tail",
    "cat": "one head, one body, four separate legs and tail",
    "horse": "one head, one body, four separate legs and tail",
    "bird": "head, body, wings, beak and legs",
}

BASE_CONSTRAINTS = (
    "plain white background",
    "single object only",
    "no duplicate object",
    "no scenery",
    "no text",
    "no watermark",
    "full object visible",
)


def object_id_for(label: str, index: int) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"
    return f"realcrop.{safe}.{index:05d}"


def clean_label(value: Any) -> str:
    label = str(value or "").split(",", 1)[0]
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"\s+", " ", label).strip().lower()
    return label


def part_hint(label: str) -> str:
    padded = f" {label} "
    for key, hint in PART_HINTS.items():
        if key in padded or label.endswith(key):
            return hint
    return "single connected object, clear outline, physically plausible structure"


def variant_specs(label: str, orientation: str, hint: str, variant_set: str) -> list[tuple[str, str, str]]:
    object_phrase = f"exactly one {label}"
    view_phrase = f", {orientation}" if orientation else ""
    specs = [
        (
            "canonical_teacher_object",
            "canonical",
            f"realistic studio product photo of {object_phrase}, fully visible and centered{view_phrase}, plain white background, single object only, no duplicate object, no scenery, no text, {hint}",
        ),
        (
            "clean_catalog_teacher_object",
            "catalog",
            f"clean catalog photo of one {label}, full object visible, centered, plain white background, no background scene, no duplicate object, {hint}",
        ),
        (
            "structure_teacher_object",
            "structure",
            f"isolated {label} object, full silhouette visible, all main parts connected and separated, no extra parts, no text, plain white background, {hint}",
        ),
    ]
    if variant_set == "bridge3":
        return specs

    scaffold = [
        (
            "front_view_teacher_object",
            "view",
            f"realistic studio product photo of {object_phrase}, front view, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "side_view_teacher_object",
            "view",
            f"realistic studio product photo of {object_phrase}, side view, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "three_quarter_teacher_object",
            "view",
            f"realistic studio product photo of {object_phrase}, front three-quarter view, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "simple_pose_teacher_object",
            "pose",
            f"realistic studio product photo of {object_phrase}, simple natural pose, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "large_centered_teacher_object",
            "scale",
            f"realistic studio product photo of {object_phrase}, large centered full-object view with margins, plain white background, {hint}, no cropped body",
        ),
    ]
    specs = specs + scaffold
    if variant_set == "scaffold8":
        return specs[:8]

    more = [
        (
            "rear_three_quarter_teacher_object",
            "view",
            f"realistic studio product photo of {object_phrase}, rear three-quarter view, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "small_centered_teacher_object",
            "scale",
            f"realistic studio product photo of {object_phrase}, small centered full object with wide margins, plain white background, {hint}, no duplicate object",
        ),
        (
            "matte_material_teacher_object",
            "material",
            f"realistic studio product photo of {object_phrase}, matte natural material, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "glossy_lighting_teacher_object",
            "lighting",
            f"realistic studio product photo of {object_phrase}, soft studio lighting, visible surface detail, fully visible and centered, plain white background, {hint}",
        ),
        (
            "left_facing_teacher_object",
            "orientation",
            f"realistic studio product photo of {object_phrase}, facing left, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "right_facing_teacher_object",
            "orientation",
            f"realistic studio product photo of {object_phrase}, facing right, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "low_camera_teacher_object",
            "camera",
            f"realistic studio product photo of {object_phrase}, slightly low camera angle, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "high_camera_teacher_object",
            "camera",
            f"realistic studio product photo of {object_phrase}, slightly high camera angle, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "plain_gray_shadow_teacher_object",
            "lighting",
            f"realistic studio product photo of {object_phrase}, subtle ground shadow, fully visible and centered, plain white background, {hint}, no duplicate object",
        ),
        (
            "silhouette_clarity_teacher_object",
            "structure",
            f"isolated {label}, clear readable silhouette, all major parts separated and attached correctly, fully visible, plain white background, {hint}",
        ),
        (
            "part_clarity_teacher_object",
            "structure",
            f"realistic studio product photo of {object_phrase}, all important parts clearly separated and connected correctly, fully visible and centered, plain white background, {hint}",
        ),
        (
            "hard_no_extra_parts_teacher_object",
            "hard_negative_guard",
            f"realistic studio product photo of {object_phrase}, physically plausible single object, no extra limbs, no extra wheels, no duplicate body, fully visible and centered, plain white background, {hint}",
        ),
    ]
    return (specs + more)[:20]


def iter_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def build(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    rows = iter_rows(source_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    teacher_rows: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = clean_label(row.get("class_label") or row.get("label"))
        if not label:
            continue
        orientation = str(row.get("orientation") or row.get("view") or "").strip().lower()
        hint = part_hint(label)
        seed = int(row.get("seed") or (args.seed + index))
        object_id = object_id_for(label, index)
        base_meta = {
            "object_id": object_id,
            "label": label,
            "class_label": row.get("class_label") or label,
            "source_dataset": row.get("source_dataset") or str(source_path),
            "source_index": index,
            "source_ref": row.get("teacher_ref") or row.get("source_ref"),
            "source_bbox": row.get("source_bbox"),
            "area_frac": row.get("area_frac"),
        }
        variants = variant_specs(label, orientation, hint, args.variant_set)
        for variant_index, (variant, axis, prompt) in enumerate(variants):
            teacher_rows.append(
                {
                    **base_meta,
                    "prompt": prompt,
                    "seed": seed + variant_index * 100000,
                    "domain": "real_crop_label_to_synthetic_teacher_object",
                    "curation": args.curation,
                    "variant_axis": axis,
                    "variant": variant,
                    "variant_index": variant_index,
                    "curriculum_stage": f"real_crop_teacher_bridge_{args.variant_set}",
                    "real_ref": row.get("teacher_ref") or row.get("source_ref"),
                    "expected_parts": hint,
                    "negative_constraints": list(BASE_CONSTRAINTS),
                }
            )
        if row.get("teacher_ref"):
            real_rows.append(
                {
                    **base_meta,
                    "prompt": row.get("prompt")
                    or f"realistic photo of one {label}, full object visible, natural real image crop, {hint}",
                    "teacher_ref": row["teacher_ref"],
                    "seed": seed,
                    "domain": "real_object_crop_target_deferred",
                    "curation": args.curation,
                    "object_id": object_id,
                    "variant_axis": "real_target",
                    "variant": "real_crop_target",
                    "variant_index": 0,
                    "curriculum_stage": "real_crop_target_deferred",
                    "expected_parts": hint,
                    "negative_constraints": ["single main object", "physically plausible object"],
                }
            )

    if args.interleave_variants:
        teacher_rows.sort(key=lambda row: (int(row.get("variant_index", 0)), int(row.get("source_index", 0))))

    teacher_output = Path(args.teacher_output)
    real_output = Path(args.real_output)
    teacher_output.parent.mkdir(parents=True, exist_ok=True)
    real_output.parent.mkdir(parents=True, exist_ok=True)
    with teacher_output.open("w", encoding="utf-8") as handle:
        for row in teacher_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with real_output.open("w", encoding="utf-8") as handle:
        for row in real_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "artifact_kind": "real_crop_teacher_bridge_prompt_corpus",
        "source": str(source_path),
        "teacher_rows": len(teacher_rows),
        "real_rows": len(real_rows),
        "teacher_output": str(teacher_output),
        "real_output": str(real_output),
        "curation": args.curation,
        "variant_set": args.variant_set,
        "interleave_variants": bool(args.interleave_variants),
    }
    teacher_output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build staged synthetic-teacher and deferred-real target prompts from real single-object crops.")
    parser.add_argument("--source", default="data/vision/prompts/coco_single_object_real_anchor_v069.jsonl")
    parser.add_argument("--teacher-output", default="data/vision/prompts/sana_realcrop_teacher_object_bridge_v0.jsonl")
    parser.add_argument("--real-output", default="data/vision/prompts/sana_realcrop_deferred_real_targets_v0.jsonl")
    parser.add_argument("--curation", default="real_crop_teacher_bridge_v0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--variant-set", choices=("bridge3", "scaffold8", "all20"), default="bridge3")
    parser.add_argument("--interleave-variants", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
