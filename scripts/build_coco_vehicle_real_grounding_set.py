#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


PROMPT_DETAIL = {
    "bus": "bus body, windows, windshield, doors and wheels",
    "truck": "truck cab, cargo body, windshield and wheels",
    "car": "car body, windshield, side windows, headlights and wheels",
    "motorcycle": "two wheels, fork, handlebar, seat and engine",
    "bicycle": "two wheels, thin frame, handlebar, pedals and seat",
}


def crop_with_margin(image: Image.Image, bbox: list[float], margin: float) -> Image.Image:
    x, y, w, h = bbox
    pad_x = w * margin
    pad_y = h * margin
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(image.width, int(x + w + pad_x))
    bottom = min(image.height, int(y + h + pad_y))
    return image.crop((left, top, right, bottom)).convert("RGB")


def bbox_touches_image_border(
    image: Image.Image,
    bbox: list[float],
    border_frac: float,
) -> bool:
    if border_frac <= 0:
        return False
    x, y, w, h = bbox
    margin_x = image.width * border_frac
    margin_y = image.height * border_frac
    return (
        x <= margin_x
        or y <= margin_y
        or x + w >= image.width - margin_x
        or y + h >= image.height - margin_y
    )


def square_pad_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Use the crop's average border color so padding does not become a new white-background domain.
    border = Image.new("RGB", (image.width, image.height), "white")
    px = image.load()
    samples = []
    for x in range(image.width):
        samples.append(px[x, 0])
        samples.append(px[x, image.height - 1])
    for y in range(image.height):
        samples.append(px[0, y])
        samples.append(px[image.width - 1, y])
    if samples:
        color = tuple(int(sum(channel) / len(samples)) for channel in zip(*samples))
        border = Image.new("RGB", (size, size), color)
    else:
        border = Image.new("RGB", (size, size), (238, 238, 238))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    border.paste(resized, (left, top))
    return border


def orientation_from_bbox(bbox: list[float]) -> str:
    _, _, w, h = bbox
    ratio = w / max(h, 1.0)
    if ratio >= 1.8:
        return "side view"
    if ratio <= 0.75:
        return "front or rear view"
    return "three-quarter view"


def prompt_for(category: str, orientation: str) -> str:
    detail = PROMPT_DETAIL.get(category, f"{category} object details")
    return (
        f"realistic photo of one {category}, full object visible in a real-world scene, "
        f"{orientation}, natural background kept in the crop, {detail}, physically plausible geometry"
    )


def save_contact_sheet(rows: list[dict], output_path: Path, thumb: int = 160, columns: int = 5) -> None:
    if not rows:
        return
    images = []
    for row in rows:
        image = Image.open(row["teacher_ref"]).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        images.append(image)
    rows_count = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows_count * (thumb + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (image, row) in enumerate(zip(images, rows)):
        x = (i % columns) * thumb
        y = (i // columns) * (thumb + 18)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + thumb + 2), f"{i:03d} {row['label']}", fill=(0, 0, 0), font=font)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build captioned COCO real-object grounding crops.")
    parser.add_argument("--output-dir", default="data/vision/real_images/coco_vehicle_real_grounding_v105")
    parser.add_argument("--output-jsonl", default="data/vision/prompts/coco_vehicle_real_grounding_v105.jsonl")
    parser.add_argument("--dataset", default="detection-datasets/coco")
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", default="bus,truck,car,motorcycle,bicycle")
    parser.add_argument("--per-category", type=int, default=24)
    parser.add_argument("--max-scan", type=int, default=60000)
    parser.add_argument("--min-area-frac", type=float, default=0.04)
    parser.add_argument("--max-area-frac", type=float, default=0.72)
    parser.add_argument("--max-other-large-objects", type=int, default=1)
    parser.add_argument("--reject-border-touch-frac", type=float, default=0.015)
    parser.add_argument("--margin", type=float, default=0.28)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=20261100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    wanted = [item.strip() for item in args.categories.split(",") if item.strip()]
    counts = {category: 0 for category in wanted}
    rows = []
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    label_feature = ds.features["objects"].feature["category"]
    id_to_name = {index: name for index, name in enumerate(label_feature.names)}
    wanted_ids = {index for index, name in id_to_name.items() if name in counts}

    for row_index, row in enumerate(ds):
        if row_index >= args.max_scan or all(count >= args.per_category for count in counts.values()):
            break
        image = row["image"].convert("RGB")
        image_area = max(1, image.width * image.height)
        categories = row["objects"]["category"]
        boxes = row["objects"]["bbox"]
        large_objects = sum(1 for box in boxes if (box[2] * box[3]) / image_area >= args.min_area_frac)
        for obj_index, (category_id, bbox) in enumerate(zip(categories, boxes)):
            if category_id not in wanted_ids:
                continue
            category = id_to_name[category_id]
            if counts[category] >= args.per_category:
                continue
            area_frac = (bbox[2] * bbox[3]) / image_area
            if area_frac < args.min_area_frac:
                continue
            if area_frac > args.max_area_frac:
                continue
            if bbox_touches_image_border(image, [float(v) for v in bbox], args.reject_border_touch_frac):
                continue
            if large_objects > args.max_other_large_objects + 1:
                continue
            crop = crop_with_margin(image, [float(v) for v in bbox], args.margin)
            if min(crop.size) < 96:
                continue
            squared = square_pad_resize(crop, args.size)
            orientation = orientation_from_bbox([float(v) for v in bbox])
            file_name = f"{category}_{counts[category]:03d}_image{row['image_id']}_obj{obj_index}.jpg"
            path = output_dir / file_name
            squared.save(path, quality=94)
            rows.append(
                {
                    "prompt": prompt_for(category, orientation),
                    "teacher_ref": str(path),
                    "label": category,
                    "class_label": category,
                    "domain": "real_captioned_object_crop",
                    "curation": "coco_vehicle_real_grounding_v105",
                    "source_dataset": args.dataset,
                    "source_image_id": int(row["image_id"]),
                    "source_object_index": int(obj_index),
                    "source_bbox": [float(value) for value in bbox],
                    "area_frac": round(float(area_frac), 5),
                    "orientation": orientation,
                    "seed": args.seed_base + len(rows),
                }
            )
            counts[category] += 1
            if all(count >= args.per_category for count in counts.values()):
                break

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    save_contact_sheet(rows, output_dir / "contact_sheet.png")
    print(json.dumps({"output_jsonl": str(output_jsonl), "rows": len(rows), "counts": Counter(r["label"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
