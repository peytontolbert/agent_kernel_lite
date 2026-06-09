#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


PROMPT_BY_CATEGORY = {
    "bus": "realistic photo of one city bus side view, single main vehicle visible, rectangular body, straight windows, two visible wheels, physically plausible geometry",
    "car": "realistic photo of one car side view, single main vehicle visible, windshield, side windows, headlights, wheels, physically plausible geometry",
    "truck": "realistic photo of one truck side view, single main vehicle visible, cab, cargo body, wheels, physically plausible geometry",
    "bicycle": "realistic photo of one bicycle side view, single main object visible, two round wheels, frame, handlebar, pedals, physically plausible geometry",
    "motorcycle": "realistic photo of one motorcycle side view, single main object visible, two wheels, fork, handlebar, seat, frame, engine, physically plausible geometry",
    "dog": "realistic photo of one dog side view, single main animal visible, one head, one body, four separate legs, tail, physically plausible anatomy",
    "horse": "realistic photo of one horse side view, single main animal visible, one head, one body, four separate legs, tail, physically plausible anatomy",
    "chair": "realistic photo of one chair, single main object visible, seat, backrest, four attached legs, physically plausible geometry",
    "fork": "realistic photo of one fork, single main object visible, straight handle, four parallel prongs, physically plausible geometry",
    "scissors": "realistic photo of one pair of scissors, single main object visible, two handles, hinge, two crossing blades, physically plausible geometry",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small COCO single-object crop anchor set.")
    parser.add_argument("--output-dir", default="data/vision/real_images/coco_single_object_v069")
    parser.add_argument("--output-jsonl", default="data/vision/prompts/coco_single_object_real_anchor_v069.jsonl")
    parser.add_argument("--dataset", default="detection-datasets/coco")
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", default="bus,car,truck,bicycle,motorcycle,dog,horse,chair,fork,scissors")
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--min-area-frac", type=float, default=0.045)
    parser.add_argument("--max-other-large-objects", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Rewrite the JSONL every N accepted rows so interrupted long scans keep usable metadata.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    wanted = [item.strip() for item in args.categories.split(",") if item.strip()]
    counts = {category: 0 for category in wanted}
    rows = []

    def write_rows() -> None:
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item) + "\n")

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
            if large_objects > args.max_other_large_objects + 1:
                continue
            crop = crop_with_margin(image, bbox, args.margin)
            if min(crop.size) < 96:
                continue
            file_name = f"{category}_{counts[category]:03d}_image{row['image_id']}_obj{obj_index}.jpg"
            path = output_dir / file_name
            crop.save(path, quality=94)
            rows.append(
                {
                    "prompt": PROMPT_BY_CATEGORY.get(category, f"realistic photo of one {category}, single main object visible"),
                    "teacher_ref": str(path),
                    "label": category,
                    "source_dataset": args.dataset,
                    "source_image_id": int(row["image_id"]),
                    "source_object_index": int(obj_index),
                    "source_bbox": [float(value) for value in bbox],
                    "area_frac": round(float(area_frac), 5),
                    "seed": 20261000 + len(rows),
                }
            )
            counts[category] += 1
            if args.flush_every > 0 and len(rows) % args.flush_every == 0:
                write_rows()
            if all(count >= args.per_category for count in counts.values()):
                break

    write_rows()
    print(json.dumps({"output_jsonl": str(output_jsonl), "rows": len(rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
