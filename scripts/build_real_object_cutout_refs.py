#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from rembg import new_session, remove


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(json.loads(line))
    return rows


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def alpha_bbox(alpha: Image.Image, threshold: int) -> tuple[int, int, int, int] | None:
    mask = alpha.point(lambda value: 255 if value >= threshold else 0)
    return mask.getbbox()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create white-background real-object cutout references.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--labels", default="")
    parser.add_argument("--model", default="u2net")
    parser.add_argument("--alpha-threshold", type=int, default=24)
    parser.add_argument("--min-foreground-frac", type=float, default=0.04)
    parser.add_argument("--max-foreground-frac", type=float, default=0.92)
    parser.add_argument(
        "--max-border-foreground-frac",
        type=float,
        default=0.08,
        help="Reject cutouts whose foreground touches too much of the crop border, a proxy for cropped objects.",
    )
    parser.add_argument("--canvas-size", type=int, default=384)
    parser.add_argument("--object-scale", type=float, default=0.82)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_dir = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}

    rows = read_jsonl(input_path)
    session = new_session(args.model)
    accepted: list[dict[str, Any]] = []
    rejected = 0

    for row in rows:
        if len(accepted) >= args.max_rows:
            break
        label = str(row.get("label") or row.get("class_label") or "").strip()
        if labels and label not in labels:
            continue
        raw_path = row.get("teacher_ref") or row.get("image_path") or row.get("path")
        if not raw_path:
            rejected += 1
            continue
        image_path = resolve_path(str(raw_path))
        if not image_path.exists():
            rejected += 1
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            cutout = remove(image, session=session).convert("RGBA")
        except Exception:
            rejected += 1
            continue
        alpha = cutout.getchannel("A")
        bbox = alpha_bbox(alpha, args.alpha_threshold)
        if bbox is None:
            rejected += 1
            continue
        foreground = alpha.point(lambda value: 255 if value >= args.alpha_threshold else 0)
        foreground_data = list(foreground.getdata())
        foreground_frac = sum(1 for value in foreground_data if value) / max(1, foreground.width * foreground.height)
        if foreground_frac < args.min_foreground_frac or foreground_frac > args.max_foreground_frac:
            rejected += 1
            continue
        width, height = foreground.size
        border_values = []
        border_values.extend(foreground.crop((0, 0, width, 1)).getdata())
        border_values.extend(foreground.crop((0, height - 1, width, height)).getdata())
        border_values.extend(foreground.crop((0, 0, 1, height)).getdata())
        border_values.extend(foreground.crop((width - 1, 0, width, height)).getdata())
        border_frac = sum(1 for value in border_values if value) / max(1, len(border_values))
        if border_frac > args.max_border_foreground_frac:
            rejected += 1
            continue

        cropped = cutout.crop(bbox)
        max_side = max(cropped.size)
        target_side = max(1, int(args.canvas_size * args.object_scale))
        scale = target_side / max(1, max_side)
        new_size = (max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale)))
        cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (args.canvas_size, args.canvas_size), (255, 255, 255, 255))
        left = (args.canvas_size - cropped.width) // 2
        top = (args.canvas_size - cropped.height) // 2
        canvas.alpha_composite(cropped, (left, top))
        rgb = canvas.convert("RGB")

        out_name = f"{label}_{len(accepted):05d}_{image_path.stem}_cutout.jpg"
        out_path = output_dir / out_name
        rgb.save(out_path, quality=94)
        prompt = (
            f"realistic photo of exactly one {label}, fully visible and centered, plain white background, "
            "single object only, no duplicate object, no scenery, physically plausible structure"
        )
        accepted.append(
            {
                **row,
                "prompt": prompt,
                "teacher_ref": str(out_path),
                "source_real_ref": str(image_path),
                "label": label,
                "class_label": label,
                "domain": "real_object_cutout_grounding",
                "curriculum_stage": "real_object_cutout_white_background",
                "cutout_model": args.model,
                "cutout_foreground_frac": round(float(foreground_frac), 6),
                "cutout_border_foreground_frac": round(float(border_frac), 6),
                "cutout_bbox": [int(value) for value in bbox],
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in accepted:
        label = str(row.get("label") or "")
        counts[label] = counts.get(label, 0) + 1
    print(
        json.dumps(
            {
                "output_jsonl": str(output_path),
                "output_dir": str(output_dir),
                "accepted_rows": len(accepted),
                "rejected_rows": rejected,
                "accepted_by_label": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
