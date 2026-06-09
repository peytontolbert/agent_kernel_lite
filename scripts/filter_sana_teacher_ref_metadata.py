#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def resolve_image_path(row: dict[str, Any], metadata_path: Path) -> Path | None:
    raw = row.get("teacher_ref") or row.get("image_ref") or row.get("path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return metadata_path.parent / path


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]
    return out


def largest_component_ratio(mask: np.ndarray) -> tuple[float, int]:
    foreground = int(mask.sum())
    if foreground <= 0:
        return 0.0, 0
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    largest = 0
    components = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        components += 1
        size = 0
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        seen[y0, x0] = True
        while queue:
            y, x = queue.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        largest = max(largest, size)
    return largest / max(foreground, 1), components


def score_image(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    image = Image.open(path).convert("RGB").resize((args.score_size, args.score_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    corners = np.concatenate(
        [
            arr[:8, :8].reshape(-1, 3),
            arr[:8, -8:].reshape(-1, 3),
            arr[-8:, :8].reshape(-1, 3),
            arr[-8:, -8:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    diff = np.abs(arr - bg).mean(axis=-1)
    mask = diff > float(args.foreground_threshold)
    foreground_fraction = float(mask.mean())
    foreground_diff_mean = float(diff[mask].mean()) if mask.any() else 0.0
    foreground_diff_p75 = float(np.percentile(diff[mask], 75)) if mask.any() else 0.0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "accepted": False,
            "reason": "empty_foreground",
            "foreground_fraction": foreground_fraction,
            "foreground_diff_mean": foreground_diff_mean,
            "foreground_diff_p75": foreground_diff_p75,
            "largest_component_ratio": 0.0,
            "components": 0,
            "border_fraction": 0.0,
        }
    border = np.zeros_like(mask, dtype=bool)
    border[: args.border_px, :] = True
    border[-args.border_px :, :] = True
    border[:, : args.border_px] = True
    border[:, -args.border_px :] = True
    border_fraction = float((mask & border).sum() / max(mask.sum(), 1))
    connected_mask = dilate(mask, args.dilation_radius)
    component_ratio, components = largest_component_ratio(connected_mask)
    reasons = []
    if foreground_fraction < args.min_foreground_fraction:
        reasons.append("too_little_foreground")
    if foreground_diff_mean < args.min_foreground_diff_mean:
        reasons.append("low_foreground_contrast")
    if foreground_diff_p75 < args.min_foreground_diff_p75:
        reasons.append("low_foreground_contrast_p75")
    if foreground_fraction > args.max_foreground_fraction:
        reasons.append("too_much_foreground")
    if border_fraction > args.max_border_fraction:
        reasons.append("border_touch")
    if component_ratio < args.min_largest_component_ratio:
        reasons.append("split_foreground")
    return {
        "accepted": not reasons,
        "reason": ",".join(reasons),
        "foreground_fraction": foreground_fraction,
        "foreground_diff_mean": foreground_diff_mean,
        "foreground_diff_p75": foreground_diff_p75,
        "largest_component_ratio": float(component_ratio),
        "components": int(components),
        "border_fraction": border_fraction,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generated Sana teacher refs for stable single-object training rows.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejects-output", default="")
    parser.add_argument("--score-size", type=int, default=128)
    parser.add_argument("--foreground-threshold", type=float, default=0.045)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.015)
    parser.add_argument("--min-foreground-diff-mean", type=float, default=0.0)
    parser.add_argument("--min-foreground-diff-p75", type=float, default=0.0)
    parser.add_argument("--max-foreground-fraction", type=float, default=0.75)
    parser.add_argument("--max-border-fraction", type=float, default=0.10)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.58)
    parser.add_argument("--dilation-radius", type=int, default=4)
    parser.add_argument("--border-px", type=int, default=3)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = Path(args.rejects_output) if args.rejects_output else output_path.with_suffix(".rejects.jsonl")

    accepted = 0
    rejected = 0
    with metadata_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as out, rejects_path.open("w", encoding="utf-8") as rej:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = resolve_image_path(row, metadata_path)
            if image_path is None or not image_path.exists():
                row["filter_reason"] = "missing_image"
                rej.write(json.dumps(row, ensure_ascii=False) + "\n")
                rejected += 1
                continue
            metrics = score_image(image_path, args)
            row.update({f"filter_{key}": value for key, value in metrics.items() if key != "accepted"})
            row["teacher_ref"] = str(image_path.resolve())
            if metrics["accepted"]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1
            else:
                rej.write(json.dumps(row, ensure_ascii=False) + "\n")
                rejected += 1
    print(json.dumps({"accepted": accepted, "rejected": rejected, "output": str(output_path), "rejects": str(rejects_path)}))


if __name__ == "__main__":
    main()
