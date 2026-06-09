#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


COLORS = [
    "black",
    "white",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "brown",
    "tan",
    "cream",
]

VIEWS = [
    "side view",
    "front view",
    "rear view",
    "front three-quarter view",
    "rear three-quarter view",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(json.loads(line))
    return rows


def image_path_for(row: dict[str, Any]) -> Path:
    raw = row.get("teacher_ref") or row.get("image_path") or row.get("path")
    if not raw:
        raise ValueError("row has no image path")
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def best_text(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_features: torch.Tensor,
    prompts: list[str],
    device: str,
) -> tuple[str, float]:
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = image_features @ text_features.T
        logit_scale = getattr(model, "logit_scale", None)
        if logit_scale is not None:
            logits = logits * logit_scale.exp().detach().to(logits.device)
        probs = logits.softmax(dim=-1)[0]
    index = int(torch.argmax(probs).item())
    return prompts[index], float(probs[index].item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Add CLIP-derived color/view attributes to real cutout prompts.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(args.device)
    model.eval()

    enriched: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label") or row.get("class_label") or "object").strip()
        image = Image.open(image_path_for(row)).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt").to(args.device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        color_prompts = [f"a photo of a {color} {label}" for color in COLORS]
        view_prompts = [f"a photo of one {label}, {view}" for view in VIEWS]
        color_prompt, color_score = best_text(model, processor, image_features, color_prompts, args.device)
        view_prompt, view_score = best_text(model, processor, image_features, view_prompts, args.device)
        color = color_prompt.replace("a photo of a ", "").rsplit(f" {label}", 1)[0]
        view = view_prompt.rsplit(", ", 1)[-1]
        prompt = (
            f"realistic photo of exactly one {color} {label}, {view}, fully visible and centered, "
            "plain white background, single object only, no duplicate object, no scenery, "
            "physically plausible structure"
        )
        enriched.append(
            {
                **row,
                "prompt": prompt,
                "clip_color": color,
                "clip_color_score": round(color_score, 6),
                "clip_view": view,
                "clip_view_score": round(view_score, 6),
                "variant_axis": "real_cutout_color_view",
                "variant": f"{color}_{view.replace(' ', '_')}",
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output_jsonl": str(output_path), "rows": len(enriched)}, indent=2))


if __name__ == "__main__":
    main()
