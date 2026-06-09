#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_LABELS = "bicycle,bus,car,chair,dog,fork,horse,motorcycle,scissors,truck"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(json.loads(line))
    return rows


def image_path_for_row(row: dict[str, Any]) -> Path | None:
    raw = next(
        (
            row.get(key)
            for key in ("teacher_ref", "real_ref", "source_ref", "image_ref", "image_path", "path")
            if row.get(key)
        ),
        None,
    )
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter real object reference rows with CLIP zero-shot label agreement.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", default="")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--min-prob", type=float, default=0.20)
    parser.add_argument("--min-margin", type=float, default=0.04)
    parser.add_argument(
        "--require-top1",
        action="store_true",
        help="Require the row label to be CLIP's top predicted label among the candidate labels.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    reject_path = Path(args.rejected_jsonl) if args.rejected_jsonl else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reject_path:
        reject_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    prompts = [f"a clear photo of one {label}" for label in labels]
    label_to_index = {label: index for index, label in enumerate(labels)}

    rows = read_jsonl(input_path)
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(args.device)
    model.eval()

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(args.device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    pending_images: list[Image.Image] = []

    def flush() -> None:
        if not pending_rows:
            return
        image_inputs = processor(images=pending_images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            logits = image_features @ text_features.T
            logit_scale = getattr(model, "logit_scale", None)
            if logit_scale is not None:
                logits = logits * logit_scale.exp().detach().to(logits.device)
            probs = logits.softmax(dim=-1).cpu()
        for row, prob in zip(pending_rows, probs, strict=True):
            label = str(row.get("label") or row.get("class_label") or "").strip()
            label_index = label_to_index.get(label)
            scored = dict(row)
            if label_index is None:
                scored["clip_filter_reason"] = "label_not_in_candidates"
                rejected.append(scored)
                continue
            values, indices = torch.sort(prob, descending=True)
            target_prob = float(prob[label_index])
            top_index = int(indices[0])
            top_prob = float(values[0])
            second_prob = float(values[1]) if len(values) > 1 else 0.0
            margin = target_prob - (top_prob if top_index != label_index else second_prob)
            scored.update(
                {
                    "clip_filter_model": args.model,
                    "clip_target_prob": target_prob,
                    "clip_top_label": labels[top_index],
                    "clip_top_prob": top_prob,
                    "clip_margin": margin,
                }
            )
            keep = target_prob >= args.min_prob and margin >= args.min_margin
            if args.require_top1:
                keep = keep and top_index == label_index
            if keep:
                scored["clip_filter_reason"] = ""
                accepted.append(scored)
            else:
                scored["clip_filter_reason"] = "low_label_agreement"
                rejected.append(scored)
        pending_rows.clear()
        pending_images.clear()

    for row in rows:
        image_path = image_path_for_row(row)
        if image_path is None:
            rejected.append({**row, "clip_filter_reason": "missing_image"})
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            rejected.append({**row, "clip_filter_reason": f"image_error:{exc}"})
            continue
        pending_rows.append(row)
        pending_images.append(image)
        if len(pending_rows) >= args.batch_size:
            flush()
    flush()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if reject_path:
        with reject_path.open("w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in accepted:
        label = str(row.get("label") or row.get("class_label") or "")
        counts[label] = counts.get(label, 0) + 1
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "accepted_rows": len(accepted),
                "rejected_rows": len(rejected),
                "output_jsonl": str(output_path),
                "rejected_jsonl": str(reject_path) if reject_path else "",
                "accepted_by_label": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
