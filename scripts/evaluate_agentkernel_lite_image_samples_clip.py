#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def as_feature_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for key in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, torch.Tensor):
            if key == "last_hidden_state":
                return candidate[:, 0]
            return candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"could not extract feature tensor from {type(value).__name__}")


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("{"):
            row = json.loads(text)
            text = str(row.get("prompt", "")).strip()
        if text:
            prompts.append(text)
    return prompts


def sample_paths(image_dir: Path) -> list[Path]:
    paths = sorted(image_dir.glob("sample_*.png"))
    if not paths:
        raise FileNotFoundError(f"no sample_*.png files found in {image_dir}")
    return paths


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    prompts = read_prompts(Path(args.prompts))
    paths = sample_paths(image_dir)
    if args.limit > 0:
        paths = paths[: args.limit]
        prompts = prompts[: args.limit]
    if len(prompts) != len(paths):
        raise ValueError(f"prompt/image count mismatch: {len(prompts)} prompts for {len(paths)} images")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    images = [Image.open(path).convert("RGB") for path in paths]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    image_features = as_feature_tensor(model.get_image_features(pixel_values=pixel_values))
    text_features = as_feature_tensor(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    similarity = image_features @ text_features.T
    diagonal = similarity.diag()
    ranks = []
    for index in range(similarity.shape[0]):
        order = torch.argsort(similarity[index], descending=True)
        rank = int((order == index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)

    rows = []
    for index, (prompt, path) in enumerate(zip(prompts, paths)):
        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "image": str(path),
                "clip_cosine": float(diagonal[index].item()),
                "clip_score_100": float(diagonal[index].item() * 100.0),
                "prompt_rank_within_batch": ranks[index],
            }
        )

    summary = {
        "image_dir": str(image_dir),
        "prompts": str(args.prompts),
        "model": args.model,
        "count": len(rows),
        "mean_clip_cosine": float(diagonal.mean().item()),
        "min_clip_cosine": float(diagonal.min().item()),
        "max_clip_cosine": float(diagonal.max().item()),
        "top1_prompt_retrieval": float(sum(rank == 1 for rank in ranks) / len(ranks)),
        "mean_prompt_rank": float(sum(ranks) / len(ranks)),
        "rows": rows,
    }
    output = Path(args.output) if args.output else image_dir / "clip_eval.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated image samples with CLIP prompt-image metrics.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
