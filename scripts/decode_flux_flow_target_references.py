#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from train_agentkernel_lite_image_flux_flow_rollout_distill import decode_flux_packed_with_vae, load_flux_vae


def sequence_key(row: dict[str, object]) -> str:
    target_id = str(row.get("target_id") or "")
    match = re.match(r"(.+)_t\d+$", target_id)
    if match:
        return match.group(1)
    return f"{row.get('prompt', '')}:{row.get('seed', '')}"


def load_final_rows(target_dir: Path, limit: int) -> list[dict[str, object]]:
    metadata_path = target_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    grouped: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "target_path" not in row:
            raise ValueError(f"metadata row missing target_path at {metadata_path}:{line_number}")
        key = sequence_key(row)
        timestep = int(row.get("timestep_index", -1))
        previous = grouped.get(key)
        if previous is None or timestep > int(previous.get("timestep_index", -1)):
            grouped[key] = row
    rows = list(grouped.values())
    rows.sort(key=lambda item: (int(item.get("seed_index", 0) or 0), str(item.get("prompt", ""))))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no final target rows found in {metadata_path}")
    return rows


def tensor_to_pil(decoded: torch.Tensor) -> Image.Image:
    image = ((decoded[0].detach().float() / 2.0) + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(image)


def make_contact_sheet(images: list[tuple[str, int, Path]], output_path: Path, thumb: int, cols: int) -> None:
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    label_h = 52
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (prompt, seed, path) in enumerate(images):
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        image = Image.open(path).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + thumb + 6), prompt[:34], fill=(0, 0, 0))
        draw.text((x + 6, y + thumb + 25), f"seed {seed}", fill=(70, 70, 70))
    sheet.save(output_path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Decode final FLUX flow target latents into teacher reference images.")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decoded-vae-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--decoded-vae-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--contact-thumb", type=int, default=192)
    parser.add_argument("--contact-cols", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=0)
    parser.add_argument("--decode-clean-terminal", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    vae = load_flux_vae(args.decoded_vae_model, device, args.decoded_vae_dtype, args.local_files_only)
    if vae is None:
        raise ValueError("--decoded-vae-model is required")

    decoded_images: list[tuple[str, int, Path]] = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(load_final_rows(target_dir, args.limit)):
            target = torch.load(target_dir / str(row["target_path"]), map_location="cpu")
            latents = target["latents"].to(device=device, dtype=torch.float32)
            if args.decode_clean_terminal:
                timestep = target["timestep"].to(device=device, dtype=torch.float32).view(-1, 1, 1)
                latents = latents - (timestep / 1000.0) * target["teacher_target"].to(device=device, dtype=torch.float32)
            width = int(row.get("width", 512) or 512)
            height = int(row.get("height", 512) or 512)
            decoded = decode_flux_packed_with_vae(vae, latents, height, width)
            if args.downsample > 1:
                decoded = F.interpolate(decoded, size=(args.downsample, args.downsample), mode="bilinear", align_corners=False)
                decoded = F.interpolate(decoded, size=(height, width), mode="bilinear", align_corners=False)
            prompt = str(row.get("prompt") or "")
            seed = int(row.get("seed", 0) or 0)
            image_path = image_dir / f"teacher_reference_{index:03d}.png"
            tensor_to_pil(decoded).save(image_path)
            decoded_images.append((prompt, seed, image_path))
            manifest.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "seed": seed,
                        "timestep_index": int(row.get("timestep_index", -1)),
                        "target_path": str(row["target_path"]),
                        "image_path": str(image_path),
                        "decode_clean_terminal": bool(args.decode_clean_terminal),
                    }
                )
                + "\n"
            )
            print(json.dumps({"prompt": prompt, "seed": seed, "path": str(image_path)}), flush=True)

    contact_path = output_dir / "contact_sheet.png"
    make_contact_sheet(decoded_images, contact_path, int(args.contact_thumb), int(args.contact_cols))
    print(json.dumps({"contact_sheet": str(contact_path), "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
