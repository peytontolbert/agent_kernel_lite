#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from train_flux_final_latent_refiner import final_rows_by_prompt_seed


class ImageDeartifactRefiner(nn.Module):
    def __init__(self, hidden: int = 96, depth: int = 8, residual_scale: float = 0.35) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, hidden, 3, padding=1), nn.SiLU()]
        for _ in range(max(int(depth) - 2, 0)):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU()]
        layers.append(nn.Conv2d(hidden, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image + self.residual_scale * self.net(image)).clamp(-1.0, 1.0)


def decode_flux_latents_tensor(pipe: Any, packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.vae.decode(latents, return_dict=False)[0]


def pil_to_tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    value = torch.from_numpy(__import__("numpy").asarray(image)).to(device=device, dtype=torch.float32)
    return (value.permute(2, 0, 1).unsqueeze(0) / 127.5) - 1.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = ((tensor[0].detach().float() + 1.0) * 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(image)


def image_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, pool: int = 5) -> torch.Tensor:
    pred_low = F.avg_pool2d(pred, kernel_size=pool, stride=1, padding=pool // 2)
    target_low = F.avg_pool2d(target, kernel_size=pool, stride=1, padding=pool // 2)
    return F.l1_loss(pred - pred_low, target - target_low)


def image_block_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = pred - target
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        groups = [residual[:, :, y::stride, x::stride].mean(dim=(-1, -2)) for y in range(stride) for x in range(stride)]
        stacked = torch.stack(groups, dim=0)
        loss = loss + (stacked - stacked.mean(dim=0, keepdim=True)).abs().mean()
        count += 1
    return loss / max(count, 1)


def make_contact_sheet(images: list[tuple[str, Path]], output_path: Path, thumb: int, cols: int) -> None:
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    label_h = 28
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (prompt, path) in enumerate(images):
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        image = Image.open(path).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb + 5), prompt[:28], fill=(0, 0, 0))
    sheet.save(output_path)


def build_pairs(args: argparse.Namespace, pipe: Any, device: torch.device) -> list[dict[str, torch.Tensor | str]]:
    prompts = read_prompts(Path(args.prompts), int(args.limit))
    seeds = [int(item) for item in Path(args.seed_csv).read_text(encoding="utf-8").strip().split(",") if item.strip()]
    final_rows = final_rows_by_prompt_seed(Path(args.target_dir))
    pairs: list[dict[str, torch.Tensor | str]] = []
    vae_dtype = next(pipe.vae.parameters()).dtype
    for index, prompt in enumerate(prompts):
        source_path = Path(args.source_dir) / f"sample_{index:03d}.png"
        source = pil_to_tensor(source_path, int(args.train_size), device)
        row = final_rows[(prompt, seeds[index])]
        target = torch.load(Path(args.target_dir) / str(row["target_path"]), map_location="cpu")
        target_latents = target["latents"].to(device=device, dtype=torch.float32)
        target_timestep = target["timestep"].to(device=device, dtype=torch.float32).view(-1, 1, 1)
        clean_latents = target_latents - (target_timestep / 1000.0) * target["teacher_target"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            target_pixels = decode_flux_latents_tensor(
                pipe,
                clean_latents.to(device=device, dtype=vae_dtype),
                int(args.height),
                int(args.width),
            ).float()
            if int(args.train_size) != int(args.height):
                target_pixels = F.interpolate(target_pixels, size=(int(args.train_size), int(args.train_size)), mode="bicubic", align_corners=False)
        pairs.append({"prompt": prompt, "source": source.detach().clone(), "target": target_pixels.detach().clone()})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny RGB image deartifact refiner for student image outputs.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--seed-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--contact-thumb", type=int, default=128)
    parser.add_argument("--contact-cols", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "samples"
    image_dir.mkdir(parents=True, exist_ok=True)
    teacher_args = argparse.Namespace(
        teacher_family="flux",
        teacher_model=args.teacher_model,
        dtype=args.dtype,
        variant="",
        local_files_only=args.local_files_only,
        quantize_transformer_4bit=False,
        bnb_4bit_quant_type="nf4",
        cpu_offload=args.cpu_offload,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    pipe = load_teacher(teacher_args)
    pipe.vae.requires_grad_(False)
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    pairs = build_pairs(args, pipe, device)
    refiner = ImageDeartifactRefiner(hidden=int(args.hidden), depth=int(args.depth), residual_scale=float(args.residual_scale)).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=float(args.lr), weight_decay=0.01)
    for step in range(1, int(args.train_steps) + 1):
        pair = pairs[(step - 1) % len(pairs)]
        source = pair["source"].to(device)
        target = pair["target"].to(device)
        refined = refiner(source)
        loss_l1 = F.l1_loss(refined, target)
        loss_hf = image_highfreq_loss(refined, target)
        loss_block = image_block_loss(refined, target)
        loss = loss_l1 + 1.5 * loss_hf + 2.0 * loss_block
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.item()), "l1": float(loss_l1.item()), "highfreq": float(loss_hf.item()), "block": float(loss_block.item())}), flush=True)
    torch.save(
        {"model": refiner.state_dict(), "config": {"hidden": int(args.hidden), "depth": int(args.depth), "residual_scale": float(args.residual_scale)}},
        output_dir / "image_deartifact_refiner.pt",
    )
    images: list[tuple[str, Path]] = []
    with torch.inference_mode():
        for index, pair in enumerate(pairs):
            refined = refiner(pair["source"].to(device))
            path = image_dir / f"sample_{index:03d}.png"
            tensor_to_pil(refined).save(path)
            images.append((str(pair["prompt"]), path))
    make_contact_sheet(images, output_dir / "contact_sheet.png", int(args.contact_thumb), int(args.contact_cols))
    print(json.dumps({"refiner": str(output_dir / "image_deartifact_refiner.pt"), "contact_sheet": str(output_dir / "contact_sheet.png")}), flush=True)


if __name__ == "__main__":
    main()
