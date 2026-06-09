#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_agentkernel_lite_image_teacher_corpus import decode_flux_latents, load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from sample_agentkernel_lite_image_flux_flow_distill import (
    load_cached_prompt_embeddings,
    load_student,
    prompt_seed_cache_key,
)
from train_agentkernel_lite_image_flux_flow_distill import pack_latent_grid, unpack_packed_latent_grid


class PackedLatentRefiner(nn.Module):
    def __init__(self, channels: int = 64, hidden: int = 192, depth: int = 6, residual_scale: float = 0.25) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU()]
        for _ in range(max(int(depth) - 2, 0)):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU()]
        layers.append(nn.Conv2d(hidden, channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = packed.shape
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError(f"packed latent token count is not square: {tokens}")
        grid = packed.float().transpose(1, 2).reshape(batch, channels, side, side)
        refined = grid + self.residual_scale * self.net(grid)
        return refined.flatten(2).transpose(1, 2).to(packed.dtype)


class FluxPackedLatentRefiner(nn.Module):
    def __init__(self, packed_channels: int = 64, hidden: int = 128, depth: int = 6, residual_scale: float = 0.25) -> None:
        super().__init__()
        if packed_channels % 4:
            raise ValueError("FLUX packed channels must be divisible by 4")
        latent_channels = packed_channels // 4
        layers: list[nn.Module] = [nn.Conv2d(latent_channels, hidden, 3, padding=1), nn.SiLU()]
        for _ in range(max(int(depth) - 2, 0)):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU()]
        layers.append(nn.Conv2d(hidden, latent_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        grid = unpack_packed_latent_grid(packed.float())
        refined = grid + self.residual_scale * self.net(grid)
        return pack_latent_grid(refined).to(dtype=packed.dtype)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class FluxPackedLatentUNetRefiner(nn.Module):
    def __init__(self, packed_channels: int = 64, hidden: int = 192, depth: int = 2, residual_scale: float = 0.25) -> None:
        super().__init__()
        if packed_channels % 4:
            raise ValueError("FLUX packed channels must be divisible by 4")
        latent_channels = packed_channels // 4
        hidden = int(hidden)
        mid = max(32, hidden // 2)
        low = max(32, hidden // 4)
        blocks = max(int(depth), 1)
        self.in_proj = nn.Conv2d(latent_channels, low, 3, padding=1)
        self.enc = nn.Sequential(*[ConvBlock(low) for _ in range(blocks)])
        self.down = nn.Conv2d(low, mid, 4, stride=2, padding=1)
        self.mid = nn.Sequential(*[ConvBlock(mid) for _ in range(blocks + 1)])
        self.up = nn.ConvTranspose2d(mid, low, 4, stride=2, padding=1)
        self.dec = nn.Sequential(*[ConvBlock(low) for _ in range(blocks)])
        self.out = nn.Conv2d(low, latent_channels, 3, padding=1)
        self.residual_scale = float(residual_scale)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        grid = unpack_packed_latent_grid(packed.float())
        skip = self.enc(self.in_proj(grid))
        mid = self.mid(self.down(skip))
        up = self.up(mid)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        refined = grid + self.residual_scale * self.out(self.dec(up + skip))
        return pack_latent_grid(refined).to(dtype=packed.dtype)


def packed_spatial_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, tokens, channels = pred.shape
    side = int(tokens**0.5)
    pred_grid = pred.float().reshape(batch, side, side, channels)
    target_grid = target.float().reshape(batch, side, side, channels)
    return F.l1_loss(pred_grid[:, 1:, :, :] - pred_grid[:, :-1, :, :], target_grid[:, 1:, :, :] - target_grid[:, :-1, :, :]) + F.l1_loss(
        pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :],
        target_grid[:, :, 1:, :] - target_grid[:, :, :-1, :],
    )


def packed_block_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, tokens, channels = pred.shape
    side = int(tokens**0.5)
    residual = pred.float().reshape(batch, side, side, channels) - target.float().reshape(batch, side, side, channels)
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        if side < stride:
            continue
        groups = [residual[:, y::stride, x::stride, :].mean(dim=(1, 2)) for y in range(stride) for x in range(stride)]
        stacked = torch.stack(groups, dim=0)
        loss = loss + (stacked - stacked.mean(dim=0, keepdim=True)).abs().mean()
        count += 1
    return loss / max(count, 1)


def flux_unpacked_spatial_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_grid = unpack_packed_latent_grid(pred.float())
    target_grid = unpack_packed_latent_grid(target.float())
    return F.l1_loss(pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :], target_grid[:, :, 1:, :] - target_grid[:, :, :-1, :]) + F.l1_loss(
        pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1],
        target_grid[:, :, :, 1:] - target_grid[:, :, :, :-1],
    )


def flux_unpacked_block_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = unpack_packed_latent_grid(pred.float()) - unpack_packed_latent_grid(target.float())
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        if residual.shape[-2] < stride or residual.shape[-1] < stride:
            continue
        groups = [residual[:, :, y::stride, x::stride].mean(dim=(-1, -2)) for y in range(stride) for x in range(stride)]
        stacked = torch.stack(groups, dim=0)
        loss = loss + (stacked - stacked.mean(dim=0, keepdim=True)).abs().mean()
        count += 1
    return loss / max(count, 1)


def decode_flux_latents_tensor(pipe: Any, packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.vae.decode(latents, return_dict=False)[0]


def image_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, pool: int = 5) -> torch.Tensor:
    pool = max(int(pool), 3)
    if pool % 2 == 0:
        pool += 1
    pred_low = F.avg_pool2d(pred, kernel_size=pool, stride=1, padding=pool // 2)
    target_low = F.avg_pool2d(target, kernel_size=pool, stride=1, padding=pool // 2)
    return F.l1_loss(pred - pred_low, target - target_low)


def image_block_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = pred - target
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        if residual.shape[-2] < stride or residual.shape[-1] < stride:
            continue
        groups = [residual[:, :, y::stride, x::stride].mean(dim=(-1, -2)) for y in range(stride) for x in range(stride)]
        stacked = torch.stack(groups, dim=0)
        loss = loss + (stacked - stacked.mean(dim=0, keepdim=True)).abs().mean()
        count += 1
    return loss / max(count, 1)


def final_rows_by_prompt_seed(target_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in (target_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["prompt"]), int(row["seed"]))
        if key not in rows or int(row["timestep_index"]) > int(rows[key]["timestep_index"]):
            rows[key] = row
    return rows


def rows_by_prompt_seed_timestep(target_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for line in (target_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(str(row["prompt"]), int(row["seed"]), int(row["timestep_index"]))] = row
    return rows


def tensor_to_pil(decoded: torch.Tensor | Image.Image) -> Image.Image:
    if isinstance(decoded, Image.Image):
        return decoded.convert("RGB")
    image = ((decoded[0].detach().float() / 2.0) + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(image)


def make_contact_sheet(images: list[tuple[str, Path]], output_path: Path, thumb: int, cols: int) -> None:
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    label_h = 36
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (prompt, path) in enumerate(images):
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        image = Image.open(path).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + thumb + 6), prompt[:32], fill=(0, 0, 0))
    sheet.save(output_path)


def build_examples_for_dataset(
    args: argparse.Namespace,
    pipe: Any,
    student: nn.Module,
    device: torch.device,
    *,
    target_dir: Path,
    prompts_path: Path,
    seed_csv_path: Path,
    limit: int,
    dataset_name: str,
) -> list[dict[str, torch.Tensor | str]]:
    prompts = read_prompts(prompts_path, int(limit))
    seeds = [int(item) for item in seed_csv_path.read_text(encoding="utf-8").strip().split(",") if item.strip()]
    if len(seeds) < len(prompts):
        raise ValueError(f"{dataset_name} has {len(prompts)} prompts but only {len(seeds)} seeds")
    if str(args.cached_initial_latent_timestep_indices).strip():
        start_indices = [int(item) for item in str(args.cached_initial_latent_timestep_indices).split(",") if item.strip()]
    else:
        start_indices = [int(args.cached_initial_latent_timestep_index)]
    embedding_cache = load_cached_prompt_embeddings(target_dir)
    final_rows = final_rows_by_prompt_seed(target_dir)
    timestep_rows = rows_by_prompt_seed_timestep(target_dir)
    examples: list[dict[str, torch.Tensor | str]] = []
    for index, prompt in enumerate(prompts):
        seed = seeds[index]
        key = prompt_seed_cache_key(prompt, seed)
        cached = embedding_cache[key]
        prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=pipe.dtype)
        pooled_prompt_embeds = cached["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
        row = final_rows[(prompt, seed)]
        target = torch.load(target_dir / str(row["target_path"]), map_location="cpu")
        target_latents = target["latents"].to(device=device, dtype=torch.float32)
        target_timestep = target["timestep"].to(device=device, dtype=torch.float32).view(-1, 1, 1)
        clean_latents = target_latents - (target_timestep / 1000.0) * target["teacher_target"].to(device=device, dtype=torch.float32)
        target_pixels = decode_flux_latents_tensor(
            pipe,
            clean_latents.to(device=device, dtype=next(pipe.vae.parameters()).dtype),
            int(args.height),
            int(args.width),
        ).float()
        for start_index in start_indices:
            if start_index >= 0:
                start_key = (prompt, seed, int(start_index))
                if start_key not in timestep_rows:
                    raise KeyError(f"missing cached timestep row for {start_key}")
                start_target = torch.load(target_dir / str(timestep_rows[start_key]["target_path"]), map_location="cpu")
                latents = start_target["latents"].to(device=device, dtype=pipe.dtype)
            else:
                latents = cached["initial_latents"].to(device=device, dtype=pipe.dtype)
            timesteps, _ = flux_timesteps(pipe, latents, int(args.steps), device)
            sampler_start_index = int(args.sampler_start_timestep_index)
            if sampler_start_index < 0 and start_index >= 0:
                sampler_start_index = int(start_index)
            if sampler_start_index > 0:
                if sampler_start_index >= len(timesteps):
                    raise ValueError(f"sampler start timestep index {sampler_start_index} >= number of steps {len(timesteps)}")
                timesteps = timesteps[sampler_start_index:]
            guidance = torch.full([1], float(args.guidance), device=device, dtype=torch.float32)
            with torch.no_grad():
                for step_index, timestep_value in enumerate(timesteps):
                    if step_index == 0 and hasattr(pipe.scheduler, "_step_index"):
                        pipe.scheduler._step_index = None
                    timestep = timestep_value.expand(latents.shape[0]).to(device)
                    pred = student(latents.float(), timestep.float(), prompt_embeds.float(), pooled_prompt_embeds.float(), guidance).to(latents.dtype)
                    latents = pipe.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
            examples.append(
                {
                    "prompt": f"{dataset_name}: {prompt} [t{start_index}]",
                    "student": latents.float().detach().clone(),
                    "target": clean_latents.detach().clone(),
                    "target_pixels": target_pixels.detach().clone(),
                    "start_index": str(start_index),
                    "dataset": dataset_name,
                }
            )
    return examples


def build_examples(args: argparse.Namespace, pipe: Any, student: nn.Module, device: torch.device) -> list[dict[str, torch.Tensor | str]]:
    specs: list[dict[str, Any]]
    if str(args.dataset_specs_jsonl).strip():
        specs = []
        for line_number, line in enumerate(Path(args.dataset_specs_jsonl).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("target_dir", "prompts", "seed_csv"):
                if key not in row:
                    raise ValueError(f"missing {key} in {args.dataset_specs_jsonl}:{line_number}")
            specs.append(row)
    else:
        specs = [
            {
                "name": "default",
                "target_dir": args.target_dir,
                "prompts": args.prompts,
                "seed_csv": args.seed_csv,
                "limit": int(args.limit),
            }
        ]
    examples: list[dict[str, torch.Tensor | str]] = []
    for index, spec in enumerate(specs):
        name = str(spec.get("name") or f"dataset{index:02d}")
        limit = int(spec.get("limit", args.limit))
        dataset_examples = build_examples_for_dataset(
            args,
            pipe,
            student,
            device,
            target_dir=Path(spec["target_dir"]),
            prompts_path=Path(spec["prompts"]),
            seed_csv_path=Path(spec["seed_csv"]),
            limit=limit,
            dataset_name=name,
        )
        print(json.dumps({"dataset": name, "examples": len(dataset_examples)}), flush=True)
        examples.extend(dataset_examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny final latent refiner on top of a FLUX packed-latent student.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dir", default="")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--seed-csv", default="")
    parser.add_argument("--dataset-specs-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cached-initial-latent-timestep-index", type=int, default=-1)
    parser.add_argument("--cached-initial-latent-timestep-indices", default="")
    parser.add_argument("--sampler-start-timestep-index", type=int, default=-1)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--init-refiner", default="")
    parser.add_argument("--refiner-architecture", choices=("conv", "flux_unet"), default="conv")
    parser.add_argument("--packing-mode", choices=("simple", "flux"), default="simple")
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.2)
    parser.add_argument("--packed-block-loss-weight", type=float, default=0.5)
    parser.add_argument("--decoded-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-image-loss-size", type=int, default=128)
    parser.add_argument("--decoded-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-block-loss-weight", type=float, default=0.0)
    parser.add_argument("--shuffle-examples", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=1337)
    parser.add_argument("--contact-thumb", type=int, default=160)
    parser.add_argument("--contact-cols", type=int, default=4)
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
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    pipe.vae.requires_grad_(False)
    student, _config = load_student(Path(args.checkpoint), device, "raw")
    examples = build_examples(args, pipe, student, device)
    print(json.dumps({"examples": len(examples), "shuffle_examples": bool(args.shuffle_examples)}), flush=True)
    if args.refiner_architecture == "flux_unet":
        if args.packing_mode != "flux":
            raise ValueError("--refiner-architecture flux_unet requires --packing-mode flux")
        refiner = FluxPackedLatentUNetRefiner(hidden=int(args.hidden), depth=int(args.depth), residual_scale=float(args.residual_scale)).to(device)
    elif args.packing_mode == "flux":
        refiner = FluxPackedLatentRefiner(hidden=int(args.hidden), depth=int(args.depth), residual_scale=float(args.residual_scale)).to(device)
    else:
        refiner = PackedLatentRefiner(hidden=int(args.hidden), depth=int(args.depth), residual_scale=float(args.residual_scale)).to(device)
    if str(args.init_refiner).strip():
        init_path = Path(args.init_refiner)
        init_payload = torch.load(init_path, map_location="cpu")
        init_config = init_payload.get("config", {})
        expected_config = {
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "residual_scale": float(args.residual_scale),
            "packing_mode": str(args.packing_mode),
            "architecture": str(args.refiner_architecture),
        }
        for key, expected in expected_config.items():
            if key in init_config and init_config[key] != expected:
                raise ValueError(f"init refiner {init_path} has {key}={init_config[key]!r}, expected {expected!r}")
        refiner.load_state_dict(init_payload["model"])
        print(json.dumps({"init_refiner": str(init_path)}), flush=True)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=float(args.lr), weight_decay=0.01)
    order = list(range(len(examples)))
    rng = random.Random(int(args.shuffle_seed))
    for step in range(1, int(args.train_steps) + 1):
        if bool(args.shuffle_examples) and (step == 1 or (step - 1) % len(order) == 0):
            rng.shuffle(order)
        example_index = order[(step - 1) % len(order)] if bool(args.shuffle_examples) else (step - 1) % len(examples)
        example = examples[example_index]
        source = example["student"].to(device)
        target = example["target"].to(device)
        refined = refiner(source)
        loss_latent = F.mse_loss(refined.float(), target.float())
        if args.packing_mode == "flux":
            loss_spatial = flux_unpacked_spatial_loss(refined, target)
            loss_block = flux_unpacked_block_loss(refined, target)
        else:
            loss_spatial = packed_spatial_loss(refined, target)
            loss_block = packed_block_loss(refined, target)
        loss_decoded = source.new_zeros(())
        loss_decoded_highfreq = source.new_zeros(())
        loss_decoded_block = source.new_zeros(())
        if (
            float(args.decoded_image_loss_weight) > 0
            or float(args.decoded_highfreq_loss_weight) > 0
            or float(args.decoded_block_loss_weight) > 0
        ):
            decoded = decode_flux_latents_tensor(
                pipe,
                refined.to(dtype=next(pipe.vae.parameters()).dtype),
                int(args.height),
                int(args.width),
            ).float()
            target_pixels = example["target_pixels"].to(device).float()
            if int(args.decoded_image_loss_size) > 0 and int(args.decoded_image_loss_size) != int(args.height):
                size = (int(args.decoded_image_loss_size), int(args.decoded_image_loss_size))
                decoded = F.interpolate(decoded, size=size, mode="bilinear", align_corners=False)
                target_pixels = F.interpolate(target_pixels, size=size, mode="bilinear", align_corners=False)
            if float(args.decoded_image_loss_weight) > 0:
                loss_decoded = F.l1_loss(decoded, target_pixels).to(device)
            if float(args.decoded_highfreq_loss_weight) > 0:
                loss_decoded_highfreq = image_highfreq_loss(decoded, target_pixels).to(device)
            if float(args.decoded_block_loss_weight) > 0:
                loss_decoded_block = image_block_loss(decoded, target_pixels).to(device)
        loss = (
            float(args.latent_loss_weight) * loss_latent
            + float(args.spatial_loss_weight) * loss_spatial
            + float(args.packed_block_loss_weight) * loss_block
            + float(args.decoded_image_loss_weight) * loss_decoded
            + float(args.decoded_highfreq_loss_weight) * loss_decoded_highfreq
            + float(args.decoded_block_loss_weight) * loss_decoded_block
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "latent_loss": float(loss_latent.item()),
                        "spatial_loss": float(loss_spatial.item()),
                        "block_loss": float(loss_block.item()),
                        "decoded_loss": float(loss_decoded.detach().item()),
                        "decoded_highfreq_loss": float(loss_decoded_highfreq.detach().item()),
                        "decoded_block_loss": float(loss_decoded_block.detach().item()),
                    }
                ),
                flush=True,
            )
        if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0:
            torch.save(
                {
                    "model": refiner.state_dict(),
                    "config": {
                        "channels": 64,
                        "hidden": int(args.hidden),
                        "depth": int(args.depth),
                        "residual_scale": float(args.residual_scale),
                        "packing_mode": str(args.packing_mode),
                        "architecture": str(args.refiner_architecture),
                    },
                    "student_checkpoint": str(args.checkpoint),
                    "step": int(step),
                    "loss": float(loss.detach().item()),
                },
                output_dir / f"flux_final_latent_refiner_step{step}.pt",
            )
    torch.save(
        {
            "model": refiner.state_dict(),
            "config": {
                "channels": 64,
                "hidden": int(args.hidden),
                "depth": int(args.depth),
                "residual_scale": float(args.residual_scale),
                "packing_mode": str(args.packing_mode),
                "architecture": str(args.refiner_architecture),
            },
            "student_checkpoint": str(args.checkpoint),
        },
        output_dir / "flux_final_latent_refiner.pt",
    )
    images: list[tuple[str, Path]] = []
    with torch.inference_mode():
        vae_dtype = next(pipe.vae.parameters()).dtype
        for index, example in enumerate(examples):
            refined = refiner(example["student"].to(device))
            image = tensor_to_pil(decode_flux_latents(pipe, refined.to(dtype=vae_dtype), 512, 512))
            path = image_dir / f"sample_{index:03d}.png"
            image.save(path)
            images.append((str(example["prompt"]), path))
    make_contact_sheet(images, output_dir / "contact_sheet.png", int(args.contact_thumb), int(args.contact_cols))
    print(json.dumps({"refiner": str(output_dir / "flux_final_latent_refiner.pt"), "contact_sheet": str(output_dir / "contact_sheet.png")}), flush=True)


if __name__ == "__main__":
    main()
