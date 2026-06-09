#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from sample_agentkernel_lite_image_flux_flow_distill import (
    load_cached_prompt_embeddings,
    load_student,
    prompt_seed_cache_key,
)
from train_flux_final_latent_refiner import (
    FluxPackedLatentRefiner,
    FluxPackedLatentUNetRefiner,
    decode_flux_latents_tensor,
    flux_unpacked_block_loss,
    flux_unpacked_spatial_loss,
)


def image_parity_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for y_offset in range(2):
        for x_offset in range(2):
            pred_parity = pred.float()[:, :, y_offset::2, x_offset::2]
            target_parity = target.float()[:, :, y_offset::2, x_offset::2]
            if int(size) > 0:
                target_size = (int(size), int(size))
                pred_parity = F.interpolate(pred_parity, size=target_size, mode="area")
                target_parity = F.interpolate(target_parity, size=target_size, mode="area")
            losses.append(F.l1_loss(pred_parity, target_parity))
            losses.append(F.l1_loss(pred_parity.mean(dim=(-2, -1)), target_parity.mean(dim=(-2, -1))))
    return torch.stack(losses).mean()


def image_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_float = pred.float()
    target_float = target.float()
    pred_dx = pred_float[:, :, :, 1:] - pred_float[:, :, :, :-1]
    target_dx = target_float[:, :, :, 1:] - target_float[:, :, :, :-1]
    pred_dy = pred_float[:, :, 1:, :] - pred_float[:, :, :-1, :]
    target_dy = target_float[:, :, 1:, :] - target_float[:, :, :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))


def image_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    pred_float = pred.float()
    target_float = target.float()
    size = max(int(size), 4)
    low_size = (size, size)
    pred_low = F.interpolate(
        F.interpolate(pred_float, size=low_size, mode="area"),
        size=pred_float.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    target_low = F.interpolate(
        F.interpolate(target_float, size=low_size, mode="area"),
        size=target_float.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return F.l1_loss(pred_float - pred_low, target_float - target_low)


def image_crop_detail_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_size: int,
    crop_count: int,
) -> torch.Tensor:
    pred_float = pred.float()
    target_float = target.float()
    height, width = pred_float.shape[-2:]
    crop_size = min(max(int(crop_size), 8), height, width)
    crop_count = max(int(crop_count), 1)
    losses: list[torch.Tensor] = []
    for _ in range(crop_count):
        top = torch.randint(0, height - crop_size + 1, (), device=pred_float.device).item()
        left = torch.randint(0, width - crop_size + 1, (), device=pred_float.device).item()
        pred_crop = pred_float[:, :, top : top + crop_size, left : left + crop_size]
        target_crop = target_float[:, :, top : top + crop_size, left : left + crop_size]
        losses.append(F.l1_loss(pred_crop, target_crop))
        losses.append(image_gradient_loss(pred_crop, target_crop))
        losses.append(image_highfreq_loss(pred_crop, target_crop, max(crop_size // 4, 8)))
    return torch.stack(losses).mean()


def parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise ValueError("at least one stage index is required")
    previous = 0
    for index in indices:
        if index <= previous:
            raise ValueError("stage indices must be strictly increasing and greater than 0")
        previous = index
    return indices


def rows_by_prompt_seed_timestep(target_dir: Path) -> dict[tuple[str, int, int], dict]:
    rows: dict[tuple[str, int, int], dict] = {}
    for line in (target_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(str(row["prompt"]), int(row["seed"]), int(row["timestep_index"]))] = row
    return rows


def split_paths(value: str) -> list[Path]:
    paths = [Path(part.strip()) for part in value.split(",") if part.strip()]
    if not paths:
        raise ValueError("at least one path is required")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny refiner from bridge T15 latents to true cached T15 latents.")
    parser.add_argument("--bridge-checkpoint", required=True)
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
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--stage-indices", default="5,10,15")
    parser.add_argument(
        "--target-index",
        type=int,
        default=-1,
        help="Optional target timestep index for the refiner target. Defaults to the final --stage-indices value.",
    )
    parser.add_argument("--bridge-output-mode", choices=("delta", "absolute"), default="absolute")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--architecture", choices=("conv", "flux_unet"), default="flux_unet")
    parser.add_argument("--resume-refiner", default="")
    parser.add_argument("--latent-loss-weight", type=float, default=40.0)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.5)
    parser.add_argument("--block-loss-weight", type=float, default=0.5)
    parser.add_argument("--decoded-loss-weight", type=float, default=2.0)
    parser.add_argument("--decoded-lowfreq-loss-weight", type=float, default=8.0)
    parser.add_argument("--decoded-loss-size", type=int, default=128)
    parser.add_argument("--decoded-parity-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-parity-size", type=int, default=64)
    parser.add_argument("--decoded-gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-highfreq-size", type=int, default=48)
    parser.add_argument("--decoded-crop-detail-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-crop-size", type=int, default=192)
    parser.add_argument("--decoded-crop-count", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dirs = split_paths(args.target_dir)
    prompt_files = split_paths(args.prompts)
    seed_files = split_paths(args.seed_csv)
    if len(prompt_files) == 1 and len(target_dirs) > 1:
        prompt_files = prompt_files * len(target_dirs)
    if len(seed_files) == 1 and len(target_dirs) > 1:
        seed_files = seed_files * len(target_dirs)
    if len(prompt_files) != len(target_dirs) or len(seed_files) != len(target_dirs):
        raise ValueError("--target-dir, --prompts, and --seed-csv must have matching comma-separated counts")
    stage_indices = parse_indices(args.stage_indices)
    target_index = int(args.target_index) if int(args.target_index) >= 0 else stage_indices[-1]

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
    bridge, _bridge_config = load_student(Path(args.bridge_checkpoint), device, "raw")
    bridge.requires_grad_(False)
    bridge.eval()
    examples: list[dict[str, torch.Tensor | str]] = []
    with torch.no_grad():
        for target_dir, prompt_file, seed_file in zip(target_dirs, prompt_files, seed_files):
            prompts = read_prompts(prompt_file, int(args.limit))
            seeds = [int(item) for item in seed_file.read_text(encoding="utf-8").strip().split(",") if item.strip()]
            if len(seeds) < len(prompts):
                raise ValueError(f"seed csv has fewer seeds than prompts: {seed_file}")
            embedding_cache = load_cached_prompt_embeddings(target_dir)
            target_rows = rows_by_prompt_seed_timestep(target_dir)
            for prompt, seed in zip(prompts, seeds):
                cache_key = prompt_seed_cache_key(prompt, seed)
                embeds = embedding_cache[cache_key]
                prompt_embeds = embeds["prompt_embeds"].to(device=device, dtype=pipe.dtype)
                pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
                latents = embeds["initial_latents"].to(device=device, dtype=pipe.dtype)
                timesteps, _ = flux_timesteps(pipe, latents, int(args.steps), device)
                guidance = torch.full([1], float(args.guidance), device=device, dtype=torch.float32)
                start_index = 0
                for stage_index in stage_indices:
                    output = bridge(
                        latents.float(),
                        timesteps[start_index].expand(latents.shape[0]).float(),
                        prompt_embeds.float(),
                        pooled_prompt_embeds.float(),
                        guidance,
                    ).to(latents.dtype)
                    if args.bridge_output_mode == "absolute":
                        latents = output
                    else:
                        latents = latents + output
                    start_index = stage_index
                row = target_rows[(prompt, seed, target_index)]
                target_payload = torch.load(target_dir / str(row["target_path"]), map_location="cpu")
                target_latents = target_payload["latents"].to(device=device, dtype=torch.float32)
                target_pixels = decode_flux_latents_tensor(
                    pipe,
                    target_latents.to(device=device, dtype=next(pipe.vae.parameters()).dtype),
                    int(args.height),
                    int(args.width),
                ).float()
                examples.append(
                    {
                        "prompt": prompt,
                        "source": latents.float().detach().clone(),
                        "target": target_latents.detach().clone(),
                        "target_pixels": target_pixels.detach().clone(),
                    }
                )
    print(json.dumps({"examples": len(examples), "target_index": target_index}), flush=True)

    if args.architecture == "flux_unet":
        refiner = FluxPackedLatentUNetRefiner(hidden=int(args.hidden), depth=int(args.depth), residual_scale=float(args.residual_scale)).to(device)
    else:
        refiner = FluxPackedLatentRefiner(hidden=int(args.hidden), depth=int(args.depth), residual_scale=float(args.residual_scale)).to(device)
    if args.resume_refiner:
        resume_payload = torch.load(args.resume_refiner, map_location="cpu")
        refiner.load_state_dict(resume_payload["model"], strict=True)
        print(json.dumps({"resume_refiner": str(args.resume_refiner)}), flush=True)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=float(args.lr), weight_decay=0.01)
    for step in range(1, int(args.train_steps) + 1):
        example = examples[(step - 1) % len(examples)]
        source = example["source"].to(device)
        target = example["target"].to(device)
        refined = refiner(source)
        latent_loss = F.mse_loss(refined.float(), target.float())
        spatial_loss = flux_unpacked_spatial_loss(refined, target)
        block_loss = flux_unpacked_block_loss(refined, target)
        decoded_loss = source.new_zeros(())
        decoded_lowfreq_loss = source.new_zeros(())
        decoded_parity_loss = source.new_zeros(())
        decoded_gradient_loss = source.new_zeros(())
        decoded_highfreq_loss = source.new_zeros(())
        decoded_crop_detail_loss = source.new_zeros(())
        if (
            float(args.decoded_loss_weight) > 0
            or float(args.decoded_lowfreq_loss_weight) > 0
            or float(args.decoded_parity_loss_weight) > 0
            or float(args.decoded_gradient_loss_weight) > 0
            or float(args.decoded_highfreq_loss_weight) > 0
            or float(args.decoded_crop_detail_loss_weight) > 0
        ):
            decoded_full = decode_flux_latents_tensor(
                pipe,
                refined.to(dtype=next(pipe.vae.parameters()).dtype),
                int(args.height),
                int(args.width),
            ).float()
            target_pixels_full = example["target_pixels"].to(device).float()
            target_pixels = target_pixels_full
            decoded = decoded_full
            if int(args.decoded_loss_size) > 0 and int(args.decoded_loss_size) != int(args.height):
                size = (int(args.decoded_loss_size), int(args.decoded_loss_size))
                decoded = F.interpolate(decoded, size=size, mode="bilinear", align_corners=False)
                target_pixels = F.interpolate(target_pixels, size=size, mode="bilinear", align_corners=False)
            decoded_loss = F.l1_loss(decoded, target_pixels)
            decoded_lowfreq_loss = F.l1_loss(
                F.interpolate(decoded, size=(32, 32), mode="area"),
                F.interpolate(target_pixels, size=(32, 32), mode="area"),
            )
            if float(args.decoded_parity_loss_weight) > 0:
                decoded_parity_loss = image_parity_lowfreq_loss(decoded, target_pixels, int(args.decoded_parity_size))
            if float(args.decoded_gradient_loss_weight) > 0:
                decoded_gradient_loss = image_gradient_loss(decoded, target_pixels)
            if float(args.decoded_highfreq_loss_weight) > 0:
                decoded_highfreq_loss = image_highfreq_loss(
                    decoded_full,
                    target_pixels_full,
                    int(args.decoded_highfreq_size),
                )
            if float(args.decoded_crop_detail_loss_weight) > 0:
                decoded_crop_detail_loss = image_crop_detail_loss(
                    decoded_full,
                    target_pixels_full,
                    int(args.decoded_crop_size),
                    int(args.decoded_crop_count),
                )
        loss = (
            float(args.latent_loss_weight) * latent_loss
            + float(args.spatial_loss_weight) * spatial_loss
            + float(args.block_loss_weight) * block_loss
            + float(args.decoded_loss_weight) * decoded_loss
            + float(args.decoded_lowfreq_loss_weight) * decoded_lowfreq_loss
            + float(args.decoded_parity_loss_weight) * decoded_parity_loss
            + float(args.decoded_gradient_loss_weight) * decoded_gradient_loss
            + float(args.decoded_highfreq_loss_weight) * decoded_highfreq_loss
            + float(args.decoded_crop_detail_loss_weight) * decoded_crop_detail_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()
        log_row = {
            "step": step,
            "prompt": str(example["prompt"]),
            "loss": float(loss.detach().item()),
            "latent_loss": float(latent_loss.detach().item()),
            "spatial_loss": float(spatial_loss.detach().item()),
            "block_loss": float(block_loss.detach().item()),
            "decoded_loss": float(decoded_loss.detach().item()),
            "decoded_lowfreq_loss": float(decoded_lowfreq_loss.detach().item()),
            "decoded_parity_loss": float(decoded_parity_loss.detach().item()),
            "decoded_gradient_loss": float(decoded_gradient_loss.detach().item()),
            "decoded_highfreq_loss": float(decoded_highfreq_loss.detach().item()),
            "decoded_crop_detail_loss": float(decoded_crop_detail_loss.detach().item()),
        }
        with (output_dir / "latent_refiner_ledger.jsonl").open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(log_row) + "\n")
        if step == 1 or step % 50 == 0:
            print(json.dumps(log_row), flush=True)
    torch.save(
        {
            "model": refiner.state_dict(),
            "config": {
                "channels": 64,
                "hidden": int(args.hidden),
                "depth": int(args.depth),
                "residual_scale": float(args.residual_scale),
                "packing_mode": "flux",
                "architecture": str(args.architecture),
            },
            "bridge_checkpoint": str(args.bridge_checkpoint),
            "resume_refiner": str(args.resume_refiner or ""),
            "stage_indices": stage_indices,
            "target_dirs": [str(path) for path in target_dirs],
        },
        output_dir / "flux_bridge_latent_refiner.pt",
    )
    print(json.dumps({"refiner": str(output_dir / "flux_bridge_latent_refiner.pt")}), flush=True)


if __name__ == "__main__":
    main()
