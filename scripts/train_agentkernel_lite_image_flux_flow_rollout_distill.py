#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from train_agentkernel_lite_image_flux_flow_distill import (
    FluxPackedStudent,
    clone_state_dict,
    infer_config,
    load_training_rows,
    save_checkpoint,
    seed_everything,
    update_ema_state,
)
from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, install_local_diffusers
from sample_agentkernel_lite_image_flux_flow_distill import load_final_latent_refiner
from train_agentkernel_lite_image_sana_latent_distill import apply_bitnet_qat_modules


class PackedLatentDiscriminator(nn.Module):
    def __init__(self, channels: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        tokens = latents.shape[1]
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError("PackedLatentDiscriminator requires a square latent token grid")
        x = latents.float().transpose(1, 2).reshape(latents.shape[0], latents.shape[2], side, side)
        return self.head(self.net(x).flatten(1)).flatten()


def sequence_key(row: dict[str, Any]) -> str:
    return str(row["target_id"]).rsplit("_t", 1)[0]


def read_prompt_file(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt = str(row.get("prompt", "")).strip()
            except json.JSONDecodeError:
                prompt = line.strip()
            if prompt:
                prompts.add(prompt)
    return prompts


def read_line_file(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                values.add(value)
    return values


def sequence_identity(sequence: list[dict[str, Any]]) -> str:
    row = sequence[0]
    prompt = str(row.get("prompt", "")).strip()
    source = Path(str(row.get("_target_dir", ""))).name
    seed = str(row.get("seed", "")).strip()
    return f"{prompt} | {source} | seed={seed}"


def sequence_id_aliases(sequence: list[dict[str, Any]]) -> set[str]:
    row = sequence[0]
    target_id = str(row.get("target_id", "")).rsplit("_t", 1)[0]
    aliases = {sequence_identity(sequence), target_id}
    source = Path(str(row.get("_target_dir", ""))).name
    if target_id:
        aliases.add(f"{source}/{target_id}")
    return aliases


def build_sequences(
    target_dirs: list[Path],
    include_prompts: set[str] | None = None,
    include_keys: set[str] | None = None,
) -> list[list[dict[str, Any]]]:
    rows = load_training_rows(target_dirs)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if include_prompts and str(row.get("prompt", "")).strip() not in include_prompts:
            continue
        grouped.setdefault(sequence_key(row), []).append(row)
    sequences = []
    for sequence in grouped.values():
        sequence.sort(key=lambda row: int(row.get("timestep_index", 0)))
        if len(sequence) >= 2:
            if include_keys and not (sequence_id_aliases(sequence) & include_keys):
                continue
            sequences.append(sequence)
    if not sequences:
        raise ValueError("no multi-timestep sequences found")
    return sequences


def load_embedding(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if "_prompt_embeds" in row and "_pooled_prompt_embeds" in row:
        return (
            row["_prompt_embeds"].to(device),
            row["_pooled_prompt_embeds"].to(device),
        )
    target_dir = Path(row["_target_dir"])
    embeds = torch.load(target_dir / row["embedding_path"], map_location="cpu")
    return (
        embeds["prompt_embeds"].to(device),
        embeds["pooled_prompt_embeds"].to(device),
    )


def load_embedding_batch(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds = []
    pooled_prompt_embeds = []
    for row in rows:
        prompt, pooled = load_embedding(row, device)
        prompt_embeds.append(prompt)
        pooled_prompt_embeds.append(pooled)
    return torch.cat(prompt_embeds, dim=0), torch.cat(pooled_prompt_embeds, dim=0)


def load_target(row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    if "_target_payload" in row:
        target = row["_target_payload"]
        return {
            "latents": target["latents"].to(device),
            "timestep": target["timestep"].reshape(()).to(device),
            "teacher_target": target["teacher_target"].to(device),
            "guidance": torch.tensor(float(target.get("guidance", row.get("guidance", 3.5))), device=device),
        }
    target_dir = Path(row["_target_dir"])
    target = torch.load(target_dir / row["target_path"], map_location="cpu")
    return {
        "latents": target["latents"].to(device),
        "timestep": target["timestep"].reshape(()).to(device),
        "teacher_target": target["teacher_target"].to(device),
        "guidance": torch.tensor(float(target.get("guidance", row.get("guidance", 3.5))), device=device),
    }


def load_target_batch(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    targets = [load_target(row, device) for row in rows]
    return {
        "latents": torch.cat([target["latents"] for target in targets], dim=0),
        "timestep": torch.stack([target["timestep"].reshape(()) for target in targets], dim=0),
        "teacher_target": torch.cat([target["teacher_target"] for target in targets], dim=0),
        "guidance": torch.stack([target["guidance"].reshape(()) for target in targets], dim=0),
    }


def teacher_step_delta(current: torch.Tensor, next_latents: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
    numerator = ((next_latents.float() - current.float()) * teacher_target.float()).flatten(1).sum(dim=1)
    denominator = teacher_target.float().square().flatten(1).sum(dim=1).clamp_min(1e-8)
    return (numerator / denominator).to(current.dtype).view(-1, 1, 1)


def closed_loop_flow_target(
    current_latents: torch.Tensor,
    teacher_next_latents: torch.Tensor,
    teacher_target: torch.Tensor,
    delta: torch.Tensor,
    mix: float,
    correction_rms_clamp: float,
) -> torch.Tensor:
    closed_target = (teacher_next_latents.float() - current_latents.float()) / delta.float().clamp_max(-1e-8)
    correction = closed_target - teacher_target.float()
    if correction_rms_clamp > 0:
        rms = correction.flatten(1).pow(2).mean(dim=1).sqrt().view(-1, 1, 1).clamp_min(1e-8)
        scale = torch.clamp(float(correction_rms_clamp) / rms, max=1.0)
        correction = correction * scale
    return teacher_target.float() + float(mix) * correction


def choose_sequence_batch(
    sequences: list[list[dict[str, Any]]],
    sequence_weights: list[float] | None,
    priority_scores: dict[str, float] | None,
    args: argparse.Namespace,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    uniform_mix = float(getattr(args, "priority_uniform_mix", 0.0))
    if priority_scores and uniform_mix > 0 and random.random() < uniform_mix:
        return random.choices(sequences, weights=sequence_weights, k=batch_size) if sequence_weights else random.choices(sequences, k=batch_size)
    if sequence_weights or priority_scores:
        weights = []
        alpha = float(getattr(args, "priority_replay_alpha", 0.0))
        min_weight = float(getattr(args, "priority_min_weight", 0.05))
        max_weight = float(getattr(args, "priority_max_weight", 20.0))
        for idx, sequence in enumerate(sequences):
            base = float(sequence_weights[idx]) if sequence_weights else 1.0
            if priority_scores:
                score = float(priority_scores.get(sequence_key(sequence[0]), 1.0))
                base *= max(min(score, max_weight), min_weight) ** alpha
            weights.append(base)
        return random.choices(sequences, weights=weights, k=batch_size)
    return random.choices(sequences, k=batch_size)


def choose_start(max_start: int, args: argparse.Namespace) -> int:
    if max_start <= 0:
        return 0
    min_start = max(0, int(getattr(args, "min_start_index", 0)))
    max_limit = int(getattr(args, "max_start_index", -1))
    upper = max_start if max_limit < 0 else min(max_start, max_limit)
    if min_start > upper:
        min_start = upper
    if random.random() < args.front_start_prob:
        return min_start
    return random.randint(min_start, upper)


def maybe_dropout_condition(
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    dropout: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dropout <= 0 or random.random() >= dropout:
        return prompt_embeds, pooled_prompt_embeds
    return torch.zeros_like(prompt_embeds), torch.zeros_like(pooled_prompt_embeds)


def add_latent_noise(latents: torch.Tensor, timestep: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.latent_noise_std <= 0:
        return latents
    scale = float(args.latent_noise_std)
    if args.latent_noise_timestep_scale:
        scale *= max(float(timestep.detach().float().mean().item()) / 1000.0, 0.05)
    return latents + torch.randn_like(latents) * scale


def real_clean_latents(sequence: list[dict[str, Any]], device: torch.device) -> torch.Tensor | None:
    if not sequence or sequence[0].get("target_kind") != "real_image_rectified_flow":
        return None
    first = load_target(sequence[0], device)
    # Real-image targets store noisy = clean + sigma * (noise - clean). The first FLUX
    # sigma is 1, so clean is the first latent minus the constant rectified-flow target.
    return first["latents"].float() - first["teacher_target"].float()


def preload_sequences(sequences: list[list[dict[str, Any]]]) -> None:
    embedding_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    target_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for sequence in sequences:
        for row in sequence:
            target_dir = Path(row["_target_dir"])
            embedding_key = (str(target_dir), str(row["embedding_path"]))
            if embedding_key not in embedding_cache:
                embeds = torch.load(target_dir / row["embedding_path"], map_location="cpu")
                embedding_cache[embedding_key] = (
                    embeds["prompt_embeds"].contiguous(),
                    embeds["pooled_prompt_embeds"].contiguous(),
                )
            row["_prompt_embeds"], row["_pooled_prompt_embeds"] = embedding_cache[embedding_key]

            target_key = (str(target_dir), str(row["target_path"]))
            if target_key not in target_cache:
                target = torch.load(target_dir / row["target_path"], map_location="cpu")
                target_cache[target_key] = {
                    "latents": target["latents"].contiguous(),
                    "timestep": target["timestep"].reshape(()).contiguous(),
                    "teacher_target": target["teacher_target"].contiguous(),
                    "guidance": float(target.get("guidance", row.get("guidance", 3.5))),
                }
            row["_target_payload"] = target_cache[target_key]


def load_prompt_weight_overrides(args: argparse.Namespace) -> dict[str, float]:
    if not args.sequence_weight_prompt_file:
        return {}
    prompts = read_prompt_file(args.sequence_weight_prompt_file)
    return {prompt: float(args.sequence_weight_multiplier) for prompt in prompts}


def load_sequence_weight_overrides(args: argparse.Namespace) -> dict[str, float]:
    if not args.sequence_weight_key_file:
        return {}
    keys = read_line_file(args.sequence_weight_key_file)
    return {key: float(args.sequence_weight_multiplier) for key in keys}


def sequence_sampling_weights(sequences: list[list[dict[str, Any]]], args: argparse.Namespace) -> list[float] | None:
    prompt_overrides = load_prompt_weight_overrides(args)
    key_overrides = load_sequence_weight_overrides(args)
    if not prompt_overrides and not key_overrides:
        return None
    weights = []
    for sequence in sequences:
        prompt = str(sequence[0].get("prompt", ""))
        weight = float(prompt_overrides.get(prompt, 1.0))
        for alias in sequence_id_aliases(sequence):
            if alias in key_overrides:
                weight = float(key_overrides[alias])
                break
        weights.append(weight)
    return weights


def direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    pred_flat = F.normalize(pred_flat, dim=-1)
    target_flat = F.normalize(target_flat, dim=-1)
    return (1.0 - (pred_flat * target_flat).sum(dim=-1)).mean()


def norm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = pred.float().flatten(1).norm(dim=-1)
    target_norm = target.float().flatten(1).norm(dim=-1).clamp_min(1e-6)
    return F.mse_loss(pred_norm / target_norm, torch.ones_like(target_norm))


def packed_spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tokens = pred.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return torch.zeros((), device=pred.device)
    pred_grid = pred.float().reshape(pred.shape[0], side, side, pred.shape[2])
    target_grid = target.float().reshape(target.shape[0], side, side, target.shape[2])
    pred_dx = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
    target_dx = target_grid[:, :, 1:, :] - target_grid[:, :, :-1, :]
    pred_dy = pred_grid[:, 1:, :, :] - pred_grid[:, :-1, :, :]
    target_dy = target_grid[:, 1:, :, :] - target_grid[:, :-1, :, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def packed_block_artifact_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tokens = pred.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return F.l1_loss(pred.float(), target.float())
    residual = pred.float().reshape(pred.shape[0], side, side, pred.shape[2]) - target.float().reshape(
        target.shape[0],
        side,
        side,
        target.shape[2],
    )
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        if side < stride:
            continue
        groups = []
        for y in range(stride):
            for x in range(stride):
                groups.append(residual[:, y::stride, x::stride, :].mean(dim=(1, 2)))
        stacked = torch.stack(groups, dim=0)
        loss = loss + (stacked - stacked.mean(dim=0, keepdim=True)).abs().mean()
        count += 1
    if count == 0:
        return residual.new_zeros(())
    return loss / float(count)


def unpack_flux_packed_latents(
    packed_latents: torch.Tensor,
    height: int = 512,
    width: int = 512,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    batch_size, _num_patches, channels = packed_latents.shape
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = packed_latents.view(batch_size, latent_height // 2, latent_width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, latent_height, latent_width)


def packed_unpacked_spatial_loss(pred: torch.Tensor, target: torch.Tensor, height: int, width: int) -> torch.Tensor:
    pred_grid = unpack_flux_packed_latents(pred.float(), height=height, width=width)
    target_grid = unpack_flux_packed_latents(target.float(), height=height, width=width)
    pred_dx = pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]
    target_dx = target_grid[:, :, :, 1:] - target_grid[:, :, :, :-1]
    pred_dy = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
    target_dy = target_grid[:, :, 1:, :] - target_grid[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def packed_unpacked_block_artifact_loss(pred: torch.Tensor, target: torch.Tensor, height: int, width: int) -> torch.Tensor:
    residual = unpack_flux_packed_latents(pred.float(), height=height, width=width) - unpack_flux_packed_latents(
        target.float(),
        height=height,
        width=width,
    )
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        if residual.shape[-2] < stride or residual.shape[-1] < stride:
            continue
        groups = [residual[:, :, y::stride, x::stride].mean(dim=(-1, -2)) for y in range(stride) for x in range(stride)]
        stacked = torch.stack(groups, dim=0)
        loss = loss + (stacked - stacked.mean(dim=0, keepdim=True)).abs().mean()
        count += 1
    if count == 0:
        return residual.new_zeros(())
    return loss / float(count)


def image_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def parse_int_list(value: str) -> list[int]:
    sizes: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            sizes.append(max(int(item), 1))
    return sizes


def image_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, pool: int = 5) -> torch.Tensor:
    pool = max(int(pool), 3)
    if pool % 2 == 0:
        pool += 1
    pred_low = F.avg_pool2d(pred, kernel_size=pool, stride=1, padding=pool // 2)
    target_low = F.avg_pool2d(target, kernel_size=pool, stride=1, padding=pool // 2)
    return F.l1_loss(pred - pred_low, target - target_low)


def image_block_artifact_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize systematic parity/block residuals that show up as checkerboard texture."""
    residual = pred - target
    loss = residual.new_zeros(())
    count = 0
    for stride in (2, 4, 8):
        if residual.shape[-2] < stride or residual.shape[-1] < stride:
            continue
        groups = []
        for y in range(stride):
            for x in range(stride):
                group = residual[:, :, y::stride, x::stride]
                if group.numel() > 0:
                    groups.append(group.mean(dim=(-1, -2)))
        if len(groups) <= 1:
            continue
        stacked = torch.stack(groups, dim=0)
        centered = stacked - stacked.mean(dim=0, keepdim=True)
        loss = loss + centered.abs().mean()
        count += 1
    if count == 0:
        return residual.new_zeros(())
    return loss / float(count)


def decoded_multiscale_losses(
    pred_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    sizes: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sizes:
        zero = pred_pixels.new_zeros(())
        return zero, zero, zero
    decoded = pred_pixels.new_zeros(())
    edge = pred_pixels.new_zeros(())
    highfreq = pred_pixels.new_zeros(())
    for size in sizes:
        pred_scaled = F.interpolate(pred_pixels, size=(size, size), mode="bilinear", align_corners=False)
        target_scaled = F.interpolate(target_pixels, size=(size, size), mode="bilinear", align_corners=False)
        decoded = decoded + F.l1_loss(pred_scaled, target_scaled)
        edge = edge + image_edge_loss(pred_scaled, target_scaled)
        highfreq = highfreq + image_highfreq_loss(pred_scaled, target_scaled)
    denominator = float(len(sizes))
    return decoded / denominator, edge / denominator, highfreq / denominator


def load_tiny_decoder(path: str, device: torch.device) -> nn.Module | None:
    if not path:
        return None
    from train_agentkernel_lite_flux_tiny_decoder_distill import TinyFluxDecoder, TinyFluxDecoderConfig

    checkpoint = torch.load(path, map_location="cpu")
    config = TinyFluxDecoderConfig(**checkpoint["config"])
    model = TinyFluxDecoder(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_flux_vae(
    model_id: str,
    device: torch.device,
    dtype_name: str,
    local_files_only: bool,
) -> nn.Module | None:
    if not model_id:
        return None
    install_local_diffusers()
    from diffusers import AutoencoderKL

    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def decode_flux_packed_with_vae(
    vae: nn.Module,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    latents = unpack_flux_packed_latents(packed_latents.float(), height=height, width=width)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    vae_dtype = next(vae.parameters()).dtype
    return vae.decode(latents.to(dtype=vae_dtype), return_dict=False)[0].float()


def packed_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    tokens = pred.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return F.mse_loss(pred.float(), target.float())
    pool = max(int(pool), 1)
    pred_grid = pred.float().transpose(1, 2).reshape(pred.shape[0], pred.shape[2], side, side)
    target_grid = target.float().transpose(1, 2).reshape(target.shape[0], target.shape[2], side, side)
    pred_low = F.avg_pool2d(pred_grid, kernel_size=pool, stride=pool)
    target_low = F.avg_pool2d(target_grid, kernel_size=pool, stride=pool)
    return F.mse_loss(pred_low, target_low)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.loss_type == "mse":
        return F.mse_loss(pred.float(), target.float())
    if args.loss_type == "huber":
        return F.huber_loss(pred.float(), target.float(), delta=float(args.huber_delta))
    raise ValueError(f"unknown loss type: {args.loss_type}")


def timestep_loss_weight(timestep: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.snr_weighting:
        return torch.ones((), device=timestep.device)
    # FLUX timesteps are stored around [0, 1000]. Treat t as noise level and
    # use a min-SNR style cap so very early high-noise targets do not dominate
    # the object-structure learning signal.
    sigma = (timestep.float().reshape(-1) / 1000.0).clamp(1e-4, 1.0)
    snr = ((1.0 - sigma) / sigma).square().clamp_min(1e-4)
    gamma = torch.tensor(float(args.snr_gamma), device=timestep.device)
    return (torch.minimum(snr, gamma) / snr).clamp(float(args.min_snr_weight), float(args.max_snr_weight)).mean()


def block_lr_scale(name: str, depth: int, max_scale: float) -> float:
    if max_scale <= 0 or abs(max_scale - 1.0) < 1e-8:
        return 1.0
    parts = name.split(".")
    if len(parts) < 2 or parts[0] != "blocks":
        return 1.0
    try:
        idx = int(parts[1])
    except ValueError:
        return 1.0
    if depth <= 1:
        return float(max_scale)
    fraction = max(0.0, min(1.0, idx / float(depth - 1)))
    return float(max_scale) ** fraction


def build_optimizer_param_groups(student: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[float, list[torch.nn.Parameter]] = defaultdict(list)
    depth = int(getattr(student.config, "depth", 1))
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = block_lr_scale(name, depth, float(args.block_lr_depth_scale))
        groups[round(scale, 6)].append(parameter)
    return [
        {"params": parameters, "lr": float(args.lr) * scale, "weight_decay": float(args.weight_decay)}
        for scale, parameters in sorted(groups.items())
    ]


def clone_trainable_grads(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter, memory_format=torch.preserve_format)
        for parameter in parameters
    ]


def grad_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    value = left[0].new_zeros(())
    for left_tensor, right_tensor in zip(left, right):
        value = value + (left_tensor.float() * right_tensor.float()).sum()
    return value


def grad_norm_sq(grad: list[torch.Tensor]) -> torch.Tensor:
    value = grad[0].new_zeros(())
    for tensor in grad:
        value = value + tensor.float().square().sum()
    return value


def apply_pcgrad_snapshots(
    parameters: list[torch.nn.Parameter],
    snapshots: list[list[torch.Tensor]],
) -> dict[str, float | int]:
    if not snapshots:
        return {"pcgrad_snapshots": 0, "pcgrad_conflicts": 0, "pcgrad_min_cosine": 0.0}
    projected = [[tensor.clone() for tensor in snapshot] for snapshot in snapshots]
    reference_norms = [grad_norm_sq(snapshot).clamp_min(1e-12) for snapshot in snapshots]
    conflicts = 0
    min_cosine = 1.0
    for idx, grad in enumerate(projected):
        for other_idx, other in enumerate(snapshots):
            if idx == other_idx:
                continue
            dot = grad_dot(grad, other)
            cosine = float(
                (dot / (grad_norm_sq(grad).sqrt() * reference_norms[other_idx].sqrt()).clamp_min(1e-12)).detach().item()
            )
            min_cosine = min(min_cosine, cosine)
            if dot.item() < 0:
                conflicts += 1
                scale = dot / reference_norms[other_idx]
                for tensor_idx in range(len(grad)):
                    grad[tensor_idx].sub_(scale.to(device=grad[tensor_idx].device, dtype=grad[tensor_idx].dtype) * other[tensor_idx])
    for parameter_idx, parameter in enumerate(parameters):
        merged = projected[0][parameter_idx]
        for grad in projected[1:]:
            merged = merged + grad[parameter_idx]
        parameter.grad = merged.to(device=parameter.device, dtype=parameter.dtype)
    return {"pcgrad_snapshots": len(snapshots), "pcgrad_conflicts": conflicts, "pcgrad_min_cosine": min_cosine}


def clone_named_tensors(named_parameters: list[tuple[str, torch.nn.Parameter]], *, grads: bool = False) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, parameter in named_parameters:
        tensor = parameter.grad if grads else parameter
        if tensor is None:
            continue
        out[name] = tensor.detach().float().cpu().clone()
    return out


def update_gradient_alignment_summary(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
) -> dict[str, float]:
    update_chunks: list[torch.Tensor] = []
    grad_chunks: list[torch.Tensor] = []
    max_relative_update = 0.0
    for name, old in before.items():
        new = after.get(name)
        grad = grads.get(name)
        if new is None or grad is None or old.shape != new.shape or old.shape != grad.shape:
            continue
        update = new.float() - old.float()
        update_chunks.append(update.flatten())
        grad_chunks.append(grad.float().flatten())
        max_relative_update = max(max_relative_update, float((update.norm() / old.float().norm().clamp_min(1e-12)).item()))
    if not update_chunks:
        return {
            "update_grad_cosine": 0.0,
            "update_norm": 0.0,
            "grad_norm": 0.0,
            "max_relative_update": 0.0,
        }
    update_vec = torch.cat(update_chunks)
    grad_vec = torch.cat(grad_chunks)
    denom = update_vec.norm() * grad_vec.norm()
    cosine = torch.dot(update_vec, grad_vec) / denom.clamp_min(1e-12)
    return {
        "update_grad_cosine": float(cosine.item()),
        "descent_update_alignment": float((-cosine).item()),
        "update_norm": float(update_vec.norm().item()),
        "grad_norm": float(grad_vec.norm().item()),
        "max_relative_update": max_relative_update,
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    target_dir = Path(args.target_dir)
    target_dirs = [target_dir] + [Path(value) for value in args.extra_target_dir]
    include_prompts = read_prompt_file(args.sequence_include_prompt_file)
    include_keys = read_line_file(args.sequence_include_key_file)
    sequences = build_sequences(target_dirs, include_prompts=include_prompts or None, include_keys=include_keys or None)
    if args.min_start_index > 0:
        required_rows = int(args.min_start_index) + max(int(args.rollout_len), 1) + 1
        before = len(sequences)
        sequences = [sequence for sequence in sequences if len(sequence) >= required_rows]
        print(
            json.dumps(
                {
                    "sequence_min_start_filter": {
                        "min_start_index": int(args.min_start_index),
                        "rollout_len": int(args.rollout_len),
                        "required_rows": required_rows,
                        "before": before,
                        "after": len(sequences),
                    }
                }
            ),
            flush=True,
        )
        if not sequences:
            raise ValueError("no sequences remain after applying min-start/rollout length filter")
    if include_prompts:
        print(json.dumps({"sequence_include_prompts": sorted(include_prompts), "matched_sequences": len(sequences)}), flush=True)
    if include_keys:
        print(json.dumps({"sequence_include_keys": sorted(include_keys), "matched_sequences": len(sequences)}), flush=True)
    sequence_weights = sequence_sampling_weights(sequences, args)
    priority_scores: dict[str, float] | None = {sequence_key(sequence[0]): 1.0 for sequence in sequences} if args.priority_replay else None
    sequence_cursor = 0
    replay_sequence_cursor = 0
    episode_sequence_batch: list[list[dict[str, Any]]] | None = None
    if args.preload_sequences:
        preload_sequences(sequences)
    rows = [row for sequence in sequences for row in sequence]
    config = infer_config(Path(rows[0]["_target_dir"]), rows, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    device = torch.device(args.device)
    tiny_decoder = load_tiny_decoder(args.tiny_decoder_checkpoint, device)
    flux_vae = load_flux_vae(
        args.decoded_vae_model,
        device,
        args.decoded_vae_dtype,
        bool(args.decoded_vae_local_files_only),
    )
    decoded_multiscale_sizes = parse_int_list(args.decoded_multiscale_downsamples)
    if decoded_multiscale_sizes:
        print(json.dumps({"decoded_multiscale_downsamples": decoded_multiscale_sizes}), flush=True)
    student = FluxPackedStudent(config).to(device)
    student.gradient_checkpointing = bool(args.gradient_checkpointing)
    if student.gradient_checkpointing:
        print(json.dumps({"gradient_checkpointing": True}), flush=True)
    bridge_recovery_student = None
    if str(getattr(args, "bridge_recovery_checkpoint", "")).strip():
        bridge_recovery_student = FluxPackedStudent(config).to(device)
        bridge_checkpoint = torch.load(args.bridge_recovery_checkpoint, map_location="cpu")
        bridge_recovery_student.load_state_dict(bridge_checkpoint.get("student", bridge_checkpoint), strict=False)
        bridge_recovery_student.requires_grad_(False)
        bridge_recovery_student.eval()
        print(
            json.dumps(
                {
                    "bridge_recovery_checkpoint": str(args.bridge_recovery_checkpoint),
                    "bridge_recovery_target_index": int(args.bridge_recovery_target_index),
                    "bridge_recovery_output_mode": str(args.bridge_recovery_output_mode),
                    "bridge_recovery_scale": float(args.bridge_recovery_scale),
                }
            ),
            flush=True,
        )
    endpoint_refiner_target = None
    if str(args.endpoint_refiner_target).strip():
        endpoint_refiner_target = load_final_latent_refiner(str(args.endpoint_refiner_target), device)
        endpoint_refiner_target.requires_grad_(False)
        endpoint_refiner_target.eval()
        print(json.dumps({"endpoint_refiner_target": str(args.endpoint_refiner_target)}), flush=True)
    endpoint_refiner_anchor_student = None
    if str(args.endpoint_refiner_anchor_resume).strip():
        endpoint_refiner_anchor_student = FluxPackedStudent(config).to(device)
        anchor_checkpoint = torch.load(args.endpoint_refiner_anchor_resume, map_location="cpu")
        missing, unexpected = endpoint_refiner_anchor_student.load_state_dict(anchor_checkpoint.get("student", anchor_checkpoint), strict=False)
        endpoint_refiner_anchor_student.requires_grad_(False)
        endpoint_refiner_anchor_student.eval()
        print(
            json.dumps(
                {
                    "endpoint_refiner_anchor_resume": str(args.endpoint_refiner_anchor_resume),
                    "missing": missing,
                    "unexpected": unexpected,
                }
            ),
            flush=True,
        )
    start_step = 0
    ema_state: dict[str, torch.Tensor] | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        missing, unexpected = student.load_state_dict(checkpoint.get("student", checkpoint), strict=False)
        if missing or unexpected:
            print(json.dumps({"resume_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
        start_step = int(checkpoint.get("step") or 0)
        # Keep continuation runs deterministic without replaying the same sample
        # order every time a checkpoint is resumed.
        seed_everything(int(args.seed) + start_step)
        if args.ema_decay > 0 and checkpoint.get("student_ema"):
            ema_state = {key: value.detach().cpu().clone() for key, value in checkpoint["student_ema"].items()}
    if args.bitnet_qat:
        modules = apply_bitnet_qat_modules(
            student,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
        )
        print(json.dumps({"bitnet_qat_enabled": {"modules": modules, "learned_scale": bool(args.bitnet_qat_learned_scale)}}), flush=True)
    if args.trainable_module_substring:
        trainable_substrings = tuple(args.trainable_module_substring)
        trainable_count = 0
        frozen_count = 0
        for name, parameter in student.named_parameters():
            trainable = any(substring in name for substring in trainable_substrings)
            parameter.requires_grad_(trainable)
            if trainable:
                trainable_count += parameter.numel()
            else:
                frozen_count += parameter.numel()
        print(
            json.dumps(
                {
                    "trainable_filter": {
                        "substrings": list(trainable_substrings),
                        "trainable_parameters": trainable_count,
                        "frozen_parameters": frozen_count,
                    }
                }
            ),
            flush=True,
        )
    if args.ema_decay > 0 and ema_state is None:
        ema_state = clone_state_dict(student)

    trainable_named_parameters = [(name, parameter) for name, parameter in student.named_parameters() if parameter.requires_grad]
    trainable_parameters = [parameter for _name, parameter in trainable_named_parameters]
    if not trainable_parameters:
        raise ValueError("no trainable student parameters selected")
    optimizer_param_groups = build_optimizer_param_groups(student, args)
    if len(optimizer_param_groups) > 1:
        print(
            json.dumps(
                {
                    "optimizer_param_groups": [
                        {"lr": group["lr"], "parameters": sum(parameter.numel() for parameter in group["params"])}
                        for group in optimizer_param_groups
                    ]
                }
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups)
    discriminator: PackedLatentDiscriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if args.latent_adv_weight > 0:
        discriminator = PackedLatentDiscriminator(
            channels=config.latent_channels,
            hidden=args.latent_adv_hidden,
        ).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.latent_adv_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )
    ledger_path = output_dir / "flux_flow_rollout_distill_ledger.jsonl"
    last_loss = 0.0
    last_flow_loss = 0.0
    last_latent_loss = 0.0
    last_direction_loss = 0.0
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        optimizer.zero_grad(set_to_none=True)
        decoded_loss_active = int(args.decoded_loss_interval) <= 1 or step % int(args.decoded_loss_interval) == 0
        total_loss_value = 0.0
        total_flow_value = 0.0
        total_latent_value = 0.0
        total_direction_value = 0.0
        total_norm_value = 0.0
        total_spatial_value = 0.0
        total_endpoint_value = 0.0
        total_lowfreq_latent_value = 0.0
        total_lowfreq_endpoint_value = 0.0
        total_latent_spatial_value = 0.0
        total_endpoint_spatial_value = 0.0
        total_endpoint_block_artifact_value = 0.0
        total_endpoint_unpacked_spatial_value = 0.0
        total_endpoint_unpacked_block_artifact_value = 0.0
        total_x0_value = 0.0
        total_lowfreq_x0_value = 0.0
        total_x0_spatial_value = 0.0
        total_decoded_value = 0.0
        total_decoded_edge_value = 0.0
        total_decoded_multiscale_value = 0.0
        total_decoded_multiscale_edge_value = 0.0
        total_decoded_highfreq_value = 0.0
        total_decoded_block_artifact_value = 0.0
        total_latent_adv_value = 0.0
        total_latent_disc_value = 0.0
        logged: dict[str, Any] = {}
        grad_accum_steps = max(int(args.grad_accum_steps), 1)
        pcgrad_snapshots: list[list[torch.Tensor]] = []
        pcgrad_stats: dict[str, float | int] = {"pcgrad_snapshots": 0, "pcgrad_conflicts": 0, "pcgrad_min_cosine": 0.0}
        for _micro_step in range(grad_accum_steps):
            use_priority = bool(priority_scores is not None and local_step >= int(args.priority_warmup_steps))
            batch_size = max(int(args.batch_size), 1)
            episode_steps = max(int(args.sequence_episode_steps), 1)
            replay_microstep = _micro_step > 0 and args.replay_microstep_mode != "none"
            if replay_microstep and args.replay_microstep_mode == "round_robin":
                sequence_batch = [sequences[(replay_sequence_cursor + offset) % len(sequences)] for offset in range(batch_size)]
                replay_sequence_cursor = (replay_sequence_cursor + batch_size) % len(sequences)
            elif replay_microstep and args.replay_microstep_mode == "random":
                sequence_batch = random.choices(sequences, weights=sequence_weights, k=batch_size) if sequence_weights else random.choices(sequences, k=batch_size)
            elif args.sequence_sampling_mode == "round_robin" and not use_priority:
                if episode_steps > 1:
                    episode_index = (local_step - 1) // episode_steps
                    batch_start = (episode_index * batch_size) % len(sequences)
                    sequence_batch = [sequences[(batch_start + offset) % len(sequences)] for offset in range(batch_size)]
                else:
                    sequence_batch = [sequences[(sequence_cursor + offset) % len(sequences)] for offset in range(batch_size)]
                    sequence_cursor = (sequence_cursor + batch_size) % len(sequences)
            elif use_priority and episode_steps > 1:
                if episode_sequence_batch is None or (local_step - 1) % episode_steps == 0:
                    episode_sequence_batch = choose_sequence_batch(
                        sequences,
                        sequence_weights,
                        priority_scores,
                        args,
                        batch_size,
                    )
                sequence_batch = episode_sequence_batch
            else:
                sequence_batch = choose_sequence_batch(
                    sequences,
                    sequence_weights,
                    priority_scores if use_priority else None,
                    args,
                    batch_size,
                )
            rollout_len = min(max(int(args.rollout_len), 1), min(len(sequence) - 1 for sequence in sequence_batch))
            start_indices = [choose_start(len(sequence) - rollout_len - 1, args) for sequence in sequence_batch]
            start_rows = [sequence[start_index] for sequence, start_index in zip(sequence_batch, start_indices)]
            prompt_embeds, pooled_prompt_embeds = load_embedding_batch(start_rows, device)
            prompt_embeds, pooled_prompt_embeds = maybe_dropout_condition(
                prompt_embeds,
                pooled_prompt_embeds,
                float(args.prompt_dropout),
            )
            current_target = load_target_batch(start_rows, device)
            current_latents = current_target["latents"].float()
            clean_latents = None
            guidance = current_target["guidance"].reshape(-1)
            if bridge_recovery_student is not None:
                bridge_target_index = int(args.bridge_recovery_target_index)
                if any(start_index != bridge_target_index for start_index in start_indices):
                    raise ValueError(
                        "--bridge-recovery-checkpoint currently requires fixed start index "
                        f"{bridge_target_index}; got {start_indices}"
                    )
                bridge_rows = [sequence[0] for sequence in sequence_batch]
                bridge_initial = load_target_batch(bridge_rows, device)
                bridge_timestep = bridge_initial["timestep"].reshape(-1)
                with torch.no_grad():
                    bridge_output = bridge_recovery_student(
                        bridge_initial["latents"].float(),
                        bridge_timestep.float(),
                        prompt_embeds.float(),
                        pooled_prompt_embeds.float(),
                        guidance.float(),
                    ).float()
                    if args.bridge_recovery_output_mode == "absolute":
                        bridge_latents = bridge_output * float(args.bridge_recovery_scale)
                    else:
                        bridge_latents = bridge_initial["latents"].float() + bridge_output * float(args.bridge_recovery_scale)
                current_latents = bridge_latents.detach()
            flow_loss = torch.zeros((), device=device)
            latent_loss = torch.zeros((), device=device)
            direction_value = torch.zeros((), device=device)
            norm_value = torch.zeros((), device=device)
            spatial_value = torch.zeros((), device=device)
            endpoint_value = torch.zeros((), device=device)
            x0_value = torch.zeros((), device=device)
            lowfreq_x0_value = torch.zeros((), device=device)
            x0_spatial_value = torch.zeros((), device=device)
            lowfreq_latent_value = torch.zeros((), device=device)
            lowfreq_endpoint_value = torch.zeros((), device=device)
            latent_spatial_value = torch.zeros((), device=device)
            endpoint_spatial_value = torch.zeros((), device=device)
            endpoint_block_artifact_value = torch.zeros((), device=device)
            endpoint_unpacked_spatial_value = torch.zeros((), device=device)
            endpoint_unpacked_block_artifact_value = torch.zeros((), device=device)
            decoded_value = torch.zeros((), device=device)
            decoded_edge_value = torch.zeros((), device=device)
            decoded_multiscale_value = torch.zeros((), device=device)
            decoded_multiscale_edge_value = torch.zeros((), device=device)
            decoded_highfreq_value = torch.zeros((), device=device)
            decoded_block_artifact_value = torch.zeros((), device=device)
            final_teacher_latents: torch.Tensor | None = None
            clean_rows = [sequence[-1] for sequence in sequence_batch]
            clean_terminal = load_target_batch(clean_rows, device)
            clean_delta = -(clean_terminal["timestep"].float() / 1000.0).view(-1, 1, 1)
            clean_latents = clean_terminal["latents"].float() + clean_delta * clean_terminal["teacher_target"].float()
            has_terminal_flow = all(
                start_index + rollout_len < len(sequence)
                for sequence, start_index in zip(sequence_batch, start_indices)
            )
            loss_denominator = rollout_len + int(bool(args.include_terminal_flow_loss and has_terminal_flow))
            for offset in range(rollout_len):
                rows = [sequence[start_index + offset] for sequence, start_index in zip(sequence_batch, start_indices)]
                next_rows = [sequence[start_index + offset + 1] for sequence, start_index in zip(sequence_batch, start_indices)]
                teacher_current = load_target_batch(rows, device)
                teacher_next = load_target_batch(next_rows, device)
                final_teacher_latents = teacher_next["latents"].float()
                timestep = teacher_current["timestep"].reshape(-1)
                model_latents = add_latent_noise(current_latents, timestep, args)
                pred = student(
                    model_latents,
                    timestep.float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance.float(),
                )
                loss_weight = timestep_loss_weight(timestep, args)
                flow_target = teacher_current["teacher_target"].float()
                delta = teacher_step_delta(
                    teacher_current["latents"],
                    teacher_next["latents"],
                    teacher_current["teacher_target"],
                )
                if args.closed_loop_flow_target:
                    flow_target = closed_loop_flow_target(
                        current_latents,
                        teacher_next["latents"],
                        teacher_current["teacher_target"],
                        delta,
                        args.closed_loop_flow_mix,
                        args.closed_loop_correction_rms_clamp,
                    )
                flow_loss = flow_loss + loss_weight * reconstruction_loss(pred, flow_target, args)
                if args.direction_loss_weight > 0:
                    direction_value = direction_value + direction_loss(pred, flow_target)
                if args.norm_loss_weight > 0:
                    norm_value = norm_value + norm_loss(pred, flow_target)
                if args.spatial_loss_weight > 0:
                    spatial_value = spatial_value + packed_spatial_gradient_loss(pred, flow_target)
                step_x0_loss = torch.zeros((), device=device)
                step_lowfreq_x0 = torch.zeros((), device=device)
                step_x0_spatial = torch.zeros((), device=device)
                if args.x0_loss_weight > 0 or args.lowfreq_x0_loss_weight > 0 or args.x0_spatial_loss_weight > 0:
                    sigma = (timestep.float() / 1000.0).view(-1, 1, 1)
                    sigma_weight = sigma.flatten(1).mean(dim=1)
                    if args.x0_min_sigma > 0:
                        sigma_weight = sigma_weight * (sigma_weight >= float(args.x0_min_sigma)).float()
                    if args.x0_sigma_power != 0:
                        sigma_weight = sigma_weight * sigma.flatten(1).mean(dim=1).clamp_min(1e-6).pow(float(args.x0_sigma_power))
                    x0_weight = sigma_weight.mean()
                    if x0_weight.item() > 0:
                        x0_pred = model_latents.float() - sigma * pred.float()
                        if args.x0_loss_weight > 0:
                            step_x0_loss = F.mse_loss(x0_pred.float(), clean_latents.float()) * x0_weight
                            x0_value = x0_value + step_x0_loss
                        if args.lowfreq_x0_loss_weight > 0:
                            step_lowfreq_x0 = packed_lowfreq_loss(x0_pred, clean_latents, args.lowfreq_pool) * x0_weight
                            lowfreq_x0_value = lowfreq_x0_value + step_lowfreq_x0
                        if args.x0_spatial_loss_weight > 0:
                            step_x0_spatial = packed_spatial_gradient_loss(x0_pred, clean_latents) * x0_weight
                            x0_spatial_value = x0_spatial_value + step_x0_spatial
                current_latents = current_latents + delta * pred.float()
                step_latent_loss = F.mse_loss(current_latents.float(), teacher_next["latents"].float())
                latent_loss = latent_loss + step_latent_loss
                step_lowfreq_latent = torch.zeros((), device=device)
                if args.lowfreq_latent_loss_weight > 0:
                    step_lowfreq_latent = packed_lowfreq_loss(current_latents, teacher_next["latents"], args.lowfreq_pool)
                    lowfreq_latent_value = lowfreq_latent_value + step_lowfreq_latent
                step_latent_spatial = torch.zeros((), device=device)
                if args.latent_spatial_loss_weight > 0:
                    step_latent_spatial = packed_spatial_gradient_loss(current_latents, teacher_next["latents"])
                    latent_spatial_value = latent_spatial_value + step_latent_spatial
                keep_final_graph = bool(
                    discriminator is not None
                    and clean_latents is not None
                    and args.latent_adv_weight > 0
                    and offset == rollout_len - 1
                )
                if args.stream_detached_backward:
                    step_direction = direction_loss(pred, flow_target) if args.direction_loss_weight > 0 else torch.zeros((), device=device)
                    step_norm = norm_loss(pred, flow_target) if args.norm_loss_weight > 0 else torch.zeros((), device=device)
                    step_spatial = packed_spatial_gradient_loss(pred, flow_target) if args.spatial_loss_weight > 0 else torch.zeros((), device=device)
                    step_loss = (
                        args.flow_loss_weight * loss_weight * reconstruction_loss(pred, flow_target, args)
                        + args.latent_loss_weight * step_latent_loss
                        + args.lowfreq_latent_loss_weight * step_lowfreq_latent
                        + args.latent_spatial_loss_weight * step_latent_spatial
                        + args.x0_loss_weight * step_x0_loss
                        + args.lowfreq_x0_loss_weight * step_lowfreq_x0
                        + args.x0_spatial_loss_weight * step_x0_spatial
                        + args.direction_loss_weight * step_direction
                        + args.norm_loss_weight * step_norm
                        + args.spatial_loss_weight * step_spatial
                    ) / max(loss_denominator, 1)
                    (step_loss / grad_accum_steps).backward(retain_graph=keep_final_graph)
                if args.detach_rollout and not keep_final_graph:
                    current_latents = current_latents.detach()
            if discriminator is not None and discriminator_optimizer is not None and clean_latents is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_for_disc = current_latents.detach()
                real_logits = discriminator(clean_latents.detach())
                fake_logits = discriminator(fake_for_disc)
                disc_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                disc_loss.backward()
                discriminator_optimizer.step()
                total_latent_disc_value += float(disc_loss.item())

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                adv_loss = F.softplus(-discriminator(current_latents)).mean()
                (args.latent_adv_weight * adv_loss / grad_accum_steps).backward()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                total_latent_adv_value += float(adv_loss.item())
                current_latents = current_latents.detach()
            if args.include_terminal_flow_loss and has_terminal_flow:
                terminal_rows = [sequence[start_index + rollout_len] for sequence, start_index in zip(sequence_batch, start_indices)]
                terminal = load_target_batch(terminal_rows, device)
                prompt_embeds_terminal, pooled_prompt_embeds_terminal = prompt_embeds, pooled_prompt_embeds
                timestep = terminal["timestep"].reshape(-1)
                pred = student(
                    add_latent_noise(current_latents, timestep, args),
                    timestep.float(),
                    prompt_embeds_terminal.float(),
                    pooled_prompt_embeds_terminal.float(),
                    guidance.float(),
                )
                terminal_flow_loss = timestep_loss_weight(timestep, args) * reconstruction_loss(pred, terminal["teacher_target"], args)
                flow_loss = flow_loss + terminal_flow_loss
                terminal_delta = -(timestep.float() / 1000.0).view(-1, 1, 1)
                terminal_teacher_latents = terminal["latents"].float() + terminal_delta * terminal["teacher_target"].float()
                current_latents = current_latents + terminal_delta * pred.float()
                final_teacher_latents = terminal_teacher_latents
                if args.direction_loss_weight > 0:
                    terminal_direction = direction_loss(pred, terminal["teacher_target"])
                    direction_value = direction_value + terminal_direction
                else:
                    terminal_direction = torch.zeros((), device=device)
                if args.norm_loss_weight > 0:
                    terminal_norm = norm_loss(pred, terminal["teacher_target"])
                    norm_value = norm_value + terminal_norm
                else:
                    terminal_norm = torch.zeros((), device=device)
                if args.spatial_loss_weight > 0:
                    terminal_spatial = packed_spatial_gradient_loss(pred, terminal["teacher_target"])
                    spatial_value = spatial_value + terminal_spatial
                else:
                    terminal_spatial = torch.zeros((), device=device)
                if args.stream_detached_backward:
                    terminal_loss = (
                        args.flow_loss_weight * terminal_flow_loss
                        + args.direction_loss_weight * terminal_direction
                        + args.norm_loss_weight * terminal_norm
                        + args.spatial_loss_weight * terminal_spatial
                    ) / max(loss_denominator, 1)
                    retain_terminal_graph = (
                        args.endpoint_loss_weight > 0
                        or args.lowfreq_endpoint_loss_weight > 0
                        or args.endpoint_spatial_loss_weight > 0
                        or args.endpoint_block_artifact_loss_weight > 0
                        or args.endpoint_unpacked_spatial_loss_weight > 0
                        or args.endpoint_unpacked_block_artifact_loss_weight > 0
                        or (
                            decoded_loss_active
                            and (
                                args.decoded_loss_weight > 0
                                or args.decoded_edge_loss_weight > 0
                                or args.decoded_multiscale_loss_weight > 0
                                or args.decoded_multiscale_edge_loss_weight > 0
                                or args.decoded_highfreq_loss_weight > 0
                                or args.decoded_block_artifact_loss_weight > 0
                            )
                        )
                    )
                    (terminal_loss / grad_accum_steps).backward(retain_graph=retain_terminal_graph)
            flow_loss = flow_loss / max(loss_denominator, 1)
            latent_loss = latent_loss / rollout_len
            lowfreq_latent_value = lowfreq_latent_value / rollout_len
            latent_spatial_value = latent_spatial_value / rollout_len
            x0_value = x0_value / rollout_len
            lowfreq_x0_value = lowfreq_x0_value / rollout_len
            x0_spatial_value = x0_spatial_value / rollout_len
            direction_value = direction_value / max(loss_denominator, 1)
            norm_value = norm_value / max(loss_denominator, 1)
            spatial_value = spatial_value / max(loss_denominator, 1)
            endpoint_refined_target_latents = None
            if final_teacher_latents is not None and endpoint_refiner_target is not None:
                with torch.no_grad():
                    if endpoint_refiner_anchor_student is not None:
                        anchor_latents = current_target["latents"].float()
                        for anchor_offset in range(rollout_len):
                            anchor_next_rows = [
                                sequence[start_index + anchor_offset + 1]
                                for sequence, start_index in zip(sequence_batch, start_indices)
                            ]
                            anchor_teacher_next = load_target_batch(anchor_next_rows, device)
                            anchor_timestep = anchor_teacher_next["timestep"].reshape(-1)
                            anchor_pred = endpoint_refiner_anchor_student(
                                anchor_latents,
                                anchor_timestep.float(),
                                prompt_embeds.float(),
                                pooled_prompt_embeds.float(),
                                guidance.float(),
                            )
                            anchor_delta = teacher_step_delta(
                                anchor_latents,
                                anchor_teacher_next["latents"],
                                anchor_teacher_next["teacher_target"],
                            )
                            anchor_latents = anchor_latents + anchor_delta * anchor_pred.float()
                        endpoint_refined_target_latents = endpoint_refiner_target(anchor_latents).float()
                    else:
                        endpoint_refined_target_latents = endpoint_refiner_target(current_latents.detach()).float()
            endpoint_target_latents = None
            if final_teacher_latents is not None:
                endpoint_target_latents = (
                    endpoint_refined_target_latents
                    if endpoint_refined_target_latents is not None
                    else (clean_latents if args.endpoint_clean_target else final_teacher_latents)
                )
            if endpoint_target_latents is not None and args.endpoint_loss_weight > 0:
                endpoint_value = F.mse_loss(current_latents.float(), endpoint_target_latents.float())
            if endpoint_target_latents is not None and args.lowfreq_endpoint_loss_weight > 0:
                lowfreq_endpoint_value = packed_lowfreq_loss(current_latents, endpoint_target_latents, args.lowfreq_pool)
            if endpoint_target_latents is not None and args.endpoint_spatial_loss_weight > 0:
                endpoint_spatial_value = packed_spatial_gradient_loss(current_latents, endpoint_target_latents)
            if endpoint_target_latents is not None and args.endpoint_block_artifact_loss_weight > 0:
                endpoint_block_artifact_value = packed_block_artifact_loss(current_latents, endpoint_target_latents)
            if endpoint_target_latents is not None and args.endpoint_unpacked_spatial_loss_weight > 0:
                endpoint_unpacked_spatial_value = packed_unpacked_spatial_loss(
                    current_latents,
                    endpoint_target_latents,
                    args.height,
                    args.width,
                )
            if endpoint_target_latents is not None and args.endpoint_unpacked_block_artifact_loss_weight > 0:
                endpoint_unpacked_block_artifact_value = packed_unpacked_block_artifact_loss(
                    current_latents,
                    endpoint_target_latents,
                    args.height,
                    args.width,
                )
            if (
                (tiny_decoder is not None or flux_vae is not None)
                and endpoint_target_latents is not None
                and decoded_loss_active
                and (
                    args.decoded_loss_weight > 0
                    or args.decoded_edge_loss_weight > 0
                    or args.decoded_multiscale_loss_weight > 0
                    or args.decoded_multiscale_edge_loss_weight > 0
                    or args.decoded_highfreq_loss_weight > 0
                    or args.decoded_block_artifact_loss_weight > 0
                )
            ):
                decoded_current_latents = current_latents
                decoded_target_latents = endpoint_target_latents
                decoded_batch_size = int(getattr(args, "decoded_loss_batch_size", 0) or 0)
                if decoded_batch_size > 0 and decoded_current_latents.shape[0] > decoded_batch_size:
                    sample_indices = torch.randperm(decoded_current_latents.shape[0], device=device)[:decoded_batch_size]
                    decoded_current_latents = decoded_current_latents.index_select(0, sample_indices)
                    decoded_target_latents = decoded_target_latents.index_select(0, sample_indices)
                if tiny_decoder is not None:
                    pred_pixels = tiny_decoder(unpack_flux_packed_latents(decoded_current_latents.float()))
                else:
                    pred_pixels = decode_flux_packed_with_vae(flux_vae, decoded_current_latents.float(), args.height, args.width)
                with torch.no_grad():
                    if tiny_decoder is not None:
                        target_pixels = tiny_decoder(unpack_flux_packed_latents(decoded_target_latents.float()))
                    else:
                        target_pixels = decode_flux_packed_with_vae(
                            flux_vae,
                            decoded_target_latents.float(),
                            args.height,
                            args.width,
                        )
                if args.decoded_loss_downsample > 1:
                    size = max(int(args.decoded_loss_downsample), 1)
                    pred_pixels_for_loss = F.interpolate(pred_pixels, size=(size, size), mode="bilinear", align_corners=False)
                    target_pixels_for_loss = F.interpolate(target_pixels, size=(size, size), mode="bilinear", align_corners=False)
                else:
                    pred_pixels_for_loss = pred_pixels
                    target_pixels_for_loss = target_pixels
                if args.decoded_loss_weight > 0:
                    decoded_value = F.l1_loss(pred_pixels_for_loss, target_pixels_for_loss)
                if args.decoded_edge_loss_weight > 0:
                    decoded_edge_value = image_edge_loss(pred_pixels_for_loss, target_pixels_for_loss)
                if args.decoded_block_artifact_loss_weight > 0:
                    decoded_block_artifact_value = image_block_artifact_loss(
                        pred_pixels_for_loss,
                        target_pixels_for_loss,
                    )
                if (
                    args.decoded_multiscale_loss_weight > 0
                    or args.decoded_multiscale_edge_loss_weight > 0
                    or args.decoded_highfreq_loss_weight > 0
                ):
                    (
                        decoded_multiscale_value,
                        decoded_multiscale_edge_value,
                        decoded_highfreq_value,
                    ) = decoded_multiscale_losses(pred_pixels, target_pixels, decoded_multiscale_sizes)
            loss = (
                args.flow_loss_weight * flow_loss
                + args.latent_loss_weight * latent_loss
                + args.lowfreq_latent_loss_weight * lowfreq_latent_value
                + args.latent_spatial_loss_weight * latent_spatial_value
                + args.x0_loss_weight * x0_value
                + args.lowfreq_x0_loss_weight * lowfreq_x0_value
                + args.x0_spatial_loss_weight * x0_spatial_value
                + args.direction_loss_weight * direction_value
                + args.norm_loss_weight * norm_value
                + args.spatial_loss_weight * spatial_value
                + args.endpoint_loss_weight * endpoint_value
                + args.lowfreq_endpoint_loss_weight * lowfreq_endpoint_value
                + args.endpoint_spatial_loss_weight * endpoint_spatial_value
                + args.endpoint_block_artifact_loss_weight * endpoint_block_artifact_value
                + args.endpoint_unpacked_spatial_loss_weight * endpoint_unpacked_spatial_value
                + args.endpoint_unpacked_block_artifact_loss_weight * endpoint_unpacked_block_artifact_value
                + (args.decoded_loss_weight * decoded_value if decoded_loss_active else 0.0)
                + (args.decoded_edge_loss_weight * decoded_edge_value if decoded_loss_active else 0.0)
                + (args.decoded_multiscale_loss_weight * decoded_multiscale_value if decoded_loss_active else 0.0)
                + (args.decoded_multiscale_edge_loss_weight * decoded_multiscale_edge_value if decoded_loss_active else 0.0)
                + (args.decoded_highfreq_loss_weight * decoded_highfreq_value if decoded_loss_active else 0.0)
                + (
                    args.decoded_block_artifact_loss_weight * decoded_block_artifact_value
                    if decoded_loss_active
                    else 0.0
                )
            )
            if args.stream_detached_backward:
                final_loss = (
                    args.endpoint_loss_weight * endpoint_value
                    + args.lowfreq_endpoint_loss_weight * lowfreq_endpoint_value
                    + args.endpoint_spatial_loss_weight * endpoint_spatial_value
                    + args.endpoint_block_artifact_loss_weight * endpoint_block_artifact_value
                    + args.endpoint_unpacked_spatial_loss_weight * endpoint_unpacked_spatial_value
                    + args.endpoint_unpacked_block_artifact_loss_weight * endpoint_unpacked_block_artifact_value
                    + (args.decoded_loss_weight * decoded_value if decoded_loss_active else 0.0)
                    + (args.decoded_edge_loss_weight * decoded_edge_value if decoded_loss_active else 0.0)
                    + (args.decoded_multiscale_loss_weight * decoded_multiscale_value if decoded_loss_active else 0.0)
                    + (args.decoded_multiscale_edge_loss_weight * decoded_multiscale_edge_value if decoded_loss_active else 0.0)
                    + (args.decoded_highfreq_loss_weight * decoded_highfreq_value if decoded_loss_active else 0.0)
                    + (
                        args.decoded_block_artifact_loss_weight * decoded_block_artifact_value
                        if decoded_loss_active
                        else 0.0
                    )
                )
                if final_loss.requires_grad:
                    (final_loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            total_loss_value += float(loss.item())
            total_flow_value += float(flow_loss.item())
            total_latent_value += float(latent_loss.item())
            total_direction_value += float(direction_value.item())
            total_norm_value += float(norm_value.item())
            total_spatial_value += float(spatial_value.item())
            total_endpoint_value += float(endpoint_value.item())
            total_lowfreq_latent_value += float(lowfreq_latent_value.item())
            total_lowfreq_endpoint_value += float(lowfreq_endpoint_value.item())
            total_latent_spatial_value += float(latent_spatial_value.item())
            total_endpoint_spatial_value += float(endpoint_spatial_value.item())
            total_endpoint_block_artifact_value += float(endpoint_block_artifact_value.item())
            total_endpoint_unpacked_spatial_value += float(endpoint_unpacked_spatial_value.item())
            total_endpoint_unpacked_block_artifact_value += float(endpoint_unpacked_block_artifact_value.item())
            total_x0_value += float(x0_value.item())
            total_lowfreq_x0_value += float(lowfreq_x0_value.item())
            total_x0_spatial_value += float(x0_spatial_value.item())
            total_decoded_value += float(decoded_value.item())
            total_decoded_edge_value += float(decoded_edge_value.item())
            total_decoded_multiscale_value += float(decoded_multiscale_value.item())
            total_decoded_multiscale_edge_value += float(decoded_multiscale_edge_value.item())
            total_decoded_highfreq_value += float(decoded_highfreq_value.item())
            total_decoded_block_artifact_value += float(decoded_block_artifact_value.item())
            if args.pcgrad_accumulation:
                pcgrad_snapshots.append(clone_trainable_grads(trainable_parameters))
                optimizer.zero_grad(set_to_none=True)
            if priority_scores is not None:
                priority_value = float(
                    loss.detach().item()
                    if args.priority_metric == "loss"
                    else (endpoint_value.detach().item() + lowfreq_endpoint_value.detach().item())
                )
                beta = float(args.priority_ema_beta)
                for selected_sequence in sequence_batch:
                    key = sequence_key(selected_sequence[0])
                    old_value = float(priority_scores.get(key, priority_value))
                    priority_scores[key] = beta * old_value + (1.0 - beta) * priority_value
            logged = {
                "prompt": start_rows[0].get("prompt", ""),
                "batch_size": len(sequence_batch),
                "start_timestep_index": start_rows[0].get("timestep_index", -1),
                "rollout_len": rollout_len,
                "priority_replay": bool(use_priority),
                "replay_microstep": bool(replay_microstep),
            }
        if args.pcgrad_accumulation:
            pcgrad_stats = apply_pcgrad_snapshots(trainable_parameters, pcgrad_snapshots)
        nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
        update_alignment: dict[str, float] = {}
        if args.interpret_update_alignment and step % int(args.log_every) == 0:
            interpret_before = clone_named_tensors(trainable_named_parameters)
            interpret_grads = clone_named_tensors(trainable_named_parameters, grads=True)
        else:
            interpret_before = {}
            interpret_grads = {}
        optimizer.step()
        if interpret_before and interpret_grads:
            update_alignment = update_gradient_alignment_summary(
                interpret_before,
                clone_named_tensors(trainable_named_parameters),
                interpret_grads,
            )
        if ema_state is not None:
            update_ema_state(ema_state, student, float(args.ema_decay))
        last_loss = total_loss_value / grad_accum_steps
        last_flow_loss = total_flow_value / grad_accum_steps
        last_latent_loss = total_latent_value / grad_accum_steps
        last_direction_loss = total_direction_value / grad_accum_steps
        last_norm_loss = total_norm_value / grad_accum_steps
        last_spatial_loss = total_spatial_value / grad_accum_steps
        last_endpoint_loss = total_endpoint_value / grad_accum_steps
        last_lowfreq_latent_loss = total_lowfreq_latent_value / grad_accum_steps
        last_lowfreq_endpoint_loss = total_lowfreq_endpoint_value / grad_accum_steps
        last_latent_spatial_loss = total_latent_spatial_value / grad_accum_steps
        last_endpoint_spatial_loss = total_endpoint_spatial_value / grad_accum_steps
        last_endpoint_block_artifact_loss = total_endpoint_block_artifact_value / grad_accum_steps
        last_endpoint_unpacked_spatial_loss = total_endpoint_unpacked_spatial_value / grad_accum_steps
        last_endpoint_unpacked_block_artifact_loss = total_endpoint_unpacked_block_artifact_value / grad_accum_steps
        last_x0_loss = total_x0_value / grad_accum_steps
        last_lowfreq_x0_loss = total_lowfreq_x0_value / grad_accum_steps
        last_x0_spatial_loss = total_x0_spatial_value / grad_accum_steps
        last_decoded_loss = total_decoded_value / grad_accum_steps
        last_decoded_edge_loss = total_decoded_edge_value / grad_accum_steps
        last_decoded_multiscale_loss = total_decoded_multiscale_value / grad_accum_steps
        last_decoded_multiscale_edge_loss = total_decoded_multiscale_edge_value / grad_accum_steps
        last_decoded_highfreq_loss = total_decoded_highfreq_value / grad_accum_steps
        last_decoded_block_artifact_loss = total_decoded_block_artifact_value / grad_accum_steps
        last_latent_adv_loss = total_latent_adv_value / grad_accum_steps
        last_latent_disc_loss = total_latent_disc_value / grad_accum_steps
        if step % args.log_every == 0 or local_step == 1:
            ledger = {
                "step": step,
                "loss": last_loss,
                "flow_loss": last_flow_loss,
                "latent_loss": last_latent_loss,
                "direction_loss": last_direction_loss,
                "norm_loss": last_norm_loss,
                "spatial_loss": last_spatial_loss,
                "endpoint_loss": last_endpoint_loss,
                "lowfreq_latent_loss": last_lowfreq_latent_loss,
                "lowfreq_endpoint_loss": last_lowfreq_endpoint_loss,
                "latent_spatial_loss": last_latent_spatial_loss,
                "endpoint_spatial_loss": last_endpoint_spatial_loss,
                "endpoint_block_artifact_loss": last_endpoint_block_artifact_loss,
                "endpoint_unpacked_spatial_loss": last_endpoint_unpacked_spatial_loss,
                "endpoint_unpacked_block_artifact_loss": last_endpoint_unpacked_block_artifact_loss,
                "x0_loss": last_x0_loss,
                "lowfreq_x0_loss": last_lowfreq_x0_loss,
                "x0_spatial_loss": last_x0_spatial_loss,
                "decoded_loss": last_decoded_loss,
                "decoded_edge_loss": last_decoded_edge_loss,
                "decoded_multiscale_loss": last_decoded_multiscale_loss,
                "decoded_multiscale_edge_loss": last_decoded_multiscale_edge_loss,
                "decoded_highfreq_loss": last_decoded_highfreq_loss,
                "decoded_block_artifact_loss": last_decoded_block_artifact_loss,
                "latent_adv_loss": last_latent_adv_loss,
                "latent_disc_loss": last_latent_disc_loss,
                "grad_accum_steps": grad_accum_steps,
                "sequences": len(sequences),
                "priority_replay_active": bool(priority_scores is not None and local_step >= int(args.priority_warmup_steps)),
                **pcgrad_stats,
                **update_alignment,
                **logged,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
            print(json.dumps(ledger), flush=True)
        if step % args.checkpoint_every == 0 or local_step == args.steps:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(output_dir, config, student, step, last_loss, args, ema_state)
            if discriminator is not None:
                torch.save(
                    {
                        "step": int(step),
                        "state_dict": discriminator.state_dict(),
                    },
                    output_dir / "latent_discriminator.pt",
                )
    save_checkpoint(output_dir, config, student, start_step + args.steps, last_loss, args, ema_state)
    if discriminator is not None:
        torch.save(
            {
                "step": int(start_step + args.steps),
                "state_dict": discriminator.state_dict(),
            },
            output_dir / "latent_discriminator.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop rollout distillation for the FLUX packed-latent student.")
    parser.add_argument("--target-dir", default="data/vision/flux1_dev_flow_targets_smoke_v0")
    parser.add_argument("--extra-target-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_flux_flow_student_rollout_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--bridge-recovery-checkpoint", default="")
    parser.add_argument("--bridge-recovery-target-index", type=int, default=15)
    parser.add_argument("--bridge-recovery-output-mode", choices=("delta", "absolute"), default="absolute")
    parser.add_argument("--bridge-recovery-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--pcgrad-accumulation", action="store_true")
    parser.add_argument("--interpret-update-alignment", action="store_true")
    parser.add_argument("--rollout-len", type=int, default=4)
    parser.add_argument("--detach-rollout", action="store_true")
    parser.add_argument("--stream-detached-backward", action="store_true")
    parser.add_argument("--closed-loop-flow-target", action="store_true")
    parser.add_argument("--closed-loop-flow-mix", type=float, default=0.25)
    parser.add_argument("--closed-loop-correction-rms-clamp", type=float, default=0.5)
    parser.add_argument("--include-terminal-flow-loss", action="store_true")
    parser.add_argument("--front-start-prob", type=float, default=0.7)
    parser.add_argument("--min-start-index", type=int, default=0)
    parser.add_argument("--max-start-index", type=int, default=-1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pos2d-scale", type=float, default=0.0)
    parser.add_argument("--timestep-scale", type=float, default=1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--block-lr-depth-scale", type=float, default=1.0)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--output-refiner-hidden", type=int, default=0)
    parser.add_argument("--output-refiner-depth", type=int, default=2)
    parser.add_argument("--output-refiner-scale", type=float, default=0.25)
    parser.add_argument("--latent-noise-std", type=float, default=0.0)
    parser.add_argument("--latent-noise-timestep-scale", action="store_true")
    parser.add_argument("--prompt-dropout", type=float, default=0.0)
    parser.add_argument("--preload-sequences", action="store_true")
    parser.add_argument("--sequence-include-prompt-file", default="")
    parser.add_argument("--sequence-include-key-file", default="")
    parser.add_argument("--sequence-weight-prompt-file", default="")
    parser.add_argument("--sequence-weight-key-file", default="")
    parser.add_argument("--sequence-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--sequence-sampling-mode", choices=("random", "round_robin"), default="random")
    parser.add_argument("--sequence-episode-steps", type=int, default=1)
    parser.add_argument("--priority-replay", action="store_true")
    parser.add_argument("--priority-replay-alpha", type=float, default=0.5)
    parser.add_argument("--priority-ema-beta", type=float, default=0.95)
    parser.add_argument("--priority-warmup-steps", type=int, default=0)
    parser.add_argument("--priority-uniform-mix", type=float, default=0.0)
    parser.add_argument("--priority-min-weight", type=float, default=0.05)
    parser.add_argument("--priority-max-weight", type=float, default=20.0)
    parser.add_argument("--priority-metric", choices=("loss", "endpoint"), default="endpoint")
    parser.add_argument("--replay-microstep-mode", choices=("none", "round_robin", "random"), default="none")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=2.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-clean-target", action="store_true")
    parser.add_argument("--endpoint-refiner-target", default="")
    parser.add_argument("--endpoint-refiner-anchor-resume", default="")
    parser.add_argument("--lowfreq-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-endpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-block-artifact-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-unpacked-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-unpacked-block-artifact-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-x0-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-min-sigma", type=float, default=0.0)
    parser.add_argument("--x0-sigma-power", type=float, default=0.0)
    parser.add_argument("--tiny-decoder-checkpoint", default="")
    parser.add_argument("--decoded-vae-model", default="")
    parser.add_argument("--decoded-vae-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--decoded-vae-local-files-only", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--decoded-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--decoded-loss-interval",
        type=int,
        default=1,
        help="Only apply decoded pixel/VAE losses every N optimizer steps; scalar logs are zero on skipped steps.",
    )
    parser.add_argument("--decoded-edge-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-loss-downsample", type=int, default=128)
    parser.add_argument("--decoded-loss-batch-size", type=int, default=0)
    parser.add_argument("--decoded-multiscale-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-multiscale-edge-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-block-artifact-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-multiscale-downsamples", default="64,128,256")
    parser.add_argument("--lowfreq-pool", type=int, default=4)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--snr-weighting", action="store_true")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--min-snr-weight", type=float, default=0.05)
    parser.add_argument("--max-snr-weight", type=float, default=1.0)
    parser.add_argument("--latent-adv-weight", type=float, default=0.0)
    parser.add_argument("--latent-adv-lr", type=float, default=1e-5)
    parser.add_argument("--latent-adv-hidden", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    parser.add_argument("--trainable-module-substring", action="append", default=[])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
