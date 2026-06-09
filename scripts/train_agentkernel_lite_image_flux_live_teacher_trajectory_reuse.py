#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import re
import shutil
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, load_teacher, teacher_family
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from train_agentkernel_lite_image_flux_flow_distill import (
    FluxPackedStudentConfig,
    clone_state_dict,
    save_checkpoint,
    seed_everything,
    update_ema_state,
)
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    PackedLatentDiscriminator,
    direction_loss,
    norm_loss,
    packed_spatial_gradient_loss,
    reconstruction_loss,
    timestep_loss_weight,
)
from train_agentkernel_lite_image_flux_live_teacher_flow import (
    PromptMixer,
    config_from_resume,
    load_student,
    parse_prompt_mix,
    stable_prompt_seed,
    teacher_predict,
)


class DecodedImageDiscriminator(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 4, hidden * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 4, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.float()).flatten(1).mean(dim=1)


class RenormLossCritic(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


def batched_step_delta(current: torch.Tensor, next_latents: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, current.ndim))
    numerator = ((next_latents.float() - current.float()) * teacher_target.float()).sum(dim=reduce_dims, keepdim=True)
    denominator = teacher_target.float().square().sum(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    return (numerator / denominator).to(current.dtype)


def packed_frequency_band_loss(pred: torch.Tensor, target: torch.Tensor, cutoff: float = 0.35) -> torch.Tensor:
    if pred.ndim != 3 or target.ndim != 3:
        return F.mse_loss(pred.float(), target.float())
    tokens = int(pred.shape[1])
    side = int(math.sqrt(tokens))
    if side * side != tokens:
        return F.mse_loss(pred.float(), target.float())
    pred_grid = pred.float().reshape(pred.shape[0], side, side, pred.shape[2]).permute(0, 3, 1, 2)
    target_grid = target.float().reshape(target.shape[0], side, side, target.shape[2]).permute(0, 3, 1, 2)
    pred_fft = torch.fft.rfft2(pred_grid, norm="ortho")
    target_fft = torch.fft.rfft2(target_grid, norm="ortho")
    fy = torch.fft.fftfreq(side, device=pred.device).abs().reshape(side, 1)
    fx = torch.fft.rfftfreq(side, device=pred.device).abs().reshape(1, side // 2 + 1)
    radius = torch.sqrt(fx.square() + fy.square())
    mask = (radius >= float(cutoff)).to(pred_grid.dtype).reshape(1, 1, side, side // 2 + 1)
    return ((pred_fft - target_fft).abs() * mask).mean()


def scheduler_step_latents(pipe: FluxPipeline, pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    # Use the FlowMatch Euler update directly instead of pipe.scheduler.step().
    # The diffusers scheduler stores mutable sigma/timestep tensors on the
    # teacher execution device; training may keep the student on a second GPU.
    device = latents.device
    model_output = pred.to(device=device, dtype=torch.float32)
    sample = latents.to(device=device, dtype=torch.float32)
    timestep_value = timestep.reshape(()).to(device=device, dtype=torch.float32)
    timesteps = pipe.scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
    sigma_index = int(torch.argmin((timesteps - timestep_value).abs()).item())
    sigma_index = min(sigma_index, sigmas.shape[0] - 2)
    dt = sigmas[sigma_index + 1] - sigmas[sigma_index]
    return (sample + dt * model_output).float()


def training_step_latents(
    pipe: FluxPipeline,
    pred: torch.Tensor,
    timestep: torch.Tensor,
    latents: torch.Tensor,
    teacher_current: torch.Tensor,
    teacher_next: torch.Tensor,
    teacher_target: torch.Tensor,
    use_scheduler_step: bool,
) -> torch.Tensor:
    if use_scheduler_step:
        return scheduler_step_latents(pipe, pred, timestep, latents)
    latent_delta = batched_step_delta(teacher_current, teacher_next, teacher_target)
    return latents + latent_delta * pred.float()


def choose_timestep_index(total_steps: int, args: argparse.Namespace) -> int:
    if total_steps <= 1:
        return 0
    max_start_index = int(getattr(args, "max_start_index", -1))
    min_start_index = max(0, int(getattr(args, "min_start_index", 0)))
    upper = total_steps - 1 if max_start_index < 0 else min(total_steps - 1, max_start_index)
    upper = max(0, upper)
    if min_start_index > upper:
        min_start_index = upper
    r = random.random()
    if r < args.front_start_prob:
        return min_start_index
    if r < args.front_start_prob + args.tail_start_prob:
        tail = max(1, int((upper + 1) * args.tail_fraction))
        return random.randint(max(min_start_index, upper + 1 - tail), upper)
    return random.randint(min_start_index, upper)


def start_index_loss_weight(start_index: int, total_steps: int, args: argparse.Namespace) -> float:
    weight = 1.0
    power = float(getattr(args, "start_index_loss_weight_power", 0.0))
    if power > 0 and total_steps > 1:
        progress = max(0.0, min(1.0, float(start_index) / float(total_steps - 1)))
        weight *= 1.0 + progress**power
    late_min = int(getattr(args, "late_start_index_min", -1))
    if late_min >= 0 and start_index >= late_min:
        weight *= float(getattr(args, "late_start_loss_multiplier", 1.0))
    return float(weight)


def renorm_critic_features(
    *,
    start_index: int,
    total_steps: int,
    start_weight: float,
    flow_value: torch.Tensor,
    dir_value: torch.Tensor,
    norm_value: torch.Tensor,
    spatial_value: torch.Tensor,
    endpoint_value: torch.Tensor,
    multistep_terminal_value: torch.Tensor,
    decoded_lowfreq_value: torch.Tensor,
    decoded_edge_value: torch.Tensor,
    on_policy_teacher_value: torch.Tensor,
    prompt_delta_value: torch.Tensor,
) -> torch.Tensor:
    device = flow_value.device
    denom = float(max(total_steps - 1, 1))

    def scalar(value: torch.Tensor) -> torch.Tensor:
        return torch.log1p(value.detach().float().reshape(()).clamp_min(0.0))

    return torch.stack(
        [
            torch.tensor(float(start_index) / denom, device=device),
            torch.tensor(float(start_weight), device=device).log1p(),
            scalar(flow_value),
            scalar(dir_value),
            scalar(norm_value),
            scalar(spatial_value),
            scalar(endpoint_value),
            scalar(multistep_terminal_value),
            scalar(decoded_lowfreq_value),
            scalar(decoded_edge_value),
            scalar(on_policy_teacher_value),
            scalar(prompt_delta_value),
        ]
    ).unsqueeze(0)


def load_balanced_channel_weights(path: str, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = torch.tensor(payload["weights"], device=device, dtype=torch.float32)
    return weights.view(1, 1, -1)


def parse_horizons(value: str) -> list[int]:
    horizons: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if horizon <= 0:
            raise ValueError("--multistep-consistency-horizons must contain positive integers")
        horizons.append(horizon)
    return sorted(set(horizons))


def parse_horizon_weights(value: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    if not value:
        return weights
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("--multistep-consistency-horizon-weights entries must be horizon:weight")
        horizon_text, weight_text = item.split(":", 1)
        horizon = int(horizon_text.strip())
        weight = float(weight_text.strip())
        if horizon <= 0:
            raise ValueError("--multistep-consistency-horizon-weights horizons must be positive")
        if weight < 0:
            raise ValueError("--multistep-consistency-horizon-weights weights must be non-negative")
        weights[horizon] = weight
    return weights


def choose_weighted_horizon(horizons: list[int], weights: dict[int, float]) -> int:
    if not weights:
        return random.choice(horizons)
    horizon_weights = [float(weights.get(horizon, 0.0)) for horizon in horizons]
    total = sum(horizon_weights)
    if total <= 0:
        return random.choice(horizons)
    pick = random.random() * total
    cumulative = 0.0
    for horizon, weight in zip(horizons, horizon_weights, strict=True):
        cumulative += weight
        if pick <= cumulative:
            return horizon
    return horizons[-1]


def load_prompt_seed_aliases(path: str) -> dict[str, str]:
    if not path:
        return {}
    aliases: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            seed_prompt = str(row.get("seed_prompt") or "").strip()
            if not prompt or not seed_prompt:
                raise ValueError(f"invalid prompt seed alias at {path}:{line_number}")
            aliases[prompt] = seed_prompt
    return aliases


def load_prompt_condition_aliases(path: str) -> dict[str, str]:
    if not path:
        return {}
    aliases: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            condition_prompt = str(row.get("condition_prompt") or row.get("seed_prompt") or "").strip()
            if not prompt or not condition_prompt:
                raise ValueError(f"invalid prompt condition alias at {path}:{line_number}")
            aliases[prompt] = condition_prompt
    return aliases


def load_prompt_negative_aliases(path: str) -> dict[str, str]:
    if not path:
        return {}
    aliases: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            negative_prompt = str(row.get("negative_prompt") or "").strip()
            if not prompt or not negative_prompt:
                raise ValueError(f"invalid prompt negative alias at {path}:{line_number}")
            aliases[prompt] = negative_prompt
            condition_prompt = str(row.get("condition_prompt") or "").strip()
            if condition_prompt:
                aliases[condition_prompt] = negative_prompt
    return aliases


def condition_prompt_alias(prompt: str | list[str], aliases: dict[str, str]) -> str | list[str]:
    if isinstance(prompt, list):
        return [aliases.get(item, item) for item in prompt]
    return aliases.get(prompt, prompt)


PROMPT_TOKEN_DROP_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "at",
    "to",
    "with",
    "and",
    "or",
    "photo",
    "image",
    "realistic",
}


def token_drop_negative_prompt(prompt: str, *, drop_count: int = 1) -> str:
    tokens = prompt.split()
    if len(tokens) <= 1:
        return prompt
    candidates = [
        index
        for index, token in enumerate(tokens)
        if token.strip(".,;:!?").lower() not in PROMPT_TOKEN_DROP_STOPWORDS
    ]
    if not candidates:
        candidates = list(range(len(tokens)))
    if drop_count <= 0 or drop_count >= len(candidates):
        drop_indexes = set(candidates)
    else:
        drop_indexes = set(random.sample(candidates, min(int(drop_count), len(candidates))))
    negative = " ".join(token for index, token in enumerate(tokens) if index not in drop_indexes).strip()
    return negative or prompt


def clamp_residual_gates(student: nn.Module, start_block: int, gate_floor: float) -> int:
    if gate_floor <= 0:
        return 0
    changed = 0
    with torch.no_grad():
        for index, block in enumerate(getattr(student, "blocks", [])):
            if index < start_block or not hasattr(block, "residual_gates"):
                continue
            gates = block.residual_gates
            before = gates.detach().clone()
            signs = torch.where(gates < 0, torch.full_like(gates, -1.0), torch.ones_like(gates))
            floored = signs * torch.clamp(gates.abs(), min=float(gate_floor))
            gates.copy_(floored)
            changed += int((before != gates).sum().item())
    return changed


def initialize_new_block_gates(student: nn.Module, source_checkpoint: dict[str, Any] | None, gate_value: float) -> int:
    if source_checkpoint is None or gate_value < 0:
        return 0
    source_config = source_checkpoint.get("config") or {}
    source_depth = int(source_config.get("depth") or 0)
    if source_depth <= 0:
        return 0
    initialized = 0
    with torch.no_grad():
        for index, block in enumerate(getattr(student, "blocks", [])):
            if index < source_depth or not hasattr(block, "residual_gates"):
                continue
            block.residual_gates.fill_(float(gate_value))
            initialized += int(block.residual_gates.numel())
    return initialized


def decode_flux_latents_tensor(pipe: Any, packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    try:
        vae_param = next(pipe.vae.parameters())
        latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    except StopIteration:
        pass
    return pipe.vae.decode(latents, return_dict=False)[0]


def as_feature_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for key in ("image_embeds", "pooler_output", "last_hidden_state"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, torch.Tensor):
            return candidate[:, 0] if key == "last_hidden_state" else candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"could not extract feature tensor from {type(value).__name__}")


def clip_pixels(decoded: torch.Tensor, size: int) -> torch.Tensor:
    image = ((decoded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    if image.shape[-1] != size or image.shape[-2] != size:
        image = F.interpolate(image, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def imagenet_pixels(decoded: torch.Tensor, size: int) -> torch.Tensor:
    image = ((decoded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    if image.shape[-1] != size or image.shape[-2] != size:
        image = F.interpolate(image, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def sobel_edges(decoded: torch.Tensor) -> torch.Tensor:
    image = ((decoded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    gray = image.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def decoded_clip_feature_loss(
    clip_vision_model: nn.Module | None,
    decoded_student: torch.Tensor,
    decoded_teacher: torch.Tensor,
    size: int,
) -> torch.Tensor:
    if clip_vision_model is None:
        return torch.zeros((), device=decoded_student.device)
    student_pixels = clip_pixels(decoded_student, size)
    teacher_pixels = clip_pixels(decoded_teacher, size)
    student_features = as_feature_tensor(clip_vision_model(pixel_values=student_pixels))
    with torch.no_grad():
        teacher_features = as_feature_tensor(clip_vision_model(pixel_values=teacher_pixels))
    student_features = F.normalize(student_features.float(), dim=-1)
    teacher_features = F.normalize(teacher_features.float(), dim=-1)
    return (1.0 - (student_features * teacher_features).sum(dim=-1)).mean()


def decoded_dino_feature_loss(
    dino_model: nn.Module | None,
    decoded_student: torch.Tensor,
    decoded_teacher: torch.Tensor,
    size: int,
) -> torch.Tensor:
    if dino_model is None:
        return torch.zeros((), device=decoded_student.device)
    student_pixels = imagenet_pixels(decoded_student, size)
    teacher_pixels = imagenet_pixels(decoded_teacher, size)
    student_features = as_feature_tensor(dino_model(pixel_values=student_pixels))
    with torch.no_grad():
        teacher_features = as_feature_tensor(dino_model(pixel_values=teacher_pixels))
    student_features = F.normalize(student_features.float(), dim=-1)
    teacher_features = F.normalize(teacher_features.float(), dim=-1)
    return (1.0 - (student_features * teacher_features).sum(dim=-1)).mean()


def decoded_adv_pixels(decoded: torch.Tensor, size: int) -> torch.Tensor:
    image = ((decoded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    if image.shape[-1] != size or image.shape[-2] != size:
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    return image * 2.0 - 1.0


@torch.no_grad()
def build_teacher_trajectory(
    pipe: Any,
    prompt: str | list[str],
    seed: int,
    args: argparse.Namespace,
    teacher_device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=teacher_device).manual_seed(seed)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=teacher_device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        len(prompt) if isinstance(prompt, list) else 1,
        num_channels_latents,
        args.height,
        args.width,
        prompt_embeds.dtype,
        teacher_device,
        generator,
        None,
    )
    timesteps, _ = flux_timesteps(pipe, latents, args.teacher_steps, teacher_device)
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([latents.shape[0]], args.guidance, device=teacher_device, dtype=torch.float32)

    trajectory: list[dict[str, torch.Tensor]] = []
    current = latents
    for timestep_value in timesteps:
        teacher_target = teacher_predict(
            pipe,
            current,
            timestep_value,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            guidance,
        )
        next_latents = pipe.scheduler.step(
            teacher_target,
            timestep_value,
            current,
            return_dict=False,
        )[0]
        trajectory.append(
            {
                "latents": current.detach().to("cpu"),
                "teacher_target": teacher_target.detach().to("cpu"),
                "teacher_next": next_latents.detach().to("cpu"),
                "timestep": timestep_value.detach().to("cpu", dtype=torch.float32),
            }
        )
        current = next_latents

    return {
        "prompt_embeds": prompt_embeds.detach().to("cpu"),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().to("cpu"),
        "text_ids": text_ids.detach().to("cpu"),
        "latent_image_ids": latent_image_ids.detach().to("cpu"),
        "guidance": float(args.guidance),
        "trajectory": trajectory,
    }


def save_manifest(output_dir: Path, config: FluxPackedStudentConfig, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_kind": "agentkernel_lite_flux_live_teacher_trajectory_reuse_training",
        "teacher_model": args.teacher_model,
        "teacher_family": teacher_family(args.teacher_model),
        "prompt_mix": parse_prompt_mix(args),
        "config": asdict(config),
        "prompt_cond_gate_init": float(args.prompt_cond_gate_init),
        "mode": "online_teacher_full_trajectory_reuse_no_disk_target_cache",
        "teacher_steps": int(args.teacher_steps),
        "windows_per_teacher": int(args.windows_per_teacher),
        "teacher_batch_size": int(args.teacher_batch_size),
        "prompt_hash_seeds": bool(args.prompt_hash_seeds),
        "prompt_seed_alias_file": args.prompt_seed_alias_file,
        "prompt_condition_alias_file": args.prompt_condition_alias_file,
        "prompt_negative_alias_file": args.prompt_negative_alias_file,
        "randomize_seeds": bool(args.randomize_seeds),
        "seed_min": int(args.seed_min),
        "seed_max": int(args.seed_max),
        "front_start_prob": float(args.front_start_prob),
        "tail_start_prob": float(args.tail_start_prob),
        "tail_fraction": float(args.tail_fraction),
        "prompt_contrast": {
            "weight": float(args.prompt_contrast_weight),
            "prob": float(args.prompt_contrast_prob),
            "every": int(args.prompt_contrast_every),
            "margin": float(args.prompt_contrast_margin),
            "token_drop_negative_prob": float(args.prompt_token_drop_negative_prob),
            "token_drop_count": int(args.prompt_token_drop_count),
            "negative_teacher_loss_weight": float(args.prompt_negative_teacher_loss_weight),
            "teacher_delta_loss_weight": float(args.prompt_teacher_delta_loss_weight),
        },
        "prompt_counterfactual_rollout": {
            "terminal_loss_weight": float(args.prompt_counterfactual_terminal_loss_weight),
            "delta_loss_weight": float(args.prompt_counterfactual_delta_loss_weight),
            "delta_direction_loss_weight": float(args.prompt_counterfactual_delta_direction_loss_weight),
            "delta_norm_loss_weight": float(args.prompt_counterfactual_delta_norm_loss_weight),
            "sensitivity_floor_loss_weight": float(args.prompt_sensitivity_floor_loss_weight),
            "sensitivity_ratio_loss_weight": float(args.prompt_sensitivity_ratio_loss_weight),
            "sensitivity_min_ratio": float(args.prompt_sensitivity_min_ratio),
            "horizon": int(args.prompt_counterfactual_horizon),
        },
        "multistep_consistency": {
            "weight": float(args.multistep_consistency_weight),
            "terminal_weight": float(args.multistep_terminal_loss_weight),
            "terminal_spatial_weight": float(args.multistep_terminal_spatial_loss_weight),
            "prob": float(args.multistep_consistency_prob),
            "horizons": parse_horizons(args.multistep_consistency_horizons),
            "horizon_weights": parse_horizon_weights(args.multistep_consistency_horizon_weights),
        },
        "training_step": {
            "use_scheduler_step_loss": bool(args.use_scheduler_step_loss),
        },
        "latent_distribution_matching": {
            "weight": float(args.latent_adv_weight),
            "prob": float(args.latent_adv_prob),
            "horizons": parse_horizons(args.latent_adv_horizons),
            "start_step": int(args.latent_adv_start_step),
            "lr": float(args.latent_adv_lr),
            "hidden": int(args.latent_adv_hidden),
        },
        "on_policy_teacher_correction": {
            "weight": float(args.on_policy_teacher_weight),
            "prob": float(args.on_policy_teacher_prob),
            "horizons": parse_horizons(args.on_policy_teacher_horizons),
            "flow_weight": float(args.on_policy_flow_weight),
            "latent_weight": float(args.on_policy_latent_weight),
        },
        "decoded_image_loss": {
            "weight": float(args.decoded_image_loss_weight),
            "device": args.decoded_device,
            "prob": float(args.decoded_image_loss_prob),
            "min_timestep_index": int(args.decoded_image_loss_min_timestep_index),
            "size": int(args.decoded_image_loss_size),
        },
        "decoded_clip_feature_loss": {
            "weight": float(args.decoded_clip_loss_weight),
            "model": str(args.decoded_clip_model),
            "size": int(args.decoded_clip_loss_size),
        },
        "decoded_dino_feature_loss": {
            "weight": float(args.decoded_dino_loss_weight),
            "model": str(args.decoded_dino_model),
            "size": int(args.decoded_dino_loss_size),
        },
        "decoded_edge_loss": {
            "weight": float(args.decoded_edge_loss_weight),
        },
        "decoded_lowfreq_loss": {
            "weight": float(args.decoded_lowfreq_loss_weight),
            "size": int(args.decoded_lowfreq_loss_size),
        },
        "decoded_adversarial_distribution_loss": {
            "weight": float(args.decoded_adv_weight),
            "prob": float(args.decoded_adv_prob),
            "size": int(args.decoded_adv_size),
            "hidden": int(args.decoded_adv_hidden),
        },
        "cache_trajectory_on_student_device": bool(args.cache_trajectory_on_student_device),
        "trajectory_cache_size": int(args.trajectory_cache_size),
        "training_args": vars(args),
    }
    (output_dir / "live_teacher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def optimizer_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "flux_trajectory_optimizer.pt"


def discriminator_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "latent_distribution_discriminator.pt"


def decoded_discriminator_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "decoded_image_discriminator.pt"


def maybe_snapshot_student_checkpoint(output_dir: Path, step: int, snapshot_every: int) -> None:
    if snapshot_every <= 0 or step % snapshot_every != 0:
        return
    checkpoint_path = output_dir / "flux_packed_student.pt"
    if not checkpoint_path.exists():
        return
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, snapshot_dir / f"flux_packed_student_step{step}.pt")


def save_optimizer_checkpoint(
    output_dir: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    decoded_discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = optimizer_checkpoint_path(output_dir)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        payload: dict[str, Any] = {"step": int(step), "optimizer": optimizer.state_dict()}
        if discriminator_optimizer is not None:
            payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        if decoded_discriminator_optimizer is not None:
            payload["decoded_discriminator_optimizer"] = decoded_discriminator_optimizer.state_dict()
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_discriminator_checkpoint(output_dir: Path, step: int, discriminator: PackedLatentDiscriminator | None) -> None:
    if discriminator is None:
        return
    path = discriminator_checkpoint_path(output_dir)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save({"step": int(step), "state_dict": discriminator.state_dict()}, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_decoded_discriminator_checkpoint(
    output_dir: Path,
    step: int,
    discriminator: DecodedImageDiscriminator | None,
) -> None:
    if discriminator is None:
        return
    path = decoded_discriminator_checkpoint_path(output_dir)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save({"step": int(step), "state_dict": discriminator.state_dict()}, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def train(args: argparse.Namespace) -> None:
    if args.randomize_seeds and int(args.seed_min) > int(args.seed_max):
        raise ValueError("--seed-min must be <= --seed-max")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "flux_live_teacher_ledger.jsonl"

    teacher_args = argparse.Namespace(
        teacher_family="flux",
        teacher_model=args.teacher_model,
        dtype=args.teacher_dtype,
        variant=args.variant,
        local_files_only=args.local_files_only,
        quantize_transformer_4bit=args.quantize_transformer_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        cpu_offload=args.cpu_offload,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    pipe = load_teacher(teacher_args)
    if hasattr(pipe, "vae"):
        for parameter in pipe.vae.parameters():
            parameter.requires_grad_(False)
    teacher_device = torch.device(getattr(pipe, "_execution_device", args.device))
    student_device = torch.device(args.student_device or str(teacher_device))
    decoded_device = torch.device(args.decoded_device or str(student_device))
    decoded_losses_enabled = (
        args.decoded_image_loss_weight > 0
        or args.decoded_clip_loss_weight > 0
        or args.decoded_dino_loss_weight > 0
        or args.decoded_edge_loss_weight > 0
        or args.decoded_lowfreq_loss_weight > 0
        or args.decoded_adv_weight > 0
    )
    if decoded_losses_enabled and hasattr(pipe, "vae"):
        pipe.vae.to(decoded_device)
    clip_vision_model: nn.Module | None = None
    if args.decoded_clip_loss_weight > 0:
        from transformers import CLIPVisionModelWithProjection

        clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
            args.decoded_clip_model,
            torch_dtype=pipe.dtype,
        ).to(decoded_device)
        clip_vision_model.eval()
        for parameter in clip_vision_model.parameters():
            parameter.requires_grad_(False)
    dino_model: nn.Module | None = None
    if args.decoded_dino_loss_weight > 0:
        from transformers import AutoModel

        dino_model = AutoModel.from_pretrained(args.decoded_dino_model, torch_dtype=pipe.dtype).to(decoded_device)
        dino_model.eval()
        for parameter in dino_model.parameters():
            parameter.requires_grad_(False)
    student, config, start_step, source_checkpoint = load_student(args, student_device)
    if args.prompt_cond_gate_init >= 0 and hasattr(student, "prompt_cond_gate"):
        with torch.no_grad():
            student.prompt_cond_gate.fill_(float(args.prompt_cond_gate_init))
        print(
            json.dumps({"prompt_cond_gate_init": float(args.prompt_cond_gate_init)}),
            flush=True,
        )
    initialized_new_gates = initialize_new_block_gates(student, source_checkpoint, float(args.new_block_gate_init))
    if initialized_new_gates:
        print(
            json.dumps(
                {
                    "new_block_gate_init": {
                        "value": float(args.new_block_gate_init),
                        "initialized_values": int(initialized_new_gates),
                    }
                }
            ),
            flush=True,
        )
    if args.trainable_name_regex:
        trainable_pattern = re.compile(args.trainable_name_regex)
        trainable_count = 0
        frozen_count = 0
        trainable_names: list[str] = []
        for name, parameter in student.named_parameters():
            is_trainable = bool(trainable_pattern.search(name))
            parameter.requires_grad_(is_trainable)
            if is_trainable:
                trainable_count += parameter.numel()
                if len(trainable_names) < 24:
                    trainable_names.append(name)
            else:
                frozen_count += parameter.numel()
        print(
            json.dumps(
                {
                    "trainable_filter": {
                        "regex": args.trainable_name_regex,
                        "trainable_params": int(trainable_count),
                        "frozen_params": int(frozen_count),
                        "sample_names": trainable_names,
                    }
                }
            ),
            flush=True,
        )
    clamped_gates = clamp_residual_gates(
        student,
        int(args.residual_gate_floor_start_block),
        float(args.residual_gate_floor),
    )
    if clamped_gates:
        print(
            json.dumps(
                {
                    "residual_gate_floor": {
                        "start_block": int(args.residual_gate_floor_start_block),
                        "floor": float(args.residual_gate_floor),
                        "clamped_values": int(clamped_gates),
                    }
                }
            ),
            flush=True,
        )
    if start_step > 0:
        seed_everything(int(args.seed) + int(start_step))
    save_manifest(output_dir, config, args)

    renorm_loss_critic: RenormLossCritic | None = None
    renorm_loss_critic_optimizer: torch.optim.Optimizer | None = None
    if args.renorm_loss_critic:
        renorm_loss_critic = RenormLossCritic(12, int(args.renorm_loss_critic_hidden)).to(student_device)
        renorm_loss_critic_optimizer = torch.optim.AdamW(
            renorm_loss_critic.parameters(),
            lr=float(args.renorm_loss_critic_lr),
            weight_decay=float(args.renorm_loss_critic_weight_decay),
        )
        print(
            json.dumps(
                {
                    "renorm_loss_critic": {
                        "features": 12,
                        "hidden": int(args.renorm_loss_critic_hidden),
                        "scale": float(args.renorm_loss_critic_scale),
                        "min_weight": float(args.renorm_loss_critic_min_weight),
                        "max_weight": float(args.renorm_loss_critic_max_weight),
                    }
                }
            ),
            flush=True,
        )
    balanced_channel_weights = load_balanced_channel_weights(args.balanced_channel_weight_file, student_device)
    if balanced_channel_weights is not None:
        print(
            json.dumps(
                {
                    "balanced_channel_weights": {
                        "path": args.balanced_channel_weight_file,
                        "channels": int(balanced_channel_weights.shape[-1]),
                        "min": float(balanced_channel_weights.min().item()),
                        "max": float(balanced_channel_weights.max().item()),
                        "mean": float(balanced_channel_weights.mean().item()),
                    }
                }
            ),
            flush=True,
        )

    ema_state: dict[str, torch.Tensor] | None = None
    if args.ema_decay > 0:
        if source_checkpoint and source_checkpoint.get("student_ema") and not args.reset_ema:
            ema_state = {key: value.detach().cpu().clone() for key, value in source_checkpoint["student_ema"].items()}
        else:
            ema_state = clone_state_dict(student)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    latent_discriminator: PackedLatentDiscriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if args.latent_adv_weight > 0:
        latent_discriminator = PackedLatentDiscriminator(
            channels=config.latent_channels,
            hidden=args.latent_adv_hidden,
        ).to(student_device)
        discriminator_optimizer = torch.optim.AdamW(
            latent_discriminator.parameters(),
            lr=args.latent_adv_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )
        discriminator_checkpoint = discriminator_checkpoint_path(output_dir)
        if args.resume and discriminator_checkpoint.exists():
            discriminator_state = torch.load(discriminator_checkpoint, map_location="cpu")
            latent_discriminator.load_state_dict(discriminator_state["state_dict"])
            print(
                json.dumps(
                    {
                        "resume_latent_distribution_discriminator": {
                            "path": str(discriminator_checkpoint),
                            "step": int(discriminator_state.get("step") or 0),
                        }
                    }
            ),
            flush=True,
        )
    decoded_discriminator: DecodedImageDiscriminator | None = None
    decoded_discriminator_optimizer: torch.optim.Optimizer | None = None
    if args.decoded_adv_weight > 0:
        decoded_discriminator = DecodedImageDiscriminator(hidden=args.decoded_adv_hidden).to(decoded_device)
        decoded_discriminator_optimizer = torch.optim.AdamW(
            decoded_discriminator.parameters(),
            lr=args.decoded_adv_lr,
            betas=(0.0, 0.99),
            weight_decay=0.0,
        )
        decoded_discriminator_checkpoint = decoded_discriminator_checkpoint_path(output_dir)
        if args.resume and decoded_discriminator_checkpoint.exists():
            decoded_discriminator_state = torch.load(decoded_discriminator_checkpoint, map_location="cpu")
            decoded_discriminator.load_state_dict(decoded_discriminator_state["state_dict"])
            print(
                json.dumps(
                    {
                        "resume_decoded_image_discriminator": {
                            "path": str(decoded_discriminator_checkpoint),
                            "step": int(decoded_discriminator_state.get("step") or 0),
                        }
                    }
                ),
                flush=True,
            )
    optimizer_path = optimizer_checkpoint_path(output_dir)
    if args.resume and optimizer_path.exists():
        optimizer_checkpoint = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        if discriminator_optimizer is not None and optimizer_checkpoint.get("discriminator_optimizer"):
            discriminator_optimizer.load_state_dict(optimizer_checkpoint["discriminator_optimizer"])
        if (
            decoded_discriminator_optimizer is not None
            and optimizer_checkpoint.get("decoded_discriminator_optimizer")
        ):
            decoded_discriminator_optimizer.load_state_dict(optimizer_checkpoint["decoded_discriminator_optimizer"])
        print(
            json.dumps(
                {
                    "resume_optimizer_state": {
                        "path": str(optimizer_path),
                        "step": int(optimizer_checkpoint.get("step") or 0),
                    }
                }
            ),
            flush=True,
        )
    prompts = PromptMixer(parse_prompt_mix(args), args.prompt_limit)
    prompt_seed_aliases = load_prompt_seed_aliases(args.prompt_seed_alias_file)
    prompt_condition_aliases = load_prompt_condition_aliases(args.prompt_condition_alias_file)
    prompt_negative_aliases = load_prompt_negative_aliases(args.prompt_negative_alias_file)
    if start_step > 0 and not args.no_resume_prompt_advance:
        prompt_draws_seen = (int(start_step) // max(int(args.windows_per_teacher), 1)) * max(
            int(args.teacher_batch_size),
            1,
        )
        prompts.advance(prompt_draws_seen)
        print(
            json.dumps(
                {
                    "resume_prompt_advance": {
                        "start_step": int(start_step),
                        "prompt_draws": int(prompt_draws_seen),
                    }
                }
            ),
            flush=True,
        )
    multistep_horizons = parse_horizons(args.multistep_consistency_horizons)
    multistep_horizon_weights = parse_horizon_weights(args.multistep_consistency_horizon_weights)
    latent_adv_horizons = parse_horizons(args.latent_adv_horizons)
    on_policy_teacher_horizons = parse_horizons(args.on_policy_teacher_horizons)
    trajectory_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

    step = start_step
    while step < start_step + args.steps:
        prompt_batch: list[str] = []
        source_batch: list[str] = []
        for _ in range(max(int(args.teacher_batch_size), 1)):
            prompt_item, source_item = prompts.next()
            prompt_batch.append(prompt_item)
            source_batch.append(source_item)
        prompt: str | list[str] = prompt_batch[0] if len(prompt_batch) == 1 else prompt_batch
        source_name: str | list[str] = source_batch[0] if len(source_batch) == 1 else source_batch
        if args.prompt_hash_seeds:
            if isinstance(prompt, list):
                # The hash mode is intended for deterministic one-prompt trajectories.
                # Batch mode still needs a stable seed; hash the ordered prompt batch.
                seed_prompt = "\n".join(prompt_seed_aliases.get(item, item) for item in prompt)
            else:
                seed_prompt = prompt_seed_aliases.get(prompt, prompt)
            seed = stable_prompt_seed(seed_prompt, int(args.seed_min), int(args.seed_max))
        elif args.randomize_seeds:
            seed = random.randint(int(args.seed_min), int(args.seed_max))
        else:
            seed = args.seed + step + 1
        prompt_key: Any = tuple(prompt) if isinstance(prompt, list) else prompt
        conditioning_prompt = condition_prompt_alias(prompt, prompt_condition_aliases)
        conditioning_prompt_key: Any = tuple(conditioning_prompt) if isinstance(conditioning_prompt, list) else conditioning_prompt
        cache_key = (
            conditioning_prompt_key,
            int(seed),
            int(args.teacher_steps),
            int(args.width),
            int(args.height),
            float(args.guidance),
            int(args.max_sequence_length),
        )
        cache_hit = False
        cached = None
        if args.trajectory_cache_size > 0:
            cached = trajectory_cache.get(cache_key)
            if cached is not None:
                cache_hit = True
                trajectory_cache.move_to_end(cache_key)
        if cached is None:
            cached = build_teacher_trajectory(pipe, conditioning_prompt, seed, args, teacher_device)
            if args.trajectory_cache_size > 0:
                trajectory_cache[cache_key] = cached
                trajectory_cache.move_to_end(cache_key)
                while len(trajectory_cache) > int(args.trajectory_cache_size):
                    trajectory_cache.popitem(last=False)
        prompt_embeds_teacher = cached["prompt_embeds"].to(teacher_device)
        pooled_prompt_embeds_teacher = cached["pooled_prompt_embeds"].to(teacher_device)
        text_ids_teacher = cached["text_ids"].to(teacher_device)
        latent_image_ids_teacher = cached["latent_image_ids"].to(teacher_device)
        guidance_teacher = None
        if pipe.transformer.config.guidance_embeds:
            guidance_teacher = torch.full(
                [prompt_embeds_teacher.shape[0]],
                float(cached["guidance"]),
                device=teacher_device,
                dtype=torch.float32,
            )
        prompt_embeds_student = cached["prompt_embeds"].to(student_device, dtype=torch.float32)
        pooled_prompt_embeds_student = cached["pooled_prompt_embeds"].to(student_device, dtype=torch.float32)
        negative_prompt_embeds_student = None
        negative_pooled_prompt_embeds_student = None
        negative_prompt_embeds_teacher = None
        negative_pooled_prompt_embeds_teacher = None
        negative_text_ids_teacher = None
        negative_cached = None
        negative_prompt = None
        prompt_negative_enabled = (
            args.prompt_contrast_weight > 0
            or args.prompt_negative_teacher_loss_weight > 0
            or args.prompt_teacher_delta_loss_weight > 0
            or args.prompt_counterfactual_terminal_loss_weight > 0
            or args.prompt_counterfactual_delta_loss_weight > 0
            or args.prompt_counterfactual_delta_direction_loss_weight > 0
            or args.prompt_counterfactual_delta_norm_loss_weight > 0
            or args.prompt_sensitivity_floor_loss_weight > 0
            or args.prompt_sensitivity_ratio_loss_weight > 0
        )
        if prompt_negative_enabled and random.random() < args.prompt_contrast_prob:
            contrast_prompt = conditioning_prompt
            if isinstance(contrast_prompt, list):
                aliased_negative = [prompt_negative_aliases.get(item, "") for item in contrast_prompt]
                has_aliased_negative = all(aliased_negative)
            else:
                aliased_negative = prompt_negative_aliases.get(contrast_prompt, "")
                has_aliased_negative = bool(aliased_negative)
            if has_aliased_negative:
                negative_prompt = aliased_negative
            elif (
                args.prompt_token_drop_negative_prob > 0
                and random.random() < args.prompt_token_drop_negative_prob
            ):
                if isinstance(contrast_prompt, list):
                    negative_prompt = [
                        token_drop_negative_prompt(item, drop_count=int(args.prompt_token_drop_count))
                        for item in contrast_prompt
                    ]
                else:
                    negative_prompt = token_drop_negative_prompt(
                        contrast_prompt,
                        drop_count=int(args.prompt_token_drop_count),
                    )
            elif isinstance(prompt, list):
                negative_prompt = [prompts.sample_negative(item) for item in prompt]
            else:
                negative_prompt = prompts.sample_negative(prompt)
            with torch.no_grad():
                negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=None,
                    device=teacher_device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
            negative_prompt_embeds_teacher = negative_prompt_embeds
            negative_pooled_prompt_embeds_teacher = negative_pooled_prompt_embeds
            negative_text_ids_teacher = negative_text_ids
            negative_prompt_embeds_student = negative_prompt_embeds.to(student_device, dtype=torch.float32)
            negative_pooled_prompt_embeds_student = negative_pooled_prompt_embeds.to(student_device, dtype=torch.float32)
            if (
                negative_prompt is not None
                and (
                    args.prompt_counterfactual_terminal_loss_weight > 0
                    or args.prompt_counterfactual_delta_loss_weight > 0
                    or args.prompt_counterfactual_delta_direction_loss_weight > 0
                    or args.prompt_counterfactual_delta_norm_loss_weight > 0
                    or args.prompt_sensitivity_floor_loss_weight > 0
                    or args.prompt_sensitivity_ratio_loss_weight > 0
                )
            ):
                negative_cache_key = (
                    tuple(negative_prompt) if isinstance(negative_prompt, list) else negative_prompt,
                    int(seed),
                    int(args.teacher_steps),
                    int(args.width),
                    int(args.height),
                    float(args.guidance),
                    int(args.max_sequence_length),
                )
                if args.trajectory_cache_size > 0:
                    negative_cached = trajectory_cache.get(negative_cache_key)
                    if negative_cached is not None:
                        trajectory_cache.move_to_end(negative_cache_key)
                if negative_cached is None:
                    negative_cached = build_teacher_trajectory(pipe, negative_prompt, seed, args, teacher_device)
                    if args.trajectory_cache_size > 0:
                        trajectory_cache[negative_cache_key] = negative_cached
                        trajectory_cache.move_to_end(negative_cache_key)
                        while len(trajectory_cache) > int(args.trajectory_cache_size):
                            trajectory_cache.popitem(last=False)
        guidance_student = torch.full([1], float(cached["guidance"]), device=student_device, dtype=torch.float32)
        trajectory: list[dict[str, torch.Tensor]] = cached["trajectory"]
        if args.cache_trajectory_on_student_device:
            for item in trajectory:
                item["latents"] = item["latents"].to(student_device, dtype=torch.float32)
                item["teacher_target"] = item["teacher_target"].to(student_device, dtype=torch.float32)
                item["teacher_next"] = item["teacher_next"].to(student_device, dtype=torch.float32)
                item["timestep"] = item["timestep"].reshape(1).to(student_device, dtype=torch.float32)
            if negative_cached is not None:
                for item in negative_cached["trajectory"]:
                    item["latents"] = item["latents"].to(student_device, dtype=torch.float32)
                    item["teacher_target"] = item["teacher_target"].to(student_device, dtype=torch.float32)
                    item["teacher_next"] = item["teacher_next"].to(student_device, dtype=torch.float32)
                    item["timestep"] = item["timestep"].reshape(1).to(student_device, dtype=torch.float32)

        for _ in range(max(int(args.windows_per_teacher), 1)):
            if step >= start_step + args.steps:
                break
            step += 1
            start_index = choose_timestep_index(len(trajectory), args)
            start_weight_value = start_index_loss_weight(start_index, len(trajectory), args)
            item = trajectory[start_index]
            current_latents = item["latents"].to(student_device, dtype=torch.float32)
            teacher_target_student = item["teacher_target"].to(student_device, dtype=torch.float32)
            teacher_next_student = item["teacher_next"].to(student_device, dtype=torch.float32)
            timestep = item["timestep"].reshape(1).to(student_device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            noisy_latents = current_latents
            if args.latent_noise_std > 0:
                scale = float(args.latent_noise_std)
                if args.latent_noise_timestep_scale:
                    scale *= max(float(timestep.detach().float().item()) / 1000.0, 0.05)
                noisy_latents = noisy_latents + torch.randn_like(noisy_latents) * scale

            pred = student(
                noisy_latents,
                timestep,
                prompt_embeds_student,
                pooled_prompt_embeds_student,
                guidance_student,
            )
            weight = timestep_loss_weight(timestep, args)
            flow_value = weight * reconstruction_loss(pred, teacher_target_student, args)
            student_next = training_step_latents(
                pipe,
                pred,
                timestep,
                current_latents,
                current_latents,
                teacher_next_student,
                teacher_target_student,
                args.use_scheduler_step_loss,
            )
            latent_value = F.mse_loss(student_next, teacher_next_student)
            dir_value = direction_loss(pred, teacher_target_student) if args.direction_loss_weight > 0 else torch.zeros((), device=student_device)
            norm_value = norm_loss(pred, teacher_target_student) if args.norm_loss_weight > 0 else torch.zeros((), device=student_device)
            spatial_value = (
                packed_spatial_gradient_loss(pred, teacher_target_student)
                if args.spatial_loss_weight > 0
                else torch.zeros((), device=student_device)
            )
            flow_frequency_value = (
                packed_frequency_band_loss(pred, teacher_target_student, args.frequency_loss_cutoff)
                if args.flow_frequency_loss_weight > 0
                else torch.zeros((), device=student_device)
            )
            endpoint_frequency_value = (
                packed_frequency_band_loss(student_next, teacher_next_student, args.frequency_loss_cutoff)
                if args.endpoint_frequency_loss_weight > 0
                else torch.zeros((), device=student_device)
            )
            contrast_value = torch.zeros((), device=student_device)
            negative_teacher_value = torch.zeros((), device=student_device)
            prompt_delta_value = torch.zeros((), device=student_device)
            if (
                negative_prompt_embeds_student is not None
                and args.prompt_contrast_weight > 0
                and step % max(int(args.prompt_contrast_every), 1) == 0
            ):
                wrong_pred = student(
                    noisy_latents,
                    timestep,
                    negative_prompt_embeds_student,
                    negative_pooled_prompt_embeds_student,
                    guidance_student,
                )
                positive_cos = F.cosine_similarity(pred.flatten(1), teacher_target_student.flatten(1), dim=1).mean()
                negative_cos = F.cosine_similarity(wrong_pred.flatten(1), teacher_target_student.flatten(1), dim=1).mean()
                contrast_value = F.relu(negative_cos - positive_cos + args.prompt_contrast_margin)
                if (
                    (args.prompt_negative_teacher_loss_weight > 0 or args.prompt_teacher_delta_loss_weight > 0)
                    and negative_prompt_embeds_teacher is not None
                    and negative_pooled_prompt_embeds_teacher is not None
                    and negative_text_ids_teacher is not None
                ):
                    with torch.no_grad():
                        negative_teacher_target = teacher_predict(
                            pipe,
                            noisy_latents.detach().to(teacher_device, dtype=prompt_embeds_teacher.dtype),
                            timestep.reshape(()).to(teacher_device),
                            negative_prompt_embeds_teacher,
                            negative_pooled_prompt_embeds_teacher,
                            negative_text_ids_teacher,
                            latent_image_ids_teacher,
                            guidance_teacher,
                        )
                    negative_teacher_target_student = negative_teacher_target.to(student_device, dtype=torch.float32)
                    negative_teacher_value = reconstruction_loss(wrong_pred, negative_teacher_target_student, args)
                    prompt_delta_value = F.mse_loss(
                        (pred.float() - wrong_pred.float()),
                        (teacher_target_student.float() - negative_teacher_target_student.float()),
                    )
            endpoint_value = F.mse_loss(student_next, teacher_next_student) if args.endpoint_loss_weight > 0 else torch.zeros((), device=student_device)
            balanced_flow_value = torch.zeros((), device=student_device)
            balanced_endpoint_value = torch.zeros((), device=student_device)
            if balanced_channel_weights is not None:
                balanced_flow_value = (
                    (pred.float() - teacher_target_student.float()).square() * balanced_channel_weights
                ).mean()
                balanced_endpoint_value = (
                    (student_next.float() - teacher_next_student.float()).square() * balanced_channel_weights
                ).mean()
            multistep_value = torch.zeros((), device=student_device)
            multistep_terminal_value = torch.zeros((), device=student_device)
            multistep_terminal_spatial_value = torch.zeros((), device=student_device)
            multistep_horizon = 0
            counterfactual_terminal_value = torch.zeros((), device=student_device)
            counterfactual_delta_value = torch.zeros((), device=student_device)
            counterfactual_delta_direction_value = torch.zeros((), device=student_device)
            counterfactual_delta_norm_value = torch.zeros((), device=student_device)
            prompt_sensitivity_floor_value = torch.zeros((), device=student_device)
            prompt_sensitivity_ratio_value = torch.zeros((), device=student_device)
            prompt_sensitivity_ratio_mean_value = torch.zeros((), device=student_device)
            student_counterfactual_delta_norm_value = torch.zeros((), device=student_device)
            teacher_counterfactual_delta_norm_value = torch.zeros((), device=student_device)
            counterfactual_horizon = 0
            latent_adv_value = torch.zeros((), device=student_device)
            latent_disc_value = torch.zeros((), device=student_device)
            latent_adv_horizon = 0
            on_policy_teacher_value = torch.zeros((), device=student_device)
            on_policy_teacher_horizon = 0
            decoded_image_value = torch.zeros((), device=student_device)
            decoded_clip_value = torch.zeros((), device=student_device)
            decoded_dino_value = torch.zeros((), device=student_device)
            decoded_edge_value = torch.zeros((), device=student_device)
            decoded_lowfreq_value = torch.zeros((), device=student_device)
            decoded_adv_value = torch.zeros((), device=student_device)
            decoded_disc_value = torch.zeros((), device=student_device)
            decoded_adv_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            multistep_terminal_latents: torch.Tensor | None = None
            multistep_terminal_target: torch.Tensor | None = None
            multistep_terminal_index = 0
            if (
                args.decoded_image_loss_weight > 0
                and start_index >= args.decoded_image_loss_min_timestep_index
                and random.random() < args.decoded_image_loss_prob
            ):
                student_decoded_latents = student_next.to(decoded_device, dtype=prompt_embeds_teacher.dtype)
                teacher_decoded_latents = teacher_next_student.to(decoded_device, dtype=prompt_embeds_teacher.dtype)
                decoded_student = decode_flux_latents_tensor(
                    pipe,
                    student_decoded_latents,
                    args.height,
                    args.width,
                ).float()
                with torch.no_grad():
                    decoded_teacher = decode_flux_latents_tensor(
                        pipe,
                        teacher_decoded_latents,
                        args.height,
                        args.width,
                    ).float()
                if args.decoded_clip_loss_weight > 0:
                    decoded_clip_value = decoded_clip_value + decoded_clip_feature_loss(
                        clip_vision_model,
                        decoded_student,
                        decoded_teacher,
                        args.decoded_clip_loss_size,
                    ).to(student_device)
                if args.decoded_dino_loss_weight > 0:
                    decoded_dino_value = decoded_dino_value + decoded_dino_feature_loss(
                        dino_model,
                        decoded_student,
                        decoded_teacher,
                        args.decoded_dino_loss_size,
                    ).to(student_device)
                if args.decoded_edge_loss_weight > 0:
                    decoded_edge_value = decoded_edge_value + F.l1_loss(
                        sobel_edges(decoded_student),
                        sobel_edges(decoded_teacher),
                    ).to(student_device)
                if args.decoded_lowfreq_loss_weight > 0:
                    low_student = F.interpolate(
                        decoded_student,
                        size=(args.decoded_lowfreq_loss_size, args.decoded_lowfreq_loss_size),
                        mode="area",
                    )
                    low_teacher = F.interpolate(
                        decoded_teacher,
                        size=(args.decoded_lowfreq_loss_size, args.decoded_lowfreq_loss_size),
                        mode="area",
                    )
                    decoded_lowfreq_value = decoded_lowfreq_value + F.l1_loss(low_student, low_teacher).to(
                        student_device
                    )
                if (
                    decoded_discriminator is not None
                    and decoded_discriminator_optimizer is not None
                    and random.random() < args.decoded_adv_prob
                ):
                    fake_image = decoded_adv_pixels(decoded_student, args.decoded_adv_size)
                    real_image = decoded_adv_pixels(decoded_teacher, args.decoded_adv_size)
                    decoded_adv_pairs.append((fake_image, real_image.detach()))
                if args.decoded_image_loss_size > 0 and decoded_student.shape[-1] != args.decoded_image_loss_size:
                    decoded_student = F.interpolate(
                        decoded_student,
                        size=(args.decoded_image_loss_size, args.decoded_image_loss_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    decoded_teacher = F.interpolate(
                        decoded_teacher,
                        size=(args.decoded_image_loss_size, args.decoded_image_loss_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                decoded_image_value = F.l1_loss(decoded_student, decoded_teacher).to(student_device)
            if args.multistep_consistency_weight > 0 and multistep_horizons and random.random() < args.multistep_consistency_prob:
                valid_horizons = [horizon for horizon in multistep_horizons if start_index + horizon <= len(trajectory)]
                if valid_horizons:
                    multistep_horizon = choose_weighted_horizon(valid_horizons, multistep_horizon_weights)
                    rollout_latents = current_latents.detach() if args.multistep_detach_rollout else current_latents
                    rollout_losses: list[torch.Tensor] = []
                    for offset in range(multistep_horizon):
                        rollout_item = trajectory[start_index + offset]
                        rollout_timestep = rollout_item["timestep"].reshape(1).to(student_device, dtype=torch.float32)
                        rollout_teacher_current = rollout_item["latents"].to(student_device, dtype=torch.float32)
                        rollout_teacher_target = rollout_item["teacher_target"].to(student_device, dtype=torch.float32)
                        rollout_teacher_next = rollout_item["teacher_next"].to(student_device, dtype=torch.float32)
                        rollout_pred = student(
                            rollout_latents,
                            rollout_timestep,
                            prompt_embeds_student,
                            pooled_prompt_embeds_student,
                            guidance_student,
                        )
                        rollout_latents = training_step_latents(
                            pipe,
                            rollout_pred,
                            rollout_timestep,
                            rollout_latents,
                            rollout_teacher_current,
                            rollout_teacher_next,
                            rollout_teacher_target,
                            args.use_scheduler_step_loss,
                        )
                        if args.multistep_detach_rollout and offset + 1 < multistep_horizon:
                            rollout_latents = rollout_latents.detach()
                        future_index = start_index + offset + 1
                        if future_index < len(trajectory):
                            teacher_future = trajectory[future_index]["latents"].to(student_device, dtype=torch.float32)
                        else:
                            teacher_future = rollout_item["teacher_next"].to(student_device, dtype=torch.float32)
                        rollout_losses.append(F.mse_loss(rollout_latents, teacher_future))
                    multistep_value = torch.stack(rollout_losses).mean()
                    multistep_terminal_latents = rollout_latents
                    terminal_index = start_index + multistep_horizon
                    if terminal_index < len(trajectory):
                        multistep_terminal_target = trajectory[terminal_index]["latents"].to(
                            student_device,
                            dtype=torch.float32,
                        )
                    else:
                        multistep_terminal_target = trajectory[-1]["teacher_next"].to(student_device, dtype=torch.float32)
                    multistep_terminal_index = start_index + multistep_horizon
                    multistep_terminal_value = F.mse_loss(multistep_terminal_latents, multistep_terminal_target)
                    if args.multistep_terminal_spatial_loss_weight > 0:
                        multistep_terminal_spatial_value = packed_spatial_gradient_loss(
                            multistep_terminal_latents,
                            multistep_terminal_target,
                        )
            if (
                negative_cached is not None
                and negative_prompt_embeds_student is not None
                and negative_pooled_prompt_embeds_student is not None
                and (
                    args.prompt_counterfactual_terminal_loss_weight > 0
                    or args.prompt_counterfactual_delta_loss_weight > 0
                    or args.prompt_counterfactual_delta_direction_loss_weight > 0
                    or args.prompt_counterfactual_delta_norm_loss_weight > 0
                    or args.prompt_sensitivity_floor_loss_weight > 0
                    or args.prompt_sensitivity_ratio_loss_weight > 0
                )
            ):
                negative_trajectory: list[dict[str, torch.Tensor]] = negative_cached["trajectory"]
                max_horizon = min(int(args.prompt_counterfactual_horizon), len(negative_trajectory) - start_index)
                if max_horizon > 0:
                    counterfactual_horizon = max_horizon
                    negative_rollout_latents = negative_trajectory[start_index]["latents"].to(
                        student_device,
                        dtype=torch.float32,
                    )
                    for offset in range(max_horizon):
                        negative_item = negative_trajectory[start_index + offset]
                        negative_timestep = negative_item["timestep"].reshape(1).to(
                            student_device,
                            dtype=torch.float32,
                        )
                        negative_teacher_current = negative_item["latents"].to(student_device, dtype=torch.float32)
                        negative_teacher_target = negative_item["teacher_target"].to(student_device, dtype=torch.float32)
                        negative_teacher_next = negative_item["teacher_next"].to(student_device, dtype=torch.float32)
                        negative_pred = student(
                            negative_rollout_latents,
                            negative_timestep,
                            negative_prompt_embeds_student,
                            negative_pooled_prompt_embeds_student,
                            guidance_student,
                        )
                        negative_rollout_latents = training_step_latents(
                            pipe,
                            negative_pred,
                            negative_timestep,
                            negative_rollout_latents,
                            negative_teacher_current,
                            negative_teacher_next,
                            negative_teacher_target,
                            args.use_scheduler_step_loss,
                        )
                    negative_terminal_index = start_index + max_horizon
                    if negative_terminal_index < len(negative_trajectory):
                        negative_terminal_target = negative_trajectory[negative_terminal_index]["latents"].to(
                            student_device,
                            dtype=torch.float32,
                        )
                    else:
                        negative_terminal_target = negative_trajectory[-1]["teacher_next"].to(
                            student_device,
                            dtype=torch.float32,
                        )
                    counterfactual_terminal_value = F.mse_loss(negative_rollout_latents, negative_terminal_target)
                    if multistep_terminal_latents is not None and multistep_terminal_target is not None:
                        student_counterfactual_delta = multistep_terminal_latents.float() - negative_rollout_latents.float()
                        teacher_counterfactual_delta = multistep_terminal_target.float() - negative_terminal_target.float()
                        student_counterfactual_delta_norm = student_counterfactual_delta.flatten(1).norm(dim=1)
                        teacher_counterfactual_delta_norm = teacher_counterfactual_delta.flatten(1).norm(dim=1)
                        student_counterfactual_delta_norm_value = student_counterfactual_delta_norm.mean()
                        teacher_counterfactual_delta_norm_value = teacher_counterfactual_delta_norm.mean()
                        sensitivity_ratio = student_counterfactual_delta_norm / teacher_counterfactual_delta_norm.clamp_min(1e-6)
                        prompt_sensitivity_ratio_mean_value = sensitivity_ratio.mean()
                        counterfactual_delta_value = F.mse_loss(
                            student_counterfactual_delta,
                            teacher_counterfactual_delta,
                        )
                        if args.prompt_counterfactual_delta_direction_loss_weight > 0:
                            counterfactual_delta_direction_value = direction_loss(
                                student_counterfactual_delta,
                                teacher_counterfactual_delta,
                            )
                        if args.prompt_counterfactual_delta_norm_loss_weight > 0:
                            counterfactual_delta_norm_value = norm_loss(
                                student_counterfactual_delta,
                                teacher_counterfactual_delta,
                            )
                        if args.prompt_sensitivity_floor_loss_weight > 0:
                            prompt_sensitivity_floor_value = F.relu(
                                float(args.prompt_sensitivity_min_ratio) * teacher_counterfactual_delta_norm
                                - student_counterfactual_delta_norm
                            ).square().mean()
                        if args.prompt_sensitivity_ratio_loss_weight > 0:
                            prompt_sensitivity_ratio_value = F.relu(
                                float(args.prompt_sensitivity_min_ratio) - sensitivity_ratio
                            ).square().mean()
            if (
                args.decoded_image_loss_weight > 0
                and multistep_terminal_latents is not None
                and multistep_terminal_target is not None
                and multistep_terminal_index >= args.decoded_image_loss_min_timestep_index
                and random.random() < args.decoded_image_loss_prob
            ):
                student_decoded_latents = multistep_terminal_latents.to(decoded_device, dtype=prompt_embeds_teacher.dtype)
                teacher_decoded_latents = multistep_terminal_target.to(decoded_device, dtype=prompt_embeds_teacher.dtype)
                decoded_student = decode_flux_latents_tensor(
                    pipe,
                    student_decoded_latents,
                    args.height,
                    args.width,
                ).float()
                with torch.no_grad():
                    decoded_teacher = decode_flux_latents_tensor(
                        pipe,
                        teacher_decoded_latents,
                        args.height,
                        args.width,
                    ).float()
                if args.decoded_clip_loss_weight > 0:
                    decoded_clip_value = decoded_clip_value + decoded_clip_feature_loss(
                        clip_vision_model,
                        decoded_student,
                        decoded_teacher,
                        args.decoded_clip_loss_size,
                    ).to(student_device)
                if args.decoded_dino_loss_weight > 0:
                    decoded_dino_value = decoded_dino_value + decoded_dino_feature_loss(
                        dino_model,
                        decoded_student,
                        decoded_teacher,
                        args.decoded_dino_loss_size,
                    ).to(student_device)
                if args.decoded_edge_loss_weight > 0:
                    decoded_edge_value = decoded_edge_value + F.l1_loss(
                        sobel_edges(decoded_student),
                        sobel_edges(decoded_teacher),
                    ).to(student_device)
                if args.decoded_lowfreq_loss_weight > 0:
                    low_student = F.interpolate(
                        decoded_student,
                        size=(args.decoded_lowfreq_loss_size, args.decoded_lowfreq_loss_size),
                        mode="area",
                    )
                    low_teacher = F.interpolate(
                        decoded_teacher,
                        size=(args.decoded_lowfreq_loss_size, args.decoded_lowfreq_loss_size),
                        mode="area",
                    )
                    decoded_lowfreq_value = decoded_lowfreq_value + F.l1_loss(low_student, low_teacher).to(
                        student_device
                    )
                if (
                    decoded_discriminator is not None
                    and decoded_discriminator_optimizer is not None
                    and random.random() < args.decoded_adv_prob
                ):
                    fake_image = decoded_adv_pixels(decoded_student, args.decoded_adv_size)
                    real_image = decoded_adv_pixels(decoded_teacher, args.decoded_adv_size)
                    decoded_adv_pairs.append((fake_image, real_image.detach()))
                if args.decoded_image_loss_size > 0 and decoded_student.shape[-1] != args.decoded_image_loss_size:
                    decoded_student = F.interpolate(
                        decoded_student,
                        size=(args.decoded_image_loss_size, args.decoded_image_loss_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    decoded_teacher = F.interpolate(
                        decoded_teacher,
                        size=(args.decoded_image_loss_size, args.decoded_image_loss_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                decoded_image_value = decoded_image_value + F.l1_loss(decoded_student, decoded_teacher).to(student_device)
            if (
                latent_discriminator is not None
                and discriminator_optimizer is not None
                and step >= args.latent_adv_start_step
                and latent_adv_horizons
                and random.random() < args.latent_adv_prob
            ):
                valid_adv_horizons = [horizon for horizon in latent_adv_horizons if start_index + horizon < len(trajectory)]
                if valid_adv_horizons:
                    latent_adv_horizon = random.choice(valid_adv_horizons)
                    adv_latents = current_latents
                    for offset in range(latent_adv_horizon):
                        adv_item = trajectory[start_index + offset]
                        adv_timestep = adv_item["timestep"].reshape(1).to(student_device, dtype=torch.float32)
                        adv_teacher_current = adv_item["latents"].to(student_device, dtype=torch.float32)
                        adv_teacher_target = adv_item["teacher_target"].to(student_device, dtype=torch.float32)
                        adv_teacher_next = adv_item["teacher_next"].to(student_device, dtype=torch.float32)
                        adv_pred = student(
                            adv_latents,
                            adv_timestep,
                            prompt_embeds_student,
                            pooled_prompt_embeds_student,
                            guidance_student,
                        )
                        adv_latents = training_step_latents(
                            pipe,
                            adv_pred,
                            adv_timestep,
                            adv_latents,
                            adv_teacher_current,
                            adv_teacher_next,
                            adv_teacher_target,
                            args.use_scheduler_step_loss,
                        )
                    adv_teacher_future = trajectory[start_index + latent_adv_horizon]["latents"].to(
                        student_device,
                        dtype=torch.float32,
                    )

                    discriminator_optimizer.zero_grad(set_to_none=True)
                    real_logits = latent_discriminator(adv_teacher_future.detach())
                    fake_logits = latent_discriminator(adv_latents.detach())
                    latent_disc_value = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                    latent_disc_value.backward()
                    nn.utils.clip_grad_norm_(latent_discriminator.parameters(), args.latent_adv_grad_clip)
                    discriminator_optimizer.step()

                    for parameter in latent_discriminator.parameters():
                        parameter.requires_grad_(False)
                    latent_adv_value = F.softplus(-latent_discriminator(adv_latents)).mean()
                    for parameter in latent_discriminator.parameters():
                        parameter.requires_grad_(True)
            if (
                args.on_policy_teacher_weight > 0
                and on_policy_teacher_horizons
                and random.random() < args.on_policy_teacher_prob
            ):
                valid_on_policy_horizons = [
                    horizon for horizon in on_policy_teacher_horizons if start_index + horizon < len(trajectory)
                ]
                if valid_on_policy_horizons:
                    on_policy_teacher_horizon = random.choice(valid_on_policy_horizons)
                    on_policy_latents = current_latents.detach() if args.on_policy_detach_rollout else current_latents
                    on_policy_losses: list[torch.Tensor] = []
                    for offset in range(on_policy_teacher_horizon):
                        op_item = trajectory[start_index + offset]
                        op_timestep = op_item["timestep"].reshape(1).to(student_device, dtype=torch.float32)
                        op_timestep_teacher = op_item["timestep"].reshape(1).to(teacher_device, dtype=torch.float32)
                        op_teacher_current = op_item["latents"].to(student_device, dtype=torch.float32)
                        op_teacher_target = op_item["teacher_target"].to(student_device, dtype=torch.float32)
                        op_teacher_next = op_item["teacher_next"].to(student_device, dtype=torch.float32)
                        teacher_latents_student = on_policy_latents.detach().to(student_device, dtype=torch.float32)
                        with torch.no_grad():
                            if student_device.type == "cuda":
                                torch.cuda.empty_cache()
                            teacher_latents = teacher_latents_student.to(teacher_device, dtype=prompt_embeds_teacher.dtype)
                            teacher_target = teacher_predict(
                                pipe,
                                teacher_latents,
                                op_timestep_teacher.reshape(()),
                                prompt_embeds_teacher,
                                pooled_prompt_embeds_teacher,
                                text_ids_teacher,
                                latent_image_ids_teacher,
                                guidance_teacher,
                            )
                        teacher_target_student = teacher_target.to(student_device, dtype=torch.float32)
                        if args.use_scheduler_step_loss:
                            teacher_next_student = scheduler_step_latents(
                                pipe,
                                teacher_target_student,
                                op_timestep,
                                teacher_latents_student,
                            )
                        else:
                            teacher_step_delta = batched_step_delta(
                                op_teacher_current,
                                op_teacher_next,
                                op_teacher_target,
                            )
                            teacher_next_student = teacher_latents_student + teacher_step_delta * teacher_target_student
                        op_pred = student(
                            teacher_latents_student,
                            op_timestep,
                            prompt_embeds_student,
                            pooled_prompt_embeds_student,
                            guidance_student,
                        )
                        on_policy_next = training_step_latents(
                            pipe,
                            op_pred,
                            op_timestep,
                            on_policy_latents,
                            teacher_latents_student,
                            teacher_next_student,
                            teacher_target_student,
                            args.use_scheduler_step_loss,
                        )
                        op_flow = reconstruction_loss(op_pred, teacher_target_student, args)
                        op_latent = F.mse_loss(on_policy_next, teacher_next_student)
                        on_policy_losses.append(
                            args.on_policy_flow_weight * op_flow
                            + args.on_policy_latent_weight * op_latent
                        )
                        on_policy_latents = on_policy_next.detach() if args.on_policy_detach_rollout else on_policy_next
                    on_policy_teacher_value = torch.stack(on_policy_losses).mean()
            if decoded_adv_pairs and decoded_discriminator is not None and decoded_discriminator_optimizer is not None:
                fake_images = torch.cat([pair[0] for pair in decoded_adv_pairs], dim=0)
                real_images = torch.cat([pair[1] for pair in decoded_adv_pairs], dim=0)
                decoded_discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = decoded_discriminator(real_images)
                fake_logits = decoded_discriminator(fake_images.detach())
                decoded_disc_step = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                decoded_disc_step.backward()
                nn.utils.clip_grad_norm_(decoded_discriminator.parameters(), args.decoded_adv_grad_clip)
                decoded_discriminator_optimizer.step()
                decoded_disc_value = decoded_disc_value + decoded_disc_step.detach().to(student_device)
                for parameter in decoded_discriminator.parameters():
                    parameter.requires_grad_(False)
                decoded_adv_value = decoded_adv_value + F.softplus(-decoded_discriminator(fake_images)).mean().to(student_device)
                for parameter in decoded_discriminator.parameters():
                    parameter.requires_grad_(True)
            base_loss = (
                args.flow_loss_weight * flow_value
                + args.latent_loss_weight * latent_value
                + args.direction_loss_weight * dir_value
                + args.norm_loss_weight * norm_value
                + args.spatial_loss_weight * spatial_value
                + args.flow_frequency_loss_weight * flow_frequency_value
                + args.endpoint_frequency_loss_weight * endpoint_frequency_value
                + args.prompt_contrast_weight * contrast_value
                + args.prompt_negative_teacher_loss_weight * negative_teacher_value
                + args.prompt_teacher_delta_loss_weight * prompt_delta_value
                + args.prompt_counterfactual_terminal_loss_weight * counterfactual_terminal_value
                + args.prompt_counterfactual_delta_loss_weight * counterfactual_delta_value
                + args.prompt_counterfactual_delta_direction_loss_weight * counterfactual_delta_direction_value
                + args.prompt_counterfactual_delta_norm_loss_weight * counterfactual_delta_norm_value
                + args.prompt_sensitivity_floor_loss_weight * prompt_sensitivity_floor_value
                + args.prompt_sensitivity_ratio_loss_weight * prompt_sensitivity_ratio_value
                + args.endpoint_loss_weight * endpoint_value
                + args.balanced_flow_loss_weight * balanced_flow_value
                + args.balanced_endpoint_loss_weight * balanced_endpoint_value
                + args.multistep_consistency_weight * multistep_value
                + args.multistep_terminal_loss_weight * multistep_terminal_value
                + args.multistep_terminal_spatial_loss_weight * multistep_terminal_spatial_value
                + args.latent_adv_weight * latent_adv_value
                + args.on_policy_teacher_weight * on_policy_teacher_value
                + args.decoded_image_loss_weight * decoded_image_value
                + args.decoded_clip_loss_weight * decoded_clip_value
                + args.decoded_dino_loss_weight * decoded_dino_value
                + args.decoded_edge_loss_weight * decoded_edge_value
                + args.decoded_lowfreq_loss_weight * decoded_lowfreq_value
                + args.decoded_adv_weight * decoded_adv_value
            )
            renorm_critic_weight = torch.ones((), device=student_device)
            renorm_critic_loss_value = torch.zeros((), device=student_device)
            renorm_critic_pred_value = torch.zeros((), device=student_device)
            renorm_critic_target_value = torch.zeros((), device=student_device)
            if renorm_loss_critic is not None and renorm_loss_critic_optimizer is not None:
                critic_features = renorm_critic_features(
                    start_index=start_index,
                    total_steps=len(trajectory),
                    start_weight=start_weight_value,
                    flow_value=flow_value,
                    dir_value=dir_value,
                    norm_value=norm_value,
                    spatial_value=spatial_value,
                    endpoint_value=endpoint_value,
                    multistep_terminal_value=multistep_terminal_value,
                    decoded_lowfreq_value=decoded_lowfreq_value,
                    decoded_edge_value=decoded_edge_value,
                    on_policy_teacher_value=on_policy_teacher_value,
                    prompt_delta_value=prompt_delta_value,
                )
                critic_target = torch.log1p(
                    endpoint_value.detach().float() * 100.0
                    + multistep_terminal_value.detach().float() * 50.0
                    + decoded_lowfreq_value.detach().float() * 10.0
                    + decoded_edge_value.detach().float() * 5.0
                    + on_policy_teacher_value.detach().float()
                ).reshape(1)
                critic_pred = renorm_loss_critic(critic_features)
                renorm_critic_loss_value = F.mse_loss(critic_pred, critic_target)
                renorm_loss_critic_optimizer.zero_grad(set_to_none=True)
                renorm_critic_loss_value.backward()
                nn.utils.clip_grad_norm_(renorm_loss_critic.parameters(), float(args.renorm_loss_critic_grad_clip))
                renorm_loss_critic_optimizer.step()
                with torch.no_grad():
                    critic_pred_for_weight = renorm_loss_critic(critic_features)
                    renorm_critic_pred_value = critic_pred_for_weight.detach().reshape(())
                    renorm_critic_target_value = critic_target.detach().reshape(())
                    renorm_critic_weight = (
                        1.0 + float(args.renorm_loss_critic_scale) * F.softplus(critic_pred_for_weight.detach())
                    ).clamp(
                        min=float(args.renorm_loss_critic_min_weight),
                        max=float(args.renorm_loss_critic_max_weight),
                    ).reshape(())
            loss = base_loss * start_weight_value * renorm_critic_weight
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            clamp_residual_gates(
                student,
                int(args.residual_gate_floor_start_block),
                float(args.residual_gate_floor),
            )
            if ema_state is not None and step % max(int(args.ema_update_every), 1) == 0:
                update_ema_state(ema_state, student, float(args.ema_decay))

            ledger = {
                "step": step,
                "prompt": prompt,
                "condition_prompt": conditioning_prompt,
                "negative_prompt": negative_prompt or "",
                "source_name": source_name,
                "seed": seed,
                "seed_prompt": seed_prompt if args.prompt_hash_seeds else "",
                "trajectory_cache_hit": bool(cache_hit),
                "trajectory_cache_size": int(len(trajectory_cache)),
                "start_timestep_index": start_index,
                "start_timestep_loss_weight": float(start_weight_value),
                "renorm_critic_weight": float(renorm_critic_weight.detach().item()),
                "renorm_critic_loss": float(renorm_critic_loss_value.detach().item()),
                "renorm_critic_pred": float(renorm_critic_pred_value.detach().item()),
                "renorm_critic_target": float(renorm_critic_target_value.detach().item()),
                "rollout_len": 1,
                "replay_kind": "trajectory_reuse",
                "windows_per_teacher": int(args.windows_per_teacher),
                "loss": float(loss.detach().item()),
                "flow_loss": float(flow_value.detach().item()),
                "latent_loss": float(latent_value.detach().item()),
                "direction_loss": float(dir_value.detach().item()),
                "norm_loss": float(norm_value.detach().item()),
                "spatial_loss": float(spatial_value.detach().item()),
                "flow_frequency_loss": float(flow_frequency_value.detach().item()),
                "endpoint_frequency_loss": float(endpoint_frequency_value.detach().item()),
                "prompt_contrast_loss": float(contrast_value.detach().item()),
                "prompt_negative_teacher_loss": float(negative_teacher_value.detach().item()),
                "prompt_teacher_delta_loss": float(prompt_delta_value.detach().item()),
                "prompt_counterfactual_terminal_loss": float(counterfactual_terminal_value.detach().item()),
                "prompt_counterfactual_delta_loss": float(counterfactual_delta_value.detach().item()),
                "prompt_counterfactual_delta_direction_loss": float(counterfactual_delta_direction_value.detach().item()),
                "prompt_counterfactual_delta_norm_loss": float(counterfactual_delta_norm_value.detach().item()),
                "prompt_sensitivity_floor_loss": float(prompt_sensitivity_floor_value.detach().item()),
                "prompt_sensitivity_ratio_loss": float(prompt_sensitivity_ratio_value.detach().item()),
                "prompt_sensitivity_ratio": float(prompt_sensitivity_ratio_mean_value.detach().item()),
                "student_counterfactual_delta_norm": float(student_counterfactual_delta_norm_value.detach().item()),
                "teacher_counterfactual_delta_norm": float(teacher_counterfactual_delta_norm_value.detach().item()),
                "prompt_counterfactual_horizon": int(counterfactual_horizon),
                "endpoint_loss": float(endpoint_value.detach().item()),
                "balanced_flow_loss": float(balanced_flow_value.detach().item()),
                "balanced_endpoint_loss": float(balanced_endpoint_value.detach().item()),
                "multistep_consistency_loss": float(multistep_value.detach().item()),
                "multistep_terminal_loss": float(multistep_terminal_value.detach().item()),
                "multistep_terminal_spatial_loss": float(multistep_terminal_spatial_value.detach().item()),
                "multistep_consistency_horizon": int(multistep_horizon),
                "latent_adv_loss": float(latent_adv_value.detach().item()),
                "latent_disc_loss": float(latent_disc_value.detach().item()),
                "latent_adv_horizon": int(latent_adv_horizon),
                "on_policy_teacher_loss": float(on_policy_teacher_value.detach().item()),
                "on_policy_teacher_horizon": int(on_policy_teacher_horizon),
                "decoded_image_loss": float(decoded_image_value.detach().item()),
                "decoded_clip_loss": float(decoded_clip_value.detach().item()),
                "decoded_dino_loss": float(decoded_dino_value.detach().item()),
                "decoded_edge_loss": float(decoded_edge_value.detach().item()),
                "decoded_lowfreq_loss": float(decoded_lowfreq_value.detach().item()),
                "decoded_adv_loss": float(decoded_adv_value.detach().item()),
                "decoded_disc_loss": float(decoded_disc_value.detach().item()),
                "mode": "live_teacher_flux_trajectory_reuse",
            }
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
            if step % args.log_every == 0 or step == start_step + 1:
                print(json.dumps(ledger), flush=True)
            if step % args.checkpoint_every == 0:
                save_checkpoint(output_dir, config, student, step, float(loss.detach().item()), args, ema_state)
                maybe_snapshot_student_checkpoint(output_dir, step, int(args.snapshot_every))
                if args.save_optimizer:
                    save_optimizer_checkpoint(
                        output_dir,
                        step,
                        optimizer,
                        discriminator_optimizer,
                        decoded_discriminator_optimizer,
                    )
                save_discriminator_checkpoint(output_dir, step, latent_discriminator)
                save_decoded_discriminator_checkpoint(output_dir, step, decoded_discriminator)

    save_checkpoint(output_dir, config, student, step, float(loss.detach().item()), args, ema_state)
    maybe_snapshot_student_checkpoint(output_dir, step, int(args.snapshot_every))
    if args.save_optimizer:
        save_optimizer_checkpoint(output_dir, step, optimizer, discriminator_optimizer, decoded_discriminator_optimizer)
    save_discriminator_checkpoint(output_dir, step, latent_discriminator)
    save_decoded_discriminator_checkpoint(output_dir, step, decoded_discriminator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast live FLUX teacher distillation using in-memory teacher trajectory reuse.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--override-resume-config", action="store_true")
    parser.add_argument("--reset-ema", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--prompt-file", default="data/vision/prompts/imagenet_object_photo_12k_v0.jsonl")
    parser.add_argument("--prompt-mix", action="append", default=[])
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--no-resume-prompt-advance", action="store_true")
    parser.add_argument("--trainable-name-regex", default="")
    parser.add_argument("--residual-gate-floor", type=float, default=0.0)
    parser.add_argument("--residual-gate-floor-start-block", type=int, default=0)
    parser.add_argument("--new-block-gate-init", type=float, default=-1.0)
    parser.add_argument("--prompt-cond-gate-init", type=float, default=-1.0)
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--teacher-steps", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=1)
    parser.add_argument("--windows-per-teacher", type=int, default=64)
    parser.add_argument("--front-start-prob", type=float, default=0.15)
    parser.add_argument("--tail-start-prob", type=float, default=0.35)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument(
        "--min-start-index",
        type=int,
        default=0,
        help="Minimum teacher trajectory index that can be selected as a training start.",
    )
    parser.add_argument(
        "--max-start-index",
        type=int,
        default=-1,
        help="Maximum teacher trajectory index that can be selected as a training start. Use to keep enough rollout horizon for phase-targeted training.",
    )
    parser.add_argument(
        "--start-index-loss-weight-power",
        type=float,
        default=0.0,
        help="If positive, multiply each row loss by 1 + normalized_start_index ** power.",
    )
    parser.add_argument(
        "--late-start-index-min",
        type=int,
        default=-1,
        help="If non-negative, rows at or after this start index get an extra multiplier.",
    )
    parser.add_argument("--late-start-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=720)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pos2d-scale", type=float, default=0.05)
    parser.add_argument("--timestep-scale", type=float, default=1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=0.10)
    parser.add_argument("--output-refiner-hidden", type=int, default=0)
    parser.add_argument("--output-refiner-depth", type=int, default=2)
    parser.add_argument("--output-refiner-scale", type=float, default=0.25)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--train-adapters-only", action="store_true")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--latent-noise-std", type=float, default=0.02)
    parser.add_argument("--latent-noise-timestep-scale", action="store_true")
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--snr-weighting", action="store_true")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--min-snr-weight", type=float, default=0.05)
    parser.add_argument("--max-snr-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=6e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=500.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.15)
    parser.add_argument("--norm-loss-weight", type=float, default=0.04)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.02)
    parser.add_argument("--flow-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--frequency-loss-cutoff", type=float, default=0.35)
    parser.add_argument("--balanced-channel-weight-file", default="")
    parser.add_argument("--balanced-flow-loss-weight", type=float, default=0.0)
    parser.add_argument("--balanced-endpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-contrast-weight", type=float, default=0.0)
    parser.add_argument("--prompt-contrast-prob", type=float, default=0.5)
    parser.add_argument("--prompt-contrast-every", type=int, default=2)
    parser.add_argument("--prompt-contrast-margin", type=float, default=0.12)
    parser.add_argument("--prompt-token-drop-negative-prob", type=float, default=0.0)
    parser.add_argument(
        "--prompt-token-drop-count",
        type=int,
        default=1,
        help="Number of non-stopword prompt tokens to remove for token-drop negatives. Use 0 to drop all content tokens.",
    )
    parser.add_argument("--prompt-negative-teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-teacher-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-counterfactual-terminal-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-counterfactual-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-counterfactual-delta-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-counterfactual-delta-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-sensitivity-floor-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-sensitivity-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-sensitivity-min-ratio", type=float, default=0.35)
    parser.add_argument("--prompt-counterfactual-horizon", type=int, default=2)
    parser.add_argument("--endpoint-loss-weight", type=float, default=250.0)
    parser.add_argument("--multistep-consistency-weight", type=float, default=0.0)
    parser.add_argument("--multistep-terminal-loss-weight", type=float, default=0.0)
    parser.add_argument("--multistep-terminal-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--multistep-consistency-prob", type=float, default=0.25)
    parser.add_argument("--multistep-consistency-horizons", default="2")
    parser.add_argument(
        "--multistep-consistency-horizon-weights",
        default="",
        help="Optional comma-separated horizon:weight list, e.g. 4:0.45,6:0.45,8:0.10.",
    )
    parser.add_argument(
        "--multistep-detach-rollout",
        action="store_true",
        help="Detach intermediate multistep rollout latents so longer horizon endpoint/decoded training fits in memory.",
    )
    parser.add_argument(
        "--use-scheduler-step-loss",
        action="store_true",
        help="Use the actual FLUX scheduler step for latent rollout losses instead of a scalar teacher-delta approximation.",
    )
    parser.add_argument("--latent-adv-weight", type=float, default=0.0)
    parser.add_argument("--latent-adv-prob", type=float, default=0.25)
    parser.add_argument("--latent-adv-horizons", default="2,4")
    parser.add_argument("--latent-adv-start-step", type=int, default=0)
    parser.add_argument("--latent-adv-lr", type=float, default=1e-5)
    parser.add_argument("--latent-adv-hidden", type=int, default=128)
    parser.add_argument("--latent-adv-grad-clip", type=float, default=1.0)
    parser.add_argument("--on-policy-teacher-weight", type=float, default=0.0)
    parser.add_argument("--on-policy-teacher-prob", type=float, default=0.05)
    parser.add_argument("--on-policy-teacher-horizons", default="1,2")
    parser.add_argument(
        "--on-policy-detach-rollout",
        action="store_true",
        help="Use detached student rollout states for on-policy teacher correction to reduce memory and train recovery at the off-policy states directly.",
    )
    parser.add_argument("--on-policy-flow-weight", type=float, default=0.5)
    parser.add_argument("--on-policy-latent-weight", type=float, default=200.0)
    parser.add_argument("--decoded-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-device", default=None)
    parser.add_argument("--decoded-image-loss-prob", type=float, default=0.05)
    parser.add_argument("--decoded-image-loss-min-timestep-index", type=int, default=48)
    parser.add_argument("--decoded-image-loss-size", type=int, default=128)
    parser.add_argument("--decoded-clip-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--decoded-clip-loss-size", type=int, default=224)
    parser.add_argument("--decoded-dino-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-dino-model", default="facebook/dinov2-base")
    parser.add_argument("--decoded-dino-loss-size", type=int, default=224)
    parser.add_argument("--decoded-edge-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-lowfreq-loss-size", type=int, default=32)
    parser.add_argument("--decoded-adv-weight", type=float, default=0.0)
    parser.add_argument("--decoded-adv-prob", type=float, default=0.25)
    parser.add_argument("--decoded-adv-size", type=int, default=128)
    parser.add_argument("--decoded-adv-hidden", type=int, default=48)
    parser.add_argument("--decoded-adv-lr", type=float, default=3e-6)
    parser.add_argument("--decoded-adv-grad-clip", type=float, default=1.0)
    parser.add_argument("--renorm-loss-critic", action="store_true")
    parser.add_argument("--renorm-loss-critic-hidden", type=int, default=64)
    parser.add_argument("--renorm-loss-critic-lr", type=float, default=1e-4)
    parser.add_argument("--renorm-loss-critic-weight-decay", type=float, default=0.0)
    parser.add_argument("--renorm-loss-critic-grad-clip", type=float, default=1.0)
    parser.add_argument("--renorm-loss-critic-scale", type=float, default=0.25)
    parser.add_argument("--renorm-loss-critic-min-weight", type=float, default=0.75)
    parser.add_argument("--renorm-loss-critic-max-weight", type=float, default=2.5)
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--ema-update-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--cache-trajectory-on-student-device", action="store_true")
    parser.add_argument("--trajectory-cache-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--prompt-seed-alias-file", default="")
    parser.add_argument("--prompt-condition-alias-file", default="")
    parser.add_argument("--prompt-negative-alias-file", default="")
    parser.add_argument("--randomize-seeds", action="store_true")
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
