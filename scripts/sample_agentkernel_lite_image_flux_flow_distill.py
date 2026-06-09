#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from generate_agentkernel_lite_image_teacher_corpus import decode_flux_latents, load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig, pack_latent_grid, unpack_packed_latent_grid


DEFAULT_FLUX_TEACHER = "black-forest-labs/FLUX.1-dev"


class SimpleFinalLatentRefiner(nn.Module):
    def __init__(self, channels: int, hidden: int, depth: int, residual_scale: float) -> None:
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
        return refined.flatten(2).transpose(1, 2).to(dtype=packed.dtype)


class FluxFinalLatentRefiner(nn.Module):
    def __init__(self, packed_channels: int, hidden: int, depth: int, residual_scale: float) -> None:
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


class FluxUNetFinalLatentRefiner(nn.Module):
    def __init__(self, packed_channels: int, hidden: int, depth: int, residual_scale: float) -> None:
        super().__init__()
        if packed_channels % 4:
            raise ValueError("FLUX packed channels must be divisible by 4")
        latent_channels = packed_channels // 4
        mid = max(32, int(hidden) // 2)
        low = max(32, int(hidden) // 4)
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


def stable_prompt_seed(prompt: str, seed_min: int, seed_max: int) -> int:
    if seed_min > seed_max:
        raise ValueError("--seed-min must be <= --seed-max")
    digest = hashlib.blake2s(prompt.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return int(seed_min + (value % (seed_max - seed_min + 1)))


def parse_seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds


def parse_step_indices(value: str) -> set[int]:
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            indices.add(int(item))
    return indices


def load_final_latent_refiner(path: str, device: torch.device) -> nn.Module | None:
    if not path:
        return None
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = checkpoint.get("config", {})
    packing_mode = str(config.get("packing_mode", "simple"))
    channels = int(config.get("channels", 64))
    hidden = int(config.get("hidden", 192))
    depth = int(config.get("depth", 6))
    residual_scale = float(config.get("residual_scale", 0.25))
    architecture = str(config.get("architecture", "conv"))
    if architecture == "flux_unet":
        refiner = FluxUNetFinalLatentRefiner(channels, hidden, depth, residual_scale)
    elif packing_mode == "flux":
        refiner: nn.Module = FluxFinalLatentRefiner(channels, hidden, depth, residual_scale)
    else:
        refiner = SimpleFinalLatentRefiner(channels, hidden, depth, residual_scale)
    refiner.load_state_dict(checkpoint["model"], strict=True)
    refiner.to(device)
    refiner.eval()
    return refiner


def load_embedded_final_latent_refiner(checkpoint_path: Path, device: torch.device) -> nn.Module | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    payload = checkpoint.get("embedded_final_latent_refiner") if isinstance(checkpoint, dict) else None
    if not payload:
        return None
    config = payload["config"]
    channels = int(config.get("channels", 64))
    hidden = int(config.get("hidden", 128))
    depth = int(config.get("depth", 4))
    residual_scale = float(config.get("residual_scale", 0.25))
    architecture = str(config.get("architecture", "conv"))
    packing_mode = str(config.get("packing_mode", "flux"))
    if architecture == "flux_unet":
        refiner: nn.Module = FluxUNetFinalLatentRefiner(channels, hidden, depth, residual_scale)
    elif packing_mode == "flux":
        refiner = FluxFinalLatentRefiner(channels, hidden, depth, residual_scale)
    else:
        refiner = SimpleFinalLatentRefiner(channels, hidden, depth, residual_scale)
    refiner.load_state_dict(payload["model"], strict=True)
    refiner.to(device)
    refiner.eval()
    print(
        json.dumps(
            {
                "embedded_final_latent_refiner": {
                    "checkpoint": str(checkpoint_path),
                    "source": payload.get("source", ""),
                    "config": config,
                }
            }
        ),
        flush=True,
    )
    return refiner


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


def checkpoint_state(checkpoint_path: Path, weights: str) -> tuple[dict[str, torch.Tensor], FluxPackedStudentConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    if weights == "ema":
        state = checkpoint.get("student_ema")
        if not state:
            raise ValueError(f"{checkpoint_path} does not contain student_ema weights")
    elif weights == "materialized":
        state = checkpoint.get("student_materialized")
        if not state:
            raise ValueError(f"{checkpoint_path} does not contain student_materialized weights")
    else:
        state = checkpoint["student"]
    return state, config


def load_checkpoint_bank(path: str) -> dict[str, Path]:
    if not path:
        return {}
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    prompt_to_checkpoint: dict[str, Path] = {}
    for entry in manifest.get("checkpoints", manifest.get("adapters", [])):
        checkpoint = Path(entry["checkpoint"])
        prompts = entry.get("prompts") or []
        for prompt in prompts:
            prompt_to_checkpoint[str(prompt).strip()] = checkpoint
    return prompt_to_checkpoint


def load_stage_bridge_bank(path: str) -> dict[int, dict[str, object]]:
    if not path:
        return {}
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = manifest.get("bridges", manifest.get("stages", []))
    bank: dict[int, dict[str, object]] = {}
    for entry in entries:
        target_index = int(entry["target_index"])
        bank[target_index] = {
            "checkpoint": Path(entry["checkpoint"]),
            "weights": str(entry.get("weights", "raw")),
            "output_mode": str(entry.get("output_mode", manifest.get("output_mode", "delta"))),
            "scale": float(entry.get("scale", manifest.get("scale", 1.0))),
        }
    return bank


def prompt_seed_cache_key(prompt: str, seed: int) -> str:
    return f"{prompt} | seed={int(seed)}"


def apply_student_state(
    student: FluxPackedStudent,
    config: FluxPackedStudentConfig,
    checkpoint_path: Path,
    weights: str,
) -> None:
    state, next_config = checkpoint_state(checkpoint_path, weights)
    if as_config_signature(next_config) != as_config_signature(config):
        raise ValueError(f"checkpoint config does not match base sampler config: {checkpoint_path}")
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"student_state": {"checkpoint": str(checkpoint_path), "missing": missing, "unexpected": unexpected}}), flush=True)
    student.eval()


def as_config_signature(config: FluxPackedStudentConfig) -> dict[str, int | float]:
    return {
        "latent_tokens": int(config.latent_tokens),
        "latent_channels": int(config.latent_channels),
        "prompt_dim": int(config.prompt_dim),
        "pooled_dim": int(config.pooled_dim),
        "max_sequence_length": int(config.max_sequence_length),
        "dim": int(config.dim),
        "depth": int(config.depth),
        "heads": int(config.heads),
        "mlp_ratio": int(config.mlp_ratio),
        "pos2d_scale": float(config.pos2d_scale),
        "timestep_scale": float(config.timestep_scale),
        "local_mixer_scale": float(config.local_mixer_scale),
        "adapter_rank": int(config.adapter_rank),
        "adapter_scale": float(config.adapter_scale),
        "output_refiner_hidden": int(getattr(config, "output_refiner_hidden", 0)),
        "output_refiner_depth": int(getattr(config, "output_refiner_depth", 2)),
        "output_refiner_scale": float(getattr(config, "output_refiner_scale", 0.25)),
    }


def load_student(
    checkpoint_path: Path,
    device: torch.device,
    weights: str,
    timestep_scale_override: float | None = None,
) -> tuple[FluxPackedStudent, FluxPackedStudentConfig]:
    state, config = checkpoint_state(checkpoint_path, weights)
    if timestep_scale_override is not None:
        config.timestep_scale = float(timestep_scale_override)
    student = FluxPackedStudent(config).to(device)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"student_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
    student.eval()
    return student, config


def load_cached_prompt_embeddings(
    target_dir: Path,
    initial_latent_policy: str = "timestep0",
    initial_latent_timestep_index: int = -1,
) -> dict[str, dict[str, torch.Tensor | int]]:
    metadata_path = target_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        seed = int(row.get("seed", 0) or 0)
        embedding_path = target_dir / row["embedding_path"]
        embeds = torch.load(embedding_path, map_location="cpu")
        target = torch.load(target_dir / row["target_path"], map_location="cpu")
        payload = {
            "prompt_embeds": embeds["prompt_embeds"],
            "pooled_prompt_embeds": embeds["pooled_prompt_embeds"],
            "seed": seed,
            "initial_latents": target["latents"],
        }
        timestep_index = int(row.get("timestep_index", 0) or 0)
        if initial_latent_timestep_index >= 0:
            if timestep_index == initial_latent_timestep_index:
                cache[prompt] = payload
                cache[prompt_seed_cache_key(prompt, seed)] = payload
        elif initial_latent_policy == "last":
            cache[prompt] = payload
            cache[prompt_seed_cache_key(prompt, seed)] = payload
        elif timestep_index == 0:
            cache[prompt] = payload
            cache[prompt_seed_cache_key(prompt, seed)] = payload
        elif initial_latent_policy == "first":
            cache.setdefault(prompt, payload)
            cache.setdefault(prompt_seed_cache_key(prompt, seed), payload)
        elif initial_latent_policy != "timestep0":
            raise ValueError(f"unknown cached initial latent policy: {initial_latent_policy}")
    if not cache:
        raise ValueError(f"no cached prompt embeddings found in {metadata_path}")
    return cache


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(Path(args.prompts), args.limit)
    if not prompts:
        raise ValueError("no prompts found")
    prompt_seed_list = parse_seed_list(args.prompt_seed_list)
    prompt_seed_aliases = load_prompt_seed_aliases(args.prompt_seed_alias_file)
    prompt_condition_aliases = load_prompt_condition_aliases(args.prompt_condition_alias_file)
    checkpoint_bank = load_checkpoint_bank(args.checkpoint_bank_manifest)
    stage_bridge_bank = load_stage_bridge_bank(args.initial_latent_bridge_bank_manifest)
    embedding_cache = None
    if args.embedding_target_dir:
        embedding_cache = load_cached_prompt_embeddings(
            Path(args.embedding_target_dir),
            args.cached_initial_latent_policy,
            int(args.cached_initial_latent_timestep_index),
        )
    teacher_args = argparse.Namespace(
        teacher_family="flux",
        teacher_model=args.teacher_model,
        dtype=args.dtype,
        variant="",
        local_files_only=args.local_files_only,
        quantize_transformer_4bit=args.quantize_transformer_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        cpu_offload=args.cpu_offload,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    pipe = load_teacher(teacher_args)
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    final_latent_refiner = load_final_latent_refiner(args.final_latent_refiner, device)
    if final_latent_refiner is None and not bool(args.disable_embedded_final_latent_refiner):
        final_latent_refiner = load_embedded_final_latent_refiner(Path(args.checkpoint), device)
    student, student_config = load_student(
        Path(args.checkpoint),
        device,
        args.weights,
        args.timestep_scale_override if args.timestep_scale_override > 0 else None,
    )
    initial_bridge: FluxPackedStudent | None = None
    if args.initial_latent_bridge_checkpoint:
        initial_bridge, _initial_bridge_config = load_student(
            Path(args.initial_latent_bridge_checkpoint),
            device,
            args.initial_latent_bridge_weights,
            args.timestep_scale_override if args.timestep_scale_override > 0 else None,
        )
        initial_bridge.eval()
    initial_bridge_refiner = load_final_latent_refiner(args.initial_latent_bridge_refiner, device)
    active_checkpoint = Path(args.checkpoint)
    max_sequence_length = int(args.max_sequence_length)
    if max_sequence_length <= 0:
        max_sequence_length = int(student_config.max_sequence_length)
    images: list[tuple[str, Path]] = []
    bridge_stage_images: list[tuple[str, Path]] = []
    adaptive_log_path = output_dir / "adaptive_sampler.jsonl"
    intermediate_step_indices = parse_step_indices(args.save_intermediate_step_indices)
    intermediate_dir = output_dir / "intermediates"
    if intermediate_step_indices:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        bridge_endpoint_decode_only = False
        prompt_checkpoint = checkpoint_bank.get(prompt)
        if prompt_checkpoint is not None and prompt_checkpoint != active_checkpoint:
            apply_student_state(student, student_config, prompt_checkpoint, args.weights)
            active_checkpoint = prompt_checkpoint
        condition_prompt = prompt_condition_aliases.get(prompt, prompt)
        explicit_seed = int(prompt_seed_list[index % len(prompt_seed_list)]) if prompt_seed_list else None
        if embedding_cache is None:
            prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
                prompt=condition_prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )
        else:
            embedding_prompt = condition_prompt if condition_prompt in embedding_cache else prompt
            exact_embedding_key = prompt_seed_cache_key(embedding_prompt, explicit_seed) if explicit_seed is not None else ""
            embedding_key = exact_embedding_key if exact_embedding_key and exact_embedding_key in embedding_cache else embedding_prompt
            if embedding_key not in embedding_cache:
                raise KeyError(f"prompt has no cached embedding in {args.embedding_target_dir}: {embedding_prompt}")
            embeds = embedding_cache[embedding_key]
            prompt_embeds = embeds["prompt_embeds"].to(device=device, dtype=pipe.dtype)
            pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
        latent_prompt = prompt_seed_aliases.get(prompt, prompt)
        if explicit_seed is not None:
            seed = explicit_seed
        elif embedding_cache is not None and args.use_cached_seeds:
            seed_prompt_key = latent_prompt if latent_prompt in embedding_cache else prompt
            seed = int(embedding_cache[seed_prompt_key]["seed"])
        elif args.prompt_hash_seeds:
            seed_prompt = prompt_seed_aliases.get(prompt, prompt)
            seed = stable_prompt_seed(seed_prompt, int(args.seed_min), int(args.seed_max))
        else:
            seed_prompt = ""
            seed = args.seed + index
        if embedding_cache is not None and args.use_cached_initial_latents:
            latent_prompt_key = latent_prompt if latent_prompt in embedding_cache else prompt
            exact_latent_key = prompt_seed_cache_key(latent_prompt_key, seed)
            latent_key = exact_latent_key if exact_latent_key in embedding_cache else latent_prompt_key
            latents = embedding_cache[latent_key]["initial_latents"].to(device=device, dtype=prompt_embeds.dtype)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
            num_channels_latents = pipe.transformer.config.in_channels // 4
            latents, _latent_image_ids = pipe.prepare_latents(
                1,
                num_channels_latents,
                args.height,
                args.width,
                prompt_embeds.dtype,
                device,
                generator,
                None,
            )
        sample_steps = int(args.adaptive_max_steps) if args.adaptive and args.adaptive_max_steps > 0 else int(args.steps)
        timesteps, _ = flux_timesteps(pipe, latents, sample_steps, device)
        sampler_start_index = int(args.sampler_start_timestep_index)
        if initial_bridge is not None or stage_bridge_bank:
            if args.initial_latent_bridge_stage_indices.strip():
                bridge_target_indices = [
                    int(part.strip())
                    for part in args.initial_latent_bridge_stage_indices.split(",")
                    if part.strip()
                ]
            else:
                bridge_target_indices = [int(args.initial_latent_bridge_target_index)]
            previous_bridge_index = 0
            for bridge_target_index in bridge_target_indices:
                if bridge_target_index <= previous_bridge_index or bridge_target_index >= len(timesteps):
                    raise ValueError(
                        "--initial-latent-bridge-stage-indices must be strictly increasing "
                        f"within [1, {len(timesteps) - 1}], got {bridge_target_indices}"
                    )
                previous_bridge_index = bridge_target_index
            guidance = torch.full([latents.shape[0]], args.guidance, device=device, dtype=torch.float32)
            bridge_start_index = 0
            for bridge_target_index in bridge_target_indices:
                stage_bridge = initial_bridge
                stage_output_mode = args.initial_latent_bridge_output_mode
                stage_scale = float(args.initial_latent_bridge_scale)
                stage_entry = stage_bridge_bank.get(int(bridge_target_index))
                if stage_entry is not None:
                    stage_bridge, _stage_bridge_config = load_student(
                        Path(stage_entry["checkpoint"]),
                        device,
                        str(stage_entry["weights"]),
                        args.timestep_scale_override if args.timestep_scale_override > 0 else None,
                    )
                    stage_bridge.eval()
                    stage_output_mode = str(stage_entry["output_mode"])
                    stage_scale = float(stage_entry["scale"])
                if stage_bridge is None:
                    raise ValueError(f"no bridge checkpoint configured for stage {bridge_target_index}")
                bridge_timestep = timesteps[bridge_start_index].expand(latents.shape[0]).to(device)
                bridge_delta = stage_bridge(
                    latents.float(),
                    bridge_timestep.float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                ).to(latents.dtype)
                if stage_output_mode == "absolute":
                    latents = bridge_delta * stage_scale
                else:
                    latents = latents + bridge_delta * stage_scale
                if stage_entry is not None:
                    del stage_bridge
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if args.save_bridge_stage_images:
                    bridge_stage_dir = output_dir / "bridge_stages"
                    bridge_stage_dir.mkdir(parents=True, exist_ok=True)
                    bridge_image = decode_flux_latents(pipe, latents, args.height, args.width)
                    bridge_path = bridge_stage_dir / f"sample_{index:03d}_t{bridge_target_index:02d}.png"
                    bridge_image.save(bridge_path)
                    bridge_stage_images.append((f"t{bridge_target_index} {prompt}", bridge_path))
                bridge_start_index = bridge_target_index
            if initial_bridge_refiner is not None:
                latents = initial_bridge_refiner(latents)
            if sampler_start_index < 0:
                sampler_start_index = bridge_target_indices[-1]
            bridge_endpoint_decode_only = bool(args.decode_bridge_endpoint_only)
        if sampler_start_index < 0 and int(args.cached_initial_latent_timestep_index) >= 0:
            sampler_start_index = int(args.cached_initial_latent_timestep_index)
        if sampler_start_index > 0 and not bridge_endpoint_decode_only:
            if sampler_start_index >= len(timesteps):
                raise ValueError(f"sampler start timestep index {sampler_start_index} >= number of steps {len(timesteps)}")
            timesteps = timesteps[sampler_start_index:]
        guidance = torch.full([latents.shape[0]], args.guidance, device=device, dtype=torch.float32)
        previous_pred: torch.Tensor | None = None
        stop_reason = "bridge_endpoint" if bridge_endpoint_decode_only else "max_steps"
        steps_used = 0
        final_latent_delta = 0.0
        final_pred_delta = 0.0
        if not bridge_endpoint_decode_only:
            for step_index, timestep_value in enumerate(timesteps):
                if step_index == 0 and hasattr(pipe.scheduler, "_step_index"):
                    pipe.scheduler._step_index = None
                timestep = timestep_value.expand(latents.shape[0]).to(device)
                pred = student(
                    latents.float(),
                    timestep.float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                ).to(latents.dtype) * float(args.prediction_scale)
                next_latents = pipe.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
                latent_delta = (next_latents.float() - latents.float()).pow(2).mean().sqrt()
                if previous_pred is None:
                    pred_delta = torch.full_like(latent_delta, float("inf"))
                else:
                    pred_delta = (pred.float() - previous_pred.float()).pow(2).mean().sqrt()
                latents = next_latents
                previous_pred = pred.detach()
                steps_used = step_index + 1
                if steps_used in intermediate_step_indices:
                    intermediate_image = decode_flux_latents(pipe, latents, args.height, args.width)
                    intermediate_image.save(intermediate_dir / f"sample_{index:03d}_step_{steps_used:03d}.png")
                final_latent_delta = float(latent_delta.detach().item())
                final_pred_delta = float(pred_delta.detach().item()) if torch.isfinite(pred_delta).all() else float("inf")
                if (
                    args.adaptive
                    and steps_used >= args.adaptive_min_steps
                    and final_latent_delta <= args.adaptive_latent_delta_threshold
                    and final_pred_delta <= args.adaptive_prediction_delta_threshold
                ):
                    stop_reason = "converged"
                    break
        if final_latent_refiner is not None:
            latents = final_latent_refiner(latents)
        image = decode_flux_latents(pipe, latents, args.height, args.width)
        path = output_dir / f"sample_{index:03d}.png"
        image.save(path)
        images.append((prompt, path))
        sample_log = {
            "prompt": prompt,
            "condition_prompt": condition_prompt,
            "seed": seed,
            "seed_prompt": prompt_seed_aliases.get(prompt, prompt) if args.prompt_hash_seeds else "",
            "path": str(path),
            "steps_used": steps_used,
            "stop_reason": stop_reason,
            "final_latent_delta": final_latent_delta,
            "final_prediction_delta": final_pred_delta,
            "adaptive": bool(args.adaptive),
            "checkpoint": str(active_checkpoint),
            "final_latent_refiner": str(args.final_latent_refiner or ""),
            "initial_latent_bridge": str(args.initial_latent_bridge_checkpoint or ""),
            "initial_latent_bridge_target_index": int(args.initial_latent_bridge_target_index),
            "initial_latent_bridge_scale": float(args.initial_latent_bridge_scale),
            "initial_latent_bridge_output_mode": str(args.initial_latent_bridge_output_mode),
            "decode_bridge_endpoint_only": bool(args.decode_bridge_endpoint_only),
            "cached_initial_latent_policy": str(args.cached_initial_latent_policy),
            "cached_initial_latent_timestep_index": int(args.cached_initial_latent_timestep_index),
            "sampler_start_timestep_index": int(sampler_start_index),
        }
        if args.adaptive:
            with adaptive_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample_log) + "\n")
        print(json.dumps(sample_log), flush=True)

    if images:
        thumb = args.contact_thumb
        cols = min(args.contact_cols, len(images))
        rows = (len(images) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 44)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (prompt, path) in enumerate(images):
            x = (index % cols) * thumb
            y = (index // cols) * (thumb + 44)
            image = Image.open(path).convert("RGB").resize((thumb, thumb))
            sheet.paste(image, (x, y))
            draw.text((x + 6, y + thumb + 6), prompt[:32], fill=(0, 0, 0))
        sheet.save(output_dir / "contact_sheet.png")
        print(json.dumps({"contact_sheet": str(output_dir / "contact_sheet.png")}), flush=True)
    if bridge_stage_images:
        thumb = args.contact_thumb
        cols = min(args.contact_cols, len(bridge_stage_images))
        rows = (len(bridge_stage_images) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 44)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (prompt, path) in enumerate(bridge_stage_images):
            x = (index % cols) * thumb
            y = (index // cols) * (thumb + 44)
            image = Image.open(path).convert("RGB").resize((thumb, thumb))
            sheet.paste(image, (x, y))
            draw.text((x + 6, y + thumb + 6), prompt[:32], fill=(0, 0, 0))
        sheet.save(output_dir / "bridge_stage_contact_sheet.png")
        print(json.dumps({"bridge_stage_contact_sheet": str(output_dir / "bridge_stage_contact_sheet.png")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a FLUX packed-latent student.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-bank-manifest", default="")
    parser.add_argument("--weights", choices=("raw", "ema", "materialized"), default="raw")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_flux_flow_student_samples")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--embedding-target-dir", default="")
    parser.add_argument("--cached-initial-latent-policy", choices=("timestep0", "first", "last"), default="timestep0")
    parser.add_argument("--cached-initial-latent-timestep-index", type=int, default=-1)
    parser.add_argument("--sampler-start-timestep-index", type=int, default=-1)
    parser.add_argument("--save-intermediate-step-indices", default="")
    parser.add_argument("--use-cached-seeds", action="store_true")
    parser.add_argument("--use-cached-initial-latents", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adaptive-max-steps", type=int, default=0)
    parser.add_argument("--adaptive-min-steps", type=int, default=24)
    parser.add_argument("--adaptive-latent-delta-threshold", type=float, default=0.0008)
    parser.add_argument("--adaptive-prediction-delta-threshold", type=float, default=0.005)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--prediction-scale", type=float, default=1.0)
    parser.add_argument("--final-latent-refiner", default="")
    parser.add_argument("--disable-embedded-final-latent-refiner", action="store_true")
    parser.add_argument("--initial-latent-bridge-checkpoint", default="")
    parser.add_argument("--initial-latent-bridge-bank-manifest", default="")
    parser.add_argument("--initial-latent-bridge-weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--initial-latent-bridge-target-index", type=int, default=15)
    parser.add_argument("--initial-latent-bridge-stage-indices", default="")
    parser.add_argument("--initial-latent-bridge-scale", type=float, default=1.0)
    parser.add_argument("--initial-latent-bridge-output-mode", choices=("delta", "absolute"), default="delta")
    parser.add_argument("--initial-latent-bridge-refiner", default="")
    parser.add_argument("--save-bridge-stage-images", action="store_true")
    parser.add_argument("--decode-bridge-endpoint-only", action="store_true")
    parser.add_argument("--timestep-scale-override", type=float, default=0.0)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--prompt-seed-list", default="")
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--prompt-seed-alias-file", default="")
    parser.add_argument("--prompt-condition-alias-file", default="")
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--contact-thumb", type=int, default=192)
    parser.add_argument("--contact-cols", type=int, default=4)
    args = parser.parse_args()
    sample(args)


if __name__ == "__main__":
    main()
