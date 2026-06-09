#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterator

import torch
from torch import nn
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, load_teacher, read_prompts, teacher_family
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from train_agentkernel_lite_image_flux_flow_distill import (
    FluxPackedStudent,
    FluxPackedStudentConfig,
    clone_state_dict,
    infer_config,
    save_checkpoint,
    seed_everything,
    update_ema_state,
)
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    direction_loss,
    norm_loss,
    packed_spatial_gradient_loss,
    reconstruction_loss,
    teacher_step_delta,
    timestep_loss_weight,
)
from train_agentkernel_lite_image_sana_latent_distill import apply_bitnet_qat_modules


def prompt_cycle(path: Path, limit: int) -> Iterator[str]:
    prompts = read_prompts(path, limit)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    while True:
        order = list(prompts)
        random.shuffle(order)
        for prompt in order:
            yield prompt


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


def parse_prompt_mix(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompt_mix:
        sources = []
        for item in args.prompt_mix:
            parts = item.split(":", 2)
            path = parts[0]
            weight = float(parts[1]) if len(parts) >= 2 and parts[1] else 1.0
            name = parts[2] if len(parts) >= 3 and parts[2] else Path(path).stem
            sources.append({"path": path, "weight": max(weight, 0.0), "name": name})
        if sources:
            return sources
    return [{"path": args.prompt_file, "weight": 1.0, "name": Path(args.prompt_file).stem}]


class PromptMixer:
    def __init__(self, sources: list[dict[str, Any]], limit: int) -> None:
        self.sources = sources
        self.prompts = [read_prompts(Path(source["path"]), limit) for source in sources]
        for source, prompts in zip(sources, self.prompts):
            if not prompts:
                raise ValueError(f"no prompts found in {source['path']}")
        self.prompt_indices: dict[str, int] = {}
        for prompts in self.prompts:
            for index, prompt in enumerate(prompts):
                self.prompt_indices.setdefault(prompt, index)
        self.orders: list[list[str]] = []
        for prompts in self.prompts:
            order = list(prompts)
            random.shuffle(order)
            self.orders.append(order)
        self.weights = [float(source["weight"]) for source in sources]

    def next(self) -> tuple[str, str]:
        index = random.choices(range(len(self.prompts)), weights=self.weights, k=1)[0]
        if not self.orders[index]:
            self.orders[index] = list(self.prompts[index])
            random.shuffle(self.orders[index])
        return self.orders[index].pop(), str(self.sources[index]["name"])

    def advance(self, count: int) -> None:
        for _ in range(max(int(count), 0)):
            self.next()

    def sample_negative(self, prompt: str) -> str:
        for _ in range(16):
            index = random.choices(range(len(self.prompts)), weights=self.weights, k=1)[0]
            candidate = random.choice(self.prompts[index])
            if candidate != prompt:
                return candidate
        return prompt

    def prompt_index(self, prompt: str) -> int:
        return int(self.prompt_indices.get(prompt, 0))


def config_from_resume(checkpoint_path: Path, args: argparse.Namespace) -> FluxPackedStudentConfig:
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("config"):
            config = FluxPackedStudentConfig(**checkpoint["config"])
            if int(getattr(args, "adapter_rank", 0)) > 0:
                config.adapter_rank = int(getattr(args, "adapter_rank", 0))
                config.adapter_scale = float(getattr(args, "adapter_scale", 1.0))
                config.adapter_dropout = float(getattr(args, "adapter_dropout", 0.0))
            if bool(getattr(args, "override_resume_config", False)):
                config.dim = int(getattr(args, "dim", config.dim))
                config.depth = int(getattr(args, "depth", config.depth))
                config.heads = int(getattr(args, "heads", config.heads))
                config.mlp_ratio = int(getattr(args, "mlp_ratio", config.mlp_ratio))
                config.dropout = float(getattr(args, "dropout", config.dropout))
                config.pos2d_scale = float(getattr(args, "pos2d_scale", config.pos2d_scale))
                config.timestep_scale = float(getattr(args, "timestep_scale", config.timestep_scale))
                config.local_mixer_scale = float(getattr(args, "local_mixer_scale", config.local_mixer_scale))
                config.output_refiner_hidden = int(getattr(args, "output_refiner_hidden", config.output_refiner_hidden))
                config.output_refiner_depth = int(getattr(args, "output_refiner_depth", config.output_refiner_depth))
                config.output_refiner_scale = float(getattr(args, "output_refiner_scale", config.output_refiner_scale))
            return config
    return FluxPackedStudentConfig(
        latent_tokens=1024,
        latent_channels=64,
        prompt_dim=4096,
        pooled_dim=768,
        max_sequence_length=args.max_sequence_length,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        pos2d_scale=args.pos2d_scale,
        timestep_scale=args.timestep_scale,
        local_mixer_scale=args.local_mixer_scale,
        adapter_rank=int(getattr(args, "adapter_rank", 0)),
        adapter_scale=float(getattr(args, "adapter_scale", 1.0)),
        adapter_dropout=float(getattr(args, "adapter_dropout", 0.0)),
        output_refiner_hidden=int(getattr(args, "output_refiner_hidden", 0)),
        output_refiner_depth=int(getattr(args, "output_refiner_depth", 2)),
        output_refiner_scale=float(getattr(args, "output_refiner_scale", 0.25)),
    )


def freeze_non_adapter_parameters(student: nn.Module) -> int:
    trainable = 0
    for name, parameter in student.named_parameters():
        is_adapter = "_adapter." in name or name.endswith("_adapter")
        parameter.requires_grad_(is_adapter)
        if is_adapter:
            trainable += parameter.numel()
    return trainable


def load_student(args: argparse.Namespace, device: torch.device) -> tuple[FluxPackedStudent, FluxPackedStudentConfig, int, dict[str, Any] | None]:
    resume = Path(args.resume) if args.resume else Path()
    config = config_from_resume(resume, args) if args.resume else config_from_resume(Path(), args)
    student = FluxPackedStudent(config).to(device)
    start_step = 0
    source_checkpoint: dict[str, Any] | None = None
    if args.resume:
        source_checkpoint = torch.load(args.resume, map_location="cpu")
        state = source_checkpoint.get("student") or source_checkpoint.get("student_materialized") or source_checkpoint
        if bool(getattr(args, "override_resume_config", False)):
            current = student.state_dict()
            filtered_state = {
                key: value
                for key, value in state.items()
                if key in current and current[key].shape == value.shape
            }
            skipped = sorted(set(state) - set(filtered_state))
            state = filtered_state
            print(
                json.dumps(
                    {
                        "partial_resume_state": {
                            "loaded_tensors": len(filtered_state),
                            "skipped_tensors": len(skipped),
                            "sample_skipped": skipped[:24],
                        }
                    }
                ),
                flush=True,
            )
        missing, unexpected = student.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(json.dumps({"resume_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
        start_step = int(source_checkpoint.get("step") or 0)
    if args.bitnet_qat:
        modules = apply_bitnet_qat_modules(
            student,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
        )
        print(json.dumps({"bitnet_qat_enabled": {"modules": modules, "learned_scale": bool(args.bitnet_qat_learned_scale)}}), flush=True)
    if bool(getattr(args, "train_adapters_only", False)):
        trainable = freeze_non_adapter_parameters(student)
        if trainable <= 0:
            raise ValueError("--train-adapters-only requires --adapter-rank > 0")
        print(json.dumps({"train_adapters_only": {"trainable_parameters": trainable}}), flush=True)
    student.gradient_checkpointing = bool(args.gradient_checkpointing)
    return student, config, start_step, source_checkpoint


@torch.no_grad()
def teacher_predict(
    pipe: Any,
    latents: torch.Tensor,
    timestep_value: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance: torch.Tensor | None,
) -> torch.Tensor:
    with pipe.transformer.cache_context("cond"):
        return pipe.transformer(
            hidden_states=latents,
            timestep=timestep_value.expand(latents.shape[0]).to(latents.dtype) / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]


def load_mentor_student(checkpoint_path: Path, device: torch.device, weights: str) -> FluxPackedStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    mentor = FluxPackedStudent(config).to(device)
    if weights == "ema":
        state = checkpoint.get("student_ema")
        if not state:
            raise ValueError(f"{checkpoint_path} does not contain student_ema weights")
    elif weights == "materialized":
        state = checkpoint.get("student_materialized")
        if not state:
            raise ValueError(f"{checkpoint_path} does not contain student_materialized weights")
    else:
        state = checkpoint.get("student", checkpoint)
    missing, unexpected = mentor.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"mentor_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
    mentor.eval()
    for parameter in mentor.parameters():
        parameter.requires_grad_(False)
    return mentor


def mentor_predict(
    mentor: FluxPackedStudent,
    mentor_device: torch.device,
    latents: torch.Tensor,
    timestep_value: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor | None,
) -> torch.Tensor:
    mentor_guidance = (
        guidance.to(mentor_device, dtype=torch.float32)
        if guidance is not None
        else torch.full([latents.shape[0]], 3.5, device=mentor_device, dtype=torch.float32)
    )
    return mentor(
        latents.to(mentor_device, dtype=torch.float32),
        timestep_value.reshape(1).to(mentor_device, dtype=torch.float32),
        prompt_embeds.to(mentor_device, dtype=torch.float32),
        pooled_prompt_embeds.to(mentor_device, dtype=torch.float32),
        mentor_guidance,
    )


def add_latent_noise(latents: torch.Tensor, timestep: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.latent_noise_std <= 0:
        return latents
    scale = float(args.latent_noise_std)
    if args.latent_noise_timestep_scale:
        scale *= max(float(timestep.detach().float().item()) / 1000.0, 0.05)
    return latents + torch.randn_like(latents) * scale


def decode_flux_latents_tensor(pipe: Any, packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
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


def save_manifest(output_dir: Path, config: FluxPackedStudentConfig, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_kind": "agentkernel_lite_flux_live_teacher_flow_training",
        "teacher_model": args.teacher_model,
        "teacher_family": teacher_family(args.teacher_model),
        "prompt_file": args.prompt_file,
        "prompt_mix": parse_prompt_mix(args),
        "config": asdict(config),
        "mode": "online_teacher_targets_no_target_cache",
        "seed_policy": {
            "randomize_seeds": bool(args.randomize_seeds),
            "prompt_hash_seeds": bool(args.prompt_hash_seeds),
            "prompt_hash_seed_replay_prob": float(args.prompt_hash_seed_replay_prob),
            "seed": int(args.seed),
            "seed_min": int(args.seed_min),
            "seed_max": int(args.seed_max),
        },
        "teacher_steps": int(args.teacher_steps),
        "rollout_len": int(args.rollout_len),
        "trajectory_teacher_targets": bool(args.trajectory_teacher_targets),
        "front_start_prob": float(args.front_start_prob),
        "decoded_image_loss": {
            "weight": float(args.decoded_image_loss_weight),
            "every": int(args.decoded_image_loss_every),
            "size": int(args.decoded_image_loss_size),
        },
    }
    (output_dir / "live_teacher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_training_checkpoint(
    output_dir: Path,
    config: FluxPackedStudentConfig,
    student: nn.Module,
    step: int,
    loss: float,
    args: argparse.Namespace,
    ema_state: dict[str, torch.Tensor] | None,
) -> None:
    save_checkpoint(output_dir, config, student, step, loss, args, ema_state)
    if bool(args.keep_step_checkpoints):
        src = output_dir / "flux_packed_student.pt"
        dst = output_dir / f"flux_packed_student_step{step:06d}.pt"
        shutil.copy2(src, dst)


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
    clip_vision_model: nn.Module | None = None
    if args.decoded_clip_loss_weight > 0:
        from transformers import CLIPVisionModelWithProjection

        clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
            args.decoded_clip_model,
            torch_dtype=pipe.dtype,
        ).to(teacher_device)
        clip_vision_model.eval()
        for parameter in clip_vision_model.parameters():
            parameter.requires_grad_(False)
    student_device = torch.device(args.student_device or str(teacher_device))
    student, config, start_step, source_checkpoint = load_student(args, student_device)
    mentor: FluxPackedStudent | None = None
    mentor_device = torch.device(args.mentor_device or str(student_device))
    if args.mentor_checkpoint:
        mentor = load_mentor_student(Path(args.mentor_checkpoint), mentor_device, args.mentor_weights)
        print(
            json.dumps(
                {
                    "mentor_student": {
                        "checkpoint": args.mentor_checkpoint,
                        "weights": args.mentor_weights,
                        "device": str(mentor_device),
                    }
                }
            ),
            flush=True,
        )
    if start_step > 0:
        seed_everything(int(args.seed) + int(start_step))
    save_manifest(output_dir, config, args)
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
    prompts = PromptMixer(parse_prompt_mix(args), args.prompt_limit)
    prompt_seed_list = parse_seed_list(args.prompt_seed_list)
    hard_examples: list[dict[str, Any]] = []
    hard_example_keys: set[tuple[str, int, int]] = set()

    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        if hard_examples and random.random() < args.hard_replay_prob:
            hard = random.choice(hard_examples)
            prompt = str(hard["prompt"])
            source_name = str(hard.get("source_name", "hard_replay"))
            seed = int(hard["seed"])
            forced_start_index = int(hard["start_timestep_index"])
            replay_kind = "hard_replay"
        else:
            prompt, source_name = prompts.next()
            if args.prompt_hash_seeds or (
                args.prompt_hash_seed_replay_prob > 0
                and random.random() < float(args.prompt_hash_seed_replay_prob)
            ):
                seed = stable_prompt_seed(prompt, int(args.seed_min), int(args.seed_max))
            elif prompt_seed_list:
                seed = int(prompt_seed_list[prompts.prompt_index(prompt) % len(prompt_seed_list)])
            elif args.prompt_index_seeds:
                seed = int(args.seed) + prompts.prompt_index(prompt)
            elif args.randomize_seeds:
                seed = random.randint(int(args.seed_min), int(args.seed_max))
            else:
                seed = args.seed + step
            forced_start_index = -1
            replay_kind = "fresh"
        generator = torch.Generator(device=teacher_device).manual_seed(seed)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=teacher_device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            if args.prompt_contrast_weight > 0 and random.random() < args.prompt_contrast_prob:
                negative_prompt = prompts.sample_negative(prompt)
                negative_prompt_embeds, negative_pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=None,
                    device=teacher_device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
            num_channels_latents = pipe.transformer.config.in_channels // 4
            latents, latent_image_ids = pipe.prepare_latents(
                1,
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
            max_start = max(0, len(timesteps) - int(args.rollout_len))
            start_index = forced_start_index if forced_start_index >= 0 else 0 if random.random() < args.front_start_prob else random.randint(0, max_start)
            teacher_path_latents: list[torch.Tensor] = []
            teacher_path_targets: list[torch.Tensor] = []
            teacher_path_next: list[torch.Tensor] = []
            if args.trajectory_teacher_targets:
                teacher_current = latents
                for teacher_index, timestep_value in enumerate(timesteps):
                    if mentor is not None:
                        teacher_target = mentor_predict(
                            mentor,
                            mentor_device,
                            teacher_current,
                            timestep_value,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            guidance,
                        ).to(teacher_device, dtype=prompt_embeds.dtype)
                    else:
                        teacher_target = teacher_predict(
                            pipe,
                            teacher_current,
                            timestep_value,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                            latent_image_ids,
                            guidance,
                        )
                    teacher_next = pipe.scheduler.step(teacher_target, timestep_value, teacher_current, return_dict=False)[0]
                    if teacher_index >= start_index and teacher_index < start_index + int(args.rollout_len):
                        teacher_path_latents.append(teacher_current.detach())
                        teacher_path_targets.append(teacher_target.detach())
                        teacher_path_next.append(teacher_next.detach())
                    teacher_current = teacher_next
                latents = teacher_path_latents[0]
            else:
                for warm_index in range(start_index):
                    if mentor is not None:
                        teacher_target = mentor_predict(
                            mentor,
                            mentor_device,
                            latents,
                            timesteps[warm_index],
                            prompt_embeds,
                            pooled_prompt_embeds,
                            guidance,
                        ).to(teacher_device, dtype=prompt_embeds.dtype)
                    else:
                        teacher_target = teacher_predict(
                            pipe,
                            latents,
                            timesteps[warm_index],
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                            latent_image_ids,
                            guidance,
                        )
                    latents = pipe.scheduler.step(teacher_target, timesteps[warm_index], latents, return_dict=False)[0]

        current_latents = latents.to(student_device, dtype=torch.float32)
        prompt_embeds_student = prompt_embeds.to(student_device, dtype=torch.float32)
        pooled_prompt_embeds_student = pooled_prompt_embeds.to(student_device, dtype=torch.float32)
        negative_prompt_embeds_student = (
            negative_prompt_embeds.to(student_device, dtype=torch.float32)
            if negative_prompt_embeds is not None
            else None
        )
        negative_pooled_prompt_embeds_student = (
            negative_pooled_prompt_embeds.to(student_device, dtype=torch.float32)
            if negative_pooled_prompt_embeds is not None
            else None
        )
        guidance_student = (
            guidance.to(student_device, dtype=torch.float32)
            if guidance is not None
            else torch.full([1], float(args.guidance), device=student_device, dtype=torch.float32)
        )
        optimizer.zero_grad(set_to_none=True)
        flow_loss = torch.zeros((), device=student_device)
        latent_loss = torch.zeros((), device=student_device)
        dir_value = torch.zeros((), device=student_device)
        norm_value = torch.zeros((), device=student_device)
        spatial_value = torch.zeros((), device=student_device)
        contrast_value = torch.zeros((), device=student_device)
        endpoint_value = torch.zeros((), device=student_device)
        decoded_image_value = torch.zeros((), device=student_device)
        decoded_clip_value = torch.zeros((), device=student_device)
        teacher_endpoint_student: torch.Tensor | None = None
        rollout_len = min(int(args.rollout_len), len(timesteps) - start_index)
        for offset in range(rollout_len):
            timestep_value = timesteps[start_index + offset]
            with torch.no_grad():
                if args.trajectory_teacher_targets:
                    teacher_latents = teacher_path_latents[offset].to(teacher_device, dtype=prompt_embeds.dtype)
                    teacher_target = teacher_path_targets[offset].to(teacher_device, dtype=prompt_embeds.dtype).float()
                    teacher_next = teacher_path_next[offset].to(teacher_device, dtype=prompt_embeds.dtype).float()
                else:
                    teacher_latents = current_latents.to(teacher_device, dtype=prompt_embeds.dtype)
                    if mentor is not None:
                        teacher_target = mentor_predict(
                            mentor,
                            mentor_device,
                            teacher_latents,
                            timestep_value,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            guidance,
                        ).to(teacher_device).float()
                    else:
                        teacher_target = teacher_predict(
                            pipe,
                            teacher_latents,
                            timestep_value,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                            latent_image_ids,
                            guidance,
                        ).float()
                    teacher_next = pipe.scheduler.step(
                        teacher_target.to(prompt_embeds.dtype),
                        timestep_value,
                        teacher_latents,
                        return_dict=False,
                    )[0].float()
            teacher_latents_student = teacher_latents.to(student_device, dtype=torch.float32)
            teacher_target_student = teacher_target.to(student_device, dtype=torch.float32)
            teacher_next_student = teacher_next.to(student_device, dtype=torch.float32)
            teacher_endpoint_student = teacher_next_student
            timestep = timestep_value.reshape(1).to(student_device)
            noisy_latents = add_latent_noise(current_latents, timestep, args)
            pred = student(
                noisy_latents,
                timestep.float(),
                prompt_embeds_student,
                pooled_prompt_embeds_student,
                guidance_student,
            )
            if negative_prompt_embeds_student is not None and offset % max(int(args.prompt_contrast_every), 1) == 0:
                wrong_pred = student(
                    noisy_latents,
                    timestep.float(),
                    negative_prompt_embeds_student,
                    negative_pooled_prompt_embeds_student,
                    guidance_student,
                )
                positive_cos = F.cosine_similarity(pred.flatten(1), teacher_target_student.flatten(1), dim=1).mean()
                negative_cos = F.cosine_similarity(wrong_pred.flatten(1), teacher_target_student.flatten(1), dim=1).mean()
                contrast_value = contrast_value + F.relu(negative_cos - positive_cos + args.prompt_contrast_margin)
            weight = timestep_loss_weight(timestep, args)
            flow_loss = flow_loss + weight * reconstruction_loss(pred, teacher_target_student, args)
            if args.direction_loss_weight > 0:
                dir_value = dir_value + direction_loss(pred, teacher_target_student)
            if args.norm_loss_weight > 0:
                norm_value = norm_value + norm_loss(pred, teacher_target_student)
            if args.spatial_loss_weight > 0:
                spatial_value = spatial_value + packed_spatial_gradient_loss(pred, teacher_target_student)
            delta_base_latents = teacher_latents_student if args.trajectory_teacher_targets else current_latents
            delta = teacher_step_delta(delta_base_latents, teacher_next_student, teacher_target_student)
            student_next = current_latents + delta * pred.float()
            latent_loss = latent_loss + F.mse_loss(student_next, teacher_next_student)
            current_latents = student_next.detach() if args.detach_rollout else student_next

        denominator = max(rollout_len, 1)
        if teacher_endpoint_student is not None and args.endpoint_loss_weight > 0:
            endpoint_value = F.mse_loss(current_latents, teacher_endpoint_student.detach())
        if (
            teacher_endpoint_student is not None
            and (args.decoded_image_loss_weight > 0 or args.decoded_clip_loss_weight > 0)
            and args.decoded_image_loss_every > 0
            and step % int(args.decoded_image_loss_every) == 0
        ):
            student_endpoint_for_decode = current_latents.to(teacher_device, dtype=prompt_embeds.dtype)
            teacher_endpoint_for_decode = teacher_endpoint_student.detach().to(teacher_device, dtype=prompt_embeds.dtype)
            with torch.no_grad():
                teacher_image = decode_flux_latents_tensor(
                    pipe,
                    teacher_endpoint_for_decode,
                    args.height,
                    args.width,
                ).float()
            student_image = decode_flux_latents_tensor(
                pipe,
                student_endpoint_for_decode,
                args.height,
                args.width,
            ).float()
            if args.decoded_image_loss_size > 0:
                size = int(args.decoded_image_loss_size)
                teacher_image = F.interpolate(teacher_image, size=(size, size), mode="bilinear", align_corners=False)
                student_image = F.interpolate(student_image, size=(size, size), mode="bilinear", align_corners=False)
            if args.decoded_image_loss_weight > 0:
                decoded_image_value = F.mse_loss(student_image, teacher_image).to(student_device)
            if args.decoded_clip_loss_weight > 0 and clip_vision_model is not None:
                clip_size = int(args.decoded_clip_loss_size)
                student_clip = clip_pixels(student_image, clip_size).to(teacher_device, dtype=pipe.dtype)
                teacher_clip = clip_pixels(teacher_image, clip_size).to(teacher_device, dtype=pipe.dtype)
                with torch.no_grad():
                    teacher_features = as_feature_tensor(clip_vision_model(teacher_clip))
                    teacher_features = F.normalize(teacher_features.float(), dim=-1)
                student_features = as_feature_tensor(clip_vision_model(student_clip))
                student_features = F.normalize(student_features.float(), dim=-1)
                decoded_clip_value = F.mse_loss(student_features, teacher_features).to(student_device)
        loss = (
            args.flow_loss_weight * flow_loss / denominator
            + args.latent_loss_weight * latent_loss / denominator
            + args.direction_loss_weight * dir_value / denominator
            + args.norm_loss_weight * norm_value / denominator
            + args.spatial_loss_weight * spatial_value / denominator
            + args.prompt_contrast_weight * contrast_value / denominator
            + args.endpoint_loss_weight * endpoint_value
            + args.decoded_image_loss_weight * decoded_image_value
            + args.decoded_clip_loss_weight * decoded_clip_value
        )
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        if ema_state is not None and step % max(int(args.ema_update_every), 1) == 0:
            update_ema_state(ema_state, student, float(args.ema_decay))
        hard_key = (prompt, int(seed), int(start_index))
        if (
            args.hard_replay_prob > 0
            and args.hard_replay_max > 0
            and float(loss.detach().item()) >= args.hard_replay_loss_threshold
            and hard_key not in hard_example_keys
            and len(hard_examples) < args.hard_replay_max
        ):
            hard_example_keys.add(hard_key)
            hard_examples.append(
                {
                    "prompt": prompt,
                    "source_name": source_name,
                    "seed": seed,
                    "start_timestep_index": start_index,
                    "loss": float(loss.detach().item()),
                }
            )

        ledger = {
            "step": step,
            "prompt": prompt,
            "source_name": source_name,
            "seed": seed,
            "start_timestep_index": start_index,
            "rollout_len": rollout_len,
            "replay_kind": replay_kind,
            "hard_replay_size": len(hard_examples),
            "loss": float(loss.detach().item()),
            "flow_loss": float((flow_loss / denominator).detach().item()),
            "latent_loss": float((latent_loss / denominator).detach().item()),
            "direction_loss": float((dir_value / denominator).detach().item()),
            "norm_loss": float((norm_value / denominator).detach().item()),
            "spatial_loss": float((spatial_value / denominator).detach().item()),
            "prompt_contrast_loss": float((contrast_value / denominator).detach().item()),
            "endpoint_loss": float(endpoint_value.detach().item()),
            "decoded_image_loss": float(decoded_image_value.detach().item()),
            "decoded_clip_loss": float(decoded_clip_value.detach().item()),
            "mode": "live_teacher_flux_flow_trajectory" if args.trajectory_teacher_targets else "live_teacher_flux_flow",
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
        if step % args.log_every == 0 or local_step == 1:
            print(json.dumps(ledger), flush=True)
        if step % args.checkpoint_every == 0 or local_step == args.steps:
            save_training_checkpoint(output_dir, config, student, step, float(loss.detach().item()), args, ema_state)

    save_training_checkpoint(output_dir, config, student, start_step + args.steps, float(loss.detach().item()), args, ema_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live FLUX teacher flow distillation for the Agent Kernel Lite packed-latent student.")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_flux_live_teacher_150m_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-ema", action="store_true")
    parser.add_argument("--prompt-file", default="data/vision/prompts/imagenet_object_photo_12k_v0.jsonl")
    parser.add_argument("--prompt-mix", action="append", default=[], help="Prompt source as path[:weight[:name]]. Use multiple times.")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--mentor-checkpoint", default="")
    parser.add_argument("--mentor-weights", choices=("raw", "ema", "materialized"), default="raw")
    parser.add_argument("--mentor-device", default="")
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
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--rollout-len", type=int, default=8)
    parser.add_argument("--trajectory-teacher-targets", action="store_true")
    parser.add_argument("--front-start-prob", type=float, default=0.45)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pos2d-scale", type=float, default=0.05)
    parser.add_argument("--timestep-scale", type=float, default=1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=0.10)
    parser.add_argument("--override-resume-config", action="store_true")
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--output-refiner-hidden", type=int, default=0)
    parser.add_argument("--output-refiner-depth", type=int, default=2)
    parser.add_argument("--output-refiner-scale", type=float, default=0.25)
    parser.add_argument("--train-adapters-only", action="store_true")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--detach-rollout", action="store_true")
    parser.add_argument("--latent-noise-std", type=float, default=0.03)
    parser.add_argument("--latent-noise-timestep-scale", action="store_true")
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--snr-weighting", action="store_true")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--min-snr-weight", type=float, default=0.05)
    parser.add_argument("--max-snr-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=500.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.1)
    parser.add_argument("--norm-loss-weight", type=float, default=0.02)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.05)
    parser.add_argument("--prompt-contrast-weight", type=float, default=0.0)
    parser.add_argument("--prompt-contrast-prob", type=float, default=0.25)
    parser.add_argument("--prompt-contrast-every", type=int, default=4)
    parser.add_argument("--prompt-contrast-margin", type=float, default=0.08)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-image-loss-every", type=int, default=0)
    parser.add_argument("--decoded-image-loss-size", type=int, default=128)
    parser.add_argument("--decoded-clip-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--decoded-clip-loss-size", type=int, default=224)
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-update-every", type=int, default=10)
    parser.add_argument("--hard-replay-prob", type=float, default=0.15)
    parser.add_argument("--hard-replay-loss-threshold", type=float, default=0.55)
    parser.add_argument("--hard-replay-max", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--prompt-seed-list", default="")
    parser.add_argument("--randomize-seeds", action="store_true")
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--prompt-index-seeds", action="store_true")
    parser.add_argument("--prompt-hash-seed-replay-prob", type=float, default=0.0)
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
