#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
from typing import Any

import torch
import torch.nn.functional as F

from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, load_teacher
from train_agentkernel_lite_image_flux_diffusiondb_zip_stream import decode_flux_latents_tensor, encode_images
from train_agentkernel_lite_image_flux_flow_distill import clone_state_dict, save_checkpoint, seed_everything, update_ema_state
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    direction_loss,
    norm_loss,
    packed_spatial_gradient_loss,
    reconstruction_loss,
    timestep_loss_weight,
)
from train_agentkernel_lite_image_flux_live_teacher_flow import load_student, teacher_predict


def clean_caption(row: dict[str, Any], caption_column: str, max_chars: int) -> str:
    value = row.get(caption_column)
    if isinstance(value, list):
        value = value[0] if value else ""
    text = " ".join(str(value or "").replace("\x00", " ").split())
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def make_sigma_schedule(pipe, clean_latents: torch.Tensor, steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps, _ = flux_timesteps(pipe, clean_latents, steps, device)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=clean_latents.dtype)
    if sigmas.shape[0] <= timesteps.shape[0]:
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return timesteps, sigmas


def choose_rollout_start(num_timesteps: int, rollout_len: int, front_prob: float, terminal_prob: float) -> int:
    max_start = max(num_timesteps - rollout_len, 0)
    if max_start == 0:
        return 0
    draw = random.random()
    if front_prob > 0 and draw < front_prob:
        return 0
    if terminal_prob > 0 and draw < front_prob + terminal_prob:
        return max_start
    return random.randrange(0, max_start + 1)


def save_training_checkpoint(
    output_dir: Path,
    config,
    student,
    step: int,
    loss: float,
    args: argparse.Namespace,
    ema_state: dict[str, torch.Tensor] | None,
    optimizer_state,
) -> None:
    save_checkpoint(output_dir, config, student, step, loss, args, ema_state, optimizer_state)
    if bool(args.keep_step_checkpoints):
        shutil.copy2(output_dir / "flux_packed_student.pt", output_dir / f"flux_packed_student_step{step:06d}.pt")


def iter_batches(args: argparse.Namespace):
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"split": args.split, "streaming": True}
    if args.use_hf_token:
        load_kwargs["token"] = True
    dataset = load_dataset(args.dataset, args.dataset_config or None, **load_kwargs)
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    batch_images = []
    batch_prompts = []
    yielded = 0
    for row in dataset:
        prompt = clean_caption(row, args.caption_column, args.max_caption_chars)
        image = row.get(args.image_column)
        if not prompt or image is None:
            continue
        words = prompt.split()
        if args.min_prompt_words > 0 and len(words) < args.min_prompt_words:
            continue
        if args.max_prompt_words > 0 and len(words) > args.max_prompt_words:
            continue
        try:
            image = image.convert("RGB")
        except Exception:
            continue
        batch_images.append(image)
        batch_prompts.append(prompt)
        if len(batch_images) >= args.batch_size:
            yield batch_images, batch_prompts
            yielded += len(batch_images)
            batch_images = []
            batch_prompts = []
            if args.max_stream_items > 0 and yielded >= args.max_stream_items:
                return
    if batch_images:
        yield batch_images, batch_prompts


def append_replay_items(
    replay_buffer: list[tuple[Any, str]],
    images: list[Any],
    prompts: list[str],
    max_size: int,
) -> None:
    if max_size <= 0:
        return
    for image, prompt in zip(images, prompts):
        replay_buffer.append((image.copy(), str(prompt)))
    if len(replay_buffer) > max_size:
        del replay_buffer[: len(replay_buffer) - max_size]


def sample_replay_batch(
    replay_buffer: list[tuple[Any, str]],
    batch_size: int,
) -> tuple[list[Any], list[str]]:
    picks = random.choices(replay_buffer, k=batch_size)
    images = [image.copy() for image, _prompt in picks]
    prompts = [prompt for _image, prompt in picks]
    return images, prompts


def packed_low_frequency_loss(a: torch.Tensor, b: torch.Tensor, pool: int) -> torch.Tensor:
    if pool <= 1 or a.ndim != 3 or b.ndim != 3 or a.shape != b.shape:
        return F.mse_loss(a.float(), b.float())
    batch, tokens, channels = a.shape
    side = int(tokens**0.5)
    if side * side != tokens:
        return F.mse_loss(a.float(), b.float())
    kernel = min(max(int(pool), 1), side)
    a_grid = a.float().reshape(batch, side, side, channels).permute(0, 3, 1, 2)
    b_grid = b.float().reshape(batch, side, side, channels).permute(0, 3, 1, 2)
    return F.mse_loss(
        F.avg_pool2d(a_grid, kernel_size=kernel, stride=kernel),
        F.avg_pool2d(b_grid, kernel_size=kernel, stride=kernel),
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_args = argparse.Namespace(
        teacher_family="flux",
        teacher_model=args.teacher_model,
        dtype=args.dtype,
        variant=args.variant,
        local_files_only=args.local_files_only,
        quantize_transformer_4bit=args.quantize_transformer_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        cpu_offload=args.cpu_offload,
        gpu_id=args.encoder_gpu_id,
        device=args.encoder_device,
    )
    pipe = load_teacher(teacher_args)
    encoder_device = torch.device(getattr(pipe, "_execution_device", args.encoder_device))
    if hasattr(pipe, "transformer") and args.teacher_score_loss_weight <= 0:
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
    elif hasattr(pipe, "transformer"):
        pipe.transformer.eval()
        for parameter in pipe.transformer.parameters():
            parameter.requires_grad_(False)
    for module_name in ("vae", "text_encoder", "text_encoder_2"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    student_device = torch.device(args.student_device)
    student, config, start_step, source_checkpoint = load_student(args, student_device)
    config.max_sequence_length = int(args.max_sequence_length)
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
    if source_checkpoint and source_checkpoint.get("optimizer") and not args.ignore_optimizer_state:
        try:
            optimizer.load_state_dict(source_checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(student_device)
        except Exception as exc:
            print(json.dumps({"optimizer_state": "ignored", "reason": str(exc)}), flush=True)

    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "hf_image_stream_manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_lite_flux_hf_image_stream_training",
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "image_column": args.image_column,
                "caption_column": args.caption_column,
                "width": args.width,
                "height": args.height,
                "target_kind": "real_image_rectified_flow_stream",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ledger_path = output_dir / "hf_image_stream_ledger.jsonl"

    step = start_step
    last_loss = 0.0
    replay_buffer: list[tuple[Any, str]] = []
    while step < start_step + args.steps:
        made_progress = False
        for images, prompts in iter_batches(args):
            if step >= start_step + args.steps:
                break
            made_progress = True
            append_replay_items(replay_buffer, images, prompts, int(args.replay_buffer_size))
            replay_kind = "stream"
            if (
                replay_buffer
                and int(args.replay_buffer_size) > 0
                and step >= start_step + int(args.replay_warmup_steps)
                and random.random() < float(args.replay_prob)
            ):
                images, prompts = sample_replay_batch(replay_buffer, len(images))
                replay_kind = "replay"
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=None,
                    device=encoder_device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
                clean_latents, target_pixels = encode_images(
                    pipe,
                    images,
                    args.height,
                    args.width,
                    encoder_device,
                    prompt_embeds.dtype,
                )
                latent_image_ids = pipe._prepare_latent_image_ids(
                    clean_latents.shape[0],
                    args.height // (pipe.vae_scale_factor * 2),
                    args.width // (pipe.vae_scale_factor * 2),
                    encoder_device,
                    prompt_embeds.dtype,
                )
                timesteps, sigmas = make_sigma_schedule(pipe, clean_latents, args.train_timesteps, encoder_device)
                rollout_len = min(max(int(args.rollout_len), 1), int(timesteps.shape[0]))
                timestep_index = choose_rollout_start(
                    int(timesteps.shape[0]),
                    rollout_len,
                    float(args.front_rollout_prob),
                    float(args.terminal_rollout_prob),
                )
                generator = torch.Generator(device=encoder_device).manual_seed(int(args.seed + step))
                noise = torch.randn(clean_latents.shape, generator=generator, device=encoder_device, dtype=clean_latents.dtype)
                sigma = sigmas[timestep_index].reshape(1, 1, 1)
                noisy_latents = sigma * noise + (1.0 - sigma) * clean_latents
                target = noise.float() - clean_latents.float()

            teacher_prompt_embeds = prompt_embeds
            teacher_pooled_prompt_embeds = pooled_prompt_embeds
            teacher_text_ids = text_ids
            teacher_latent_image_ids = latent_image_ids
            prompt_embeds = prompt_embeds.to(student_device, dtype=torch.float32)
            pooled_prompt_embeds = pooled_prompt_embeds.to(student_device, dtype=torch.float32)
            noisy_latents = noisy_latents.to(student_device, dtype=torch.float32)
            target = target.to(student_device, dtype=torch.float32)
            clean_latents = clean_latents.to(student_device, dtype=torch.float32)
            target_pixels = target_pixels.to(encoder_device, dtype=torch.float32)
            noise = noise.to(student_device, dtype=torch.float32)
            timesteps = timesteps.to(student_device, dtype=torch.float32)
            sigmas = sigmas.to(student_device, dtype=torch.float32)
            guidance = torch.full([noisy_latents.shape[0]], float(args.guidance), device=student_device, dtype=torch.float32)

            if args.prompt_dropout > 0 and random.random() < args.prompt_dropout:
                prompt_embeds = torch.zeros_like(prompt_embeds)
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

            optimizer.zero_grad(set_to_none=True)
            current_latents = noisy_latents
            flow = torch.zeros((), device=student_device)
            dir_value = torch.zeros((), device=student_device)
            norm_value = torch.zeros((), device=student_device)
            spatial_value = torch.zeros((), device=student_device)
            latent_value = torch.zeros((), device=student_device)
            latent_low_frequency_value = torch.zeros((), device=student_device)
            x0_clean_value = torch.zeros((), device=student_device)
            x0_low_frequency_value = torch.zeros((), device=student_device)
            terminal_value = torch.zeros((), device=student_device)
            terminal_low_frequency_value = torch.zeros((), device=student_device)
            decoded_value = torch.zeros((), device=student_device)
            decoded_low_frequency_value = torch.zeros((), device=student_device)
            teacher_score_value = torch.zeros((), device=student_device)

            for rollout_offset in range(rollout_len):
                current_index = timestep_index + rollout_offset
                timestep = timesteps[current_index].reshape(1).repeat(current_latents.shape[0])
                pred = student(current_latents, timestep, prompt_embeds, pooled_prompt_embeds, guidance)
                if args.teacher_score_loss_weight > 0:
                    with torch.no_grad():
                        teacher_guidance = None
                        if pipe.transformer.config.guidance_embeds:
                            teacher_guidance = torch.full(
                                [current_latents.shape[0]],
                                float(args.guidance),
                                device=encoder_device,
                                dtype=torch.float32,
                            )
                        teacher_pred = teacher_predict(
                            pipe,
                            current_latents.detach().to(encoder_device, dtype=teacher_prompt_embeds.dtype),
                            timesteps[current_index].to(encoder_device),
                            teacher_prompt_embeds,
                            teacher_pooled_prompt_embeds,
                            teacher_text_ids,
                            teacher_latent_image_ids,
                            teacher_guidance,
                        ).float()
                    teacher_score_value = teacher_score_value + F.mse_loss(
                        pred.float(),
                        teacher_pred.to(student_device, dtype=torch.float32),
                    )
                weight = timestep_loss_weight(timestep.mean(), args)
                flow = flow + weight * reconstruction_loss(pred, target, args)
                if args.direction_loss_weight > 0:
                    dir_value = dir_value + direction_loss(pred, target)
                if args.norm_loss_weight > 0:
                    norm_value = norm_value + norm_loss(pred, target)
                if args.spatial_loss_weight > 0:
                    spatial_value = spatial_value + packed_spatial_gradient_loss(pred, target)

                if args.x0_clean_loss_weight > 0 or args.x0_low_frequency_loss_weight > 0:
                    sigma_current = sigmas[current_index].reshape(1, 1, 1)
                    x0_pred = current_latents.float() - sigma_current.float() * pred.float()
                    if args.x0_clean_loss_weight > 0:
                        x0_clean_value = x0_clean_value + F.mse_loss(x0_pred, clean_latents.float())
                    if args.x0_low_frequency_loss_weight > 0:
                        x0_low_frequency_value = x0_low_frequency_value + packed_low_frequency_loss(
                            x0_pred,
                            clean_latents,
                            int(args.latent_low_frequency_pool),
                        )

                next_sigma = sigmas[current_index + 1].reshape(1, 1, 1)
                next_target_latents = next_sigma * noise + (1.0 - next_sigma) * clean_latents
                delta = sigmas[current_index + 1] - sigmas[current_index]
                current_latents = current_latents + delta.reshape(1, 1, 1) * pred.float()
                latent_value = latent_value + F.mse_loss(current_latents.float(), next_target_latents.float())
                if args.latent_low_frequency_loss_weight > 0:
                    latent_low_frequency_value = latent_low_frequency_value + packed_low_frequency_loss(
                        current_latents,
                        next_target_latents,
                        int(args.latent_low_frequency_pool),
                    )
                if args.detach_rollout and rollout_offset + 1 < rollout_len:
                    current_latents = current_latents.detach()

            if int(timestep_index + rollout_len) >= int(timesteps.shape[0]):
                terminal_value = F.mse_loss(current_latents.float(), clean_latents.float())
                if args.terminal_low_frequency_loss_weight > 0:
                    terminal_low_frequency_value = packed_low_frequency_loss(
                        current_latents,
                        clean_latents,
                        int(args.latent_low_frequency_pool),
                    )
                if (
                    (args.decoded_image_loss_weight > 0 or args.decoded_low_frequency_loss_weight > 0)
                    and args.decoded_image_loss_every > 0
                    and step % int(args.decoded_image_loss_every) == 0
                ):
                    decoded = decode_flux_latents_tensor(
                        pipe,
                        current_latents.to(encoder_device, dtype=next(pipe.vae.parameters()).dtype),
                        args.height,
                        args.width,
                    ).float()
                    target_for_decode = target_pixels
                    if int(args.decoded_image_loss_size) > 0 and int(args.decoded_image_loss_size) != int(args.height):
                        size = (int(args.decoded_image_loss_size), int(args.decoded_image_loss_size))
                        decoded = F.interpolate(decoded, size=size, mode="bilinear", align_corners=False)
                        target_for_decode = F.interpolate(target_for_decode, size=size, mode="bilinear", align_corners=False)
                    if args.decoded_image_loss_weight > 0:
                        decoded_value = F.l1_loss(decoded, target_for_decode).to(student_device)
                    if args.decoded_low_frequency_loss_weight > 0:
                        low_size = int(args.decoded_low_frequency_size)
                        if low_size > 0:
                            decoded_low = F.interpolate(decoded, size=(low_size, low_size), mode="bilinear", align_corners=False)
                            target_low = F.interpolate(target_for_decode, size=(low_size, low_size), mode="bilinear", align_corners=False)
                        else:
                            decoded_low = decoded
                            target_low = target_for_decode
                        decoded_low_frequency_value = F.l1_loss(decoded_low, target_low).to(student_device)

            denominator = float(max(rollout_len, 1))
            flow = flow / denominator
            dir_value = dir_value / denominator
            norm_value = norm_value / denominator
            spatial_value = spatial_value / denominator
            latent_value = latent_value / denominator
            latent_low_frequency_value = latent_low_frequency_value / denominator
            x0_clean_value = x0_clean_value / denominator
            x0_low_frequency_value = x0_low_frequency_value / denominator
            teacher_score_value = teacher_score_value / denominator
            loss = (
                args.flow_loss_weight * flow
                + args.latent_loss_weight * latent_value
                + args.latent_low_frequency_loss_weight * latent_low_frequency_value
                + args.x0_clean_loss_weight * x0_clean_value
                + args.x0_low_frequency_loss_weight * x0_low_frequency_value
                + args.terminal_clean_loss_weight * terminal_value
                + args.terminal_low_frequency_loss_weight * terminal_low_frequency_value
                + args.decoded_image_loss_weight * decoded_value
                + args.decoded_low_frequency_loss_weight * decoded_low_frequency_value
                + args.teacher_score_loss_weight * teacher_score_value
                + args.direction_loss_weight * dir_value
                + args.norm_loss_weight * norm_value
                + args.spatial_loss_weight * spatial_value
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            if ema_state is not None and step % max(int(args.ema_update_every), 1) == 0:
                update_ema_state(ema_state, student, float(args.ema_decay))

            step += 1
            last_loss = float(loss.detach().item())
            if step % args.log_every == 0 or step == start_step + 1 or float(decoded_value.detach().item()) > 0:
                record = {
                    "step": step,
                    "loss": last_loss,
                    "flow_loss": float(flow.detach().item()),
                    "direction_loss": float(dir_value.detach().item()),
                    "norm_loss": float(norm_value.detach().item()),
                    "spatial_loss": float(spatial_value.detach().item()),
                    "latent_loss": float(latent_value.detach().item()),
                    "latent_low_frequency_loss": float(latent_low_frequency_value.detach().item()),
                    "x0_clean_loss": float(x0_clean_value.detach().item()),
                    "x0_low_frequency_loss": float(x0_low_frequency_value.detach().item()),
                    "teacher_score_loss": float(teacher_score_value.detach().item()),
                    "terminal_clean_loss": float(terminal_value.detach().item()),
                    "terminal_low_frequency_loss": float(terminal_low_frequency_value.detach().item()),
                    "decoded_image_loss": float(decoded_value.detach().item()),
                    "decoded_low_frequency_loss": float(decoded_low_frequency_value.detach().item()),
                    "timestep_index": int(timestep_index),
                    "rollout_len": int(rollout_len),
                    "batch_size": int(noisy_latents.shape[0]),
                    "prompt": prompts[:2],
                    "dataset": args.dataset,
                    "replay_kind": replay_kind,
                    "replay_buffer_size": len(replay_buffer),
                    "mode": "hf_real_image_rollout_stream",
                }
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % args.checkpoint_every == 0:
                save_training_checkpoint(
                    output_dir,
                    config,
                    student,
                    step,
                    last_loss,
                    args,
                    ema_state,
                    optimizer.state_dict() if args.save_optimizer else None,
                )
        if not made_progress:
            raise RuntimeError("stream produced no usable image-caption batches")

    save_training_checkpoint(
        output_dir,
        config,
        student,
        step,
        last_loss,
        args,
        ema_state,
        optimizer.state_dict() if args.save_optimizer else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FLUX student on streaming HF image-caption rows without storing the dataset.")
    parser.add_argument("--dataset", default="regisss/coco_2017")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_flux_hf_image_stream_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--encoder-device", default="cuda:0")
    parser.add_argument("--encoder-gpu-id", type=int, default=0)
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--train-timesteps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--max-stream-items", type=int, default=0)
    parser.add_argument("--min-prompt-words", type=int, default=0)
    parser.add_argument("--max-prompt-words", type=int, default=36)
    parser.add_argument("--max-caption-chars", type=int, default=220)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--ema-update-every", type=int, default=5)
    parser.add_argument("--reset-ema", action="store_true")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.5)
    parser.add_argument("--latent-low-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-low-frequency-pool", type=int, default=4)
    parser.add_argument("--x0-clean-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-low-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-clean-loss-weight", type=float, default=0.35)
    parser.add_argument("--terminal-low-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-image-loss-weight", type=float, default=10.0)
    parser.add_argument("--decoded-image-loss-every", type=int, default=2)
    parser.add_argument("--decoded-image-loss-size", type=int, default=128)
    parser.add_argument("--decoded-low-frequency-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-low-frequency-size", type=int, default=32)
    parser.add_argument("--teacher-score-loss-weight", type=float, default=0.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.05)
    parser.add_argument("--norm-loss-weight", type=float, default=0.01)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.01)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--snr-weighting", action="store_true")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--min-snr-weight", type=float, default=0.10)
    parser.add_argument("--max-snr-weight", type=float, default=1.0)
    parser.add_argument("--prompt-dropout", type=float, default=0.03)
    parser.add_argument("--replay-buffer-size", type=int, default=0)
    parser.add_argument("--replay-prob", type=float, default=0.0)
    parser.add_argument("--replay-warmup-steps", type=int, default=0)
    parser.add_argument("--rollout-len", type=int, default=4)
    parser.add_argument("--front-rollout-prob", type=float, default=0.65)
    parser.add_argument("--terminal-rollout-prob", type=float, default=0.20)
    parser.add_argument("--detach-rollout", action="store_true")
    parser.add_argument("--dim", type=int, default=720)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pos2d-scale", type=float, default=0.0)
    parser.add_argument("--timestep-scale", type=float, default=1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=0.0)
    parser.add_argument("--override-resume-config", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--ignore-optimizer-state", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--use-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
