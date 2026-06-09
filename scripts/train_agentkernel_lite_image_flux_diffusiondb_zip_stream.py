#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
import torch
import torch.nn.functional as F

from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, load_teacher
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    direction_loss,
    norm_loss,
    packed_spatial_gradient_loss,
    reconstruction_loss,
    timestep_loss_weight,
)
from train_agentkernel_lite_image_flux_live_teacher_flow import load_student
from train_agentkernel_lite_image_flux_flow_distill import clone_state_dict, save_checkpoint, seed_everything, update_ema_state

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

try:
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
except Exception:
    pass


def list_zip_shards(repo_id: str, prefix: str, token: str | bool | None) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    shards = [name for name in files if name.startswith(prefix) and name.endswith(".zip")]
    if not shards:
        raise ValueError(f"no zip shards found for {repo_id}/{prefix}")
    return sorted(shards)


STYLE_SOUP_TERMS = (
    "artstation",
    "deviantart",
    "octane render",
    "unreal engine",
    "cinematic lighting",
    "trending on",
    "award winning",
    "masterpiece",
    "greg rutkowski",
    "wlop",
    "alphonse mucha",
    "by ",
    "8 k",
    "8k",
    "4 k",
    "4k",
    "anime",
    "cartoon",
    "comic",
    "manga",
    "minecraft",
    "movie poster",
    "poster",
    "ukiyo",
    "vector art",
    "digital art",
    "concept art",
    "illustration",
    "painting",
    "painted",
    "render",
)

STRICT_NON_PHOTO_TERMS = STYLE_SOUP_TERMS + (
    "anthropomorphic",
    "censorship",
    "chibi",
    "creepy",
    "dnd",
    "emoji",
    "etching",
    "furry",
    "fursona",
    "hollow knight",
    "horror",
    "meme",
    "monochrome",
    "outline drawing",
    "pattern",
    "pixar",
    "portrait",
    "slenderman",
    "spooky",
    "star wars",
    "stylized",
    "texture",
    "texture map",
    "tv show",
    "woman",
    "women",
    "man ",
    " men",
    "boy",
    "girl",
    "person",
    "people",
    "celebrity",
    "jesus",
    "jungkook",
    "freddie mercury",
    "chris cornell",
)

STRICT_PHOTO_TERMS = (
    "photo",
    "photograph",
    "photography",
    "photorealistic",
    "realistic",
    "real photo",
    "product photo",
    "macro photo",
    "studio photo",
    "dslr",
    "polaroid",
)

STRICT_OBJECT_TERMS = (
    "animal",
    "airplane",
    "apple",
    "appliance",
    "banana",
    "beach",
    "bear",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "building",
    "camera",
    "car",
    "cat",
    "chair",
    "city",
    "coral",
    "cow",
    "dinosaur",
    "dog",
    "fish",
    "food",
    "flower",
    "forest",
    "frog",
    "fruit",
    "furniture",
    "guitar",
    "house",
    "instrument",
    "keyboard",
    "lamp",
    "lake",
    "landscape",
    "meadow",
    "mountain",
    "mushroom",
    "ocean",
    "octopus",
    "orange",
    "plane",
    "plant",
    "river",
    "rose",
    "shoe",
    "squirrel",
    "statue",
    "table",
    "tool",
    "toy",
    "train",
    "truck",
    "vehicle",
    "violin",
)


def prompt_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def has_word_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def prompt_is_allowed(prompt: str, args: argparse.Namespace) -> bool:
    words = prompt.split()
    if args.min_prompt_words > 0 and len(words) < int(args.min_prompt_words):
        return False
    if args.max_prompt_chars > 0 and len(prompt) > int(args.max_prompt_chars):
        return False
    if args.max_prompt_words > 0 and len(words) > int(args.max_prompt_words):
        return False
    mode = str(getattr(args, "prompt_filter_mode", "none"))
    if mode == "clean-photo":
        lower = prompt.lower()
        if "http://" in lower or "https://" in lower:
            return False
        if lower.count(",") > int(args.max_prompt_commas):
            return False
        if sum(not char.isalnum() and not char.isspace() for char in lower) > max(8, len(lower) // 12):
            return False
        if any(term in lower for term in STYLE_SOUP_TERMS):
            return False
        grounding_terms = ("photo", "photograph", "realistic", "portrait", "close up", "macro", "product photo")
        simple_caption = len(words) <= int(args.clean_prompt_simple_word_limit)
        if not simple_caption and not any(term in lower for term in grounding_terms):
            return False
    if mode == "object-photo":
        lower = prompt.lower()
        ascii_ratio = sum(ord(char) < 128 for char in prompt) / max(len(prompt), 1)
        if ascii_ratio < 0.96:
            return False
        if "http://" in lower or "https://" in lower:
            return False
        if lower.count(",") > int(args.max_prompt_commas):
            return False
        if any(term in lower for term in STRICT_NON_PHOTO_TERMS):
            return False
        if any(char in lower for char in ("@", "#", "|", "{", "}", "[", "]")):
            return False
        if sum(not char.isalnum() and not char.isspace() for char in lower) > max(4, len(lower) // 20):
            return False
        has_object_grounding = any(has_word_phrase(lower, term) for term in STRICT_OBJECT_TERMS)
        simple_caption = len(words) <= int(args.clean_prompt_simple_word_limit)
        if not has_object_grounding:
            return False
        has_photo_grounding = any(term in lower for term in STRICT_PHOTO_TERMS)
        if not (has_photo_grounding or simple_caption):
            return False
    return True


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
    if bool(getattr(args, "keep_step_checkpoints", False)):
        src = output_dir / "flux_packed_student.pt"
        dst = output_dir / f"flux_packed_student_step{step:06d}.pt"
        shutil.copy2(src, dst)


def load_zip_items(path: Path, args: argparse.Namespace | int = 0) -> list[tuple[str, str]]:
    if isinstance(args, int):
        args = argparse.Namespace(
            max_items_per_shard=args,
            min_prompt_words=0,
            max_prompt_chars=0,
            max_prompt_words=0,
            max_images_per_prompt=0,
            prompt_filter_mode="none",
            max_prompt_commas=99,
            clean_prompt_simple_word_limit=16,
        )
    with zipfile.ZipFile(path) as archive:
        metadata_name = next((name for name in archive.namelist() if name.endswith(".json")), "")
        if not metadata_name:
            raise ValueError(f"zip shard has no metadata json: {path}")
        metadata = json.loads(archive.read(metadata_name))
        items: list[tuple[str, str]] = []
        prompt_counts: dict[str, int] = {}
        for image_name, record in metadata.items():
            prompt = " ".join(str(record.get("p") or "").replace("\x00", " ").split())
            key = prompt_key(prompt)
            if (
                image_name in archive.NameToInfo
                and prompt
                and prompt_is_allowed(prompt, args)
                and (
                    int(args.max_images_per_prompt) <= 0
                    or prompt_counts.get(key, 0) < int(args.max_images_per_prompt)
                )
            ):
                items.append((image_name, prompt))
                prompt_counts[key] = prompt_counts.get(key, 0) + 1
            if args.max_items_per_shard and len(items) >= int(args.max_items_per_shard):
                break
        return items


def read_image_batch(path: Path, items: list[tuple[str, str]]) -> tuple[list[Image.Image], list[str]]:
    images: list[Image.Image] = []
    prompts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for image_name, prompt in items:
            try:
                image = Image.open(BytesIO(archive.read(image_name))).convert("RGB")
            except Exception:
                continue
            images.append(image)
            prompts.append(prompt)
    return images, prompts


def encode_images(
    pipe,
    images: list[Image.Image],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = pipe.image_processor.preprocess(images, height=height, width=width)
    pixels = pixels.to(device=device, dtype=dtype)
    latents = pipe.vae.encode(pixels).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    batch, channels, latent_height, latent_width = latents.shape
    return pipe._pack_latents(latents, batch, channels, latent_height, latent_width), pixels


def decode_flux_latents_tensor(pipe, packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.vae.decode(latents, return_dict=False)[0]


def make_sigma_schedule(pipe, clean_latents: torch.Tensor, steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps, _ = flux_timesteps(pipe, clean_latents, steps, device)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=clean_latents.dtype)
    if sigmas.shape[0] <= timesteps.shape[0]:
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return timesteps, sigmas


def choose_rollout_start(num_timesteps: int, rollout_len: int, terminal_prob: float) -> int:
    max_start = max(num_timesteps - rollout_len, 0)
    if max_start == 0:
        return 0
    if terminal_prob > 0 and random.random() < terminal_prob:
        return max_start
    return random.randrange(0, max_start + 1)


def choose_rollout_start_from_args(num_timesteps: int, rollout_len: int, args: argparse.Namespace) -> int:
    max_start = max(num_timesteps - rollout_len, 0)
    if max_start == 0:
        return 0
    if args.front_rollout_prob > 0 and random.random() < float(args.front_rollout_prob):
        return 0
    return choose_rollout_start(num_timesteps, rollout_len, float(args.terminal_rollout_prob))


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    token: str | bool | None = True if args.use_hf_token else None
    shards = list_zip_shards(args.dataset_repo, args.shard_prefix, token)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    random.shuffle(shards)

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
    if hasattr(pipe, "transformer"):
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
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
    (output_dir / "real_image_stream_manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_lite_flux_diffusiondb_zip_stream_training",
                "dataset_repo": args.dataset_repo,
                "shard_prefix": args.shard_prefix,
                "width": args.width,
                "height": args.height,
                "teacher_model_for_vae_text": args.teacher_model,
                "target_kind": "real_image_rectified_flow_stream",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ledger_path = output_dir / "diffusiondb_real_image_stream_ledger.jsonl"

    step = start_step
    estimated_items_per_shard = int(args.max_items_per_shard) if int(args.max_items_per_shard) > 0 else 1000
    estimated_steps_per_shard = max(1, estimated_items_per_shard // max(int(args.batch_size), 1))
    shard_index = (int(start_step) // estimated_steps_per_shard) % max(len(shards), 1)
    resume_shuffle_salt = int(start_step)
    while step < start_step + args.steps:
        shard_name = shards[shard_index % len(shards)]
        shard_index += 1
        shard_path = Path(
            hf_hub_download(
                args.dataset_repo,
                repo_type="dataset",
                filename=shard_name,
                local_dir=str(cache_dir),
                token=token,
            )
        )
        items = load_zip_items(shard_path, args)
        if not args.disable_item_shuffle:
            random.Random(args.seed + resume_shuffle_salt + shard_index).shuffle(items)
        for offset in range(0, len(items), args.batch_size):
            if step >= start_step + args.steps:
                break
            batch_items = items[offset : offset + args.batch_size]
            images, prompts = read_image_batch(shard_path, batch_items)
            if not images:
                continue
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
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
                timesteps, sigmas = make_sigma_schedule(pipe, clean_latents, args.train_timesteps, encoder_device)
                rollout_len = min(max(int(args.rollout_len), 1), int(timesteps.shape[0]))
                if rollout_len > 1:
                    timestep_index = choose_rollout_start_from_args(int(timesteps.shape[0]), rollout_len, args)
                else:
                    timestep_index = random.randrange(0, len(timesteps))
                noise_seed = int(args.fixed_noise_seed) if int(args.fixed_noise_seed) >= 0 else int(args.seed + step)
                generator = torch.Generator(device=encoder_device).manual_seed(noise_seed)
                noise = torch.randn(clean_latents.shape, generator=generator, device=encoder_device, dtype=clean_latents.dtype)
                sigma = sigmas[timestep_index].reshape(1, 1, 1)
                noisy_latents = sigma * noise + (1.0 - sigma) * clean_latents
                target = noise.float() - clean_latents.float()

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
            terminal_value = torch.zeros((), device=student_device)
            decoded_value = torch.zeros((), device=student_device)
            for rollout_offset in range(rollout_len):
                current_index = timestep_index + rollout_offset
                timestep = timesteps[current_index].reshape(1).repeat(current_latents.shape[0])
                pred = student(current_latents, timestep, prompt_embeds, pooled_prompt_embeds, guidance)
                weight = timestep_loss_weight(timestep.mean(), args)
                flow = flow + weight * reconstruction_loss(pred, target, args)
                if args.direction_loss_weight > 0:
                    dir_value = dir_value + direction_loss(pred, target)
                if args.norm_loss_weight > 0:
                    norm_value = norm_value + norm_loss(pred, target)
                if args.spatial_loss_weight > 0:
                    spatial_value = spatial_value + packed_spatial_gradient_loss(pred, target)

                next_sigma = sigmas[current_index + 1].reshape(1, 1, 1)
                next_target_latents = next_sigma * noise + (1.0 - next_sigma) * clean_latents
                delta = sigmas[current_index + 1] - sigmas[current_index]
                current_latents = current_latents + delta.reshape(1, 1, 1) * pred.float()
                latent_value = latent_value + F.mse_loss(current_latents.float(), next_target_latents.float())
                if args.detach_rollout and rollout_offset + 1 < rollout_len:
                    current_latents = current_latents.detach()

            if int(timestep_index + rollout_len) >= int(timesteps.shape[0]):
                terminal_value = F.mse_loss(current_latents.float(), clean_latents.float())
                if (
                    args.decoded_image_loss_weight > 0
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
                    decoded_value = F.l1_loss(decoded, target_for_decode).to(student_device)
            denominator = float(max(rollout_len, 1))
            flow = flow / denominator
            dir_value = dir_value / denominator
            norm_value = norm_value / denominator
            spatial_value = spatial_value / denominator
            latent_value = latent_value / denominator
            loss = (
                args.flow_loss_weight * flow
                + args.latent_loss_weight * latent_value
                + args.terminal_clean_loss_weight * terminal_value
                + args.decoded_image_loss_weight * decoded_value
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
            should_log = step % args.log_every == 0 or step == start_step + 1
            if float(decoded_value.detach().item()) > 0:
                should_log = True
            if should_log:
                record = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "flow_loss": float(flow.detach().item()),
                    "direction_loss": float(dir_value.detach().item()),
                    "norm_loss": float(norm_value.detach().item()),
                    "spatial_loss": float(spatial_value.detach().item()),
                    "latent_loss": float(latent_value.detach().item()),
                    "terminal_clean_loss": float(terminal_value.detach().item()),
                    "decoded_image_loss": float(decoded_value.detach().item()),
                    "timestep_index": int(timestep_index),
                    "rollout_len": int(rollout_len),
                    "batch_size": int(noisy_latents.shape[0]),
                    "unique_prompts": len({prompt_key(prompt) for prompt in prompts}),
                    "shard": shard_name,
                    "prompt": prompts[:2],
                    "prompt_filter_mode": str(args.prompt_filter_mode),
                    "fixed_noise_seed": int(args.fixed_noise_seed),
                    "mode": "diffusiondb_real_image_rollout_stream" if rollout_len > 1 else "diffusiondb_real_image_stream",
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
                    float(loss.detach().item()),
                    args,
                    ema_state,
                    optimizer.state_dict() if args.save_optimizer else None,
                )
        if args.delete_shards_after_use:
            try:
                shard_path.unlink()
            except FileNotFoundError:
                pass
    save_training_checkpoint(
        output_dir,
        config,
        student,
        step,
        float(loss.detach().item()),
        args,
        ema_state,
        optimizer.state_dict() if args.save_optimizer else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream DiffusionDB zip shards and train the FLUX student on real image rectified-flow targets.")
    parser.add_argument("--dataset-repo", default="poloclub/diffusiondb")
    parser.add_argument("--shard-prefix", default="diffusiondb-large-part-1/")
    parser.add_argument("--cache-dir", default="/dev/shm/diffusiondb_zip_cache")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_flux_diffusiondb_real_stream_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--encoder-device", default="cuda:0")
    parser.add_argument("--encoder-gpu-id", type=int, default=0)
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--train-timesteps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-items-per-shard", type=int, default=1000)
    parser.add_argument("--min-prompt-words", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--max-prompt-words", type=int, default=0)
    parser.add_argument("--max-images-per-prompt", type=int, default=0)
    parser.add_argument("--prompt-filter-mode", choices=("none", "clean-photo", "object-photo"), default="none")
    parser.add_argument("--max-prompt-commas", type=int, default=3)
    parser.add_argument("--clean-prompt-simple-word-limit", type=int, default=16)
    parser.add_argument("--disable-item-shuffle", action="store_true")
    parser.add_argument("--fixed-noise-seed", type=int, default=-1)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--ema-update-every", type=int, default=5)
    parser.add_argument("--reset-ema", action="store_true")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.5)
    parser.add_argument("--terminal-clean-loss-weight", type=float, default=0.25)
    parser.add_argument("--decoded-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-image-loss-every", type=int, default=0)
    parser.add_argument("--decoded-image-loss-size", type=int, default=256)
    parser.add_argument("--direction-loss-weight", type=float, default=0.05)
    parser.add_argument("--norm-loss-weight", type=float, default=0.01)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.01)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--snr-weighting", action="store_true")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--min-snr-weight", type=float, default=0.10)
    parser.add_argument("--max-snr-weight", type=float, default=1.0)
    parser.add_argument("--prompt-dropout", type=float, default=0.05)
    parser.add_argument("--rollout-len", type=int, default=1)
    parser.add_argument("--front-rollout-prob", type=float, default=0.0)
    parser.add_argument("--terminal-rollout-prob", type=float, default=0.35)
    parser.add_argument("--detach-rollout", action="store_true")
    parser.add_argument("--dim", type=int, default=720)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pos2d-scale", type=float, default=0.0)
    parser.add_argument("--timestep-scale", type=float, default=1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--ignore-optimizer-state", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--delete-shards-after-use", action="store_true")
    parser.add_argument("--use-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
