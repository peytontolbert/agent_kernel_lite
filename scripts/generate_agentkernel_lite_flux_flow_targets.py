#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from generate_agentkernel_lite_image_teacher_corpus import (
    DEFAULT_FLUX_TEACHER,
    load_teacher,
    read_prompts,
    teacher_family,
)


def install_local_diffusers() -> None:
    candidates = [
        "/data/webgl-game/repos/diffusers/src",
        "/data/repositories/diffusers/src",
    ]
    for candidate in candidates:
        if Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


def stable_id(prompt: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"{index}\n{seed}\n{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"flux_flow_{index:08d}_{digest}"


def stable_prompt_seed(prompt: str, seed_min: int, seed_max: int) -> int:
    if seed_min > seed_max:
        raise ValueError("--seed-min must be <= --seed-max")
    digest = hashlib.blake2s(prompt.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return int(seed_min + (value % (seed_max - seed_min + 1)))


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def clean_prompt(value: Any, min_words: int, max_chars: int) -> str:
    text = " ".join(str(value or "").replace("\x00", "").split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    if len(text.split()) < min_words:
        return ""
    lowered = text.lower()
    blocked = ("nsfw", "nude", "naked", "porn", "gore", "blood", "sexy", "explicit", "suggestive", "cleavage", "lingerie")
    if any(term in lowered for term in blocked):
        return ""
    return text


def read_streaming_prompts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"split": args.prompt_split, "streaming": True}
    if args.prompt_trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.prompt_config:
        dataset = load_dataset(args.prompt_dataset, args.prompt_config, **load_kwargs)
    else:
        dataset = load_dataset(args.prompt_dataset, **load_kwargs)
    if args.prompt_shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.prompt_shuffle_buffer, seed=args.seed)
    columns = [column.strip() for column in args.prompt_columns.split(",") if column.strip()]
    prompts: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for row in dataset:
        if skipped < args.prompt_skip:
            skipped += 1
            continue
        prompt = ""
        for column in columns:
            if column in row:
                prompt = clean_prompt(row[column], args.min_prompt_words, args.max_prompt_chars)
                if prompt:
                    break
        key = prompt.lower()
        if prompt and key not in seen:
            seen.add(key)
            prompts.append(prompt)
            if args.limit and len(prompts) >= args.limit:
                break
    return prompts


def load_prompt_list(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        limit = args.limit + args.prompt_skip if args.limit else 0
        prompts = read_prompts(Path(args.prompts), limit)
        return prompts[args.prompt_skip :]
    return read_streaming_prompts(args)


def flux_timesteps(pipe: Any, latents: torch.Tensor, steps: int, device: torch.device) -> tuple[torch.Tensor, int]:
    install_local_diffusers()
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

    sigmas = np.linspace(1.0, 1 / steps, steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    return retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)


@torch.inference_mode()
def generate_flow_targets(args: argparse.Namespace) -> None:
    prompts = load_prompt_list(args)
    if not prompts:
        raise ValueError("no prompts found")

    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    targets_dir = output_dir / "targets"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    args.teacher_family = "flux"
    pipe = load_teacher(args)
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    metadata_path = output_dir / "metadata.jsonl"
    done_ids: set[str] = set()
    if args.resume and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_ids.add(json.loads(line)["target_id"])

    rows_written = 0
    prompt_embedding_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for prompt_index, prompt in enumerate(prompts):
      global_prompt_index = int(args.prompt_skip) + prompt_index
      for seed_index in range(args.seeds_per_prompt):
        if args.prompt_hash_seeds:
            base_seed = stable_prompt_seed(prompt, int(args.seed_min), int(args.seed_max))
            seed = base_seed + seed_index
        else:
            seed = args.seed + global_prompt_index * args.seeds_per_prompt + seed_index
        item_id = stable_id(prompt, seed, global_prompt_index * args.seeds_per_prompt + seed_index)
        saved_timestep_indices = list(range(0, args.steps, args.target_stride))
        expected_target_ids = [f"{item_id}_t{timestep_index:03d}" for timestep_index in saved_timestep_indices]
        if args.resume and expected_target_ids and all(
            target_id in done_ids and (targets_dir / f"{target_id}.pt").exists()
            for target_id in expected_target_ids
        ):
            continue
        embedding_path = embeddings_dir / f"{item_id}.pt"
        if not embedding_path.exists():
            cached_embeds = prompt_embedding_cache.get(prompt)
            if cached_embeds is None:
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
                prompt_embedding_cache[prompt] = (
                    prompt_embeds.detach().clone(),
                    pooled_prompt_embeds.detach().clone(),
                    text_ids.detach().clone(),
                )
            else:
                prompt_embeds, pooled_prompt_embeds, text_ids = cached_embeds
            torch.save(
                {
                    "prompt": prompt,
                    "prompt_embeds": prompt_embeds.detach().to("cpu", dtype=torch.float16),
                    "pooled_prompt_embeds": pooled_prompt_embeds.detach().to("cpu", dtype=torch.float16),
                    "text_ids": text_ids.detach().to("cpu", dtype=torch.float16),
                    "max_sequence_length": args.max_sequence_length,
                },
                embedding_path,
            )
        else:
            embeds = torch.load(embedding_path, map_location=device)
            prompt_embeds = embeds["prompt_embeds"].to(device=device, dtype=dtype)
            pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(device=device, dtype=dtype)
            text_ids = embeds["text_ids"].to(device=device, dtype=dtype)

        generator = torch.Generator(device=device).manual_seed(seed)
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = pipe.prepare_latents(
            1,
            num_channels_latents,
            args.height,
            args.width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        timesteps, _ = flux_timesteps(pipe, latents, args.steps, device)
        guidance = None
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full([latents.shape[0]], args.guidance, device=device, dtype=torch.float32)

        for timestep_index, timestep_value in enumerate(timesteps):
            if args.max_timestep_index >= 0 and timestep_index > int(args.max_timestep_index):
                break
            if timestep_index % args.target_stride:
                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
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
                latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                continue

            target_id = f"{item_id}_t{timestep_index:03d}"
            target_path = targets_dir / f"{target_id}.pt"
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
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
            if target_id not in done_ids or not target_path.exists():
                atomic_torch_save(
                    {
                        "latents": latents.detach().to("cpu", dtype=torch.float16),
                        "timestep": timestep_value.detach().to("cpu", dtype=torch.float32),
                        "teacher_target": noise_pred.detach().to("cpu", dtype=torch.float16),
                        "latent_image_ids": latent_image_ids.detach().to("cpu", dtype=torch.float16),
                        "guidance": float(args.guidance),
                        "prompt": prompt,
                        "embedding_path": str(embedding_path.relative_to(output_dir)),
                        "latent_format": "flux_packed_latents",
                    },
                    target_path,
                )
                row = {
                    "target_id": target_id,
                    "prompt": prompt,
                    "target_path": str(target_path.relative_to(output_dir)),
                    "embedding_path": str(embedding_path.relative_to(output_dir)),
                    "teacher_model": args.teacher_model,
                    "teacher_family": teacher_family(args.teacher_model),
                    "seed": seed,
                    "seed_index": seed_index,
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "timestep_index": timestep_index,
                    "guidance": args.guidance,
                }
                with metadata_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
                print(json.dumps({"target": target_id, "prompt": prompt, "timestep_index": timestep_index}), flush=True)
            latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]

    rows = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    manifest = {
        "artifact_kind": "agentkernel_lite_flux_flow_targets",
        "teacher_model": args.teacher_model,
        "teacher_family": teacher_family(args.teacher_model),
        "rows": len(rows),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "target_stride": args.target_stride,
        "metadata": "metadata.jsonl",
        "embeddings": "embeddings",
        "targets": "targets",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": rows_written, "total_rows": len(rows), "output_dir": str(output_dir)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FLUX teacher flow targets for Agent Kernel Lite distillation.")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--prompt-dataset", default="poloclub/diffusiondb")
    parser.add_argument("--prompt-config", default="2m_text_only")
    parser.add_argument("--prompt-split", default="train")
    parser.add_argument("--prompt-columns", default="prompt,caption,text")
    parser.add_argument("--prompt-skip", type=int, default=0)
    parser.add_argument("--prompt-shuffle-buffer", type=int, default=0)
    parser.add_argument("--prompt-trust-remote-code", action="store_true")
    parser.add_argument("--min-prompt-words", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=420)
    parser.add_argument("--output-dir", default="data/vision/flux1_dev_flow_targets_v0")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--target-stride", type=int, default=4)
    parser.add_argument("--max-timestep-index", type=int, default=-1)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    args = parser.parse_args()
    generate_flow_targets(args)


if __name__ == "__main__":
    main()
