#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from generate_agentkernel_lite_flux_flow_targets import atomic_torch_save, flux_timesteps
from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, load_teacher, teacher_family


def stable_id(prompt: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"real\n{index}\n{seed}\n{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"flux_real_{index:08d}_{digest}"


def clean_caption(row: dict[str, Any], columns: list[str], max_chars: int) -> str:
    for column in columns:
        value = row.get(column)
        if isinstance(value, list) and value:
            value = value[0]
        text = " ".join(str(value or "").replace("\x00", " ").split())
        if text:
            return text[:max_chars].rsplit(" ", 1)[0].strip()
    return ""


def encode_flux_image(pipe: Any, image: Any, width: int, height: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pixels = pipe.image_processor.preprocess(image.convert("RGB"), height=height, width=width)
    pixels = pixels.to(device=device, dtype=dtype)
    latents = pipe.vae.encode(pixels).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    batch, channels, latent_height, latent_width = latents.shape
    return pipe._pack_latents(latents, batch, channels, latent_height, latent_width)


@torch.inference_mode()
def generate_targets(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    targets_dir = output_dir / "targets"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    done_ids: set[str] = set()
    if args.resume and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_ids.add(json.loads(line)["target_id"])

    args.teacher_family = "flux"
    pipe = load_teacher(args)
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    columns = [value.strip() for value in args.caption_columns.split(",") if value.strip()]

    dataset = load_dataset(args.dataset, args.dataset_config or None, split=args.split, streaming=True)
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    rows_written = 0
    seen = 0
    for source_index, row in enumerate(dataset):
        if seen < args.skip:
            seen += 1
            continue
        prompt = clean_caption(row, columns, args.max_caption_chars)
        image = row.get(args.image_column)
        if not prompt or image is None:
            continue
        item_index = seen - args.skip
        if args.limit and item_index >= args.limit:
            break
        seen += 1
        seed = args.seed + item_index
        item_id = stable_id(prompt, seed, item_index)
        expected_target_ids = [f"{item_id}_t{timestep_index:03d}" for timestep_index in range(args.steps)]
        if args.resume and expected_target_ids and all(
            target_id in done_ids and (targets_dir / f"{target_id}.pt").exists()
            for target_id in expected_target_ids
        ):
            continue

        embedding_path = embeddings_dir / f"{item_id}.pt"
        if not embedding_path.exists():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )
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

        clean_latents = encode_flux_image(pipe, image, args.width, args.height, device, prompt_embeds.dtype)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(clean_latents.shape, generator=generator, device=device, dtype=clean_latents.dtype)
        timesteps, _ = flux_timesteps(pipe, clean_latents, args.steps, device)
        sigmas = pipe.scheduler.sigmas.to(device=device, dtype=clean_latents.dtype)
        target = noise.float() - clean_latents.float()

        for timestep_index, timestep_value in enumerate(timesteps):
            target_id = f"{item_id}_t{timestep_index:03d}"
            target_path = targets_dir / f"{target_id}.pt"
            sigma = sigmas[timestep_index].reshape(1, 1, 1)
            noisy_latents = sigma * noise + (1.0 - sigma) * clean_latents
            if target_id not in done_ids or not target_path.exists():
                atomic_torch_save(
                    {
                        "latents": noisy_latents.detach().to("cpu", dtype=torch.float16),
                        "timestep": timestep_value.detach().to("cpu", dtype=torch.float32),
                        "teacher_target": target.detach().to("cpu", dtype=torch.float16),
                        "guidance": float(args.guidance),
                        "prompt": prompt,
                        "embedding_path": str(embedding_path.relative_to(output_dir)),
                        "latent_format": "flux_packed_latents",
                        "target_kind": "real_image_rectified_flow",
                    },
                    target_path,
                )
                with metadata_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "target_id": target_id,
                                "prompt": prompt,
                                "target_path": str(target_path.relative_to(output_dir)),
                                "embedding_path": str(embedding_path.relative_to(output_dir)),
                                "teacher_model": args.teacher_model,
                                "teacher_family": teacher_family(args.teacher_model),
                                "source_dataset": args.dataset,
                                "source_index": source_index,
                                "seed": seed,
                                "width": args.width,
                                "height": args.height,
                                "steps": args.steps,
                                "timestep_index": timestep_index,
                                "guidance": args.guidance,
                                "target_kind": "real_image_rectified_flow",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                rows_written += 1
                print(json.dumps({"target": target_id, "prompt": prompt, "timestep_index": timestep_index}), flush=True)

    rows = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "agentkernel_lite_flux_real_image_flow_targets",
                "teacher_model": args.teacher_model,
                "teacher_family": teacher_family(args.teacher_model),
                "rows": len(rows),
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "metadata": "metadata.jsonl",
                "embeddings": "embeddings",
                "targets": "targets",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"wrote": rows_written, "total_rows": len(rows), "output_dir": str(output_dir)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FLUX-compatible rectified-flow targets from real image-caption data.")
    parser.add_argument("--dataset", default="jxie/flickr8k")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-columns", default="caption_0,caption,caption_1,text")
    parser.add_argument("--output-dir", default="data/vision/flux_real_image_flow_targets_v0")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--max-caption-chars", type=int, default=240)
    parser.add_argument("--seed", type=int, default=70700000)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    args = parser.parse_args()
    generate_targets(args)


if __name__ == "__main__":
    main()
