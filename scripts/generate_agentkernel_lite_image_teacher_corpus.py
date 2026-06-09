#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


DEFAULT_SANA_TEACHER = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers"
DEFAULT_FLUX_TEACHER = "black-forest-labs/FLUX.1-dev"
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, noisy, watermark, text artifacts"


def install_local_diffusers() -> None:
    candidates = [
        os.environ.get("DIFFUSERS_SRC", ""),
        "/data/webgl-game/repos/diffusers/src",
        "/data/repositories/diffusers/src",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


def import_sana_pipeline() -> tuple[type[Any], str]:
    install_local_diffusers()
    try:
        from diffusers import SanaPipeline

        return SanaPipeline, "sana"
    except Exception:
        try:
            from diffusers import AutoPipelineForText2Image

            return AutoPipelineForText2Image, "auto"
        except Exception as error:
            raise RuntimeError(
                "Could not import diffusers SanaPipeline or AutoPipelineForText2Image. "
                "Install diffusers or set DIFFUSERS_SRC to a local checkout."
            ) from error


def import_flux_pipeline() -> tuple[type[Any], type[Any], type[Any]]:
    install_local_diffusers()
    try:
        from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
    except Exception as error:
        raise RuntimeError(
            "Could not import FluxPipeline, FluxTransformer2DModel, or BitsAndBytesConfig. "
            "Install a recent diffusers checkout and bitsandbytes, or set DIFFUSERS_SRC."
        ) from error
    return BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel


def teacher_family(model_id: str) -> str:
    if "FLUX.1" in model_id or "black-forest-labs/FLUX" in model_id:
        return "flux"
    if "Sana_Sprint_0.6B" in model_id or "SANA_Sprint_0.6B" in model_id:
        return "sana-sprint-0.6b-teacher"
    if "Sana_600M" in model_id:
        return "sana-0.6b"
    return "teacher"


def read_prompt_records(path: Path, limit: int, base_seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line) if path.suffix.lower() == ".jsonl" else None
                if row is None:
                    data = json.loads(line)
                    rows = data if isinstance(data, list) else data.get("prompts", [])
                    for index, item in enumerate(rows):
                        if isinstance(item, dict):
                            record = dict(item)
                            prompt = str(item.get("prompt", item.get("text", item.get("caption", "")))).strip()
                        else:
                            record = {}
                            prompt = str(item).strip()
                        if not prompt:
                            continue
                        record["prompt"] = prompt
                        record.setdefault("source_index", index)
                        if record.get("seed") is None and record.get("seed_offset") is not None:
                            record["seed"] = int(base_seed) + int(record["seed_offset"])
                        records.append(record)
                        if limit and len(records) >= limit:
                            break
                    break
                prompt = str(row.get("prompt", row.get("text", row.get("caption", "")))).strip()
                if not prompt:
                    continue
                row = dict(row)
                row["prompt"] = prompt
                row.setdefault("source_index", len(records))
                if row.get("seed") is None and row.get("seed_offset") is not None:
                    row["seed"] = int(base_seed) + int(row["seed_offset"])
                records.append(row)
                if limit and len(records) >= limit:
                    break
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                prompt = line.strip()
                if prompt and not prompt.startswith("#"):
                    records.append({"prompt": prompt, "source_index": len(records)})
                if limit and len(records) >= limit:
                    break
    return [record for record in records if record.get("prompt")]


def read_prompts(path: Path, limit: int) -> list[str]:
    """Backward-compatible helper used by older sampling/training scripts."""
    return [str(record["prompt"]) for record in read_prompt_records(path, limit, base_seed=0)]


def stable_id(prompt: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"{index}\n{seed}\n{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"teacher_{index:08d}_{digest}"


def load_teacher(args: argparse.Namespace) -> Any:
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.teacher_family == "flux":
        BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel = import_flux_pipeline()
        transformer = None
        if args.quantize_transformer_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                args.teacher_model,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
                local_files_only=args.local_files_only,
            )
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if transformer is not None:
            kwargs["transformer"] = transformer
        if args.local_files_only:
            kwargs["local_files_only"] = True
        pipe = FluxPipeline.from_pretrained(args.teacher_model, **kwargs)
        if args.cpu_offload:
            pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)
        else:
            pipe = pipe.to(args.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        return pipe

    pipeline_cls, pipeline_kind = import_sana_pipeline()
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    if args.variant:
        kwargs["variant"] = args.variant
    if args.local_files_only:
        kwargs["local_files_only"] = True
    if pipeline_kind == "auto":
        pipe = pipeline_cls.from_pretrained(args.teacher_model, **kwargs)
    else:
        pipe = pipeline_cls.from_pretrained(args.teacher_model, **kwargs)
    pipe = pipe.to(args.device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe


def decode_flux_latents(pipe: Any, packed_latents: torch.Tensor, height: int, width: int) -> Any:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.inference_mode()
def generate_corpus(args: argparse.Namespace) -> None:
    prompt_records = read_prompt_records(Path(args.prompts), args.limit, args.seed)
    if not prompt_records:
        raise ValueError("no prompts found")
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    latents_dir = output_dir / "latents"
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.output_latents:
        latents_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_teacher(args)
    metadata_path = output_dir / "metadata.jsonl"
    csv_path = output_dir / "metadata.csv"
    done_ids = set()
    if metadata_path.exists() and args.resume:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    rows: list[dict[str, Any]] = []
    for index, prompt_record in enumerate(prompt_records):
        prompt = str(prompt_record["prompt"])
        for sample_index in range(args.images_per_prompt):
            row_seed = prompt_record.get("seed")
            if row_seed is None:
                seed = args.seed + index * args.images_per_prompt + sample_index
            else:
                seed = int(row_seed) + sample_index
            item_id = stable_id(prompt, seed, index * args.images_per_prompt + sample_index)
            image_path = images_dir / f"{item_id}.png"
            latent_path = latents_dir / f"{item_id}.pt"
            if item_id in done_ids and image_path.exists() and (not args.output_latents or latent_path.exists()):
                continue
            generator = torch.Generator(device=args.device).manual_seed(seed)
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance,
                "generator": generator,
            }
            if args.max_sequence_length:
                call_kwargs["max_sequence_length"] = args.max_sequence_length
            if args.teacher_family == "flux":
                call_kwargs.pop("negative_prompt", None)
                if args.output_latents:
                    result = pipe(**call_kwargs, output_type="latent")
                    packed_latents = result.images.detach().to("cpu", dtype=torch.float16)
                    torch.save(
                        {
                            "packed_latents": packed_latents,
                            "prompt": prompt,
                            "seed": seed,
                            "width": args.width,
                            "height": args.height,
                            "teacher_model": args.teacher_model,
                            "teacher_family": teacher_family(args.teacher_model),
                            "steps": args.steps,
                            "guidance": args.guidance,
                            "latent_format": "flux_packed_latents",
                        },
                        latent_path,
                    )
                    if args.output_images:
                        image = decode_flux_latents(pipe, result.images, args.height, args.width)
                        image.save(image_path)
                else:
                    result = pipe(**call_kwargs)
                    image = result.images[0]
                    image.save(image_path)
            else:
                result = pipe(**call_kwargs)
                image = result.images[0]
                image.save(image_path)
            row = {
                "id": item_id,
                "prompt": prompt,
                "path": str(image_path.relative_to(output_dir)) if image_path.exists() else "",
                "teacher_ref": str(image_path.resolve()) if image_path.exists() else "",
                "latent_path": str(latent_path.relative_to(output_dir)) if latent_path.exists() else "",
                "teacher_model": args.teacher_model,
                "teacher_family": teacher_family(args.teacher_model),
                "seed": seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "guidance": args.guidance,
            }
            for key, value in prompt_record.items():
                if key not in row and key not in {"text", "caption"}:
                    row[key] = value
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(json.dumps({"generated": row["id"], "prompt": prompt, "path": row["path"]}), flush=True)

    all_rows: list[dict[str, Any]] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            all_rows = [json.loads(line) for line in handle if line.strip()]
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames: list[str] = []
            for row in all_rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_image_teacher_corpus",
        "teacher_model": args.teacher_model,
        "teacher_family": teacher_family(args.teacher_model),
        "rows": len(all_rows),
        "width": args.width,
        "height": args.height,
        "metadata": "metadata.jsonl",
        "images": "images" if args.output_images else "",
        "latents": "latents" if args.output_latents else "",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a real image teacher corpus for Agent Kernel Lite image distillation.")
    parser.add_argument("--prompts", required=True, help="Text/jsonl/json prompt file.")
    parser.add_argument("--output-dir", default="data/vision/teacher_sana_sprint_0_6b_1024px_v0")
    parser.add_argument("--teacher-model", default=DEFAULT_SANA_TEACHER)
    parser.add_argument("--teacher-family", choices=("sana", "flux"), default="sana")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variant", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--images-per-prompt", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--output-latents", action="store_true")
    parser.add_argument("--output-images", action="store_true", default=True)
    args = parser.parse_args()
    generate_corpus(args)


if __name__ == "__main__":
    main()
