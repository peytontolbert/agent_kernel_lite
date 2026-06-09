#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig


def stable_prompt_seed(prompt: str, seed_min: int, seed_max: int) -> int:
    digest = hashlib.blake2s(prompt.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return int(seed_min + (value % (seed_max - seed_min + 1)))


def packed_frequency_energy(latents: torch.Tensor) -> dict[str, float]:
    tokens = int(latents.shape[1])
    side = int(tokens**0.5)
    if side * side != tokens:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    grid = latents.float().reshape(latents.shape[0], side, side, latents.shape[2]).permute(0, 3, 1, 2)
    fft = torch.fft.rfft2(grid, norm="ortho").abs().square()
    fy = torch.fft.fftfreq(side, device=latents.device).abs().reshape(side, 1)
    fx = torch.fft.rfftfreq(side, device=latents.device).abs().reshape(1, side // 2 + 1)
    radius = torch.sqrt(fx.square() + fy.square()).reshape(1, 1, side, side // 2 + 1)
    total = fft.mean().clamp_min(1e-12)
    return {
        "low": float((fft * (radius < 0.12)).mean().div(total).item()),
        "mid": float((fft * ((radius >= 0.12) & (radius < 0.28))).mean().div(total).item()),
        "high": float((fft * (radius >= 0.28)).mean().div(total).item()),
    }


def load_student(checkpoint_path: Path, device: torch.device, weights: str) -> tuple[FluxPackedStudent, FluxPackedStudentConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    student = FluxPackedStudent(config).to(device)
    if weights == "ema":
        state = checkpoint.get("student_ema")
        if not state:
            raise ValueError(f"{checkpoint_path} does not contain student_ema")
    else:
        state = checkpoint["student"]
    student.load_state_dict(state, strict=True)
    student.eval()
    return student, config


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fresh-T0 FLUX student rollout wiring and prompt sensitivity.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-sequence-length", type=int, default=96)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    args = parser.parse_args()

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
    student, config = load_student(Path(args.checkpoint), device, args.weights)
    prompts = read_prompts(Path(args.prompts), int(args.limit))
    output_rows = []
    for prompt in prompts:
        seed = stable_prompt_seed(prompt, int(args.seed_min), int(args.seed_max))
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=int(args.max_sequence_length),
        )
        zero_prompt_embeds = torch.zeros_like(prompt_embeds)
        zero_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
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
        timesteps, _ = flux_timesteps(pipe, latents, int(args.steps), device)
        guidance = torch.full([latents.shape[0]], float(args.guidance), device=device, dtype=torch.float32)
        rows = []
        for step_index, timestep_value in enumerate(timesteps):
            timestep = timestep_value.expand(latents.shape[0]).to(device)
            pred = student(
                latents.float(),
                timestep.float(),
                prompt_embeds.float(),
                pooled_prompt_embeds.float(),
                guidance,
            )
            zero_pred = student(
                latents.float(),
                timestep.float(),
                zero_prompt_embeds.float(),
                zero_pooled_prompt_embeds.float(),
                guidance,
            )
            prompt_delta = pred.float() - zero_pred.float()
            next_latents = pipe.scheduler.step(pred.to(latents.dtype), timestep_value, latents, return_dict=False)[0]
            delta = next_latents.float() - latents.float()
            freq = packed_frequency_energy(next_latents)
            rows.append(
                {
                    "step_index": int(step_index),
                    "timestep": float(timestep_value.item()),
                    "latent_rms": float(next_latents.float().square().mean().sqrt().item()),
                    "step_delta_rms": float(delta.square().mean().sqrt().item()),
                    "pred_rms": float(pred.float().square().mean().sqrt().item()),
                    "prompt_delta_rms": float(prompt_delta.square().mean().sqrt().item()),
                    "prompt_delta_ratio": float(
                        prompt_delta.flatten(1).norm(dim=1).mean().div(pred.float().flatten(1).norm(dim=1).mean().clamp_min(1e-12)).item()
                    ),
                    "freq_low_ratio": freq["low"],
                    "freq_mid_ratio": freq["mid"],
                    "freq_high_ratio": freq["high"],
                }
            )
            latents = next_latents
        output_rows.append({"prompt": prompt, "seed": seed, "steps": rows, "config_max_sequence_length": int(config.max_sequence_length)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": args.checkpoint, "rows": output_rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "prompts": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()
