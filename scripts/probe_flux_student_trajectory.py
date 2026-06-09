#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from sample_agentkernel_lite_image_flux_flow_distill import (
    DEFAULT_FLUX_TEACHER,
    load_prompt_seed_aliases,
    load_student,
    stable_prompt_seed,
)


def tensor_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().flatten(1).norm(dim=-1).mean().item())


def tensor_rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item())


@torch.inference_mode()
def forward_with_block_stats(
    student: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    config = student.config
    x = student.latent_in(latents.float()) + student.pos[:, : latents.shape[1]]
    if config.pos2d_scale:
        x = x + student.pos2d[:, : latents.shape[1]].to(device=x.device, dtype=x.dtype) * float(config.pos2d_scale)

    prompt_input = prompt_embeds.float()
    pooled_input = pooled_prompt_embeds.float()
    guidance_input = student.guidance_mlp[0].weight.new_empty(guidance.shape[0], config.dim)
    from train_agentkernel_lite_image_flux_flow_distill import timestep_embedding

    guidance_input = timestep_embedding(guidance.float(), config.dim)
    prompt = student.prompt_proj(prompt_input)
    if student.prompt_adapter is not None:
        prompt = prompt + student.prompt_adapter(prompt_input)
    cond = student.time_mlp(timestep_embedding(timestep.float() * float(config.timestep_scale), config.dim))
    pooled = student.pooled_proj(pooled_input)
    if student.pooled_adapter is not None:
        pooled = pooled + student.pooled_adapter(pooled_input)
    guided = student.guidance_mlp(guidance_input)
    if student.guidance_adapter is not None:
        guided = guided + student.guidance_adapter(guidance_input)
    cond = cond + pooled + guided

    rows: list[dict[str, float]] = []
    for block_index, block in enumerate(student.blocks):
        block_input = x
        shift1, scale1, shift2, scale2, shift3, scale3 = block.cond(cond).chunk(6, dim=-1)

        y = block.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        self_update = block.residual_gates[0] * block.self_attn(y)
        x = x + self_update

        local_update = None
        if block.local_mixer is not None:
            local_update = float(block.local_mixer_scale) * block.local_mixer(x)
            x = x + local_update

        y = block.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        cross = block.cross_attn(y, prompt)
        if block.cross_attn_adapter is not None:
            cross = cross + block.cross_attn_adapter(y)
        cross_update = block.residual_gates[1] * cross
        x = x + cross_update

        y = block.norm3(x) * (1 + scale3[:, None, :]) + shift3[:, None, :]
        mlp = block.dropout(block.mlp(y))
        if block.mlp_adapter is not None:
            mlp = mlp + block.mlp_adapter(y)
        mlp_update = block.residual_gates[2] * mlp
        x = x + mlp_update

        total_update = x - block_input
        denom = max(tensor_norm(total_update), 1e-8)
        rows.append(
            {
                "block": float(block_index),
                "input_norm": tensor_norm(block_input),
                "output_norm": tensor_norm(x),
                "total_update_norm": tensor_norm(total_update),
                "self_update_norm": tensor_norm(self_update),
                "cross_update_norm": tensor_norm(cross_update),
                "mlp_update_norm": tensor_norm(mlp_update),
                "local_update_norm": tensor_norm(local_update) if local_update is not None else 0.0,
                "cross_update_share": tensor_norm(cross_update) / denom,
                "mlp_update_share": tensor_norm(mlp_update) / denom,
                "self_update_share": tensor_norm(self_update) / denom,
                "gate_self": float(block.residual_gates[0].detach().float().item()),
                "gate_cross": float(block.residual_gates[1].detach().float().item()),
                "gate_mlp": float(block.residual_gates[2].detach().float().item()),
            }
        )

    return student.latent_out(student.norm(x)), rows


def mean_rows(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(row[key] for row in rows) / len(rows))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Probe per-block trajectory behavior in a FLUX packed student.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("raw", "ema", "materialized"), default="raw")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="tmp/flux_student_trajectory_probe.json")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--prompt-seed-alias-file", default="")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts), args.limit)
    aliases = load_prompt_seed_aliases(args.prompt_seed_alias_file)
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

    output: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "steps": int(args.steps),
        "config": vars(config),
        "prompts": [],
    }

    for index, prompt in enumerate(prompts):
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=int(config.max_sequence_length),
        )
        if args.prompt_hash_seeds:
            seed_prompt = aliases.get(prompt, prompt)
            seed = stable_prompt_seed(seed_prompt, int(args.seed_min), int(args.seed_max))
        else:
            seed_prompt = ""
            seed = int(args.seed) + index
        generator = torch.Generator(device=device).manual_seed(seed)
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, _latent_image_ids = pipe.prepare_latents(
            1,
            num_channels_latents,
            int(args.height),
            int(args.width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        timesteps, _ = flux_timesteps(pipe, latents, int(args.steps), device)
        guidance = torch.full([latents.shape[0]], float(args.guidance), device=device, dtype=torch.float32)
        step_rows: list[dict[str, Any]] = []
        for step_index, timestep_value in enumerate(timesteps):
            timestep = timestep_value.expand(latents.shape[0]).to(device)
            pred, block_rows = forward_with_block_stats(
                student,
                latents.float(),
                timestep.float(),
                prompt_embeds.float(),
                pooled_prompt_embeds.float(),
                guidance,
            )
            next_latents = pipe.scheduler.step(pred.to(latents.dtype), timestep_value, latents, return_dict=False)[0]
            step_rows.append(
                {
                    "step_index": int(step_index),
                    "timestep": float(timestep_value.detach().float().item()),
                    "prediction_rms": tensor_rms(pred),
                    "latent_delta_rms": tensor_rms(next_latents.float() - latents.float()),
                    "mean_total_update_norm": mean_rows(block_rows, "total_update_norm"),
                    "mean_self_update_share": mean_rows(block_rows, "self_update_share"),
                    "mean_cross_update_share": mean_rows(block_rows, "cross_update_share"),
                    "mean_mlp_update_share": mean_rows(block_rows, "mlp_update_share"),
                    "blocks": block_rows,
                }
            )
            latents = next_latents
        output["prompts"].append(
            {
                "prompt": prompt,
                "seed": int(seed),
                "seed_prompt": seed_prompt,
                "mean_prediction_rms": mean_rows(step_rows, "prediction_rms"),
                "mean_latent_delta_rms": mean_rows(step_rows, "latent_delta_rms"),
                "mean_total_update_norm": mean_rows(step_rows, "mean_total_update_norm"),
                "mean_self_update_share": mean_rows(step_rows, "mean_self_update_share"),
                "mean_cross_update_share": mean_rows(step_rows, "mean_cross_update_share"),
                "mean_mlp_update_share": mean_rows(step_rows, "mean_mlp_update_share"),
                "steps": step_rows,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "prompts": len(output["prompts"])}), flush=True)


if __name__ == "__main__":
    main()
