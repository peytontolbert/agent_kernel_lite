#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from sample_agentkernel_lite_image_flux_flow_distill import (
    DEFAULT_FLUX_TEACHER,
    load_prompt_condition_aliases,
    load_prompt_seed_aliases,
    load_student,
    stable_prompt_seed,
)
from train_agentkernel_lite_image_flux_flow_distill import timestep_embedding
from train_agentkernel_lite_image_flux_live_teacher_trajectory_reuse import build_teacher_trajectory


def rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().float().flatten(1), b.detach().float().flatten(1), dim=1).mean().item())


def step_delta(current: torch.Tensor, next_latents: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, current.ndim))
    numerator = ((next_latents.float() - current.float()) * teacher_target.float()).sum(dim=reduce_dims, keepdim=True)
    denominator = teacher_target.float().square().sum(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    return numerator / denominator


@torch.inference_mode()
def student_predict_with_ablation(
    student: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor,
    *,
    ablate_blocks: set[int],
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    config = student.config
    x = student.latent_in(latents.float()) + student.pos[:, : latents.shape[1]]
    if config.pos2d_scale:
        x = x + student.pos2d[:, : latents.shape[1]].to(device=x.device, dtype=x.dtype) * float(config.pos2d_scale)

    prompt_input = prompt_embeds.float()
    pooled_input = pooled_prompt_embeds.float()
    guidance_input = timestep_embedding(guidance.float(), config.dim)
    prompt = student.prompt_proj(prompt_input)
    if student.prompt_adapter is not None:
        prompt = prompt + student.prompt_adapter(prompt_input)
    prompt_cond_gate = getattr(student, "prompt_cond_gate", None)
    if hasattr(student, "prompt_cond_proj") and prompt_cond_gate is not None:
        prompt_cond = student.prompt_cond_proj(prompt.mean(dim=1))
    else:
        prompt_cond = torch.zeros(prompt.shape[0], config.dim, device=prompt.device, dtype=prompt.dtype)
        prompt_cond_gate = 0.0
    pooled = student.pooled_proj(pooled_input)
    if student.pooled_adapter is not None:
        pooled = pooled + student.pooled_adapter(pooled_input)
    guided = student.guidance_mlp(guidance_input)
    if student.guidance_adapter is not None:
        guided = guided + student.guidance_adapter(guidance_input)
    cond = student.time_mlp(timestep_embedding(timestep.float() * float(config.timestep_scale), config.dim))
    cond = cond + pooled + guided + prompt_cond_gate * prompt_cond

    rows: list[dict[str, float]] = []
    for block_index, block in enumerate(student.blocks):
        block_input = x
        shift1, scale1, shift2, scale2, shift3, scale3 = block.cond(cond).chunk(6, dim=-1)

        y = block.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        self_update = block.residual_gates[0] * block.self_attn(y)
        if block_index not in ablate_blocks:
            x = x + self_update

        local_update = torch.zeros_like(x)
        if block.local_mixer is not None:
            local_update = float(block.local_mixer_scale) * block.local_mixer(x)
            if block_index not in ablate_blocks:
                x = x + local_update

        y = block.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        cross = block.cross_attn(y, prompt)
        if block.cross_attn_adapter is not None:
            cross = cross + block.cross_attn_adapter(y)
        cross_update = block.residual_gates[1] * cross
        if block_index not in ablate_blocks:
            x = x + cross_update

        y = block.norm3(x) * (1 + scale3[:, None, :]) + shift3[:, None, :]
        mlp = block.dropout(block.mlp(y))
        if block.mlp_adapter is not None:
            mlp = mlp + block.mlp_adapter(y)
        mlp_update = block.residual_gates[2] * mlp
        if block_index not in ablate_blocks:
            x = x + mlp_update

        total_update = x - block_input
        rows.append(
            {
                "block": float(block_index),
                "input_rms": rms(block_input),
                "output_rms": rms(x),
                "self_update_rms": rms(self_update),
                "cross_update_rms": rms(cross_update),
                "mlp_update_rms": rms(mlp_update),
                "local_update_rms": rms(local_update),
                "total_update_rms": rms(total_update),
                "gate_self": float(block.residual_gates[0].detach().float().item()),
                "gate_cross": float(block.residual_gates[1].detach().float().item()),
                "gate_mlp": float(block.residual_gates[2].detach().float().item()),
            }
        )

    return student.latent_out(student.norm(x)), rows


@torch.inference_mode()
def rollout_terminal(
    pipe: Any,
    student: torch.nn.Module,
    trajectory: list[dict[str, torch.Tensor]],
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor,
    device: torch.device,
    *,
    ablate_blocks: set[int],
) -> tuple[torch.Tensor, list[list[dict[str, float]]]]:
    latents = trajectory[0]["latents"].to(device=device, dtype=prompt_embeds.dtype)
    all_rows: list[list[dict[str, float]]] = []
    for item in trajectory:
        timestep_value = item["timestep"].reshape(()).to(device=device)
        timestep = timestep_value.expand(latents.shape[0]).to(device=device, dtype=torch.float32)
        pred, rows = student_predict_with_ablation(
            student,
            latents.float(),
            timestep,
            prompt_embeds.float(),
            pooled_prompt_embeds.float(),
            guidance,
            ablate_blocks=ablate_blocks,
        )
        teacher_current = item["latents"].to(device=device, dtype=torch.float32)
        teacher_next = item["teacher_next"].to(device=device, dtype=torch.float32)
        teacher_target = item["teacher_target"].to(device=device, dtype=torch.float32)
        delta = step_delta(teacher_current, teacher_next, teacher_target)
        latents = (latents.float() + delta * pred.float()).to(prompt_embeds.dtype)
        all_rows.append(rows)
    return latents.float(), all_rows


def summarize_block_rows(rows_by_step: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not rows_by_step:
        return []
    depth = len(rows_by_step[0])
    out: list[dict[str, float]] = []
    keys = [key for key in rows_by_step[0][0] if key != "block"]
    for block_index in range(depth):
        row: dict[str, float] = {"block": float(block_index)}
        for key in keys:
            row[key] = float(sum(step_rows[block_index][key] for step_rows in rows_by_step) / len(rows_by_step))
        out.append(row)
    return out


def prompt_token_ablations(prompt: str) -> list[tuple[int, str, str]]:
    tokens = prompt.split()
    rows: list[tuple[int, str, str]] = []
    for index, token in enumerate(tokens):
        ablated = " ".join(tokens[:index] + tokens[index + 1 :]).strip()
        if ablated:
            rows.append((index, token, ablated))
    return rows


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Latent rollout interpretability scan for FLUX packed students.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("raw", "ema", "materialized"), default="raw")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--prompt-seed-alias-file", default="")
    parser.add_argument("--prompt-condition-alias-file", default="")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--token-occlusion", action="store_true")
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts), args.limit)
    seed_aliases = load_prompt_seed_aliases(args.prompt_seed_alias_file)
    condition_aliases = load_prompt_condition_aliases(args.prompt_condition_alias_file)
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
    student.eval()

    output: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "config": vars(config),
        "teacher_steps": int(args.teacher_steps),
        "prompts": [],
    }
    depth = len(student.blocks)

    for index, prompt in enumerate(prompts):
        condition_prompt = condition_aliases.get(prompt, prompt)
        if args.prompt_hash_seeds:
            seed_prompt = seed_aliases.get(prompt, prompt)
            seed = stable_prompt_seed(seed_prompt, int(args.seed_min), int(args.seed_max))
        else:
            seed_prompt = ""
            seed = int(args.seed) + index
        cached = build_teacher_trajectory(pipe, condition_prompt, int(seed), args, device)
        trajectory = cached["trajectory"]
        prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=pipe.dtype)
        pooled_prompt_embeds = cached["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
        guidance = torch.full([prompt_embeds.shape[0]], float(cached["guidance"]), device=device, dtype=torch.float32)
        teacher_terminal = trajectory[-1]["teacher_next"].to(device=device, dtype=torch.float32)
        base_terminal, base_rows = rollout_terminal(
            pipe,
            student,
            trajectory,
            prompt_embeds,
            pooled_prompt_embeds,
            guidance,
            device,
            ablate_blocks=set(),
        )
        base_mse = float(F.mse_loss(base_terminal, teacher_terminal).item())
        block_scan = []
        for block_index in range(depth):
            ablated_terminal, _rows = rollout_terminal(
                pipe,
                student,
                trajectory,
                prompt_embeds,
                pooled_prompt_embeds,
                guidance,
                device,
                ablate_blocks={block_index},
            )
            ablated_mse = float(F.mse_loss(ablated_terminal, teacher_terminal).item())
            block_scan.append(
                {
                    "block": int(block_index),
                    "ablated_terminal_mse": ablated_mse,
                    "mse_delta_vs_base": ablated_mse - base_mse,
                    "ablated_terminal_cosine": cosine(ablated_terminal, teacher_terminal),
                }
            )
        block_scan.sort(key=lambda row: row["mse_delta_vs_base"], reverse=True)
        token_occlusion = []
        if args.token_occlusion and isinstance(condition_prompt, str):
            for token_index, token, ablated_prompt in prompt_token_ablations(condition_prompt):
                ablated_cached = build_teacher_trajectory(pipe, ablated_prompt, int(seed), args, device)
                ablated_trajectory = ablated_cached["trajectory"]
                ablated_prompt_embeds = ablated_cached["prompt_embeds"].to(device=device, dtype=pipe.dtype)
                ablated_pooled_prompt_embeds = ablated_cached["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
                ablated_guidance = torch.full(
                    [ablated_prompt_embeds.shape[0]],
                    float(ablated_cached["guidance"]),
                    device=device,
                    dtype=torch.float32,
                )
                ablated_teacher_terminal = ablated_trajectory[-1]["teacher_next"].to(device=device, dtype=torch.float32)
                ablated_student_terminal, _ = rollout_terminal(
                    pipe,
                    student,
                    ablated_trajectory,
                    ablated_prompt_embeds,
                    ablated_pooled_prompt_embeds,
                    ablated_guidance,
                    device,
                    ablate_blocks=set(),
                )
                teacher_delta = float(F.mse_loss(ablated_teacher_terminal, teacher_terminal).item())
                student_delta = float(F.mse_loss(ablated_student_terminal, base_terminal).item())
                token_occlusion.append(
                    {
                        "index": int(token_index),
                        "token": token,
                        "ablated_prompt": ablated_prompt,
                        "teacher_terminal_delta_mse": teacher_delta,
                        "student_terminal_delta_mse": student_delta,
                        "student_teacher_terminal_mse": float(
                            F.mse_loss(ablated_student_terminal, ablated_teacher_terminal).item()
                        ),
                        "student_to_teacher_sensitivity_ratio": student_delta / max(teacher_delta, 1e-8),
                    }
                )
            token_occlusion.sort(key=lambda row: row["teacher_terminal_delta_mse"], reverse=True)
        output["prompts"].append(
            {
                "prompt": prompt,
                "condition_prompt": condition_prompt,
                "seed": int(seed),
                "seed_prompt": seed_prompt,
                "base_terminal_mse": base_mse,
                "base_terminal_rmse": base_mse**0.5,
                "base_terminal_cosine": cosine(base_terminal, teacher_terminal),
                "base_terminal_rms": rms(base_terminal),
                "teacher_terminal_rms": rms(teacher_terminal),
                "block_importance": block_scan,
                "block_update_summary": summarize_block_rows(base_rows),
                "token_occlusion": token_occlusion,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "prompts": len(output["prompts"]), "depth": depth}), flush=True)


if __name__ == "__main__":
    main()
