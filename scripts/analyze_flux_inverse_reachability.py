#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from sample_agentkernel_lite_image_flux_flow_distill import load_student, stable_prompt_seed
from train_agentkernel_lite_image_flux_live_teacher_trajectory_reuse import build_teacher_trajectory


DEFAULT_FLUX_TEACHER = "black-forest-labs/FLUX.1-dev"


def flatten_project(x: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    flat = x.float().flatten(1)
    return flat @ projection.to(device=flat.device, dtype=flat.dtype)


def ridge_r2(x: torch.Tensor, y: torch.Tensor, ridge: float) -> float:
    if x.shape[0] < 3:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    weights = torch.linalg.solve(x.T @ x + float(ridge) * eye, x.T @ y)
    pred = x @ weights
    ss_res = (y - pred).square().sum()
    ss_tot = y.square().sum().clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).item())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse reachability probe for FLUX student rollouts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=96)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=999_999_999)
    parser.add_argument("--student-indices", default="0,2,4,6,8,10,12")
    parser.add_argument("--teacher-target-indices", default="12,15,20,23")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.teacher_steps = int(args.steps)

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
    student, student_config = load_student(Path(args.checkpoint), device, "raw", None)
    prompts = read_prompts(Path(args.prompts), args.limit)
    student_indices = [int(item) for item in args.student_indices.split(",") if item.strip()]
    teacher_target_indices = [int(item) for item in args.teacher_target_indices.split(",") if item.strip()]
    max_index = max(max(student_indices), max(teacher_target_indices))
    if max_index >= int(args.steps):
        raise ValueError("--steps must exceed the largest requested index")

    rows: list[dict[str, object]] = []
    student_projected: dict[int, list[torch.Tensor]] = {idx: [] for idx in student_indices}
    teacher_projected: dict[int, list[torch.Tensor]] = {idx: [] for idx in teacher_target_indices}
    projection: torch.Tensor | None = None

    for prompt in prompts:
        seed = stable_prompt_seed(prompt, int(args.seed_min), int(args.seed_max))
        teacher_cached = build_teacher_trajectory(pipe, prompt, seed, args, device)
        teacher_traj = teacher_cached["trajectory"]
        prompt_embeds = teacher_cached["prompt_embeds"].to(device=device, dtype=pipe.dtype)
        pooled_prompt_embeds = teacher_cached["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
        latents = teacher_traj[0]["latents"].to(device=device, dtype=pipe.dtype)
        timesteps, _ = flux_timesteps(pipe, latents, int(args.steps), device)
        guidance = torch.full([1], float(args.guidance), device=device, dtype=torch.float32)
        student_states: dict[int, torch.Tensor] = {0: latents.detach().float()}
        if projection is None:
            flat_dim = int(latents.flatten(1).shape[1])
            generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
            projection = torch.randn(flat_dim, int(args.projection_dim), generator=generator) / (flat_dim ** 0.5)
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
            ).to(latents.dtype)
            latents = pipe.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
            completed = step_index + 1
            if completed in student_indices:
                student_states[completed] = latents.detach().float()
        for idx in student_indices:
            state = student_states.get(idx)
            if state is not None:
                student_projected[idx].append(flatten_project(state, projection).cpu())
        for idx in teacher_target_indices:
            if idx < len(teacher_traj):
                target = teacher_traj[idx]["latents"]
            else:
                target = teacher_traj[-1]["teacher_next"]
            teacher_projected[idx].append(flatten_project(target, projection).cpu())
        for idx in student_indices:
            state = student_states.get(idx)
            if state is None:
                continue
            row = {"prompt": prompt, "seed": seed, "student_index": idx}
            for target_idx in teacher_target_indices:
                target = teacher_traj[target_idx]["latents"] if target_idx < len(teacher_traj) else teacher_traj[-1]["teacher_next"]
                row[f"mse_to_teacher_t{target_idx}"] = float(
                    torch.mean((state.cpu().float() - target.cpu().float()).square()).item()
                )
            rows.append(row)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "examples": len(prompts),
        "student_indices": student_indices,
        "teacher_target_indices": teacher_target_indices,
        "projection_dim": int(args.projection_dim),
        "ridge_regression_warning": (
            "underdetermined: increase --limit or reduce --projection-dim"
            if len(prompts) <= int(args.projection_dim)
            else ""
        ),
        "rows": rows,
        "inverse_linear_r2": {},
    }
    for student_idx, xs in student_projected.items():
        if not xs:
            continue
        x = torch.cat(xs, dim=0)
        for target_idx, ys in teacher_projected.items():
            if not ys:
                continue
            y = torch.cat(ys, dim=0)
            summary["inverse_linear_r2"][f"student_t{student_idx}_to_teacher_t{target_idx}"] = ridge_r2(
                x,
                y,
                float(args.ridge),
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"inverse_reachability": str(output)}), flush=True)


if __name__ == "__main__":
    main()
