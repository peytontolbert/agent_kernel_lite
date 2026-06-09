#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    build_sequences,
    load_embedding,
    load_target,
    packed_lowfreq_loss,
    teacher_step_delta,
)


def pick_sequences(target_dir: str, limit: int, seed: int) -> list[list[dict[str, Any]]]:
    sequences = build_sequences([Path(target_dir)])
    rng = random.Random(seed)
    rng.shuffle(sequences)
    return sequences[:limit]


def cosine_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=1).mean()


def rollout_metrics(
    student: FluxPackedStudent,
    sequence: list[dict[str, Any]],
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
) -> dict[str, float]:
    prompt_embeds, pooled_prompt_embeds = load_embedding(sequence[0], device)
    current = load_target(sequence[0], device)
    current_latents = current["latents"].float()
    guidance = current["guidance"].reshape(1)
    flow_cos = []
    flow_mse = []
    latent_mse = []
    lowfreq_mse = []
    final_teacher = None
    for offset in range(min(rollout_len, len(sequence) - 1)):
        row = load_target(sequence[offset], device)
        next_row = load_target(sequence[offset + 1], device)
        pred = student(
            current_latents,
            row["timestep"].reshape(1).float(),
            prompt_embeds.float(),
            pooled_prompt_embeds.float(),
            guidance.float(),
        )
        delta = teacher_step_delta(row["latents"], next_row["latents"], row["teacher_target"])
        current_latents = current_latents + delta * pred.float()
        final_teacher = next_row["latents"].float()
        flow_cos.append(float(cosine_alignment(pred, row["teacher_target"]).item()))
        flow_mse.append(float(F.mse_loss(pred.float(), row["teacher_target"].float()).item()))
        latent_mse.append(float(F.mse_loss(current_latents.float(), final_teacher).item()))
        lowfreq_mse.append(float(packed_lowfreq_loss(current_latents, final_teacher, lowfreq_pool).item()))
    return {
        "flow_cos": sum(flow_cos) / max(len(flow_cos), 1),
        "flow_mse": sum(flow_mse) / max(len(flow_mse), 1),
        "latent_mse": sum(latent_mse) / max(len(latent_mse), 1),
        "lowfreq_mse": sum(lowfreq_mse) / max(len(lowfreq_mse), 1),
        "endpoint_mse": float(F.mse_loss(current_latents.float(), final_teacher).item()) if final_teacher is not None else 0.0,
        "endpoint_lowfreq_mse": float(packed_lowfreq_loss(current_latents, final_teacher, lowfreq_pool).item()) if final_teacher is not None else 0.0,
    }


def rollout_train_loss(
    student: FluxPackedStudent,
    sequence: list[dict[str, Any]],
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
) -> torch.Tensor:
    prompt_embeds, pooled_prompt_embeds = load_embedding(sequence[0], device)
    current = load_target(sequence[0], device)
    current_latents = current["latents"].float()
    guidance = current["guidance"].reshape(1)
    losses = []
    for offset in range(min(rollout_len, len(sequence) - 1)):
        row = load_target(sequence[offset], device)
        next_row = load_target(sequence[offset + 1], device)
        pred = student(
            current_latents,
            row["timestep"].reshape(1).float(),
            prompt_embeds.float(),
            pooled_prompt_embeds.float(),
            guidance.float(),
        )
        delta = teacher_step_delta(row["latents"], next_row["latents"], row["teacher_target"])
        current_latents = current_latents + delta * pred.float()
        losses.append(F.huber_loss(pred.float(), row["teacher_target"].float(), delta=0.1))
        losses.append(50.0 * F.mse_loss(current_latents.float(), next_row["latents"].float()))
        losses.append(25.0 * packed_lowfreq_loss(current_latents, next_row["latents"], lowfreq_pool))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def grad_vector(
    student: FluxPackedStudent,
    sequences: list[list[dict[str, Any]]],
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
) -> tuple[torch.Tensor, float]:
    student.zero_grad(set_to_none=True)
    loss = torch.stack([rollout_train_loss(student, seq, device, rollout_len, lowfreq_pool) for seq in sequences]).mean()
    loss.backward()
    parts = []
    for parameter in student.parameters():
        if parameter.grad is not None:
            parts.append(parameter.grad.detach().float().flatten().cpu())
    return torch.cat(parts), float(loss.detach().item())


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row[key]) for row in rows if key in row) / max(sum(1 for row in rows if key in row), 1) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute-aligned transfer metrics for FLUX student overfit experiments.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-target-dir", required=True)
    parser.add_argument("--query-near-target-dir", required=True)
    parser.add_argument("--query-far-target-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--grad-limit", type=int, default=1)
    parser.add_argument("--rollout-len", type=int, default=24)
    parser.add_argument("--lowfreq-pool", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    student = FluxPackedStudent(config).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.train()

    support = pick_sequences(args.support_target_dir, args.limit, args.seed)
    near = pick_sequences(args.query_near_target_dir, args.limit, args.seed + 1)
    far = pick_sequences(args.query_far_target_dir, args.limit, args.seed + 2)

    student.eval()
    with torch.inference_mode():
        support_rows = [rollout_metrics(student, seq, device, args.rollout_len, args.lowfreq_pool) for seq in support]
        near_rows = [rollout_metrics(student, seq, device, args.rollout_len, args.lowfreq_pool) for seq in near]
        far_rows = [rollout_metrics(student, seq, device, args.rollout_len, args.lowfreq_pool) for seq in far]

    student.train()
    support_grad, support_grad_loss = grad_vector(student, support[: args.grad_limit], device, args.rollout_len, args.lowfreq_pool)
    far_grad, far_grad_loss = grad_vector(student, far[: args.grad_limit], device, args.rollout_len, args.lowfreq_pool)
    grad_cos = F.cosine_similarity(support_grad, far_grad, dim=0).item()

    out = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step") or 0),
        "config": asdict(config),
        "support": mean_metrics(support_rows),
        "query_near": mean_metrics(near_rows),
        "query_far": mean_metrics(far_rows),
        "gradient_cos_support_far": float(grad_cos),
        "support_grad_loss": support_grad_loss,
        "query_far_grad_loss": far_grad_loss,
        "memorization_gap_endpoint_lowfreq": mean_metrics(support_rows)["endpoint_lowfreq_mse"]
        - mean_metrics(far_rows)["endpoint_lowfreq_mse"],
    }
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
