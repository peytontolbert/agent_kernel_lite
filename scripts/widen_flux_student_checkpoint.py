#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig


def copy_overlap(target: torch.Tensor, source: torch.Tensor, zero_new_values: bool = False) -> tuple[torch.Tensor, bool]:
    if target.ndim != source.ndim:
        return target, False
    slices = tuple(slice(0, min(t, s)) for t, s in zip(target.shape, source.shape))
    updated = torch.zeros_like(target) if zero_new_values else target.clone()
    updated[slices] = source[slices].to(dtype=target.dtype)
    return updated, True


def widen_state(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    zero_new_values: bool = False,
) -> dict[str, torch.Tensor]:
    widened: dict[str, torch.Tensor] = {}
    copied = 0
    partial = 0
    skipped = 0
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            widened[name] = target_tensor
            skipped += 1
            continue
        if target_tensor.shape == source_tensor.shape:
            widened[name] = source_tensor.to(dtype=target_tensor.dtype)
            copied += 1
            continue
        updated, ok = copy_overlap(target_tensor, source_tensor, zero_new_values=zero_new_values)
        widened[name] = updated
        if ok:
            partial += 1
        else:
            skipped += 1
    print({"copied_exact": copied, "copied_partial": partial, "kept_random": skipped}, flush=True)
    return widened


def main() -> None:
    parser = argparse.ArgumentParser(description="Widen a trained FLUX packed student checkpoint into a larger config.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--pos2d-scale", type=float, default=0.03)
    parser.add_argument("--local-mixer-scale", type=float, default=0.05)
    parser.add_argument("--reset-step", type=int, default=0)
    parser.add_argument("--zero-new-block-gates", action="store_true")
    parser.add_argument("--zero-new-values", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_ckpt: dict[str, Any] = torch.load(source_path, map_location="cpu")
    source_config = dict(source_ckpt["config"])
    target_config = FluxPackedStudentConfig(
        latent_tokens=int(source_config.get("latent_tokens", 1024)),
        latent_channels=int(source_config.get("latent_channels", 64)),
        prompt_dim=int(source_config.get("prompt_dim", 4096)),
        pooled_dim=int(source_config.get("pooled_dim", 768)),
        max_sequence_length=int(source_config.get("max_sequence_length", 512)),
        dim=int(args.dim),
        depth=int(args.depth),
        heads=int(args.heads),
        mlp_ratio=int(args.mlp_ratio),
        dropout=float(args.dropout),
        pos2d_scale=float(args.pos2d_scale),
        timestep_scale=float(source_config.get("timestep_scale", 1.0)),
        local_mixer_scale=float(args.local_mixer_scale),
        adapter_rank=0,
        adapter_scale=1.0,
        adapter_dropout=0.0,
    )
    target_model = FluxPackedStudent(target_config)
    target_state = target_model.state_dict()
    source_state = source_ckpt["student"]
    widened = widen_state(target_state, source_state, zero_new_values=bool(args.zero_new_values))
    if args.zero_new_block_gates:
        source_depth = int(source_config.get("depth", 0) or 0)
        for block_index in range(source_depth, int(args.depth)):
            key = f"blocks.{block_index}.residual_gates"
            if key in widened:
                widened[key] = torch.zeros_like(widened[key])

    output_ckpt: dict[str, Any] = {
        "config": target_config.__dict__,
        "student": widened,
        "step": int(args.reset_step),
        "loss": float(source_ckpt.get("loss", 0.0) or 0.0),
        "source_checkpoint": str(source_path),
        "widened_from_config": source_config,
    }
    if source_ckpt.get("student_ema"):
        widened_ema = widen_state(target_state, source_ckpt["student_ema"], zero_new_values=bool(args.zero_new_values))
        if args.zero_new_block_gates:
            source_depth = int(source_config.get("depth", 0) or 0)
            for block_index in range(source_depth, int(args.depth)):
                key = f"blocks.{block_index}.residual_gates"
                if key in widened_ema:
                    widened_ema[key] = torch.zeros_like(widened_ema[key])
        output_ckpt["student_ema"] = widened_ema
    torch.save(output_ckpt, output_path)
    print({"output": str(output_path), "config": target_config.__dict__}, flush=True)


if __name__ == "__main__":
    main()
