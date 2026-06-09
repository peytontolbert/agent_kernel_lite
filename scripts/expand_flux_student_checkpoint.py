#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig


def copy_overlap(key: str, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if source.ndim > 1:
        expanded = torch.zeros_like(target.detach())
    elif key.endswith(".weight"):
        expanded = target.detach().clone()
    else:
        expanded = torch.zeros_like(target.detach())
    slices = tuple(slice(0, min(dst, src)) for dst, src in zip(expanded.shape, source.shape))
    expanded[slices].copy_(source[slices].to(device=expanded.device, dtype=expanded.dtype))
    return expanded


def expanded_state(
    target_model: FluxPackedStudent,
    source_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    target_state = target_model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    expanded_keys: list[str] = []
    missing_keys: list[str] = []
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            copied[key] = target_tensor.detach().cpu()
            missing_keys.append(key)
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            copied[key] = source_tensor.detach().cpu().to(dtype=target_tensor.dtype)
        elif source_tensor.ndim == target_tensor.ndim:
            copied[key] = copy_overlap(key, target_tensor.cpu(), source_tensor.cpu())
            expanded_keys.append(key)
        else:
            copied[key] = target_tensor.detach().cpu()
            missing_keys.append(key)
    return copied, expanded_keys, missing_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Width-expand a FLUX packed student checkpoint by overlapping tensor copies.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--mlp-ratio", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=-1.0)
    parser.add_argument("--pos2d-scale", type=float, default=-1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=-1.0)
    args = parser.parse_args()

    source_path = Path(args.source)
    checkpoint = torch.load(source_path, map_location="cpu")
    source_config = FluxPackedStudentConfig(**checkpoint["config"])
    target_config = FluxPackedStudentConfig(
        **{
            **asdict(source_config),
            "dim": int(args.dim),
            "depth": int(args.depth) if args.depth > 0 else int(source_config.depth),
            "heads": int(args.heads),
            "mlp_ratio": int(args.mlp_ratio) if args.mlp_ratio > 0 else int(source_config.mlp_ratio),
            "dropout": float(args.dropout) if args.dropout >= 0 else float(source_config.dropout),
            "pos2d_scale": float(args.pos2d_scale) if args.pos2d_scale >= 0 else float(source_config.pos2d_scale),
            "local_mixer_scale": float(args.local_mixer_scale)
            if args.local_mixer_scale >= 0
            else float(source_config.local_mixer_scale),
        }
    )
    target_model = FluxPackedStudent(target_config)
    student_state, expanded_keys, missing_keys = expanded_state(target_model, checkpoint["student"])
    payload = {
        "config": asdict(target_config),
        "student": student_state,
        "step": int(checkpoint.get("step") or 0),
        "source_checkpoint": str(source_path),
        "expanded_from_config": checkpoint["config"],
        "expanded_keys": expanded_keys,
        "missing_keys": missing_keys,
    }
    if checkpoint.get("student_ema"):
        ema_state, ema_expanded_keys, ema_missing_keys = expanded_state(target_model, checkpoint["student_ema"])
        payload["student_ema"] = ema_state
        payload["ema_expanded_keys"] = ema_expanded_keys
        payload["ema_missing_keys"] = ema_missing_keys
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    params = sum(parameter.numel() for parameter in target_model.parameters())
    print(
        {
            "output": str(output_path),
            "parameters": params,
            "source_dim": source_config.dim,
            "target_dim": target_config.dim,
            "depth": target_config.depth,
            "heads": target_config.heads,
            "expanded_keys": len(expanded_keys),
            "missing_keys": len(missing_keys),
        }
    )


if __name__ == "__main__":
    main()
