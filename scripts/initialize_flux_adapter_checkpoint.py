#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a fixed low-rank adapter initialization to a dense FLUX student checkpoint.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-rank", type=int, default=512)
    parser.add_argument("--adapter-scale", type=float, default=512.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()

    seed_everything(int(args.seed))
    checkpoint = torch.load(args.base, map_location="cpu")
    base_config = FluxPackedStudentConfig(**checkpoint["config"])
    config = FluxPackedStudentConfig(
        **{
            **asdict(base_config),
            "adapter_rank": int(args.adapter_rank),
            "adapter_scale": float(args.adapter_scale),
            "adapter_dropout": float(args.adapter_dropout),
        }
    )
    model = FluxPackedStudent(config)
    state = model.state_dict()
    copied = 0
    for key, tensor in checkpoint["student"].items():
        if key in state and tuple(state[key].shape) == tuple(tensor.shape):
            state[key] = tensor.detach().cpu().to(dtype=state[key].dtype)
            copied += 1
    payload = {
        "architecture": checkpoint.get("architecture", "flux_packed_student"),
        "step": int(checkpoint.get("step") or 0),
        "loss": checkpoint.get("loss"),
        "config": asdict(config),
        "base_checkpoint": str(args.base),
        "adapter_initialization_seed": int(args.seed),
        "copied_dense_tensors": copied,
        "student": state,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        {
            "output": str(output),
            "base": str(args.base),
            "adapter_rank": int(args.adapter_rank),
            "adapter_seed": int(args.seed),
            "copied_dense_tensors": copied,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )


if __name__ == "__main__":
    main()
