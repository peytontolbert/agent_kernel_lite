#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig


def parse_weighted_checkpoint(value: str) -> tuple[float, Path]:
    if "=" not in value:
        return 1.0, Path(value)
    weight, path = value.split("=", 1)
    return float(weight), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge adapter tensors from multiple FLUX student checkpoints onto one base.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", action="append", required=True, help="optional_weight=/path/to/checkpoint.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-rank", type=int, default=512)
    parser.add_argument("--adapter-scale", type=float, default=512.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    args = parser.parse_args()

    base_checkpoint = torch.load(args.base, map_location="cpu")
    base_config = FluxPackedStudentConfig(**base_checkpoint["config"])
    config = FluxPackedStudentConfig(
        **{
            **asdict(base_config),
            "adapter_rank": int(args.adapter_rank),
            "adapter_scale": float(args.adapter_scale),
            "adapter_dropout": float(args.adapter_dropout),
        }
    )
    model = FluxPackedStudent(config)
    merged_state = model.state_dict()
    base_state = base_checkpoint["student"]
    for key, value in list(merged_state.items()):
        if "adapter" not in key and key in base_state and tuple(base_state[key].shape) == tuple(value.shape):
            merged_state[key] = base_state[key].detach().cpu().to(dtype=value.dtype)

    weighted_adapters = [parse_weighted_checkpoint(value) for value in args.adapter]
    total_weight = sum(weight for weight, _path in weighted_adapters)
    if total_weight <= 0:
        raise ValueError("adapter weights must sum to a positive value")
    adapter_sums: dict[str, torch.Tensor] = {}
    adapter_sources: dict[str, int] = {}
    for weight, path in weighted_adapters:
        checkpoint = torch.load(path, map_location="cpu")
        for key, tensor in checkpoint["student"].items():
            if "adapter" not in key:
                continue
            if key not in merged_state:
                continue
            if tuple(tensor.shape) != tuple(merged_state[key].shape):
                continue
            adapter_sums[key] = adapter_sums.get(key, torch.zeros_like(merged_state[key])) + tensor.detach().cpu().to(
                dtype=merged_state[key].dtype
            ) * float(weight)
            adapter_sources[key] = adapter_sources.get(key, 0) + 1
    for key, tensor in adapter_sums.items():
        merged_state[key] = tensor / float(total_weight)

    payload = {
        "config": asdict(config),
        "student": merged_state,
        "step": int(max(torch.load(path, map_location="cpu").get("step") or 0 for _weight, path in weighted_adapters)),
        "base_checkpoint": str(args.base),
        "adapter_checkpoints": [{"weight": weight, "path": str(path)} for weight, path in weighted_adapters],
        "merged_adapter_tensors": len(adapter_sums),
        "adapter_sources": adapter_sources,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        {
            "output": str(output),
            "base": str(args.base),
            "adapters": len(weighted_adapters),
            "merged_adapter_tensors": len(adapter_sums),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )


if __name__ == "__main__":
    main()
