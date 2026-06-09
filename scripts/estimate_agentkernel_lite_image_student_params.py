#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def import_training_module():
    root = Path(__file__).resolve().parents[1]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from train_agentkernel_lite_image_latent_flow import LatentFlowConfig, LatentFlowDiT, TinyAutoencoder

    return LatentFlowConfig, LatentFlowDiT, TinyAutoencoder


def count_params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Agent Kernel Lite latent-flow image student parameter counts.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--class-count", type=int, default=0)
    parser.add_argument("--bitlinear", action="store_true")
    args = parser.parse_args()
    LatentFlowConfig, LatentFlowDiT, TinyAutoencoder = import_training_module()
    config = LatentFlowConfig(
        image_size=args.image_size,
        downsample=args.downsample,
        latent_channels=args.latent_channels,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        class_count=args.class_count,
        bitlinear=args.bitlinear,
    )
    autoencoder = TinyAutoencoder(config)
    flow = LatentFlowDiT(config)
    autoencoder_params = count_params(autoencoder)
    flow_params = count_params(flow)
    print(json.dumps({
        "config": config.__dict__,
        "autoencoder_params": autoencoder_params,
        "flow_params": flow_params,
        "total_params": autoencoder_params + flow_params,
        "flow_ternary_weight_estimate_mb": round(flow_params * 1.58 / 8 / 1024 / 1024, 2),
        "total_fp16_estimate_mb": round((autoencoder_params + flow_params) * 2 / 1024 / 1024, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
