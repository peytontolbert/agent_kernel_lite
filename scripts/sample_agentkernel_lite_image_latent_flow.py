#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import torch
from torchvision import utils


def import_student_modules():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_agentkernel_lite_image_latent_flow import LatentFlowConfig, LatentFlowDiT, TinyAutoencoder

    return LatentFlowConfig, LatentFlowDiT, TinyAutoencoder


def prompt_ids(prompt: str, *, vocab_size: int, tokens: int, device: torch.device) -> torch.Tensor | None:
    if vocab_size <= 0 or tokens <= 0:
        return None
    words = re.findall(r"[a-z0-9]+", prompt.lower())[:tokens]
    if not words:
        words = ["image"]
    ids = []
    for word in words:
        value = 2166136261
        for char in word:
            value ^= ord(char)
            value = (value * 16777619) & 0xFFFFFFFF
        ids.append(value % vocab_size)
    while len(ids) < tokens:
        ids.append(0)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def sample(autoencoder, flow, config, *, prompt: str, batch_size: int, steps: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(
        batch_size,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        device=device,
        generator=generator,
    )
    ids = prompt_ids(prompt, vocab_size=config.prompt_vocab_size, tokens=config.prompt_tokens, device=device)
    if ids is not None:
        ids = ids.expand(batch_size, -1)
    for index in range(steps):
        t = torch.full((batch_size,), index / max(steps - 1, 1), device=device)
        v = flow(z, t, None, ids)
        z = z + v / steps
    return autoencoder.decode(z).clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample an Agent Kernel Lite latent-flow image checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--prompt", default="a realistic photo")
    args = parser.parse_args()

    LatentFlowConfig, LatentFlowDiT, TinyAutoencoder = import_student_modules()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = LatentFlowConfig(**checkpoint["config"])
    autoencoder = TinyAutoencoder(config).to(device)
    flow = LatentFlowDiT(config).to(device)
    autoencoder.load_state_dict(checkpoint["autoencoder"])
    flow.load_state_dict(checkpoint["flow"])
    autoencoder.eval()
    flow.eval()
    images = sample(autoencoder, flow, config, prompt=args.prompt, batch_size=args.batch_size, steps=args.steps, seed=args.seed, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(images * 0.5 + 0.5, output, nrow=min(args.batch_size, 4))


if __name__ == "__main__":
    main()
