#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_rows(target_dir: Path) -> list[dict]:
    metadata_path = target_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if "target_path" in row]


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate balanced packed-channel relevance from FLUX flow targets.")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--late-start-index", type=int, default=14)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-weight", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=2.0)
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    rows = load_rows(target_dir)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError(f"no rows in {target_dir}")

    target_sum = None
    target_sq_sum = None
    late_energy_sum = None
    latent_late_sq_sum = None
    count = 0
    late_count = 0

    for row in rows:
        payload = torch.load(target_dir / row["target_path"], map_location="cpu")
        target = payload["teacher_target"].float().reshape(-1, payload["teacher_target"].shape[-1])
        latent = payload["latents"].float().reshape(-1, payload["latents"].shape[-1])
        channel_sum = target.mean(dim=0)
        channel_sq_sum = target.square().mean(dim=0)
        if target_sum is None:
            target_sum = torch.zeros_like(channel_sum)
            target_sq_sum = torch.zeros_like(channel_sq_sum)
            late_energy_sum = torch.zeros_like(channel_sq_sum)
            latent_late_sq_sum = torch.zeros_like(channel_sq_sum)
        target_sum += channel_sum
        target_sq_sum += channel_sq_sum
        count += 1
        if int(row.get("timestep_index", 0)) >= int(args.late_start_index):
            late_energy_sum += channel_sq_sum
            latent_late_sq_sum += latent.square().mean(dim=0)
            late_count += 1

    assert target_sum is not None and target_sq_sum is not None and late_energy_sum is not None
    mean = target_sum / max(count, 1)
    controllability = (target_sq_sum / max(count, 1) - mean.square()).clamp_min(0.0)
    observability = late_energy_sum / max(late_count, 1)
    if latent_late_sq_sum is not None and late_count:
        observability = observability + 0.25 * (latent_late_sq_sum / late_count)
    balanced = torch.sqrt(controllability * observability).clamp_min(1e-12)
    weights = balanced / balanced.mean().clamp_min(1e-12)
    weights = weights.clamp(float(args.min_weight), float(args.max_weight))

    report = {
        "target_dir": str(target_dir),
        "rows": int(count),
        "late_rows": int(late_count),
        "late_start_index": int(args.late_start_index),
        "min_weight": float(weights.min().item()),
        "max_weight": float(weights.max().item()),
        "mean_weight": float(weights.mean().item()),
        "top_channels": [
            {"channel": int(index), "weight": float(weights[index].item())}
            for index in torch.argsort(weights, descending=True)[:12]
        ],
        "weights": [float(value) for value in weights.tolist()],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "late_rows", "min_weight", "max_weight", "mean_weight")}))


if __name__ == "__main__":
    main()
