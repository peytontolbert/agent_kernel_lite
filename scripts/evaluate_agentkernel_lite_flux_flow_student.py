#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean, median
from typing import Any

import torch
import torch.nn.functional as F

from train_agentkernel_lite_image_flux_flow_distill import (
    FluxPackedStudent,
    FluxPackedStudentConfig,
    load_batch,
    load_rows,
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def load_checkpoint(path: Path, device: torch.device) -> tuple[FluxPackedStudent, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    model = FluxPackedStudent(config).to(device)
    state = checkpoint.get("student_materialized") or checkpoint.get("student") or checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"resume_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    rows = load_rows(Path(args.target_dir))
    rng = random.Random(args.seed)
    if args.limit and len(rows) > args.limit:
        rows = rng.sample(rows, args.limit)

    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    losses: list[float] = []
    for index in range(0, len(rows), args.batch_size):
        batch_rows = rows[index : index + args.batch_size]
        latents, timesteps, targets, prompt_embeds, pooled_prompt_embeds, guidance = load_batch(Path(args.target_dir), batch_rows, device)
        pred = model(latents, timesteps, prompt_embeds, pooled_prompt_embeds, guidance)
        per_item = F.mse_loss(pred.float(), targets.float(), reduction="none").flatten(1).mean(dim=1)
        losses.extend(float(value) for value in per_item.detach().cpu())

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step") or 0),
        "target_dir": str(args.target_dir),
        "rows": len(rows),
        "seed": args.seed,
        "loss_mean": mean(losses) if losses else 0.0,
        "loss_median": median(losses) if losses else 0.0,
        "loss_p90": percentile(losses, 0.90),
        "loss_max": max(losses) if losses else 0.0,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FLUX flow student checkpoint on a fixed target-row probe.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
