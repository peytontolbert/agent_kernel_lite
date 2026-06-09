#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig


def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    x = t.detach().float().reshape(-1)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(x.abs().mean().item()),
        "max_abs": float(x.abs().max().item()),
    }


def load_student(path: Path) -> tuple[FluxPackedStudent, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = FluxPackedStudent(FluxPackedStudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["student"], strict=True)
    model.eval()
    return model, ckpt


def gate_report(model: FluxPackedStudent) -> list[dict]:
    rows = []
    for index, block in enumerate(model.blocks):
        gates = block.residual_gates.detach().float()
        rows.append(
            {
                "block": index,
                "attn_gate": float(gates[0].item()),
                "cross_gate": float(gates[1].item()),
                "mlp_gate": float(gates[2].item()),
                "gate_abs_mean": float(gates.abs().mean().item()),
            }
        )
    return rows


def state_delta_report(current: dict[str, torch.Tensor], base: dict[str, torch.Tensor], limit: int) -> list[dict]:
    rows = []
    for name, cur in current.items():
        old = base.get(name)
        if old is None or cur.shape != old.shape or not cur.is_floating_point():
            continue
        delta = cur.float() - old.float()
        denom = old.float().norm().clamp_min(1e-8)
        rows.append(
            {
                "name": name,
                "delta_norm": float(delta.norm().item()),
                "relative_delta": float((delta.norm() / denom).item()),
                "delta_abs_mean": float(delta.abs().mean().item()),
                "weight_abs_mean": float(cur.float().abs().mean().item()),
            }
        )
    rows.sort(key=lambda row: row["relative_delta"], reverse=True)
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect learning signals in a FLUX packed student checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--top-deltas", type=int, default=24)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    model, ckpt = load_student(Path(args.checkpoint))
    report = {
        "checkpoint": str(args.checkpoint),
        "step": int(ckpt.get("step") or 0),
        "loss": float(ckpt.get("loss") or 0.0),
        "config": ckpt.get("config"),
        "gates": gate_report(model),
    }
    if args.base_checkpoint:
        _base_model, base_ckpt = load_student(Path(args.base_checkpoint))
        report["base_checkpoint"] = str(args.base_checkpoint)
        report["top_state_deltas"] = state_delta_report(
            ckpt["student"],
            base_ckpt["student"],
            int(args.top_deltas),
        )
    text = json.dumps(report, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
