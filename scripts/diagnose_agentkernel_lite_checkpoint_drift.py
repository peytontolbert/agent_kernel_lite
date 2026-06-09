#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {str(name): tensor for name, tensor in value.items() if isinstance(tensor, torch.Tensor)}
        return {str(name): tensor for name, tensor in checkpoint.items() if isinstance(tensor, torch.Tensor)}
    raise TypeError(f"unsupported checkpoint type at {path}: {type(checkpoint)!r}")


def _group_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in {"encoder", "decoder"} and parts[1].isdigit():
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


def summarize_drift(base_path: Path, candidate_path: Path, *, top_k: int) -> dict[str, Any]:
    base = _load_state(base_path)
    candidate = _load_state(candidate_path)
    rows: list[dict[str, Any]] = []
    group_totals: dict[str, dict[str, float]] = {}
    compared = 0
    skipped: list[str] = []
    for name, base_tensor in base.items():
        cand_tensor = candidate.get(name)
        if not isinstance(cand_tensor, torch.Tensor) or tuple(cand_tensor.shape) != tuple(base_tensor.shape):
            skipped.append(name)
            continue
        if not torch.is_floating_point(base_tensor):
            continue
        compared += 1
        base_f = base_tensor.float()
        cand_f = cand_tensor.float()
        delta = cand_f - base_f
        delta_l2 = float(delta.norm().item())
        base_l2 = float(base_f.norm().item())
        rel = float(delta_l2 / max(base_l2, 1e-12))
        max_abs = float(delta.abs().max().item())
        rows.append(
            {
                "name": name,
                "shape": list(base_tensor.shape),
                "delta_l2": delta_l2,
                "base_l2": base_l2,
                "relative_delta": rel,
                "max_abs_delta": max_abs,
            }
        )
        group = group_totals.setdefault(_group_name(name), {"delta_l2_sq": 0.0, "base_l2_sq": 0.0, "max_abs_delta": 0.0, "tensors": 0})
        group["delta_l2_sq"] += delta_l2 * delta_l2
        group["base_l2_sq"] += base_l2 * base_l2
        group["max_abs_delta"] = max(group["max_abs_delta"], max_abs)
        group["tensors"] += 1
    top = sorted(rows, key=lambda row: (row["relative_delta"], row["max_abs_delta"]), reverse=True)[: int(top_k)]
    groups = []
    for name, values in group_totals.items():
        delta_l2 = values["delta_l2_sq"] ** 0.5
        base_l2 = values["base_l2_sq"] ** 0.5
        groups.append(
            {
                "name": name,
                "tensors": int(values["tensors"]),
                "delta_l2": delta_l2,
                "base_l2": base_l2,
                "relative_delta": float(delta_l2 / max(base_l2, 1e-12)),
                "max_abs_delta": values["max_abs_delta"],
            }
        )
    groups = sorted(groups, key=lambda row: (row["relative_delta"], row["max_abs_delta"]), reverse=True)[: int(top_k)]
    return {
        "base_checkpoint": str(base_path.resolve()),
        "candidate_checkpoint": str(candidate_path.resolve()),
        "compared_float_tensors": compared,
        "skipped_tensors": skipped[:50],
        "top_tensor_drift": top,
        "top_group_drift": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = summarize_drift(
        Path(args.base_checkpoint).expanduser().resolve(),
        Path(args.candidate_checkpoint).expanduser().resolve(),
        top_k=int(args.top_k),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
