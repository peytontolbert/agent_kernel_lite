#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch


BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def load_student(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("student")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a student state dict")
    return state, checkpoint


def family_name(name: str) -> str:
    match = BLOCK_RE.match(name)
    if not match:
        return name.split(".", 1)[0]
    layer = int(match.group(1))
    rest = match.group(2)
    part = rest.split(".", 1)[0]
    return f"blocks.{layer:02d}.{part}"


def coarse_family(name: str) -> str:
    match = BLOCK_RE.match(name)
    if not match:
        return name.split(".", 1)[0]
    return f"blocks.{match.group(2).split('.', 1)[0]}"


def tensor_stats(name: str, base: torch.Tensor, current: torch.Tensor) -> dict[str, Any]:
    base_f = base.float()
    current_f = current.float()
    delta = current_f - base_f
    delta_norm = float(delta.norm().item())
    base_norm = float(base_f.norm().item())
    current_norm = float(current_f.norm().item())
    rel = delta_norm / max(base_norm, 1e-12)
    return {
        "name": name,
        "shape": list(base.shape),
        "numel": int(base.numel()),
        "delta_l2": delta_norm,
        "base_l2": base_norm,
        "current_l2": current_norm,
        "relative_l2": rel,
        "delta_mean_abs": float(delta.abs().mean().item()),
        "delta_max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "family": family_name(name),
        "coarse_family": coarse_family(name),
    }


def add_group(groups: dict[str, dict[str, Any]], key: str, row: dict[str, Any]) -> None:
    group = groups.setdefault(
        key,
        {
            "name": key,
            "tensors": 0,
            "numel": 0,
            "delta_l2_sq": 0.0,
            "base_l2_sq": 0.0,
            "delta_mean_abs_weighted": 0.0,
            "max_relative_l2": 0.0,
            "max_delta_tensor": "",
        },
    )
    group["tensors"] += 1
    group["numel"] += int(row["numel"])
    group["delta_l2_sq"] += float(row["delta_l2"]) ** 2
    group["base_l2_sq"] += float(row["base_l2"]) ** 2
    group["delta_mean_abs_weighted"] += float(row["delta_mean_abs"]) * int(row["numel"])
    if float(row["relative_l2"]) > float(group["max_relative_l2"]):
        group["max_relative_l2"] = float(row["relative_l2"])
        group["max_delta_tensor"] = str(row["name"])


def finish_groups(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups.values():
        delta_l2 = group.pop("delta_l2_sq") ** 0.5
        base_l2 = group.pop("base_l2_sq") ** 0.5
        weighted = group.pop("delta_mean_abs_weighted")
        group["delta_l2"] = float(delta_l2)
        group["base_l2"] = float(base_l2)
        group["relative_l2"] = float(delta_l2 / max(base_l2, 1e-12))
        group["delta_mean_abs"] = float(weighted / max(int(group["numel"]), 1))
        rows.append(group)
    return sorted(rows, key=lambda row: float(row["delta_l2"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report where a FLUX packed student moved between checkpoints.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    base_state, base_checkpoint = load_student(Path(args.base))
    current_state, current_checkpoint = load_student(Path(args.current))
    common = sorted(set(base_state) & set(current_state))
    if not common:
        raise ValueError("checkpoints have no common student tensors")

    missing_from_current = sorted(set(base_state) - set(current_state))
    missing_from_base = sorted(set(current_state) - set(base_state))
    tensor_rows = []
    family_groups: dict[str, dict[str, Any]] = {}
    coarse_groups: dict[str, dict[str, Any]] = {}
    total_delta_sq = 0.0
    total_base_sq = 0.0
    compared_numel = 0
    skipped_shape_mismatch = []
    for name in common:
        base = base_state[name]
        current = current_state[name]
        if not isinstance(base, torch.Tensor) or not isinstance(current, torch.Tensor):
            continue
        if base.shape != current.shape:
            skipped_shape_mismatch.append({"name": name, "base_shape": list(base.shape), "current_shape": list(current.shape)})
            continue
        row = tensor_stats(name, base, current)
        tensor_rows.append(row)
        add_group(family_groups, row["family"], row)
        add_group(coarse_groups, row["coarse_family"], row)
        total_delta_sq += float(row["delta_l2"]) ** 2
        total_base_sq += float(row["base_l2"]) ** 2
        compared_numel += int(row["numel"])

    by_delta = sorted(tensor_rows, key=lambda row: float(row["delta_l2"]), reverse=True)
    by_relative = sorted(tensor_rows, key=lambda row: float(row["relative_l2"]), reverse=True)
    family_rows = finish_groups(family_groups)
    coarse_rows = finish_groups(coarse_groups)
    output = {
        "base": str(args.base),
        "current": str(args.current),
        "base_step": int(base_checkpoint.get("step") or 0),
        "current_step": int(current_checkpoint.get("step") or 0),
        "compared_tensors": len(tensor_rows),
        "compared_numel": compared_numel,
        "global_delta_l2": float(total_delta_sq**0.5),
        "global_base_l2": float(total_base_sq**0.5),
        "global_relative_l2": float((total_delta_sq**0.5) / max(total_base_sq**0.5, 1e-12)),
        "top_tensors_by_delta_l2": by_delta[: args.top_k],
        "top_tensors_by_relative_l2": by_relative[: args.top_k],
        "top_families_by_delta_l2": family_rows[: args.top_k],
        "top_coarse_families_by_delta_l2": coarse_rows[: args.top_k],
        "missing_from_current": missing_from_current[: args.top_k],
        "missing_from_base": missing_from_base[: args.top_k],
        "skipped_shape_mismatch": skipped_shape_mismatch[: args.top_k],
    }
    text = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
