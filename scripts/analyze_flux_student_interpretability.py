#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F

TRANSFORMER_10 = Path("/data/transformer_10")
if TRANSFORMER_10.exists():
    sys.path.insert(0, str(TRANSFORMER_10))

try:
    from interpret.activation_cache import CaptureSpec
    from interpret.tracer import ActivationTracer
except Exception:  # pragma: no cover - optional local toolkit
    class CaptureSpec:  # type: ignore[no-redef]
        def __init__(self, move_to_cpu: bool = True, dtype: torch.dtype | None = None) -> None:
            self.move_to_cpu = move_to_cpu
            self.dtype = dtype

    class _SimpleCache:
        def __init__(self, spec: CaptureSpec) -> None:
            self.spec = spec
            self.store: dict[str, torch.Tensor] = {}

        def put(self, key: str, value: torch.Tensor) -> None:
            value = value.detach().clone()
            if self.spec.dtype is not None:
                value = value.to(self.spec.dtype)
            if self.spec.move_to_cpu:
                value = value.cpu()
            self.store[key] = value

        def items(self):
            return self.store.items()

    class ActivationTracer:  # type: ignore[no-redef]
        def __init__(self, model: torch.nn.Module, *, spec: CaptureSpec) -> None:
            self.model = model
            self.spec = spec
            self.names: list[str] = []
            self.handles = []
            self.cache = _SimpleCache(spec)

        def add_modules(self, names: list[str]) -> None:
            self.names.extend(names)

        def trace(self):
            tracer = self

            class _Context:
                def __enter__(self):
                    modules = dict(tracer.model.named_modules())
                    for name in tracer.names:
                        module = modules.get(name)
                        if module is None:
                            continue

                        def hook(_module, _inputs, output, *, key=name):
                            if isinstance(output, torch.Tensor):
                                tracer.cache.put(key, output)

                        tracer.handles.append(module.register_forward_hook(hook))
                    return tracer.cache

                def __exit__(self, _exc_type, _exc, _tb):
                    for handle in tracer.handles:
                        handle.remove()
                    tracer.handles.clear()

            return _Context()

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    build_sequences,
    load_embedding,
    load_target,
    read_line_file,
    sequence_id_aliases,
    teacher_step_delta,
)


def load_prompt_set(path: str) -> set[str]:
    if not path:
        return set()
    prompts: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt = str(row.get("prompt", "")).strip()
            except json.JSONDecodeError:
                prompt = line.strip()
            if prompt:
                prompts.add(prompt)
    return prompts


def bucket_sequences(
    sequences: list[list[dict[str, Any]]],
    anchor_prompts: set[str],
    anchor_keys: set[str],
    limit_per_bucket: int,
    seed: int,
) -> dict[str, list[list[dict[str, Any]]]]:
    buckets = {"anchor": [], "general": []}
    for sequence in sequences:
        prompt = str(sequence[0].get("prompt", ""))
        bucket = "anchor" if prompt in anchor_prompts or bool(sequence_id_aliases(sequence) & anchor_keys) else "general"
        buckets[bucket].append(sequence)
    rng = random.Random(seed)
    for key, values in buckets.items():
        rng.shuffle(values)
        buckets[key] = values[: max(int(limit_per_bucket), 0)]
    return buckets


def lowfreq_mse(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    tokens = pred.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return F.mse_loss(pred.float(), target.float())
    pred_grid = pred.float().transpose(1, 2).reshape(pred.shape[0], pred.shape[2], side, side)
    target_grid = target.float().transpose(1, 2).reshape(target.shape[0], target.shape[2], side, side)
    pred_low = F.avg_pool2d(pred_grid, kernel_size=pool, stride=pool)
    target_low = F.avg_pool2d(target_grid, kernel_size=pool, stride=pool)
    return F.mse_loss(pred_low, target_low)


def cosine_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.float().flatten(1)
    target_flat = target.float().flatten(1)
    return F.cosine_similarity(pred_flat, target_flat, dim=1).mean()


def tensor_norm_ratio(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return delta.float().flatten(1).norm(dim=1).mean() / reference.float().flatten(1).norm(dim=1).mean().clamp_min(1e-8)


@torch.inference_mode()
def trace_first_step(
    student: FluxPackedStudent,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor,
) -> dict[str, float]:
    if ActivationTracer is None or CaptureSpec is None:
        return {}
    tracer = ActivationTracer(student, spec=CaptureSpec(move_to_cpu=True, dtype=torch.float32))
    tracer.add_modules([f"blocks.{idx}" for idx in range(len(student.blocks))])
    with tracer.trace() as cache:
        _ = student(latents, timestep, prompt_embeds, pooled_prompt_embeds, guidance)
    norms = {}
    for key, value in cache.items():
        norms[f"{key}.output_norm"] = float(value.float().norm(dim=-1).mean().item())
        norms[f"{key}.output_std"] = float(value.float().std(unbiased=False).item())
    return norms


@torch.inference_mode()
def analyze_sequence(
    student: FluxPackedStudent,
    sequence: list[dict[str, Any]],
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
) -> dict[str, Any]:
    prompt = str(sequence[0].get("prompt", ""))
    prompt_embeds, pooled_prompt_embeds = load_embedding(sequence[0], device)
    current = load_target(sequence[0], device)
    current_latents = current["latents"].float()
    guidance = current["guidance"].reshape(1)

    first_pred = student(
        current_latents,
        current["timestep"].reshape(1).float(),
        prompt_embeds.float(),
        pooled_prompt_embeds.float(),
        guidance.float(),
    )
    zero_pred = student(
        current_latents,
        current["timestep"].reshape(1).float(),
        torch.zeros_like(prompt_embeds).float(),
        torch.zeros_like(pooled_prompt_embeds).float(),
        guidance.float(),
    )
    trace = trace_first_step(
        student,
        current_latents,
        current["timestep"].reshape(1).float(),
        prompt_embeds.float(),
        pooled_prompt_embeds.float(),
        guidance.float(),
    )

    flow_cos = []
    flow_mse = []
    latent_mse = []
    lowfreq_latent = []
    final_teacher = None
    actual_rollout_len = min(int(rollout_len), len(sequence) - 1)
    for offset in range(actual_rollout_len):
        row = sequence[offset]
        next_row = sequence[offset + 1]
        teacher_current = load_target(row, device)
        teacher_next = load_target(next_row, device)
        pred = student(
            current_latents,
            teacher_current["timestep"].reshape(1).float(),
            prompt_embeds.float(),
            pooled_prompt_embeds.float(),
            guidance.float(),
        )
        delta = teacher_step_delta(
            teacher_current["latents"],
            teacher_next["latents"],
            teacher_current["teacher_target"],
        )
        current_latents = current_latents + delta * pred.float()
        final_teacher = teacher_next["latents"].float()
        flow_cos.append(float(cosine_alignment(pred, teacher_current["teacher_target"]).item()))
        flow_mse.append(float(F.mse_loss(pred.float(), teacher_current["teacher_target"].float()).item()))
        latent_mse.append(float(F.mse_loss(current_latents.float(), final_teacher).item()))
        lowfreq_latent.append(float(lowfreq_mse(current_latents, final_teacher, lowfreq_pool).item()))

    endpoint_mse = float(F.mse_loss(current_latents.float(), final_teacher).item()) if final_teacher is not None else 0.0
    endpoint_lowfreq_mse = float(lowfreq_mse(current_latents, final_teacher, lowfreq_pool).item()) if final_teacher is not None else 0.0
    return {
        "prompt": prompt,
        "rollout_len": actual_rollout_len,
        "first_step_prompt_influence_ratio": float(tensor_norm_ratio(first_pred - zero_pred, first_pred).item()),
        "first_step_prompt_cos_gain": float(
            (
                cosine_alignment(first_pred, current["teacher_target"])
                - cosine_alignment(zero_pred, current["teacher_target"])
            ).item()
        ),
        "flow_cos_mean": float(sum(flow_cos) / max(len(flow_cos), 1)),
        "flow_mse_mean": float(sum(flow_mse) / max(len(flow_mse), 1)),
        "latent_mse_mean": float(sum(latent_mse) / max(len(latent_mse), 1)),
        "lowfreq_latent_mse_mean": float(sum(lowfreq_latent) / max(len(lowfreq_latent), 1)),
        "endpoint_mse": endpoint_mse,
        "endpoint_lowfreq_mse": endpoint_lowfreq_mse,
        "trace": trace,
    }


def mean_dict(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = float(sum(values) / max(len(values), 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretability probes for FLUX packed-latent student coherence.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--extra-target-dir", action="append", default=[])
    parser.add_argument("--anchor-prompt-file", default="")
    parser.add_argument("--anchor-key-file", default="")
    parser.add_argument("--sequence-include-key-file", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-bucket", type=int, default=8)
    parser.add_argument("--rollout-len", type=int, default=24)
    parser.add_argument("--lowfreq-pool", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    student = FluxPackedStudent(config).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()

    target_dirs = [Path(args.target_dir)] + [Path(value) for value in args.extra_target_dir]
    include_keys = read_line_file(args.sequence_include_key_file)
    sequences = build_sequences(target_dirs, include_keys=include_keys or None)
    anchor_prompts = load_prompt_set(args.anchor_prompt_file)
    anchor_keys = read_line_file(args.anchor_key_file)
    buckets = bucket_sequences(sequences, anchor_prompts, anchor_keys, args.limit_per_bucket, args.seed)

    results = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step") or 0),
        "config": asdict(config),
        "buckets": {},
    }
    summary_keys = [
        "first_step_prompt_influence_ratio",
        "first_step_prompt_cos_gain",
        "flow_cos_mean",
        "flow_mse_mean",
        "latent_mse_mean",
        "lowfreq_latent_mse_mean",
        "endpoint_mse",
        "endpoint_lowfreq_mse",
    ]
    for bucket, bucket_sequences_ in buckets.items():
        rows = [
            analyze_sequence(student, sequence, device, args.rollout_len, args.lowfreq_pool)
            for sequence in bucket_sequences_
        ]
        results["buckets"][bucket] = {
            "count": len(rows),
            "summary": mean_dict(rows, summary_keys),
            "rows": rows,
        }

    text = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
