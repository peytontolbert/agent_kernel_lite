#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    build_sequences,
    load_embedding,
    load_target,
    packed_lowfreq_loss,
    read_prompt_file,
    teacher_step_delta,
)


def block_index(parameter_name: str) -> int | None:
    parts = parameter_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def grad_norms_by_block(model: torch.nn.Module) -> dict[str, float]:
    sums: dict[int, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        idx = block_index(name)
        if idx is None or parameter.grad is None:
            continue
        sums[idx] += float(parameter.grad.detach().float().pow(2).sum().item())
    return {str(idx): value**0.5 for idx, value in sorted(sums.items())}


def top_blocks(norms: dict[str, float], topk: int) -> list[dict[str, float | int]]:
    rows = sorted(((int(key), float(value)) for key, value in norms.items()), key=lambda item: item[1], reverse=True)
    return [{"block": idx, "grad_norm": value} for idx, value in rows[:topk]]


def highfreq_loss(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    full = F.mse_loss(pred.float(), target.float())
    low = packed_lowfreq_loss(pred, target, pool)
    return (full - low).clamp_min(0.0)


def prompt_tags(prompt: str) -> list[str]:
    text = prompt.lower()
    tags = []
    for word in ("dog", "cat", "car", "airplane", "bicycle", "ramen", "apple"):
        if word in text:
            tags.append(f"object:{word}")
    for word in ("red", "blue"):
        if word in text:
            tags.append(f"color:{word}")
    if "bowl" in text or "ramen" in text:
        tags.append("scene:container_food")
    if not tags:
        tags.append("object:unknown")
    return tags


def rollout_endpoint(
    model: FluxPackedStudent,
    sequence: list[dict[str, Any]],
    device: torch.device,
    rollout_len: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    prompt_embeds, pooled_prompt_embeds = load_embedding(sequence[0], device)
    current = load_target(sequence[0], device)
    current_latents = current["latents"].float()
    guidance = current["guidance"].reshape(1).float()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    final_teacher = current_latents
    for offset in range(min(int(rollout_len), len(sequence) - 1)):
        row_payload = load_target(sequence[offset], device)
        next_payload = load_target(sequence[offset + 1], device)
        pred = model(
            current_latents,
            row_payload["timestep"].reshape(1).float(),
            prompt_embeds.float(),
            pooled_prompt_embeds.float(),
            guidance,
        )
        delta = teacher_step_delta(row_payload["latents"], next_payload["latents"], row_payload["teacher_target"])
        current_latents = current_latents + delta * pred.float()
        final_teacher = next_payload["latents"].float()
        preds.append(pred)
        targets.append(row_payload["teacher_target"].float())
    return current_latents, final_teacher, preds, targets


def analyze_sequence(
    model: FluxPackedStudent,
    sequence: list[dict[str, Any]],
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
    topk: int,
) -> dict[str, Any]:
    prompt = str(sequence[0].get("prompt", ""))
    endpoint, teacher_endpoint, preds, targets = rollout_endpoint(model, sequence, device, rollout_len)
    losses = {
        "silhouette_lowfreq": packed_lowfreq_loss(endpoint, teacher_endpoint, lowfreq_pool),
        "texture_highfreq": highfreq_loss(endpoint, teacher_endpoint, lowfreq_pool),
        "endpoint_full": F.mse_loss(endpoint.float(), teacher_endpoint.float()),
        "flow_field": torch.stack([F.mse_loss(pred.float(), target.float()) for pred, target in zip(preds, targets)]).mean()
        if preds
        else torch.zeros((), device=device),
    }
    rows: dict[str, Any] = {
        "prompt": prompt,
        "tags": prompt_tags(prompt),
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
        "top_blocks": {},
    }
    for name, loss in losses.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        norms = grad_norms_by_block(model)
        rows["top_blocks"][name] = top_blocks(norms, topk)
    model.zero_grad(set_to_none=True)
    return rows


def aggregate_by_tag(rows: list[dict[str, Any]], loss_name: str, topk: int) -> dict[str, list[dict[str, float | int]]]:
    by_tag: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        block_rows = row["top_blocks"][loss_name]
        for tag in row["tags"]:
            counts[tag] += 1
            for block_row in block_rows:
                by_tag[tag][int(block_row["block"])] += float(block_row["grad_norm"])
    out = {}
    for tag, scores in by_tag.items():
        averaged = {str(block): value / max(counts[tag], 1) for block, value in scores.items()}
        out[tag] = top_blocks(averaged, topk)
    return out


def flattened_grad_vector(model: torch.nn.Module, name_filter: str) -> torch.Tensor:
    chunks = []
    for name, parameter in model.named_parameters():
        if name_filter and name_filter not in name:
            continue
        if not parameter.requires_grad or parameter.grad is None:
            continue
        chunks.append(parameter.grad.detach().float().flatten().cpu())
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks, dim=0)


def prompt_gradient_cosines(
    model: FluxPackedStudent,
    sequences: list[list[dict[str, Any]]],
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
    loss_name: str,
    name_filter: str,
) -> dict[str, Any]:
    vectors = []
    labels = []
    losses_by_prompt = {}
    for sequence in sequences:
        prompt = str(sequence[0].get("prompt", ""))
        source = Path(str(sequence[0].get("_target_dir", ""))).name
        seed = sequence[0].get("seed", "")
        label = f"{prompt} | {source} | seed={seed}"
        endpoint, teacher_endpoint, preds, targets = rollout_endpoint(model, sequence, device, rollout_len)
        losses = {
            "silhouette_lowfreq": packed_lowfreq_loss(endpoint, teacher_endpoint, lowfreq_pool),
            "texture_highfreq": highfreq_loss(endpoint, teacher_endpoint, lowfreq_pool),
            "endpoint_full": F.mse_loss(endpoint.float(), teacher_endpoint.float()),
            "flow_field": torch.stack([F.mse_loss(pred.float(), target.float()) for pred, target in zip(preds, targets)]).mean()
            if preds
            else torch.zeros((), device=device),
        }
        loss = losses[loss_name]
        model.zero_grad(set_to_none=True)
        loss.backward()
        vector = flattened_grad_vector(model, name_filter)
        norm = vector.norm().clamp_min(1e-12)
        vectors.append(vector / norm)
        labels.append(label)
        losses_by_prompt[label] = float(loss.detach().item())
        model.zero_grad(set_to_none=True)
    matrix = []
    negative_pairs = []
    for i, left in enumerate(vectors):
        row = []
        for j, right in enumerate(vectors):
            cosine = float((left * right).sum().item())
            row.append(cosine)
            if j > i and cosine < 0:
                negative_pairs.append({"a": labels[i], "b": labels[j], "cosine": cosine})
        matrix.append(row)
    return {
        "loss_name": loss_name,
        "parameter_filter": name_filter,
        "prompts": labels,
        "losses": losses_by_prompt,
        "cosine_matrix": matrix,
        "negative_pairs": negative_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Concept-gradient map for the FLUX packed-latent student.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--extra-target-dir", action="append", default=[])
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--rollout-len", type=int, default=23)
    parser.add_argument("--lowfreq-pool", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--grad-cosine-loss", choices=("silhouette_lowfreq", "texture_highfreq", "endpoint_full", "flow_field"), default="")
    parser.add_argument("--grad-cosine-filter", default="adapter")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    model = FluxPackedStudent(config).to(device)
    model.load_state_dict(checkpoint["student"], strict=True)
    model.gradient_checkpointing = True
    model.train()

    include_prompts = read_prompt_file(args.prompt_file)
    target_dirs = [Path(args.target_dir)] + [Path(value) for value in args.extra_target_dir]
    sequences = build_sequences(target_dirs, include_prompts=include_prompts or None)[: max(int(args.limit), 1)]
    rows = [analyze_sequence(model, sequence, device, args.rollout_len, args.lowfreq_pool, args.topk) for sequence in sequences]
    result = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step") or 0),
        "config": asdict(config),
        "rows": rows,
        "tag_summary": {
            loss_name: aggregate_by_tag(rows, loss_name, args.topk)
            for loss_name in ("silhouette_lowfreq", "texture_highfreq", "endpoint_full", "flow_field")
        },
    }
    if args.grad_cosine_loss:
        result["gradient_cosines"] = prompt_gradient_cosines(
            model,
            sequences,
            device,
            args.rollout_len,
            args.lowfreq_pool,
            args.grad_cosine_loss,
            args.grad_cosine_filter,
        )
    text = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
