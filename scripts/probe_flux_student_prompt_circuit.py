#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from sample_agentkernel_lite_image_flux_flow_distill import (
    DEFAULT_FLUX_TEACHER,
    load_prompt_condition_aliases,
    load_prompt_seed_aliases,
    load_student,
    stable_prompt_seed,
)
from train_agentkernel_lite_image_flux_live_teacher_trajectory_reuse import (
    batched_step_delta,
    build_teacher_trajectory,
    token_drop_negative_prompt,
)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a.detach().float(), b.detach().float()).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().float().flatten(1), b.detach().float().flatten(1), dim=1).mean().item())


def module_candidates(student: torch.nn.Module, pattern: str) -> list[str]:
    names: list[str] = []
    for block_index, block in enumerate(getattr(student, "blocks", [])):
        for part in ("self_attn", "cross_attn", "mlp"):
            name = f"blocks.{block_index}.{part}"
            if pattern == "all" or part in pattern.split(","):
                if hasattr(block, part):
                    names.append(name)
    return names


def tensor_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def replace_tensor_output(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement.to(device=output.device, dtype=output.dtype)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        values = list(output)
        values[0] = replacement.to(device=values[0].device, dtype=values[0].dtype)
        return tuple(values)
    return output


def parse_step_windows(spec: str, total_steps: int) -> list[tuple[str, set[int] | None]]:
    windows: list[tuple[str, set[int] | None]] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            label, value = item.split(":", 1)
            label = label.strip() or value.strip()
        else:
            label = item
            value = item
        value = value.strip()
        if value in {"*", "all"}:
            windows.append((label, None))
            continue
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = max(0, int(start_text))
            end = min(total_steps - 1, int(end_text))
            if end < start:
                continue
            windows.append((label, set(range(start, end + 1))))
            continue
        step = int(value)
        if 0 <= step < total_steps:
            windows.append((label, {step}))
    return windows or [("all", None)]


@torch.inference_mode()
def rollout(
    student: torch.nn.Module,
    trajectory: list[dict[str, torch.Tensor]],
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor,
    device: torch.device,
    *,
    capture_modules: list[str] | None = None,
    patch_modules: list[str] | None = None,
    patch_step_set: set[int] | None = None,
    clean_cache: dict[tuple[str, int], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[tuple[str, int], torch.Tensor]]:
    modules = dict(student.named_modules())
    capture_modules = capture_modules or []
    patch_modules = patch_modules or []
    captured: dict[tuple[str, int], torch.Tensor] = {}
    counters: dict[str, int] = {name: 0 for name in set(capture_modules) | set(patch_modules)}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_capture_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            index = counters[name]
            counters[name] = index + 1
            value = tensor_output(output)
            if value is not None:
                captured[(name, index)] = value.detach().to("cpu")
            return output

        return hook

    def make_patch_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            index = counters[name]
            counters[name] = index + 1
            if patch_step_set is not None and index not in patch_step_set:
                return output
            replacement = None if clean_cache is None else clean_cache.get((name, index))
            if replacement is None:
                return output
            return replace_tensor_output(output, replacement)

        return hook

    for name in capture_modules:
        handles.append(modules[name].register_forward_hook(make_capture_hook(name)))
    for name in patch_modules:
        handles.append(modules[name].register_forward_hook(make_patch_hook(name)))
    try:
        latents = trajectory[0]["latents"].to(device=device, dtype=torch.float32)
        for item in trajectory:
            timestep = item["timestep"].reshape(1).to(device=device, dtype=torch.float32)
            pred = student(
                latents,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                guidance,
            )
            teacher_current = item["latents"].to(device=device, dtype=torch.float32)
            teacher_next = item["teacher_next"].to(device=device, dtype=torch.float32)
            teacher_target = item["teacher_target"].to(device=device, dtype=torch.float32)
            delta = batched_step_delta(teacher_current, teacher_next, teacher_target)
            latents = latents + delta * pred.float()
    finally:
        for handle in handles:
            handle.remove()
    return latents.float(), captured


def encode_prompt(pipe: Any, prompt: str, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )
    return prompt_embeds.to(device=device, dtype=torch.float32), pooled_prompt_embeds.to(device=device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt circuit recovery probe for FLUX packed-latent student.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("raw", "ema", "materialized"), default="raw")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--prompt-hash-seeds", action="store_true")
    parser.add_argument("--prompt-seed-alias-file", default="")
    parser.add_argument("--prompt-condition-alias-file", default="")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--seed-min", type=int, default=90000000)
    parser.add_argument("--seed-max", type=int, default=99999999)
    parser.add_argument("--drop-count", type=int, default=2)
    parser.add_argument("--candidate-parts", default="cross_attn")
    parser.add_argument(
        "--patch-step-windows",
        default="all:*",
        help="Comma-separated patch windows. Examples: all:*,early:0-7,mid:8-15,late:16-23",
    )
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher_args = argparse.Namespace(
        teacher_model=args.teacher_model,
        teacher_family="flux",
        dtype=args.dtype,
        variant="",
        local_files_only=args.local_files_only,
        quantize_transformer_4bit=args.quantize_transformer_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        cpu_offload=args.cpu_offload,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    pipe = load_teacher(teacher_args)
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    student, config = load_student(Path(args.checkpoint), device, args.weights)
    student.eval()

    prompt_seed_aliases = load_prompt_seed_aliases(args.prompt_seed_alias_file)
    prompt_condition_aliases = load_prompt_condition_aliases(args.prompt_condition_alias_file)
    candidates = module_candidates(student, args.candidate_parts)
    prompts = read_prompts(Path(args.prompts), args.limit)
    output: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "config": config.__dict__,
        "candidate_modules": candidates,
        "prompts": [],
    }
    guidance = torch.full([1], float(args.guidance), device=device, dtype=torch.float32)
    for index, prompt in enumerate(prompts):
        condition_prompt = prompt_condition_aliases.get(prompt, prompt)
        seed_prompt = prompt_seed_aliases.get(prompt, prompt)
        seed = stable_prompt_seed(seed_prompt, int(args.seed_min), int(args.seed_max)) if args.prompt_hash_seeds else int(args.seed) + index
        corrupted_prompt = token_drop_negative_prompt(condition_prompt, drop_count=int(args.drop_count))
        trajectory = build_teacher_trajectory(pipe, condition_prompt, int(seed), args, device)["trajectory"]
        step_windows = parse_step_windows(args.patch_step_windows, len(trajectory))
        clean_prompt_embeds, clean_pooled = encode_prompt(pipe, condition_prompt, args, device)
        corrupted_prompt_embeds, corrupted_pooled = encode_prompt(pipe, corrupted_prompt, args, device)
        clean_terminal, clean_cache = rollout(
            student,
            trajectory,
            clean_prompt_embeds,
            clean_pooled,
            guidance,
            device,
            capture_modules=candidates,
        )
        corrupted_terminal, _ = rollout(
            student,
            trajectory,
            corrupted_prompt_embeds,
            corrupted_pooled,
            guidance,
            device,
        )
        clean_corrupted_mse = F.mse_loss(clean_terminal, corrupted_terminal).clamp_min(1e-8)
        rows: list[dict[str, Any]] = []
        for module_name in candidates:
            for window_label, patch_step_set in step_windows:
                patched_terminal, _ = rollout(
                    student,
                    trajectory,
                    corrupted_prompt_embeds,
                    corrupted_pooled,
                    guidance,
                    device,
                    patch_modules=[module_name],
                    patch_step_set=patch_step_set,
                    clean_cache=clean_cache,
                )
                patched_mse = F.mse_loss(clean_terminal, patched_terminal)
                recovery = float(((clean_corrupted_mse - patched_mse) / clean_corrupted_mse).detach().cpu().item())
                rows.append(
                    {
                        "module": module_name,
                        "step_window": window_label,
                        "recovery": recovery,
                        "patched_mse_to_clean": float(patched_mse.detach().cpu().item()),
                        "patched_cosine_to_clean": cosine(clean_terminal, patched_terminal),
                    }
                )
        rows.sort(key=lambda row: float(row["recovery"]), reverse=True)
        output["prompts"].append(
            {
                "prompt": prompt,
                "condition_prompt": condition_prompt,
                "corrupted_prompt": corrupted_prompt,
                "seed": int(seed),
                "clean_corrupted_mse": float(clean_corrupted_mse.detach().cpu().item()),
                "clean_corrupted_cosine": cosine(clean_terminal, corrupted_terminal),
                "module_phase_rows": rows,
                "top_modules": rows[: min(12, len(rows))],
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "prompts": len(output["prompts"]), "candidates": len(candidates)}), flush=True)


if __name__ == "__main__":
    main()
