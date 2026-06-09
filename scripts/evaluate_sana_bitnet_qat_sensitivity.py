#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def import_training_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_agentkernel_lite_image_sana_latent_distill import (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        load_teacher,
        load_cached_latent,
        load_latent_cache_rows,
        make_student,
        materialize_bitnet_qat_weights,
        student_predict,
        teacher_predict,
    )

    return {
        "SanaLatentStudentConfig": SanaLatentStudentConfig,
        "apply_bitnet_qat_modules": apply_bitnet_qat_modules,
        "encode_prompts": encode_prompts,
        "load_teacher": load_teacher,
        "load_cached_latent": load_cached_latent,
        "load_latent_cache_rows": load_latent_cache_rows,
        "make_student": make_student,
        "materialize_bitnet_qat_weights": materialize_bitnet_qat_weights,
        "student_predict": student_predict,
        "teacher_predict": teacher_predict,
    }


def read_patterns(path: Path) -> list[str]:
    patterns = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                patterns.append(text)
    return patterns


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate Sana BitNet-QAT include patterns by teacher-target MSE.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patterns-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--latent-cache-dir", default="data/vision/sana_latent_cache_2k_v0")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--noise-train-steps", type=int, default=20)
    parser.add_argument("--threshold-ratio", type=float, default=0.7)
    parser.add_argument("--teacher-model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers")
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--student-architecture", choices=("custom", "sana_transformer"), default="sana_transformer")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--sana-num-layers", type=int, default=28)
    parser.add_argument("--sana-num-attention-heads", type=int, default=36)
    parser.add_argument("--sana-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-num-cross-attention-heads", type=int, default=16)
    parser.add_argument("--sana-cross-attention-head-dim", type=int, default=72)
    parser.add_argument("--sana-mlp-ratio", type=float, default=2.5)
    parser.add_argument("--sana-qk-norm", default="rms_norm_across_heads")
    parser.add_argument("--seed", type=int, default=20260504)
    args = parser.parse_args()

    mod = import_training_module()
    teacher = mod["load_teacher"](args)
    checkpoint = torch.load(args.checkpoint, map_location=args.student_device)
    config = mod["SanaLatentStudentConfig"](**{**checkpoint.get("config", {}), "patch_size": args.patch_size})
    rows = mod["load_latent_cache_rows"](Path(args.latent_cache_dir))[: args.rows]
    if not rows:
        raise ValueError(f"no cached rows found in {args.latent_cache_dir}")

    device = torch.device(args.student_device)
    base_state = checkpoint.get("student_materialized") or checkpoint["student"]
    base_student = mod["make_student"](config, args).to(device)
    base_student.load_state_dict(base_state, strict=True)
    base_student.eval()
    prompts = [row["prompt"] for row in rows]
    prompt_embeds, prompt_mask = mod["encode_prompts"](teacher, prompts, args)
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)
    final_latents = torch.cat(
        [mod["load_cached_latent"](Path(args.latent_cache_dir), row, torch.device(args.teacher_device)) for row in rows],
        dim=0,
    )
    teacher.scheduler.set_timesteps(args.noise_train_steps, device=final_latents.device)
    timesteps = teacher.scheduler.timesteps[: min(args.rows, len(teacher.scheduler.timesteps))]
    if len(timesteps) < final_latents.shape[0]:
        repeats = (final_latents.shape[0] + len(timesteps) - 1) // len(timesteps)
        timesteps = timesteps.repeat(repeats)
    timesteps = timesteps[: final_latents.shape[0]].to(final_latents.device)
    generator = torch.Generator(device=final_latents.device).manual_seed(args.seed)
    noise = torch.randn(final_latents.shape, generator=generator, device=final_latents.device)
    zt = teacher.scheduler.add_noise(final_latents, noise, timesteps)
    target = mod["teacher_predict"](
        teacher,
        zt.to(teacher.transformer.device),
        timesteps.to(teacher.transformer.device),
        prompt_embeds.to(teacher.transformer.device),
        prompt_mask.to(teacher.transformer.device),
    ).to(device)
    zt = zt.to(device)
    timesteps = timesteps.to(device)

    base_pred = mod["student_predict"](base_student, zt, timesteps.float(), prompt_embeds, prompt_mask, args)
    base_loss = float(F.mse_loss(base_pred, target).item())
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patterns = read_patterns(Path(args.patterns_file))
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pattern": "__base__", "modules": 0, "loss": base_loss, "delta": 0.0}) + "\n")
        for pattern in patterns:
            student = mod["make_student"](config, args).to(device)
            student.load_state_dict(base_state, strict=True)
            modules = mod["apply_bitnet_qat_modules"](
                student,
                threshold_ratio=args.threshold_ratio,
                include=tuple(item.strip() for item in pattern.split(",") if item.strip()),
                exclude=(),
            )
            mod["materialize_bitnet_qat_weights"](student)
            student.eval()
            pred = mod["student_predict"](student, zt, timesteps.float(), prompt_embeds, prompt_mask, args)
            loss = float(F.mse_loss(pred, target).item())
            row = {"pattern": pattern, "modules": int(modules), "loss": loss, "delta": loss - base_loss}
            handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
