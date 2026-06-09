#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torchvision import utils


def import_training_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_agentkernel_lite_image_sana_latent_distill import (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        teacher_trajectory_targets,
        load_teacher,
        make_student,
        student_predict_cfg,
    )

    return (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        teacher_trajectory_targets,
        load_teacher,
        make_student,
        student_predict_cfg,
    )


def read_prompts(path: Path) -> list[tuple[str, int | None]]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                if text.startswith("{"):
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        prompts.append(text)
                    else:
                        prompt = row.get("prompt") or row.get("caption") or row.get("text")
                        if prompt:
                            seed = row.get("seed")
                            prompts.append((str(prompt), int(seed) if seed is not None else None))
                else:
                    prompts.append((text, None))
    return prompts


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Agent Kernel Lite Sana latent distillation checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-file", default="data/vision/prompts/sana_distill_seed_prompts.txt")
    parser.add_argument("--teacher-model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers")
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoded-loss-device", default="")
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument(
        "--sample-guidance",
        type=float,
        default=None,
        help="Student CFG guidance. Defaults to the checkpoint's recorded sample_guidance, or 1.0 for guided-target students.",
    )
    parser.add_argument("--teacher-guidance", type=float, default=4.5)
    parser.add_argument("--distill-guided-targets", action="store_true")
    parser.add_argument("--fixed-teacher-seed", type=int, default=-1)
    parser.add_argument(
        "--start-from-teacher-index",
        type=int,
        default=-1,
        help="If >=0, initialize from the teacher trajectory at this scheduler index and roll out the student from there.",
    )
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--use-materialized-bitnet", action="store_true")
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--bitnet-qat-include", default="")
    parser.add_argument("--bitnet-qat-exclude", default="")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--student-architecture", choices=("custom", "sana_transformer"), default="sana_transformer")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--sana-num-layers", type=int, default=8)
    parser.add_argument("--sana-num-attention-heads", type=int, default=12)
    parser.add_argument("--sana-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-num-cross-attention-heads", type=int, default=12)
    parser.add_argument("--sana-cross-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-mlp-ratio", type=float, default=2.5)
    parser.add_argument("--sana-qk-norm", default="")
    parser.add_argument("--student-output-log-scale", action="store_true")
    parser.add_argument("--student-output-log-scale-init", type=float, default=0.0)
    args = parser.parse_args()

    (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        teacher_trajectory_targets,
        load_teacher,
        make_student,
        student_predict_cfg,
    ) = import_training_module()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    training_args = checkpoint.get("training_args") or {}
    if args.sample_guidance is None:
        if "sample_guidance" in training_args:
            args.sample_guidance = float(training_args["sample_guidance"])
        elif (
            training_args.get("distill_guided_targets")
            and float(training_args.get("train_student_cfg_guidance", 1.0) or 1.0) <= 1.0
        ):
            args.sample_guidance = 1.0
        else:
            args.sample_guidance = 4.5
    if checkpoint.get("student_architecture"):
        args.student_architecture = str(checkpoint["student_architecture"])
    for name in (
        "sana_num_layers",
        "sana_num_attention_heads",
        "sana_attention_head_dim",
        "sana_num_cross_attention_heads",
        "sana_cross_attention_head_dim",
        "sana_mlp_ratio",
        "sana_qk_norm",
        "student_output_log_scale",
        "student_output_log_scale_init",
    ):
        if name in training_args:
            setattr(args, name, training_args[name])
    if (
        training_args.get("distill_guided_targets")
        and float(training_args.get("train_student_cfg_guidance", 1.0) or 1.0) <= 1.0
        and args.sample_guidance > 1.0
    ):
        print(
            json.dumps(
                {
                    "warning": (
                        "checkpoint conditional branch was trained against guided teacher targets; "
                        "sampling with CFG > 1 applies guidance a second time. Use --sample-guidance 1.0 "
                        "unless the checkpoint was trained with --train-student-cfg-guidance."
                    ),
                    "sample_guidance": args.sample_guidance,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    config_dict = checkpoint.get("config", {})
    config = SanaLatentStudentConfig(**{**config_dict, "resolution": args.resolution, "patch_size": args.patch_size})
    student = make_student(config, args).to(args.student_device)
    if args.bitnet_qat:
        apply_bitnet_qat_modules(
            student,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
            include=tuple(item.strip() for item in args.bitnet_qat_include.split(",") if item.strip()),
            exclude=tuple(item.strip() for item in args.bitnet_qat_exclude.split(",") if item.strip()),
        )
    student_state = checkpoint.get("student_materialized") if args.use_materialized_bitnet else None
    if student_state is None:
        student_state = checkpoint["student"]
    missing, unexpected = student.load_state_dict(student_state, strict=False)
    if missing or unexpected:
        print(json.dumps({"student_state_load": {"missing": missing, "unexpected": unexpected}}), file=sys.stderr, flush=True)
    student.eval()
    prompts = read_prompts(Path(args.prompt_file))
    if args.prompt_offset > 0:
        prompts = prompts[args.prompt_offset :]
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    scheduler = teacher.scheduler
    device = torch.device(args.student_device)

    for index, (prompt, prompt_seed) in enumerate(prompts):
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = encode_prompts(teacher, [prompt], args)
        prompt_embeds = prompt_embeds.to(device)
        prompt_mask = prompt_mask.to(device)
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        negative_prompt_mask = negative_prompt_mask.to(device)
        sample_seed = prompt_seed if prompt_seed is not None else args.seed + args.prompt_offset + index
        if args.start_from_teacher_index >= 0:
            args.trajectory_steps = args.sample_steps
            examples = teacher_trajectory_targets(
                teacher,
                prompt_embeds.to(teacher.transformer.device),
                prompt_mask.to(teacher.transformer.device),
                negative_prompt_embeds.to(teacher.transformer.device),
                negative_prompt_mask.to(teacher.transformer.device),
                args,
                config,
                step=sample_seed,
                seed_override=sample_seed,
            )
            start_index = int(args.start_from_teacher_index)
            if start_index < 0 or start_index >= len(examples):
                raise ValueError(f"--start-from-teacher-index must be in [0, {len(examples) - 1}]")
            scheduler.set_timesteps(args.sample_steps, device=device)
            # DPM-style schedulers keep internal multistep model-output history.
            # When sampling from a cached teacher trajectory index, replay the
            # teacher prefix first so the scheduler state matches the latent.
            latents = examples[start_index][0].to(device)
            for replay_index in range(start_index):
                replay_latents, _, replay_target, _ = examples[replay_index]
                latents = scheduler.step(
                    replay_target.to(device),
                    scheduler.timesteps[replay_index],
                    replay_latents.to(device),
                    return_dict=False,
                )[0]
            timestep_values = list(scheduler.timesteps)[start_index:]
        else:
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            latent_channels = student.config.in_channels if args.student_architecture == "sana_transformer" else student.config.latent_channels
            latent_size = student.config.sample_size if args.student_architecture == "sana_transformer" else student.config.latent_size
            latents = torch.randn(1, latent_channels, latent_size, latent_size, generator=generator, device=device)
            scheduler.set_timesteps(args.sample_steps, device=device)
            timestep_values = list(scheduler.timesteps)
        for timestep_value in timestep_values:
            timestep = timestep_value.expand(latents.shape[0]).to(device)
            pred = student_predict_cfg(
                student,
                latents,
                timestep,
                prompt_embeds,
                prompt_mask,
                negative_prompt_embeds,
                negative_prompt_mask,
                args.sample_guidance,
                args,
            )
            latents = scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
        image_latents = latents.to(teacher.vae.device, dtype=teacher.vae.dtype)
        decoded = teacher.vae.decode(image_latents / teacher.vae.config.scaling_factor, return_dict=False)[0]
        image = teacher.image_processor.postprocess(decoded, output_type="pt")[0]
        utils.save_image(image, output_dir / f"sample_{index:03d}.png")


if __name__ == "__main__":
    main()
