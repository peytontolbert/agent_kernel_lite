#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def import_training_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_agentkernel_lite_image_sana_latent_distill import (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        decode_latents_for_loss,
        encode_prompts,
        load_teacher,
        make_student,
        student_predict_cfg,
        teacher_trajectory_targets,
    )

    return (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        decode_latents_for_loss,
        encode_prompts,
        load_teacher,
        make_student,
        student_predict_cfg,
        teacher_trajectory_targets,
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        row = json.loads(text) if text.startswith("{") else {"prompt": text}
        prompt = row.get("prompt") or row.get("caption") or row.get("text")
        if prompt:
            row["prompt"] = str(prompt)
            rows.append(row)
    return rows


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    # VAE-decoded tensors are in [-1, 1]; map to display range [0, 1].
    tensor = (tensor.detach().float() / 2.0 + 0.5).clamp(0, 1).cpu()
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def image_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float().cpu() - b.float().cpu()).abs().mean().item())


def lowfreq_l1(a: torch.Tensor, b: torch.Tensor, size: int = 64) -> float:
    a_small = F.interpolate(a[None].float(), size=(size, size), mode="bilinear", align_corners=False)[0]
    b_small = F.interpolate(b[None].float(), size=(size, size), mode="bilinear", align_corners=False)[0]
    return image_l1(a_small, b_small)


def draw_cell(image: Image.Image, label: str, width: int, label_h: int = 42) -> Image.Image:
    canvas = Image.new("RGB", (width, width + label_h), "white")
    canvas.paste(image.resize((width, width), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, width + 4), label[:90], fill=(0, 0, 0))
    return canvas


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Decode teacher/student rollout checkpoints at selected SANA steps.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--student-device", default="cuda:2")
    parser.add_argument("--decoded-loss-device", default="cuda:1")
    parser.add_argument("--teacher-dtype", default="float16")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--trajectory-steps", type=int, default=12)
    parser.add_argument("--teacher-guidance", type=float, default=4.5)
    parser.add_argument("--sample-guidance", type=float, default=2.0)
    parser.add_argument("--train-student-cfg-guidance", type=float, default=2.0)
    parser.add_argument("--distill-guided-targets", action="store_true")
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--student-architecture", choices=("custom", "sana_transformer"), default="sana_transformer")
    parser.add_argument("--student-gradient-checkpointing", action="store_true")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--sana-num-layers", type=int, default=12)
    parser.add_argument("--sana-num-attention-heads", type=int, default=16)
    parser.add_argument("--sana-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-num-cross-attention-heads", type=int, default=16)
    parser.add_argument("--sana-cross-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-mlp-ratio", type=float, default=2.5)
    parser.add_argument("--sana-qk-norm", default="")
    parser.add_argument("--decoded-size", type=int, default=192)
    parser.add_argument("--decoded-loss-latent-size", type=int, default=8)
    parser.add_argument("--indices", default="3,7,11", help="Zero-based denoising step indices to decode.")
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-resolution-binning", action="store_true")
    parser.add_argument("--fixed-teacher-seed", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--student-output-log-scale", action="store_true")
    parser.add_argument("--student-output-log-scale-init", type=float, default=0.0)
    args = parser.parse_args()

    (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        decode_latents_for_loss,
        encode_prompts,
        load_teacher,
        make_student,
        student_predict_cfg,
        teacher_trajectory_targets,
    ) = import_training_module()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.prompt_file))[args.prompt_offset : args.prompt_offset + args.max_prompts]
    teacher = load_teacher(args)
    state = torch.load(args.checkpoint, map_location="cpu")
    training_args = state.get("training_args") or {}
    if state.get("student_architecture"):
        args.student_architecture = str(state["student_architecture"])
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
    config_dict = state.get("config", {})
    config = SanaLatentStudentConfig(**{**config_dict, "resolution": args.resolution, "patch_size": args.patch_size})
    student = make_student(config, args).to(args.student_device)
    if args.bitnet_qat:
        apply_bitnet_qat_modules(student)
    student_state = state.get("student_materialized") or state.get("student") or state.get("model") or state
    missing, unexpected = student.load_state_dict(student_state, strict=False)
    if missing or unexpected:
        print(json.dumps({"student_state_load": {"missing": missing, "unexpected": unexpected}}), file=sys.stderr, flush=True)
    student.eval()

    decode_indices = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
    all_rows: list[dict[str, Any]] = []
    sheet_rows: list[Image.Image] = []
    cell_w = args.decoded_size
    for prompt_index, row in enumerate(rows):
        prompt = row["prompt"]
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = encode_prompts(teacher, [prompt], args)
        seed = int(row.get("seed", args.seed + prompt_index))
        examples = teacher_trajectory_targets(
            teacher,
            prompt_embeds.to(teacher.transformer.device),
            prompt_mask.to(teacher.transformer.device),
            negative_prompt_embeds.to(teacher.transformer.device),
            negative_prompt_mask.to(teacher.transformer.device),
            args,
            config,
            prompt_index,
            seed,
        )
        latents = examples[0][0].to(args.student_device)
        teacher.scheduler.set_timesteps(args.trajectory_steps, device=args.student_device)
        row_cells: list[Image.Image] = []
        for step_index, timestep_value in enumerate(teacher.scheduler.timesteps):
            timestep = timestep_value.expand(latents.shape[0]).to(args.student_device)
            pred = student_predict_cfg(
                student,
                latents,
                timestep.float(),
                prompt_embeds.to(args.student_device),
                prompt_mask.to(args.student_device),
                negative_prompt_embeds.to(args.student_device),
                negative_prompt_mask.to(args.student_device),
                args.sample_guidance,
                args,
            )
            latents = teacher.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
            if step_index not in decode_indices:
                continue
            teacher_latents = examples[step_index][3].to(args.student_device)
            student_dec = decode_latents_for_loss(teacher, latents, args.decoded_size, args.decoded_loss_latent_size)[0]
            teacher_dec = decode_latents_for_loss(teacher, teacher_latents, args.decoded_size, args.decoded_loss_latent_size)[0]
            metrics = {
                "prompt_index": prompt_index,
                "step_index": step_index,
                "prompt": prompt,
                "l1": round(image_l1(student_dec, teacher_dec), 5),
                "lowfreq_l1": round(lowfreq_l1(student_dec, teacher_dec), 5),
            }
            all_rows.append(metrics)
            row_cells.append(draw_cell(tensor_to_image(teacher_dec), f"teacher t{step_index + 1}", cell_w))
            row_cells.append(draw_cell(tensor_to_image(student_dec), f"student t{step_index + 1} l1={metrics['l1']}", cell_w))
        if row_cells:
            prompt_label = Image.new("RGB", (cell_w, cell_w + 42), "white")
            ImageDraw.Draw(prompt_label).text((4, 4), f"{prompt_index:02d} {prompt[:80]}", fill=(0, 0, 0))
            cells = [prompt_label] + row_cells
            row_canvas = Image.new("RGB", (cell_w * len(cells), cell_w + 42), "white")
            for idx, cell in enumerate(cells):
                row_canvas.paste(cell, (idx * cell_w, 0))
            sheet_rows.append(row_canvas)
    if sheet_rows:
        sheet = Image.new("RGB", (max(row.width for row in sheet_rows), sum(row.height for row in sheet_rows)), "white")
        y = 0
        for row_img in sheet_rows:
            sheet.paste(row_img, (0, y))
            y += row_img.height
        sheet.save(output_dir / "rollout_steps.png")
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    print(output_dir / "rollout_steps.png")


if __name__ == "__main__":
    main()
