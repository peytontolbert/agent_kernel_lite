#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import utils


def import_training_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_agentkernel_lite_image_sana_latent_distill import (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        load_teacher,
        make_student,
        student_predict_cfg,
    )

    return SanaLatentStudentConfig, apply_bitnet_qat_modules, encode_prompts, load_teacher, make_student, student_predict_cfg


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                row = json.loads(text)
            else:
                row = {"prompt": text}
            prompt = row.get("prompt") or row.get("caption") or row.get("text")
            if prompt:
                row["prompt"] = str(prompt)
                rows.append(row)
    return rows


def resolve_path(path_text: str | None, root: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return path if path.exists() else None


def load_reference(row: dict[str, Any], root: Path, size: int) -> Image.Image | None:
    for key in ("teacher_ref", "real_ref", "source_ref", "image_ref", "image_path", "path"):
        path = resolve_path(row.get(key), root)
        if path is not None:
            return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return None


@torch.no_grad()
def sample_student(
    *,
    teacher: Any,
    student: torch.nn.Module,
    prompt: str,
    seed: int,
    args: argparse.Namespace,
    encode_prompts,
    student_predict_cfg,
) -> torch.Tensor:
    device = torch.device(args.student_device)
    prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = encode_prompts(teacher, [prompt], args)
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)
    negative_prompt_embeds = negative_prompt_embeds.to(device)
    negative_prompt_mask = negative_prompt_mask.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_channels = student.config.in_channels if args.student_architecture == "sana_transformer" else student.config.latent_channels
    latent_size = student.config.sample_size if args.student_architecture == "sana_transformer" else student.config.latent_size
    latents = torch.randn(1, latent_channels, latent_size, latent_size, generator=generator, device=device)
    teacher.scheduler.set_timesteps(args.sample_steps, device=device)
    for timestep_value in teacher.scheduler.timesteps:
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
        latents = teacher.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
    image_latents = latents.to(teacher.vae.device, dtype=teacher.vae.dtype)
    decoded = teacher.vae.decode(image_latents / teacher.vae.config.scaling_factor, return_dict=False)[0]
    return teacher.image_processor.postprocess(decoded, output_type="pt")[0].detach().cpu()


@torch.no_grad()
def sample_teacher(teacher: Any, prompt: str, seed: int, args: argparse.Namespace) -> Image.Image:
    generator = torch.Generator(device=args.teacher_device).manual_seed(seed)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": args.resolution,
        "width": args.resolution,
        "num_inference_steps": args.teacher_steps,
        "guidance_scale": args.teacher_guidance,
        "generator": generator,
    }
    if args.max_sequence_length:
        kwargs["max_sequence_length"] = args.max_sequence_length
    try:
        result = teacher(**kwargs)
    except TypeError:
        kwargs.pop("guidance_scale", None)
        result = teacher(**kwargs)
    return result.images[0].convert("RGB")


def image_l1(a: Image.Image, b: Image.Image) -> float:
    at = torch.ByteTensor(torch.ByteStorage.from_buffer(a.tobytes())).float().view(a.height, a.width, 3) / 255.0
    bt = torch.ByteTensor(torch.ByteStorage.from_buffer(b.tobytes())).float().view(b.height, b.width, 3) / 255.0
    return float((at - bt).abs().mean().item())


def lowfreq_l1(a: Image.Image, b: Image.Image, size: int = 64) -> float:
    a_small = a.resize((size, size), Image.Resampling.BILINEAR)
    b_small = b.resize((size, size), Image.Resampling.BILINEAR)
    return image_l1(a_small, b_small)


def image_edge_l1(a: Image.Image, b: Image.Image, size: int = 256) -> float:
    at = image_tensor(a.resize((size, size), Image.Resampling.BILINEAR)).permute(2, 0, 1)
    bt = image_tensor(b.resize((size, size), Image.Resampling.BILINEAR)).permute(2, 0, 1)
    gray_a = 0.299 * at[0] + 0.587 * at[1] + 0.114 * at[2]
    gray_b = 0.299 * bt[0] + 0.587 * bt[1] + 0.114 * bt[2]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    def edges(x: torch.Tensor) -> torch.Tensor:
        x = x[None, None]
        ex = torch.nn.functional.conv2d(x, kx.view(1, 1, 3, 3), padding=1)
        ey = torch.nn.functional.conv2d(x, ky.view(1, 1, 3, 3), padding=1)
        return torch.sqrt(ex.square() + ey.square() + 1e-8).squeeze()
    return float((edges(gray_a) - edges(gray_b)).abs().mean().item())


def color_hist_js(a: Image.Image, b: Image.Image, bins: int = 64) -> float:
    at = image_tensor(a)
    bt = image_tensor(b)
    values = []
    for channel in range(3):
        ha = torch.histc(at[..., channel], bins=bins, min=0.0, max=1.0)
        hb = torch.histc(bt[..., channel], bins=bins, min=0.0, max=1.0)
        ha = ha / (ha.sum() + 1e-8)
        hb = hb / (hb.sum() + 1e-8)
        m = 0.5 * (ha + hb)
        js = 0.5 * (
            torch.sum(ha * torch.log((ha + 1e-8) / (m + 1e-8)))
            + torch.sum(hb * torch.log((hb + 1e-8) / (m + 1e-8)))
        )
        values.append(js)
    return float(torch.stack(values).mean().item())


class ClipMetric:
    def __init__(self, model_name: str, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def image_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def text_features(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**inputs)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def score(self, prompt: str, teacher_image: Image.Image, student_image: Image.Image) -> dict[str, float]:
        image_features = self.image_features([teacher_image, student_image])
        text_feature = self.text_features([prompt])
        teacher_text = (image_features[0:1] * text_feature).sum(dim=-1)
        student_text = (image_features[1:2] * text_feature).sum(dim=-1)
        teacher_student = (image_features[0:1] * image_features[1:2]).sum(dim=-1)
        return {
            "clip_teacher_text": round(float(teacher_text.item()), 5),
            "clip_student_text": round(float(student_text.item()), 5),
            "clip_student_teacher": round(float(teacher_student.item()), 5),
            "clip_text_gap": round(float((teacher_text - student_text).item()), 5),
        }


def image_tensor(image: Image.Image) -> torch.Tensor:
    return torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).float().view(image.height, image.width, 3) / 255.0


def foreground_mask(image: Image.Image, threshold: float = 0.10) -> torch.Tensor:
    tensor = image_tensor(image)
    white_distance = 1.0 - tensor.amin(dim=-1)
    saturation = tensor.max(dim=-1).values - tensor.min(dim=-1).values
    return ((white_distance > threshold) | (saturation > threshold * 0.8)).float()


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        b = torch.nn.functional.interpolate(
            b[None, None], size=a.shape, mode="nearest"
        ).squeeze(0).squeeze(0)
    inter = torch.minimum(a, b).sum()
    union = torch.maximum(a, b).sum()
    if float(union.item()) <= 1e-6:
        return 1.0
    return float((inter / union).item())


def bbox_from_mask(mask: torch.Tensor) -> tuple[float, float, float, float] | None:
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0:
        return None
    h, w = mask.shape
    return (
        float(xs.min().item()) / max(w - 1, 1),
        float(ys.min().item()) / max(h - 1, 1),
        float(xs.max().item()) / max(w - 1, 1),
        float(ys.max().item()) / max(h - 1, 1),
    )


def bbox_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-8 else inter / union


def projection_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        b = torch.nn.functional.interpolate(
            b[None, None], size=a.shape, mode="nearest"
        ).squeeze(0).squeeze(0)
    ax = a.mean(dim=0)
    bx = b.mean(dim=0)
    ay = a.mean(dim=1)
    by = b.mean(dim=1)
    return float(0.5 * ((ax - bx).abs().mean() + (ay - by).abs().mean()).item())


def geometry_metrics(candidate: Image.Image, reference: Image.Image) -> dict[str, float]:
    candidate_mask = foreground_mask(candidate)
    reference_mask = foreground_mask(reference)
    miou = mask_iou(candidate_mask, reference_mask)
    biou = bbox_iou(bbox_from_mask(candidate_mask), bbox_from_mask(reference_mask))
    proj = projection_l1(candidate_mask, reference_mask)
    area_ratio = float(candidate_mask.mean().item() / max(reference_mask.mean().item(), 1e-6))
    return {
        "mask_iou": round(miou, 5),
        "bbox_iou": round(biou, 5),
        "projection_l1": round(proj, 5),
        "foreground_area_ratio": round(area_ratio, 5),
    }


def make_sheet(
    *,
    rows: list[dict[str, Any]],
    refs: list[Image.Image | None],
    teachers: list[Image.Image],
    students: list[Image.Image],
    metrics: list[dict[str, Any]],
    output_path: Path,
    cell: int,
    run_label: str = "",
) -> None:
    label_h = 92
    header_h = 34 if run_label else 0
    cols = 3
    sheet = Image.new("RGB", (cols * cell, header_h + len(rows) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    if run_label:
        draw.text((6, 6), run_label[:220], fill=(0, 0, 0), font=font)
    headers = ("dataset_ref", "sana_teacher", "student")
    for row_i, row in enumerate(rows):
        y = header_h + row_i * (cell + label_h)
        images = [
            refs[row_i].resize((cell, cell), Image.Resampling.LANCZOS) if refs[row_i] is not None else Image.new("RGB", (cell, cell), (235, 235, 235)),
            teachers[row_i].resize((cell, cell), Image.Resampling.LANCZOS),
            students[row_i].resize((cell, cell), Image.Resampling.LANCZOS),
        ]
        for col_i, image in enumerate(images):
            x = col_i * cell
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + cell + 3), headers[col_i], fill=(0, 0, 0), font=font)
        prompt = row["prompt"]
        meta = metrics[row_i]
        gate = meta.get("student_geometry_gate")
        label = (
            f"{row_i:02d} seed={meta['seed']} "
            f"s/t={meta.get('student_teacher_l1')} lf={meta.get('student_teacher_lowfreq_l1')} "
            f"edge={meta.get('student_teacher_edge_l1')} js={meta.get('student_teacher_color_js')} "
            f"clip={meta.get('clip_student_teacher')} txt={meta.get('clip_student_text')} "
            f"t/ref={meta.get('teacher_ref_l1')} s/ref={meta.get('student_ref_l1')} "
            f"mask={meta.get('student_ref_mask_iou', meta.get('student_teacher_mask_iou'))} "
            f"bbox={meta.get('student_ref_bbox_iou', meta.get('student_teacher_bbox_iou'))} "
            f"proj={meta.get('student_ref_projection_l1', meta.get('student_teacher_projection_l1'))} "
            f"gate={gate}"
        )
        wrapped = textwrap.wrap(label + " " + prompt, width=92)[:4]
        draw.text((4, y + cell + 18), "\n".join(wrapped), fill=(0, 0, 0), font=font)
    sheet.save(output_path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dataset refs, SANA teacher generations, and SANA student outputs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers")
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoded-loss-device", default="")
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--teacher-guidance", type=float, default=4.5)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--sample-guidance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--min-mask-iou", type=float, default=0.35)
    parser.add_argument("--min-bbox-iou", type=float, default=0.45)
    parser.add_argument("--max-projection-l1", type=float, default=0.20)
    parser.add_argument("--max-student-ref-l1", type=float, default=0.16)
    parser.add_argument("--max-teacher-l1-ratio", type=float, default=2.0)
    parser.add_argument("--clip-metrics", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip-device", default="")
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--use-materialized-bitnet", action="store_true")
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--bitnet-qat-include", default="")
    parser.add_argument("--bitnet-qat-exclude", default="")
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
        load_teacher,
        make_student,
        student_predict_cfg,
    ) = import_training_module()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.prompt_file))
    rows = rows[args.prompt_offset :]
    if args.max_prompts:
        rows = rows[: args.max_prompts]
    if not rows:
        raise ValueError("no prompts found")

    teacher = load_teacher(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    training_args = checkpoint.get("training_args") or {}
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
    training_context_keys = (
        "teacher_steps",
        "trajectory_steps",
        "sample_steps",
        "teacher_guidance",
        "sample_guidance",
        "distill_guided_targets",
        "train_student_cfg_guidance",
        "trajectory_rollout_start_indices",
        "trajectory_rollout_start_index",
        "trajectory_rollout_prefix_steps",
        "decoded_rollout_start_index",
        "decoded_rollout_prefix_steps",
        "decoded_rollout_clip_loss_weight",
        "use_teacher_final_decoded_targets",
    )
    training_context = {key: training_args.get(key) for key in training_context_keys if key in training_args}
    run_label = (
        f"teacher_steps={args.teacher_steps} sample_steps={args.sample_steps} "
        f"teacher_guidance={args.teacher_guidance} sample_guidance={args.sample_guidance} "
        f"trained_guided={training_context.get('distill_guided_targets')} "
        f"trained_cfg={training_context.get('train_student_cfg_guidance')} "
        f"train_traj={training_context.get('trajectory_steps')} "
        f"rollout={training_context.get('trajectory_rollout_start_indices') or training_context.get('trajectory_rollout_start_index')}"
    )
    clip_metric = None
    if args.clip_metrics:
        clip_device = args.clip_device or args.decoded_loss_device or args.teacher_device
        clip_metric = ClipMetric(args.clip_model, clip_device)

    refs: list[Image.Image | None] = []
    teachers: list[Image.Image] = []
    students: list[Image.Image] = []
    metrics: list[dict[str, Any]] = []
    root = Path.cwd()
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        seed = int(row.get("seed") if row.get("seed") is not None else args.seed + args.prompt_offset + index)
        ref = load_reference(row, root, args.resolution)
        teacher_image = sample_teacher(teacher, prompt, seed, args)
        student_tensor = sample_student(
            teacher=teacher,
            student=student,
            prompt=prompt,
            seed=seed,
            args=args,
            encode_prompts=encode_prompts,
            student_predict_cfg=student_predict_cfg,
        )
        student_path = output_dir / f"student_{index:03d}.png"
        teacher_path = output_dir / f"teacher_{index:03d}.png"
        ref_path = output_dir / f"ref_{index:03d}.png"
        utils.save_image(student_tensor, student_path)
        teacher_image.save(teacher_path)
        student_image = Image.open(student_path).convert("RGB")
        if ref is not None:
            ref.save(ref_path)
        metric: dict[str, Any] = {
            "index": index,
            "prompt": prompt,
            "seed": seed,
            "source_ref": row.get("teacher_ref") or row.get("real_ref") or row.get("source_ref"),
        }
        metric["student_teacher_l1"] = round(image_l1(student_image.resize((args.resolution, args.resolution)), teacher_image.resize((args.resolution, args.resolution))), 5)
        metric["student_teacher_lowfreq_l1"] = round(lowfreq_l1(student_image, teacher_image), 5)
        metric["student_teacher_edge_l1"] = round(image_edge_l1(student_image, teacher_image), 5)
        metric["student_teacher_color_js"] = round(color_hist_js(student_image.resize((256, 256)), teacher_image.resize((256, 256))), 5)
        if clip_metric is not None:
            metric.update(clip_metric.score(prompt, teacher_image, student_image))
        for key, value in geometry_metrics(
            student_image.resize((args.resolution, args.resolution)),
            teacher_image.resize((args.resolution, args.resolution)),
        ).items():
            metric[f"student_teacher_{key}"] = value
        if ref is not None:
            metric["teacher_ref_l1"] = round(image_l1(teacher_image.resize((args.resolution, args.resolution)), ref), 5)
            metric["student_ref_l1"] = round(image_l1(student_image.resize((args.resolution, args.resolution)), ref), 5)
            for prefix, values in (
                ("teacher_ref", geometry_metrics(teacher_image.resize((args.resolution, args.resolution)), ref)),
                ("student_ref", geometry_metrics(student_image.resize((args.resolution, args.resolution)), ref)),
            ):
                for key, value in values.items():
                    metric[f"{prefix}_{key}"] = value
            metric["student_geometry_gate"] = bool(
                metric["student_ref_mask_iou"] >= args.min_mask_iou
                and metric["student_ref_bbox_iou"] >= args.min_bbox_iou
                and metric["student_ref_projection_l1"] <= args.max_projection_l1
                and metric["student_ref_l1"] <= args.max_student_ref_l1
                and metric["student_ref_l1"] <= metric["teacher_ref_l1"] * args.max_teacher_l1_ratio
            )
        else:
            metric["teacher_ref_l1"] = None
            metric["student_ref_l1"] = None
        refs.append(ref)
        teachers.append(teacher_image)
        students.append(student_image)
        metrics.append(metric)
        print(json.dumps(metric), flush=True)

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric) + "\n")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        manifest = vars(args).copy()
        manifest["checkpoint_training_context"] = training_context
        manifest["checkpoint_step"] = checkpoint.get("step")
        manifest["run_label"] = run_label
        json.dump(manifest, handle, indent=2, sort_keys=True)
    make_sheet(
        rows=rows,
        refs=refs,
        teachers=teachers,
        students=students,
        metrics=metrics,
        output_path=output_dir / "contact_sheet.png",
        cell=180,
        run_label=run_label,
    )
    print(output_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
