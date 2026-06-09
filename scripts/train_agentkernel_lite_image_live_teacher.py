#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterator

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, utils


DEFAULT_TEACHER = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers"


def install_local_diffusers() -> None:
    candidates = [
        os.environ.get("DIFFUSERS_SRC", ""),
        "/data/webgl-game/repos/diffusers/src",
        "/data/repositories/diffusers/src",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


def install_local_scripts() -> None:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def import_teacher_pipeline() -> type[Any]:
    install_local_diffusers()
    try:
        from diffusers import SanaPipeline

        return SanaPipeline
    except Exception:
        from diffusers import AutoPipelineForText2Image

        return AutoPipelineForText2Image


def import_student_modules():
    install_local_scripts()
    from train_agentkernel_lite_image_latent_flow import LatentFlowConfig, LatentFlowDiT, TinyAutoencoder, edge_loss

    return LatentFlowConfig, LatentFlowDiT, TinyAutoencoder, edge_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_prompt(value: Any, min_words: int = 5, max_chars: int = 420) -> str:
    text = " ".join(str(value or "").replace("\x00", "").split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    if len(text.split()) < min_words:
        return ""
    lowered = text.lower()
    if any(term in lowered for term in ("nsfw", "nude", "naked", "porn", "gore")):
        return ""
    return text


def prompt_ids(prompt: str, *, vocab_size: int, tokens: int, device: torch.device) -> torch.Tensor | None:
    if vocab_size <= 0 or tokens <= 0:
        return None
    words = re.findall(r"[a-z0-9]+", prompt.lower())[:tokens]
    if not words:
        words = ["image"]
    ids = []
    for word in words:
        value = 2166136261
        for char in word:
            value ^= ord(char)
            value = (value * 16777619) & 0xFFFFFFFF
        ids.append(value % vocab_size)
    while len(ids) < tokens:
        ids.append(0)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def prompt_stream(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.prompt_file:
        path = Path(args.prompt_file)
        index = 0
        while True:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    if path.suffix.lower() == ".jsonl":
                        row = json.loads(line)
                        prompt = clean_prompt(row.get("prompt", row.get("text", row.get("caption", ""))), min_words=args.min_prompt_words, max_chars=args.max_prompt_chars)
                    else:
                        prompt = clean_prompt(line, min_words=args.min_prompt_words, max_chars=args.max_prompt_chars)
                    if prompt:
                        yield {"prompt": prompt, "source_index": index, "source_dataset": str(path)}
                        index += 1
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"split": args.prompt_split, "streaming": True}
    if args.prompt_trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.prompt_config:
        dataset = load_dataset(args.prompt_dataset, args.prompt_config, **load_kwargs)
    else:
        dataset = load_dataset(args.prompt_dataset, **load_kwargs)
    columns = [column.strip() for column in args.prompt_columns.split(",") if column.strip()]
    index = 0
    for row in dataset:
        prompt = ""
        for column in columns:
            if column in row:
                prompt = clean_prompt(row[column], min_words=args.min_prompt_words, max_chars=args.max_prompt_chars)
            image_nsfw = float(row.get("image_nsfw") or 0.0)
            prompt_nsfw = float(row.get("prompt_nsfw") or 0.0)
            if prompt and image_nsfw <= args.max_nsfw_score and prompt_nsfw <= args.max_nsfw_score:
                break
            prompt = ""
        if prompt:
            yield {"prompt": prompt, "source_index": index, "source_dataset": args.prompt_dataset}
        index += 1


def load_teacher(args: argparse.Namespace) -> Any:
    pipeline_cls = import_teacher_pipeline()
    dtype = torch.float16 if args.teacher_dtype == "float16" else torch.bfloat16 if args.teacher_dtype == "bfloat16" else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.teacher_variant:
        kwargs["variant"] = args.teacher_variant
    if args.local_files_only:
        kwargs["local_files_only"] = True
    teacher = pipeline_cls.from_pretrained(args.teacher_model, **kwargs)
    teacher = teacher.to(args.teacher_device)
    if hasattr(teacher, "set_progress_bar_config"):
        teacher.set_progress_bar_config(disable=True)
    if hasattr(teacher, "enable_vae_slicing"):
        teacher.enable_vae_slicing()
    if hasattr(teacher, "enable_vae_tiling"):
        teacher.enable_vae_tiling()
    return teacher


def image_to_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


@torch.inference_mode()
def generate_teacher_image(teacher: Any, prompt: str, seed: int, args: argparse.Namespace) -> Image.Image:
    generator = torch.Generator(device=args.teacher_device).manual_seed(seed)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": args.teacher_resolution,
        "width": args.teacher_resolution,
        "num_inference_steps": args.teacher_steps,
        "guidance_scale": args.teacher_guidance,
        "generator": generator,
    }
    if args.negative_prompt:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.max_sequence_length:
        kwargs["max_sequence_length"] = args.max_sequence_length
    return teacher(**kwargs).images[0]


def save_checkpoint(output_dir: Path, config: Any, autoencoder: nn.Module, flow: nn.Module, step: int, loss: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "agentkernel-lite-live-teacher-latent-flow-v0",
            "mode": "live_teacher_flow",
            "step": int(step),
            "loss": float(loss),
            "config": asdict(config),
            "autoencoder": autoencoder.state_dict(),
            "flow": flow.state_dict(),
        },
        output_dir / "latent_flow.pt",
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    LatentFlowConfig, LatentFlowDiT, TinyAutoencoder, edge_loss = import_student_modules()
    student_device = torch.device(args.student_device)
    config = LatentFlowConfig(
        image_size=args.student_resolution,
        downsample=args.downsample,
        latent_channels=args.latent_channels,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        class_count=0,
        bitlinear=args.bitlinear,
        prompt_vocab_size=args.prompt_vocab_size,
        prompt_tokens=args.prompt_tokens,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "live_teacher_ledger.jsonl"
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    teacher = load_teacher(args)
    autoencoder = TinyAutoencoder(config).to(student_device)
    flow = LatentFlowDiT(config).to(student_device)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=student_device)
        autoencoder.load_state_dict(checkpoint["autoencoder"])
        missing, unexpected = flow.load_state_dict(checkpoint["flow"], strict=False)
        if missing or unexpected:
            print(json.dumps({"resume_flow_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
        start_step = int(checkpoint.get("step") or 0)

    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
    flow_optimizer = torch.optim.AdamW(flow.parameters(), lr=args.flow_lr, weight_decay=args.weight_decay)
    if args.freeze_autoencoder:
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
    prompts = prompt_stream(args)
    last_loss = 0.0
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        row = next(prompts)
        prompt = row["prompt"]
        seed = args.seed + step
        teacher_image = generate_teacher_image(teacher, prompt, seed, args)
        image = image_to_tensor(teacher_image, args.student_resolution, student_device)

        recon = autoencoder(image)
        ae_loss = F.l1_loss(recon, image) + args.edge_weight * edge_loss(recon, image)
        if not args.freeze_autoencoder and args.ae_weight > 0:
            ae_optimizer.zero_grad(set_to_none=True)
            (ae_loss * args.ae_weight).backward()
            nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip)
            ae_optimizer.step()

        with torch.no_grad():
            z1 = autoencoder.encode(image)
        flow_loss = torch.zeros((), device=student_device)
        if args.flow_weight > 0:
            ids = prompt_ids(prompt, vocab_size=config.prompt_vocab_size, tokens=config.prompt_tokens, device=student_device)
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], device=student_device)
            zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
            target = z1 - z0
            pred = flow(zt, t, None, ids)
            flow_loss = F.mse_loss(pred, target)
            flow_optimizer.zero_grad(set_to_none=True)
            (flow_loss * args.flow_weight).backward()
            nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            flow_optimizer.step()

        last_loss = float(ae_loss.item() + flow_loss.item())
        ledger = {
            "step": step,
            "source_dataset": row["source_dataset"],
            "source_index": row["source_index"],
            "prompt": prompt,
            "seed": seed,
            "teacher_model": args.teacher_model,
            "teacher_resolution": args.teacher_resolution,
            "student_resolution": args.student_resolution,
            "ae_loss": float(ae_loss.item()),
            "flow_loss": float(flow_loss.item()),
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(ledger), flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            utils.save_image(torch.cat([image.detach().cpu(), recon.detach().cpu()], dim=0) * 0.5 + 0.5, output_dir / f"live_teacher_recon_step_{step:06d}.png", nrow=2)
        if step % args.checkpoint_every == 0 or local_step == args.steps:
            save_checkpoint(output_dir, config, autoencoder, flow, step, last_loss)
    save_checkpoint(output_dir, config, autoencoder, flow, start_step + args.steps, last_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Sana teacher distillation for Agent Kernel Lite image student.")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_live_sana_sprint_512_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--prompt-dataset", default="bitmind/JourneyDB")
    parser.add_argument("--prompt-config", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt-split", default="train")
    parser.add_argument("--prompt-columns", default="prompt,caption,text")
    parser.add_argument("--prompt-trust-remote-code", action="store_true")
    parser.add_argument("--min-prompt-words", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=420)
    parser.add_argument("--max-nsfw-score", type=float, default=0.2)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--teacher-variant", default="")
    parser.add_argument("--teacher-resolution", type=int, default=512)
    parser.add_argument("--student-resolution", type=int, default=512)
    parser.add_argument("--teacher-steps", type=int, default=4)
    parser.add_argument("--teacher-guidance", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bitlinear", action="store_true")
    parser.add_argument("--prompt-vocab-size", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--ae-lr", type=float, default=2e-4)
    parser.add_argument("--flow-lr", type=float, default=2e-4)
    parser.add_argument("--ae-weight", type=float, default=1.0)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--freeze-autoencoder", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--edge-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
