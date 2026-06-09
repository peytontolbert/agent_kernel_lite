#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
import random
import zipfile

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, import_flux_pipeline
from train_agentkernel_lite_image_flux_diffusiondb_zip_stream import load_zip_items, read_image_batch
from train_agentkernel_lite_image_flux_flow_distill import seed_everything


@dataclass
class TinyFluxDecoderConfig:
    in_channels: int = 16
    base_channels: int = 192
    channel_mults: tuple[int, ...] = (4, 3, 2, 1)
    blocks_per_stage: int = 2
    out_channels: int = 3
    latent_size: int = 64
    image_size: int = 512


def list_zip_shards(repo_id: str, prefix: str, token: str | bool | None) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    shards = [name for name in files if name.startswith(prefix) and name.endswith(".zip")]
    if not shards:
        raise ValueError(f"no zip shards found for {repo_id}/{prefix}")
    return sorted(shards)


def load_flux_vae_pipeline(args: argparse.Namespace):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    _BitsAndBytesConfig, FluxPipeline, _FluxTransformer2DModel = import_flux_pipeline()
    kwargs = {
        "torch_dtype": dtype,
        "transformer": None,
        "text_encoder": None,
        "text_encoder_2": None,
        "tokenizer": None,
        "tokenizer_2": None,
    }
    if args.local_files_only:
        kwargs["local_files_only"] = True
    pipe = FluxPipeline.from_pretrained(args.teacher_model, **kwargs)
    pipe.vae.to(args.teacher_device)
    pipe.vae.eval()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class TinyFluxDecoder(nn.Module):
    def __init__(self, config: TinyFluxDecoderConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base_channels * mult for mult in config.channel_mults]
        self.in_conv = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        stages: list[nn.Module] = []
        for index, channel in enumerate(channels):
            for _ in range(config.blocks_per_stage):
                stages.append(ResBlock(channel))
            if index + 1 < len(channels):
                stages.append(nn.Upsample(scale_factor=2, mode="nearest"))
                stages.append(nn.Conv2d(channel, channels[index + 1], 3, padding=1))
        self.stages = nn.Sequential(*stages)
        self.out = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], config.out_channels, 3, padding=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.out(self.stages(self.in_conv(latents.float()))).clamp(-1.25, 1.25)


def preprocess_images(pipe, images: list[Image.Image], height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return pipe.image_processor.preprocess(images, height=height, width=width).to(device=device, dtype=dtype)


@torch.no_grad()
def encode_flux_latents(pipe, pixels: torch.Tensor) -> torch.Tensor:
    latents = pipe.vae.encode(pixels).latent_dist.sample()
    return (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


@torch.no_grad()
def decode_flux_latents(pipe, flux_latents: torch.Tensor) -> torch.Tensor:
    latents = (flux_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.vae.decode(latents, return_dict=False)[0]


def save_checkpoint(output_dir: Path, model: TinyFluxDecoder, config: TinyFluxDecoderConfig, step: int, loss: float, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_kind": "agentkernel_lite_flux_tiny_decoder_distill_checkpoint",
            "step": int(step),
            "loss": float(loss),
            "config": asdict(config),
            "model": model.state_dict(),
            "args": vars(args),
        },
        output_dir / "tiny_flux_decoder.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token: str | bool | None = True if args.use_hf_token else None
    shards = list_zip_shards(args.dataset_repo, args.shard_prefix, token)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    random.shuffle(shards)

    pipe = load_flux_vae_pipeline(args)
    device = torch.device(args.teacher_device)
    pipe.vae.eval()
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)

    config = TinyFluxDecoderConfig(base_channels=args.base_channels, blocks_per_stage=args.blocks_per_stage)
    model = TinyFluxDecoder(config).to(args.student_device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        start_step = int(checkpoint.get("step", 0))
    else:
        start_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ledger_path = output_dir / "tiny_decoder_ledger.jsonl"

    step = start_step
    shard_index = start_step % max(len(shards), 1)
    while step < start_step + args.steps:
        shard_name = shards[shard_index % len(shards)]
        shard_index += 1
        shard_path = Path(hf_hub_download(args.dataset_repo, repo_type="dataset", filename=shard_name, local_dir=str(cache_dir), token=token))
        items = load_zip_items(shard_path, args.max_items_per_shard)
        random.Random(args.seed + step + shard_index).shuffle(items)
        for offset in range(0, len(items), args.batch_size):
            if step >= start_step + args.steps:
                break
            images, prompts = read_image_batch(shard_path, items[offset : offset + args.batch_size])
            if not images:
                continue
            with torch.no_grad():
                pixels = preprocess_images(pipe, images, args.height, args.width, device, next(pipe.vae.parameters()).dtype)
                flux_latents = encode_flux_latents(pipe, pixels)
                if args.teacher_decode_weight > 0:
                    teacher_pixels = decode_flux_latents(pipe, flux_latents).float()
                else:
                    teacher_pixels = None
            latents = flux_latents.to(args.student_device, dtype=torch.float32)
            target_pixels = pixels.to(args.student_device, dtype=torch.float32)
            pred = model(latents)
            if pred.shape[-2:] != target_pixels.shape[-2:]:
                pred = F.interpolate(pred, size=target_pixels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.l1_loss(pred, target_pixels)
            if args.teacher_decode_weight > 0 and teacher_pixels is not None:
                loss = loss + float(args.teacher_decode_weight) * F.l1_loss(pred, teacher_pixels.to(args.student_device, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                record = {"step": step, "loss": float(loss.detach().item()), "shard": shard_name, "prompt": prompts[:2]}
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % args.checkpoint_every == 0:
                save_checkpoint(output_dir, model, config, step, float(loss.detach().item()), args)
        if args.delete_shards_after_use:
            try:
                shard_path.unlink()
            except FileNotFoundError:
                pass
    save_checkpoint(output_dir, model, config, step, float(loss.detach().item()), args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill the FLUX VAE decoder into a small mobile/browser decoder.")
    parser.add_argument("--dataset-repo", default="poloclub/diffusiondb")
    parser.add_argument("--shard-prefix", default="diffusiondb-large-part-1/")
    parser.add_argument("--cache-dir", default="/dev/shm/diffusiondb_zip_cache")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_flux_tiny_decoder_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-items-per-shard", type=int, default=1000)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--base-channels", type=int, default=192)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--teacher-decode-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--delete-shards-after-use", action="store_true")
    parser.add_argument("--use-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
