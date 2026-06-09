#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


@dataclass
class LatentFlowConfig:
    image_size: int = 128
    downsample: int = 8
    channels: int = 3
    latent_channels: int = 8
    patch_size: int = 2
    dim: int = 384
    depth: int = 8
    heads: int = 8
    mlp_ratio: int = 4
    class_count: int = 0
    bitlinear: bool = False
    prompt_vocab_size: int = 0
    prompt_tokens: int = 0

    @property
    def latent_size(self) -> int:
        return self.image_size // self.downsample


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def ternary_weight_ste(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    threshold = 0.7 * scale
    quantized = torch.where(weight > threshold, scale, torch.where(weight < -threshold, -scale, torch.zeros_like(weight)))
    return weight + (quantized - weight).detach()


class BitLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, ternary_weight_ste(self.weight), self.bias)


def linear(in_features: int, out_features: int, *, bitlinear: bool) -> nn.Module:
    return (BitLinear if bitlinear else nn.Linear)(in_features, out_features)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyAutoencoder(nn.Module):
    def __init__(self, config: LatentFlowConfig) -> None:
        super().__init__()
        if config.downsample not in {4, 8, 16}:
            raise ValueError("downsample must be one of 4, 8, or 16")
        widths = [64, 128, 192, 256, 320]
        downs = int(math.log2(config.downsample))
        enc = [nn.Conv2d(config.channels, widths[0], 3, padding=1), ResBlock(widths[0])]
        for index in range(downs):
            enc += [
                nn.Conv2d(widths[index], widths[index + 1], 4, stride=2, padding=1),
                ResBlock(widths[index + 1]),
            ]
        enc += [nn.GroupNorm(8, widths[downs]), nn.SiLU(), nn.Conv2d(widths[downs], config.latent_channels, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        dec = [nn.Conv2d(config.latent_channels, widths[downs], 3, padding=1), ResBlock(widths[downs])]
        for index in reversed(range(downs)):
            dec += [
                nn.ConvTranspose2d(widths[index + 1], widths[index], 4, stride=2, padding=1),
                ResBlock(widths[index]),
            ]
        dec += [nn.GroupNorm(8, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], config.channels, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, *, bitlinear: bool) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = linear(dim, dim * 3, bitlinear=bitlinear)
        self.out = linear(dim, dim, bitlinear=bitlinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(batch, tokens, dim))


class FlowBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, *, bitlinear: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, bitlinear=bitlinear)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(linear(dim, hidden, bitlinear=bitlinear), nn.GELU(), linear(hidden, dim, bitlinear=bitlinear))
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.cond(cond).chunk(4, dim=-1)
        y = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + self.attn(y)
        y = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        return x + self.mlp(y)


class LatentFlowDiT(nn.Module):
    def __init__(self, config: LatentFlowConfig) -> None:
        super().__init__()
        self.config = config
        latent_size = config.latent_size
        if latent_size % config.patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        patch_dim = config.latent_channels * config.patch_size * config.patch_size
        side = latent_size // config.patch_size
        self.side = side
        self.patch_in = linear(patch_dim, config.dim, bitlinear=config.bitlinear)
        self.patch_out = linear(config.dim, patch_dim, bitlinear=config.bitlinear)
        self.pos = nn.Parameter(torch.zeros(1, side * side, config.dim))
        self.time_mlp = nn.Sequential(nn.Linear(config.dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.class_embed = nn.Embedding(max(config.class_count, 1), config.dim)
        self.prompt_embed = nn.Embedding(config.prompt_vocab_size, config.dim) if config.prompt_vocab_size > 0 else None
        self.blocks = nn.ModuleList([FlowBlock(config.dim, config.heads, config.mlp_ratio, bitlinear=config.bitlinear) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.dim)
        nn.init.normal_(self.pos, std=0.02)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        return F.unfold(z, kernel_size=p, stride=p).transpose(1, 2)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        patches = patches.transpose(1, 2)
        return F.fold(patches, output_size=(self.config.latent_size, self.config.latent_size), kernel_size=p, stride=p)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor | None = None,
        prompt_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.patch_in(self.patchify(z)) + self.pos
        cond = self.time_mlp(timestep_embedding(t * 1000.0, self.config.dim))
        if labels is not None and self.config.class_count > 0:
            cond = cond + self.class_embed(labels.clamp(0, self.config.class_count - 1))
        if prompt_ids is not None and self.prompt_embed is not None:
            cond = cond + self.prompt_embed(prompt_ids.clamp(0, self.config.prompt_vocab_size - 1)).mean(dim=1)
        for block in self.blocks:
            tokens = block(tokens, cond)
        return self.unpatchify(self.patch_out(self.norm(tokens)))


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def make_dataset(args: argparse.Namespace, config: LatentFlowConfig) -> torch.utils.data.Dataset:
    transform = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset == "imagefolder":
        return datasets.ImageFolder(args.data_dir, transform=transform)
    if args.dataset == "teacher_jsonl":
        metadata_path = Path(args.teacher_metadata)
        root = Path(args.teacher_root or metadata_path.parent)
        rows = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("path") and row.get("prompt"):
                        rows.append(row)
        if not rows:
            raise ValueError(f"no teacher rows found in {metadata_path}")

        class TeacherJsonlDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(rows)

            def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
                from PIL import Image

                row = rows[int(index)]
                image = Image.open(root / row["path"]).convert("RGB")
                return transform(image), 0

        return TeacherJsonlDataset()
    if args.dataset == "cifar10":
        return datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    if args.dataset == "hf":
        try:
            from datasets import load_dataset
        except Exception as error:
            raise RuntimeError("Hugging Face dataset loading requires the 'datasets' package") from error
        hf_dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        image_column = args.hf_image_column

        class HfImageDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(hf_dataset)

            def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
                item = hf_dataset[int(index)]
                image = item[image_column].convert("RGB")
                label = int(item.get(args.hf_label_column, 0) or 0) if args.hf_label_column else 0
                return transform(image), label

        return HfImageDataset()
    if args.dataset == "synthetic":
        class SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return int(args.synthetic_size)

            def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
                generator = torch.Generator().manual_seed(args.seed + int(index))
                image = torch.zeros(3, config.image_size, config.image_size)
                yy, xx = torch.meshgrid(torch.arange(config.image_size), torch.arange(config.image_size), indexing="ij")
                image[:] = torch.rand(3, generator=generator)[:, None, None] * 0.4
                for shape_index in range(4):
                    color = torch.rand(3, generator=generator)[:, None]
                    cx = int(torch.randint(config.image_size // 8, config.image_size - config.image_size // 8, (1,), generator=generator).item())
                    cy = int(torch.randint(config.image_size // 8, config.image_size - config.image_size // 8, (1,), generator=generator).item())
                    radius = int(torch.randint(max(3, config.image_size // 16), max(4, config.image_size // 4), (1,), generator=generator).item())
                    mask = ((xx - cx).pow(2) + (yy - cy).pow(2)) < radius**2
                    image[:, mask] = color
                    if shape_index % 2 == 0:
                        band = (yy - cy).abs() < max(2, radius // 3)
                        image[:, band] = torch.maximum(image[:, band], color * 0.7)
                return image.clamp(0, 1) * 2 - 1, 0

        return SyntheticDataset()
    raise ValueError(f"unknown dataset: {args.dataset}")


def save_checkpoint(path: Path, *, config: LatentFlowConfig, autoencoder: TinyAutoencoder, flow: LatentFlowDiT | None, step: int, loss: float, mode: str) -> None:
    payload = {
        "architecture": "agentkernel-lite-latent-rectified-flow-v0",
        "mode": mode,
        "step": int(step),
        "loss": float(loss),
        "config": asdict(config),
        "autoencoder": autoencoder.state_dict(),
    }
    if flow is not None:
        payload["flow"] = flow.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_autoencoder(path: str, autoencoder: TinyAutoencoder, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder"])
    return checkpoint


@torch.no_grad()
def sample_flow(autoencoder: TinyAutoencoder, flow: LatentFlowDiT, config: LatentFlowConfig, labels: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    z = torch.randn(labels.shape[0], config.latent_channels, config.latent_size, config.latent_size, device=device)
    for index in range(steps):
        t = torch.full((labels.shape[0],), index / max(steps - 1, 1), device=device)
        v = flow(z, t, labels)
        z = z + v / steps
    return autoencoder.decode(z).clamp(-1, 1)


def train_autoencoder(args: argparse.Namespace, config: LatentFlowConfig, loader: DataLoader, device: torch.device) -> None:
    autoencoder = TinyAutoencoder(config).to(device)
    if args.resume:
        load_autoencoder(args.resume, autoencoder, device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    last_loss = 0.0
    output_dir = Path(args.output_dir)
    while step < args.steps:
        for images, _labels in loader:
            images = images.to(device, non_blocking=True)
            recon = autoencoder(images)
            loss = F.l1_loss(recon, images) + args.edge_weight * edge_loss(recon, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            last_loss = float(loss.item())
            if step % args.log_every == 0:
                print(json.dumps({"mode": "autoencoder", "step": step, "loss": last_loss}), flush=True)
            if step % args.sample_every == 0 or step == args.steps:
                sample_path = output_dir / f"autoencoder_recon_step_{step:06d}.png"
                utils.save_image(torch.cat([images[:8], recon[:8]], dim=0) * 0.5 + 0.5, sample_path, nrow=8)
                save_checkpoint(output_dir / "autoencoder.pt", config=config, autoencoder=autoencoder, flow=None, step=step, loss=last_loss, mode="autoencoder")
            if step >= args.steps:
                break
    save_checkpoint(output_dir / "autoencoder.pt", config=config, autoencoder=autoencoder, flow=None, step=step, loss=last_loss, mode="autoencoder")


def train_flow(args: argparse.Namespace, config: LatentFlowConfig, loader: DataLoader, device: torch.device) -> None:
    autoencoder = TinyAutoencoder(config).to(device)
    if not args.autoencoder_checkpoint:
        raise ValueError("--autoencoder-checkpoint is required for --mode flow")
    load_autoencoder(args.autoencoder_checkpoint, autoencoder, device)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    flow = LatentFlowDiT(config).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        flow.load_state_dict(checkpoint["flow"])
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    last_loss = 0.0
    output_dir = Path(args.output_dir)
    while step < args.steps:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) if config.class_count > 0 else None
            with torch.no_grad():
                z1 = autoencoder.encode(images)
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], device=device)
            zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
            target = z1 - z0
            pred = flow(zt, t, labels)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            last_loss = float(loss.item())
            if step % args.log_every == 0:
                print(json.dumps({"mode": "flow", "step": step, "loss": last_loss}), flush=True)
            if step % args.sample_every == 0 or step == args.steps:
                sample_labels = torch.arange(0, min(8, max(config.class_count, 1)), device=device)
                samples = sample_flow(autoencoder, flow, config, sample_labels, args.sample_steps, device)
                utils.save_image(samples * 0.5 + 0.5, output_dir / f"latent_flow_samples_step_{step:06d}.png", nrow=4)
                save_checkpoint(output_dir / "latent_flow.pt", config=config, autoencoder=autoencoder, flow=flow, step=step, loss=last_loss, mode="flow")
            if step >= args.steps:
                break
    save_checkpoint(output_dir / "latent_flow.pt", config=config, autoencoder=autoencoder, flow=flow, step=step, loss=last_loss, mode="flow")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Agent Kernel Lite's next image model: latent rectified flow for browser/mobile export.")
    parser.add_argument("--mode", choices=("autoencoder", "flow"), required=True)
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_latent_flow_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--autoencoder-checkpoint", default="")
    parser.add_argument("--dataset", choices=("imagefolder", "teacher_jsonl", "cifar10", "hf", "synthetic"), default="imagefolder")
    parser.add_argument("--data-dir", default="data/images")
    parser.add_argument("--teacher-metadata", default="")
    parser.add_argument("--teacher-root", default="")
    parser.add_argument("--hf-dataset", default="")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-image-column", default="image")
    parser.add_argument("--hf-label-column", default="")
    parser.add_argument("--synthetic-size", type=int, default=10000)
    parser.add_argument("--device", default="")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--class-count", type=int, default=0)
    parser.add_argument("--bitlinear", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--edge-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = LatentFlowConfig(
        image_size=args.image_size,
        downsample=args.downsample,
        latent_channels=args.latent_channels,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        class_count=args.class_count,
        bitlinear=args.bitlinear,
    )
    dataset = make_dataset(args, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if args.mode == "autoencoder":
        train_autoencoder(args, config, loader, device)
    else:
        train_flow(args, config, loader, device)


if __name__ == "__main__":
    main()
