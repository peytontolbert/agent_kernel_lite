#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class BitDiTConfig:
    image_size: int = 32
    patch_size: int = 4
    channels: int = 3
    class_count: int = 10
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: int = 4
    timesteps: int = 1000
    bitlinear: bool = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
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
    cls = BitLinear if bitlinear else nn.Linear
    return cls(in_features, out_features)


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
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, tokens, dim)
        return self.out(attn)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, *, bitlinear: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, bitlinear=bitlinear)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            linear(dim, hidden, bitlinear=bitlinear),
            nn.GELU(),
            linear(hidden, dim, bitlinear=bitlinear),
        )
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.cond(conditioning).chunk(4, dim=-1)
        y = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + self.attn(y)
        y = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        return x + self.mlp(y)


class BitDiT(nn.Module):
    def __init__(self, config: BitDiTConfig) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.channels * config.patch_size * config.patch_size
        side = config.image_size // config.patch_size
        self.side = side
        self.patch_in = linear(patch_dim, config.dim, bitlinear=config.bitlinear)
        self.pos = nn.Parameter(torch.zeros(1, side * side, config.dim))
        self.class_embed = nn.Embedding(config.class_count, config.dim)
        self.time_mlp = nn.Sequential(nn.Linear(config.dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.blocks = nn.ModuleList(
            [DiTBlock(config.dim, config.heads, config.mlp_ratio, bitlinear=config.bitlinear) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.dim)
        self.patch_out = linear(config.dim, patch_dim, bitlinear=config.bitlinear)
        nn.init.normal_(self.pos, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        patches = F.unfold(images, kernel_size=p, stride=p).transpose(1, 2)
        return patches

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        patches = patches.transpose(1, 2)
        return F.fold(patches, output_size=(self.config.image_size, self.config.image_size), kernel_size=p, stride=p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_in(self.patchify(x)) + self.pos
        conditioning = self.time_mlp(timestep_embedding(t, self.config.dim)) + self.class_embed(labels)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        return self.unpatchify(self.patch_out(self.norm(tokens)))


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device) -> None:
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        b = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return a * x0 + b * noise


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        original = {}
        model_state = model.state_dict()
        for name, value in self.shadow.items():
            if name in model_state:
                original[name] = model_state[name].detach().clone()
                model_state[name].copy_(value)
        return original

    @torch.no_grad()
    def restore(self, model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        model_state = model.state_dict()
        for name, value in original.items():
            model_state[name].copy_(value)


class SyntheticPromptImageDataset(torch.utils.data.Dataset):
    def __init__(self, *, size: int, image_size: int, seed: int) -> None:
        self.size = int(size)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.seed + int(index))
        label = int(torch.randint(0, len(CLASSES), (1,), generator=generator).item())
        image = torch.zeros(3, self.image_size, self.image_size)
        base = torch.rand(3, generator=generator) * 0.45
        image[:] = base[:, None, None]
        yy, xx = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing="ij")
        shape_count = 2 + label % 4
        for shape_index in range(shape_count):
            color = torch.rand(3, generator=generator) * 1.6 - 0.3
            cx = int(torch.randint(6, max(7, self.image_size - 6), (1,), generator=generator).item())
            cy = int(torch.randint(6, max(7, self.image_size - 6), (1,), generator=generator).item())
            radius = int(torch.randint(4, max(5, self.image_size // 3), (1,), generator=generator).item())
            if (label + shape_index) % 3 == 0:
                mask = ((xx - cx).abs() < radius) & ((yy - cy).abs() < max(2, radius // 2))
            elif (label + shape_index) % 3 == 1:
                mask = (xx - cx).pow(2) + (yy - cy).pow(2) < radius**2
            else:
                mask = ((xx - cx).abs() + (yy - cy).abs()) < radius
            image[:, mask] = color[:, None]
        image = image.clamp(0, 1)
        return image * 2 - 1, label


class HfCifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, *, split: str, image_size: int) -> None:
        try:
            from datasets import load_dataset
        except Exception as error:  # pragma: no cover - dependency is environment-specific.
            raise RuntimeError("Hugging Face CIFAR-10 loading requires the 'datasets' package") from error
        self.dataset = load_dataset("cifar10", split=split)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[int(index)]
        return self.transform(item["img"].convert("RGB")), int(item["label"])


@torch.no_grad()
def sample(model: BitDiT, labels: torch.Tensor, schedule: DiffusionSchedule, steps: int, *, sampler: str) -> torch.Tensor:
    model.eval()
    device = labels.device
    x = torch.randn(labels.shape[0], model.config.channels, model.config.image_size, model.config.image_size, device=device)
    if sampler == "ddpm":
        indices = torch.arange(schedule.timesteps - 1, -1, -1, device=device, dtype=torch.long)
    else:
        indices = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()
    for t_value in indices:
        t = torch.full((labels.shape[0],), int(t_value.item()), device=device, dtype=torch.long)
        eps = model(x, t, labels)
        if sampler == "ddpm":
            beta = schedule.betas[t].view(-1, 1, 1, 1)
            alpha = schedule.alphas[t].view(-1, 1, 1, 1)
            alpha_bar = schedule.alpha_bars[t].view(-1, 1, 1, 1)
            mean = (x - beta * eps / (1 - alpha_bar).sqrt().clamp_min(1e-6)) / alpha.sqrt().clamp_min(1e-6)
            if t_value > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
        else:
            alpha_bar = schedule.alpha_bars[t].view(-1, 1, 1, 1)
            pred_x0 = ((x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp_min(1e-6)).clamp(-1, 1)
            if t_value > 0:
                next_t = torch.clamp(t_value - max(schedule.timesteps // steps, 1), min=0)
                next_alpha_bar = schedule.alpha_bars[next_t].view(1, 1, 1, 1)
                x = next_alpha_bar.sqrt() * pred_x0 + (1 - next_alpha_bar).sqrt() * eps
            else:
                x = pred_x0
    return x.clamp(-1, 1)


def save_checkpoint(
    model: BitDiT,
    output_dir: Path,
    step: int,
    loss: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ModelEma | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "step": step,
        "loss": loss,
        "classes": CLASSES,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, output_dir / "model.pt")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "agentkernel-lite-image-bitdit",
                "model_id": output_dir.name,
                "architecture": "bitdit-ddpm-pixel-v0",
                "config": asdict(model.config),
                "classes": CLASSES,
                "checkpoint": "model.pt",
                "status": "training_checkpoint",
                "step": step,
                "loss": loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    config = BitDiTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        timesteps=args.timesteps,
        bitlinear=not args.dense,
    )
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if args.dataset == "synthetic":
        dataset = SyntheticPromptImageDataset(size=args.synthetic_size, image_size=config.image_size, seed=args.seed)
    elif args.dataset == "hf_cifar10":
        dataset = HfCifar10Dataset(split="train", image_size=config.image_size)
    else:
        try:
            dataset = datasets.CIFAR10(
                root=str(_repo_root() / "data" / "vision"),
                train=True,
                download=True,
                transform=transform,
            )
        except Exception as error:
            if not args.allow_synthetic_fallback:
                raise
            print(json.dumps({"warning": f"CIFAR unavailable; using synthetic fallback: {error}"}), flush=True)
            dataset = SyntheticPromptImageDataset(size=args.synthetic_size, image_size=config.image_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    model = BitDiT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None
    schedule = DiffusionSchedule(config.timesteps, device)
    teacher_model = None
    if args.teacher_checkpoint:
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        teacher_config = BitDiTConfig(**teacher_checkpoint["config"])
        teacher_model = BitDiT(teacher_config).to(device)
        teacher_model.load_state_dict(teacher_checkpoint["model"])
        if teacher_checkpoint.get("ema"):
            teacher_ema = ModelEma(teacher_model, 0.0)
            teacher_ema.load_state_dict(teacher_checkpoint["ema"])
            teacher_ema.copy_to(teacher_model)
        teacher_model.eval()
        print(json.dumps({"teacher": str(args.teacher_checkpoint), "step": int(teacher_checkpoint.get("step") or 0)}), flush=True)
    step = 0
    last_loss = float("nan")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_config = checkpoint.get("config") or {}
        if checkpoint_config and checkpoint_config != asdict(config):
            raise ValueError(f"resume config mismatch: {checkpoint_config} != {asdict(config)}")
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None and checkpoint.get("ema"):
            ema.load_state_dict(checkpoint["ema"])
        step = int(checkpoint.get("step") or 0)
        last_loss = float(checkpoint.get("loss") or last_loss)
        print(json.dumps({"resume": str(args.resume), "step": step, "loss": last_loss}), flush=True)
    model.train()
    while step < args.steps:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = schedule.q_sample(images, t, noise)
            pred = model(noisy, t, labels)
            noise_loss = F.mse_loss(pred, noise)
            distill_loss = pred.new_tensor(0.0)
            if teacher_model is not None and args.teacher_weight > 0:
                with torch.no_grad():
                    teacher_pred = teacher_model(noisy, t, labels)
                distill_loss = F.mse_loss(pred, teacher_pred)
            loss = noise_loss + float(args.teacher_weight) * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            step += 1
            last_loss = float(loss.detach().cpu())
            if step % args.log_every == 0 or step == 1:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": last_loss,
                            "noise_loss": float(noise_loss.detach().cpu()),
                            "distill_loss": float(distill_loss.detach().cpu()),
                        }
                    ),
                    flush=True,
                )
            if step % args.sample_every == 0 or step == args.steps:
                labels_sample = torch.arange(0, min(10, config.class_count), device=device)
                original = ema.copy_to(model) if ema is not None else None
                images_sample = sample(model, labels_sample, schedule, args.sample_steps, sampler=args.sampler)
                if ema is not None and original is not None:
                    ema.restore(model, original)
                sample_path = output_dir / f"samples_step_{step:06d}.png"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_image((images_sample + 1) * 0.5, sample_path, nrow=5)
                save_checkpoint(model, output_dir, step, last_loss, optimizer=optimizer, ema=ema)
            if step >= args.steps:
                break
    save_checkpoint(model, output_dir, step, last_loss, optimizer=optimizer, ema=ema)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small AgentKernel Lite BitDiT image model.")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_bitdit_cifar_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--dataset", choices=("cifar10", "hf_cifar10", "synthetic"), default="hf_cifar10")
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--synthetic-size", type=int, default=50000)
    parser.add_argument("--device", default="")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dense", action="store_true", help="Use dense Linear layers instead of BitLinear STE.")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
