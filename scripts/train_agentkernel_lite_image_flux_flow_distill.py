#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from train_agentkernel_lite_image_sana_latent_distill import (
    apply_bitnet_qat_modules,
    materialize_bitnet_qat_weights,
)


@dataclass
class FluxPackedStudentConfig:
    latent_tokens: int = 1024
    latent_channels: int = 64
    prompt_dim: int = 4096
    pooled_dim: int = 768
    max_sequence_length: int = 512
    dim: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    pos2d_scale: float = 0.0
    timestep_scale: float = 1.0
    local_mixer_scale: float = 0.0
    adapter_rank: int = 0
    adapter_scale: float = 1.0
    adapter_dropout: float = 0.0
    output_refiner_hidden: int = 0
    output_refiner_depth: int = 2
    output_refiner_scale: float = 0.25
    output_log_scale_init: float = 0.0
    prompt_output_scale_clip: float = 0.0
    output_scale_mode: str = "output"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class LowRankAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, scale: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(scale) / float(rank)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) * self.scale


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def fixed_2d_sincos_position(tokens: int, dim: int) -> torch.Tensor:
    side = int(math.sqrt(tokens))
    if side * side != tokens or dim % 4:
        return torch.zeros(1, tokens, dim)
    half = dim // 2
    quarter = dim // 4
    freqs = torch.exp(-math.log(10000) * torch.arange(quarter) / max(quarter - 1, 1))
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    x = x.reshape(-1).float()[:, None] * freqs[None, :]
    y = y.reshape(-1).float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
    if emb.shape[1] != dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb.unsqueeze(0)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = linear(dim, dim * 3)
        self.out = linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.dropout(self.out(y.transpose(1, 2).reshape(batch, tokens, dim)))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = linear(dim, dim)
        self.k = linear(dim, dim)
        self.v = linear(dim, dim)
        self.out = linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        context_tokens = context.shape[1]
        q = self.q(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(batch, context_tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(batch, context_tokens, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.dropout(self.out(y.transpose(1, 2).reshape(batch, tokens, dim)))


class LocalTokenMixer(nn.Module):
    def __init__(self, dim: int, tokens: int, dropout: float = 0.0) -> None:
        super().__init__()
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            raise ValueError("LocalTokenMixer requires a square latent token grid")
        self.side = side
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        y = self.norm(x).transpose(1, 2).reshape(batch, dim, self.side, self.side)
        y = self.pointwise(self.act(self.depthwise(y)))
        y = y.reshape(batch, dim, tokens).transpose(1, 2)
        return self.dropout(y)


def unpack_packed_latent_grid(packed: torch.Tensor) -> torch.Tensor:
    batch, tokens, channels = packed.shape
    side = int(math.sqrt(tokens))
    if side * side != tokens or channels % 4:
        raise ValueError("packed latent grid must be square with channels divisible by 4")
    grid = packed.view(batch, side, side, channels // 4, 2, 2)
    grid = grid.permute(0, 3, 1, 4, 2, 5)
    return grid.reshape(batch, channels // 4, side * 2, side * 2)


def pack_latent_grid(grid: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = grid.shape
    if height % 2 or width % 2:
        raise ValueError("latent grid height/width must be divisible by 2")
    packed = grid.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


class OutputLatentRefiner(nn.Module):
    def __init__(self, channels: int, hidden: int, depth: int, scale: float) -> None:
        super().__init__()
        latent_channels = channels // 4
        layers: list[nn.Module] = [nn.Conv2d(latent_channels, hidden, kernel_size=3, padding=1), nn.SiLU()]
        for _ in range(max(int(depth) - 2, 0)):
            layers += [nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU()]
        layers.append(nn.Conv2d(hidden, latent_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.scale = float(scale)
        final = self.net[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        grid = unpack_packed_latent_grid(packed.float())
        refined = grid + self.scale * self.net(grid)
        return pack_latent_grid(refined).to(dtype=packed.dtype)


class FluxPackedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int,
        tokens: int,
        dropout: float = 0.0,
        local_mixer_scale: float = 0.0,
        adapter_rank: int = 0,
        adapter_scale: float = 1.0,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(linear(dim, hidden), nn.GELU(), linear(hidden, dim))
        self.dropout = nn.Dropout(dropout)
        self.cond = nn.Sequential(nn.SiLU(), linear(dim, dim * 6))
        self.residual_gates = nn.Parameter(torch.ones(3))
        self.local_mixer_scale = float(local_mixer_scale)
        self.local_mixer = LocalTokenMixer(dim, tokens, dropout) if self.local_mixer_scale else None
        self.cross_attn_adapter = (
            LowRankAdapter(dim, dim, adapter_rank, adapter_scale, adapter_dropout) if adapter_rank > 0 else None
        )
        self.mlp_adapter = LowRankAdapter(dim, dim, adapter_rank, adapter_scale, adapter_dropout) if adapter_rank > 0 else None

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2, shift3, scale3 = self.cond(cond).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + self.residual_gates[0] * self.self_attn(y)
        if self.local_mixer is not None:
            x = x + self.local_mixer_scale * self.local_mixer(x)
        y = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        cross = self.cross_attn(y, prompt)
        if self.cross_attn_adapter is not None:
            cross = cross + self.cross_attn_adapter(y)
        x = x + self.residual_gates[1] * cross
        y = self.norm3(x) * (1 + scale3[:, None, :]) + shift3[:, None, :]
        mlp = self.dropout(self.mlp(y))
        if self.mlp_adapter is not None:
            mlp = mlp + self.mlp_adapter(y)
        return x + self.residual_gates[2] * mlp


class FluxPackedStudent(nn.Module):
    def __init__(self, config: FluxPackedStudentConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_in = linear(config.latent_channels, config.dim)
        self.latent_out = linear(config.dim, config.latent_channels)
        self.pos = nn.Parameter(torch.zeros(1, config.latent_tokens, config.dim))
        self.prompt_proj = nn.Sequential(linear(config.prompt_dim, config.dim), nn.GELU(), linear(config.dim, config.dim))
        self.prompt_cond_proj = nn.Sequential(linear(config.dim, config.dim), nn.SiLU(), linear(config.dim, config.dim))
        self.prompt_cond_gate = nn.Parameter(torch.zeros(()))
        self.time_mlp = nn.Sequential(linear(config.dim, config.dim), nn.SiLU(), linear(config.dim, config.dim))
        self.pooled_proj = nn.Sequential(linear(config.pooled_dim, config.dim), nn.SiLU(), linear(config.dim, config.dim))
        self.guidance_mlp = nn.Sequential(linear(config.dim, config.dim), nn.SiLU(), linear(config.dim, config.dim))
        self.prompt_adapter = (
            LowRankAdapter(config.prompt_dim, config.dim, config.adapter_rank, config.adapter_scale, config.adapter_dropout)
            if config.adapter_rank > 0
            else None
        )
        self.pooled_adapter = (
            LowRankAdapter(config.pooled_dim, config.dim, config.adapter_rank, config.adapter_scale, config.adapter_dropout)
            if config.adapter_rank > 0
            else None
        )
        self.guidance_adapter = (
            LowRankAdapter(config.dim, config.dim, config.adapter_rank, config.adapter_scale, config.adapter_dropout)
            if config.adapter_rank > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                FluxPackedBlock(
                    config.dim,
                    config.heads,
                    config.mlp_ratio,
                    config.latent_tokens,
                    config.dropout,
                    config.local_mixer_scale,
                    config.adapter_rank,
                    config.adapter_scale,
                    config.adapter_dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.dim)
        self.output_refiner = (
            OutputLatentRefiner(
                config.latent_channels,
                int(config.output_refiner_hidden),
                int(config.output_refiner_depth),
                float(config.output_refiner_scale),
            )
            if int(config.output_refiner_hidden) > 0
            else None
        )
        self.output_log_scale = nn.Parameter(torch.tensor(float(config.output_log_scale_init)))
        self.prompt_output_scale = linear(config.dim, 1)
        nn.init.zeros_(self.prompt_output_scale.weight)
        nn.init.zeros_(self.prompt_output_scale.bias)
        self.gradient_checkpointing = False
        self.register_buffer("pos2d", fixed_2d_sincos_position(config.latent_tokens, config.dim), persistent=False)
        nn.init.normal_(self.pos, std=0.02)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        guidance: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.latent_in(latents.float()) + self.pos[:, : latents.shape[1]]
        if self.config.pos2d_scale:
            x = x + self.pos2d[:, : latents.shape[1]].to(device=x.device, dtype=x.dtype) * float(self.config.pos2d_scale)
        prompt_input = prompt_embeds.float()
        pooled_input = pooled_prompt_embeds.float()
        guidance_input = timestep_embedding(guidance.float(), self.config.dim)
        prompt = self.prompt_proj(prompt_input)
        if self.prompt_adapter is not None:
            prompt = prompt + self.prompt_adapter(prompt_input)
        prompt_cond = self.prompt_cond_proj(prompt.mean(dim=1))
        cond = self.time_mlp(timestep_embedding(timestep.float() * float(self.config.timestep_scale), self.config.dim))
        pooled = self.pooled_proj(pooled_input)
        if self.pooled_adapter is not None:
            pooled = pooled + self.pooled_adapter(pooled_input)
        guided = self.guidance_mlp(guidance_input)
        if self.guidance_adapter is not None:
            guided = guided + self.guidance_adapter(guidance_input)
        cond = cond + pooled + guided + self.prompt_cond_gate * prompt_cond
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, prompt, cond, use_reentrant=False)
            else:
                x = block(x, prompt, cond)
        hidden = self.norm(x)
        output = self.latent_out(hidden)
        if self.output_refiner is not None:
            output = self.output_refiner(output)
        log_scale = self.output_log_scale.float()
        prompt_log_scale = self.prompt_output_scale((prompt_cond + pooled).float()).squeeze(-1)
        if float(self.config.prompt_output_scale_clip) > 0:
            prompt_log_scale = prompt_log_scale.clamp(
                -float(self.config.prompt_output_scale_clip),
                float(self.config.prompt_output_scale_clip),
            )
        total_scale = torch.exp(log_scale + prompt_log_scale).to(dtype=output.dtype)
        scale_view = total_scale.view(-1, 1, 1)
        if str(getattr(self.config, "output_scale_mode", "output")) == "delta_from_input":
            output = latents.to(output.dtype) + (output - latents.to(output.dtype)) * scale_view
        else:
            output = output * scale_view
        if return_hidden:
            return output, hidden
        return output


def load_rows(target_dir: Path) -> list[dict[str, Any]]:
    metadata_path = target_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no target rows found in {metadata_path}")
    for row in rows:
        row["_target_dir"] = str(target_dir)
    return rows


def load_training_rows(target_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_dir in target_dirs:
        rows.extend(load_rows(target_dir))
    if not rows:
        raise ValueError("no target rows found")
    return rows


def timestep_sampling_weights(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[float] | None:
    mode = args.timestep_sampling_mode
    if mode == "uniform":
        return None
    weights: list[float] = []
    floor = float(args.min_timestep_sampling_weight)
    power = float(args.timestep_sampling_power)
    for row in rows:
        steps = max(float(row.get("steps", 24) or 24), 1.0)
        index = float(row.get("timestep_index", 0) or 0)
        denom = max(steps - 1.0, 1.0)
        position = max(0.0, min(1.0, index / denom))
        if mode == "front":
            weight = (1.0 - position) ** power
        elif mode == "tail":
            weight = position**power
        elif mode == "ends":
            weight = (abs(position - 0.5) * 2.0) ** power
        else:
            raise ValueError(f"unknown timestep sampling mode: {mode}")
        weights.append(max(floor, float(weight)))
    return weights


def infer_config(target_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> FluxPackedStudentConfig:
    target = torch.load(target_dir / rows[0]["target_path"], map_location="cpu")
    embeds = torch.load(target_dir / rows[0]["embedding_path"], map_location="cpu")
    return FluxPackedStudentConfig(
        latent_tokens=int(target["latents"].shape[1]),
        latent_channels=int(target["latents"].shape[2]),
        prompt_dim=int(embeds["prompt_embeds"].shape[2]),
        pooled_dim=int(embeds["pooled_prompt_embeds"].shape[1]),
        max_sequence_length=int(embeds.get("max_sequence_length", embeds["prompt_embeds"].shape[1])),
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=float(args.dropout),
        pos2d_scale=float(args.pos2d_scale),
        timestep_scale=float(args.timestep_scale),
        local_mixer_scale=float(args.local_mixer_scale),
        adapter_rank=int(getattr(args, "adapter_rank", 0)),
        adapter_scale=float(getattr(args, "adapter_scale", 1.0)),
        adapter_dropout=float(getattr(args, "adapter_dropout", 0.0)),
        output_refiner_hidden=int(getattr(args, "output_refiner_hidden", 0)),
        output_refiner_depth=int(getattr(args, "output_refiner_depth", 2)),
        output_refiner_scale=float(getattr(args, "output_refiner_scale", 0.25)),
    )


def load_batch(target_dir: Path, rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = []
    timesteps = []
    targets = []
    prompts = []
    pooled = []
    guidance = []
    for row in rows:
        if "_target_payload" in row:
            target = row["_target_payload"]
        else:
            row_target_dir = Path(row.get("_target_dir") or target_dir)
            target = torch.load(row_target_dir / row["target_path"], map_location="cpu")
        if "_prompt_embeds" in row and "_pooled_prompt_embeds" in row:
            embeds = {
                "prompt_embeds": row["_prompt_embeds"],
                "pooled_prompt_embeds": row["_pooled_prompt_embeds"],
            }
        else:
            row_target_dir = Path(row.get("_target_dir") or target_dir)
            embeds = torch.load(row_target_dir / row["embedding_path"], map_location="cpu")
        latents.append(target["latents"][0])
        timesteps.append(target["timestep"].reshape(()))
        targets.append(target["teacher_target"][0])
        prompts.append(embeds["prompt_embeds"][0])
        pooled.append(embeds["pooled_prompt_embeds"][0])
        guidance.append(torch.tensor(float(target.get("guidance", row.get("guidance", 3.5)))))
    return (
        torch.stack(latents).to(device),
        torch.stack(timesteps).to(device),
        torch.stack(targets).to(device),
        torch.stack(prompts).to(device),
        torch.stack(pooled).to(device),
        torch.stack(guidance).to(device),
    )


def preload_rows(rows: list[dict[str, Any]]) -> None:
    embedding_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    target_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for row in rows:
        target_dir = Path(row["_target_dir"])
        embedding_key = (str(target_dir), str(row["embedding_path"]))
        if embedding_key not in embedding_cache:
            embeds = torch.load(target_dir / row["embedding_path"], map_location="cpu")
            embedding_cache[embedding_key] = (
                embeds["prompt_embeds"].contiguous(),
                embeds["pooled_prompt_embeds"].contiguous(),
            )
        row["_prompt_embeds"], row["_pooled_prompt_embeds"] = embedding_cache[embedding_key]

        target_key = (str(target_dir), str(row["target_path"]))
        if target_key not in target_cache:
            target = torch.load(target_dir / row["target_path"], map_location="cpu")
            target_cache[target_key] = {
                "latents": target["latents"].contiguous(),
                "timestep": target["timestep"].reshape(()).contiguous(),
                "teacher_target": target["teacher_target"].contiguous(),
                "guidance": float(target.get("guidance", row.get("guidance", 3.5))),
            }
        row["_target_payload"] = target_cache[target_key]


def direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    pred_flat = F.normalize(pred_flat, dim=-1)
    target_flat = F.normalize(target_flat, dim=-1)
    return (1.0 - (pred_flat * target_flat).sum(dim=-1)).mean()


def norm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = pred.float().flatten(1).norm(dim=-1)
    target_norm = target.float().flatten(1).norm(dim=-1).clamp_min(1e-6)
    return F.mse_loss(pred_norm / target_norm, torch.ones_like(target_norm))


def packed_spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tokens = pred.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return torch.zeros((), device=pred.device)
    pred_grid = pred.float().reshape(pred.shape[0], side, side, pred.shape[2])
    target_grid = target.float().reshape(target.shape[0], side, side, target.shape[2])
    pred_dx = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
    target_dx = target_grid[:, :, 1:, :] - target_grid[:, :, :-1, :]
    pred_dy = pred_grid[:, 1:, :, :] - pred_grid[:, :-1, :, :]
    target_dy = target_grid[:, 1:, :, :] - target_grid[:, :-1, :, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def update_ema_state(ema_state: dict[str, torch.Tensor], module: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for key, value in module.state_dict().items():
            source = value.detach().cpu()
            if key not in ema_state:
                ema_state[key] = source.clone()
            elif torch.is_floating_point(ema_state[key]):
                ema_state[key].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema_state[key].copy_(source)


def save_checkpoint(
    output_dir: Path,
    config: FluxPackedStudentConfig,
    student: nn.Module,
    step: int,
    loss: float,
    args: argparse.Namespace,
    ema_state: dict[str, torch.Tensor] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_state = None
    materialized_modules = 0
    if args.save_materialized_bitnet:
        materialized = FluxPackedStudent(config).to("cpu")
        materialized_modules = apply_bitnet_qat_modules(
            materialized,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
        )
        materialized.load_state_dict({key: value.detach().cpu() for key, value in student.state_dict().items()}, strict=True)
        materialize_bitnet_qat_weights(materialized)
        materialized_state = {
            key: value
            for key, value in materialized.state_dict().items()
            if not key.endswith(".weight_scale")
        }
    checkpoint_path = output_dir / "flux_packed_student.pt"
    tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    payload = {
        "architecture": "agentkernel-lite-flux-packed-flow-student-v0",
        "step": int(step),
        "loss": float(loss),
        "config": asdict(config),
        "bitnet_qat": bool(args.bitnet_qat),
        "ema_decay": float(args.ema_decay) if ema_state is not None else 0.0,
        "materialized_bitnet_qat_modules": int(materialized_modules),
        "student": {key: value.detach().cpu() for key, value in student.state_dict().items()},
        "student_ema": ema_state,
        "student_materialized": materialized_state,
        "optimizer": optimizer_state,
    }
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(checkpoint_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    target_dir = Path(args.target_dir)
    target_dirs = [target_dir] + [Path(value) for value in args.extra_target_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_training_rows(target_dirs)
    if args.preload_rows:
        preload_rows(rows)
    config = infer_config(Path(rows[0]["_target_dir"]), rows, args)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    device = torch.device(args.device)
    student = FluxPackedStudent(config).to(device)
    start_step = 0
    ema_state: dict[str, torch.Tensor] | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        missing, unexpected = student.load_state_dict(checkpoint.get("student", checkpoint), strict=False)
        if missing or unexpected:
            print(json.dumps({"resume_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
        start_step = int(checkpoint.get("step") or 0)
        if args.ema_decay > 0 and checkpoint.get("student_ema"):
            ema_state = {key: value.detach().cpu().clone() for key, value in checkpoint["student_ema"].items()}
    if args.bitnet_qat:
        modules = apply_bitnet_qat_modules(
            student,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
        )
        print(json.dumps({"bitnet_qat_enabled": {"modules": modules, "learned_scale": bool(args.bitnet_qat_learned_scale)}}), flush=True)
    trainable_params = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.ema_decay > 0 and ema_state is None:
        ema_state = clone_state_dict(student)
    sample_weights = timestep_sampling_weights(rows, args)
    ledger_path = output_dir / "flux_flow_distill_ledger.jsonl"
    last_loss = 0.0
    last_flow_loss = 0.0
    last_direction_loss = 0.0
    last_norm_loss = 0.0
    last_spatial_loss = 0.0
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        flow_total = 0.0
        direction_total = 0.0
        norm_total = 0.0
        spatial_total = 0.0
        logged_rows: list[dict[str, Any]] = []
        grad_accum_steps = max(int(args.grad_accum_steps), 1)
        for _micro_step in range(grad_accum_steps):
            batch_rows = random.choices(rows, weights=sample_weights, k=args.batch_size)
            logged_rows = batch_rows
            latents, timesteps, targets, prompt_embeds, pooled_prompt_embeds, guidance = load_batch(target_dir, batch_rows, device)
            pred = student(latents, timesteps, prompt_embeds, pooled_prompt_embeds, guidance)
            flow_value = F.mse_loss(pred.float(), targets.float())
            direction_value = direction_loss(pred, targets) if args.direction_loss_weight > 0 else torch.zeros((), device=device)
            norm_value = norm_loss(pred, targets) if args.norm_loss_weight > 0 else torch.zeros((), device=device)
            spatial_value = packed_spatial_gradient_loss(pred, targets) if args.spatial_loss_weight > 0 else torch.zeros((), device=device)
            loss = (
                args.flow_loss_weight * flow_value
                + args.direction_loss_weight * direction_value
                + args.norm_loss_weight * norm_value
                + args.spatial_loss_weight * spatial_value
            )
            (loss / grad_accum_steps).backward()
            loss_total += float(loss.item())
            flow_total += float(flow_value.item())
            direction_total += float(direction_value.item())
            norm_total += float(norm_value.item())
            spatial_total += float(spatial_value.item())
        nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        if ema_state is not None:
            update_ema_state(ema_state, student, float(args.ema_decay))
        last_loss = loss_total / grad_accum_steps
        last_flow_loss = flow_total / grad_accum_steps
        last_direction_loss = direction_total / grad_accum_steps
        last_norm_loss = norm_total / grad_accum_steps
        last_spatial_loss = spatial_total / grad_accum_steps
        if step % args.log_every == 0 or local_step == 1:
            ledger = {
                "step": step,
                "loss": last_loss,
                "flow_loss": last_flow_loss,
                "direction_loss": last_direction_loss,
                "norm_loss": last_norm_loss,
                "spatial_loss": last_spatial_loss,
                "batch_size": args.batch_size,
                "grad_accum_steps": grad_accum_steps,
                "target_rows": len(rows),
                "timestep_sampling_mode": args.timestep_sampling_mode,
                "prompt": logged_rows[0].get("prompt", "") if logged_rows else "",
                "timestep_index": logged_rows[0].get("timestep_index", -1) if logged_rows else -1,
            }
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
            print(json.dumps(ledger), flush=True)
        if step % args.checkpoint_every == 0 or local_step == args.steps:
            save_checkpoint(output_dir, config, student, step, last_loss, args, ema_state)
    save_checkpoint(output_dir, config, student, start_step + args.steps, last_loss, args, ema_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FLUX packed-latent student on teacher flow targets.")
    parser.add_argument("--target-dir", default="data/vision/flux1_dev_flow_targets_smoke_v0")
    parser.add_argument("--extra-target-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_flux_flow_student_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--preload-rows", action="store_true")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pos2d-scale", type=float, default=0.0)
    parser.add_argument("--timestep-scale", type=float, default=1.0)
    parser.add_argument("--local-mixer-scale", type=float, default=0.0)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--output-refiner-hidden", type=int, default=0)
    parser.add_argument("--output-refiner-depth", type=int, default=2)
    parser.add_argument("--output-refiner-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--timestep-sampling-mode", choices=("uniform", "front", "tail", "ends"), default="uniform")
    parser.add_argument("--timestep-sampling-power", type=float, default=2.0)
    parser.add_argument("--min-timestep-sampling-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
