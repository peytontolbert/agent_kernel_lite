#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import torch
from torch import nn
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, import_flux_pipeline
from train_agentkernel_lite_image_flux_diffusiondb_zip_stream import load_zip_items
from train_agentkernel_lite_image_flux_flow_distill import seed_everything


@dataclass
class TinyFluxPromptEncoderConfig:
    t5_vocab_size: int
    clip_vocab_size: int
    max_sequence_length: int = 128
    clip_sequence_length: int = 77
    dim: int = 768
    depth: int = 8
    heads: int = 12
    mlp_ratio: int = 4
    prompt_dim: int = 4096
    pooled_dim: int = 768
    dropout: float = 0.05


def list_zip_shards(repo_id: str, prefix: str, token: str | bool | None) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    shards = [name for name in files if name.startswith(prefix) and name.endswith(".zip")]
    if not shards:
        raise ValueError(f"no zip shards found for {repo_id}/{prefix}")
    return sorted(shards)


def prompt_batch_from_zip(path: Path, limit: int = 0) -> list[str]:
    prompts: list[str] = []
    for _image_name, prompt in load_zip_items(path, limit):
        prompts.append(prompt)
    return prompts


def load_flux_text_pipeline(args: argparse.Namespace):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    _BitsAndBytesConfig, FluxPipeline, _FluxTransformer2DModel = import_flux_pipeline()
    kwargs = {
        "torch_dtype": dtype,
        "transformer": None,
        "vae": None,
    }
    if args.local_files_only:
        kwargs["local_files_only"] = True
    pipe = FluxPipeline.from_pretrained(args.teacher_model, **kwargs)
    for name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.to(args.teacher_device)
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(y)
        return x + self.dropout(self.mlp(self.norm2(x)))


class TinyFluxPromptEncoder(nn.Module):
    def __init__(self, config: TinyFluxPromptEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.t5_embed = nn.Embedding(config.t5_vocab_size, config.dim)
        self.t5_pos = nn.Parameter(torch.zeros(1, config.max_sequence_length, config.dim))
        self.t5_blocks = nn.ModuleList(
            [TransformerBlock(config.dim, config.heads, config.mlp_ratio, config.dropout) for _ in range(config.depth)]
        )
        self.t5_norm = nn.LayerNorm(config.dim)
        self.prompt_out = nn.Sequential(
            nn.Linear(config.dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.prompt_dim),
        )

        self.clip_embed = nn.Embedding(config.clip_vocab_size, config.dim)
        self.clip_pos = nn.Parameter(torch.zeros(1, config.clip_sequence_length, config.dim))
        clip_depth = max(2, config.depth // 2)
        self.clip_blocks = nn.ModuleList(
            [TransformerBlock(config.dim, config.heads, config.mlp_ratio, config.dropout) for _ in range(clip_depth)]
        )
        self.clip_norm = nn.LayerNorm(config.dim)
        self.pooled_out = nn.Linear(config.dim, config.pooled_dim)
        nn.init.normal_(self.t5_pos, std=0.02)
        nn.init.normal_(self.clip_pos, std=0.02)

    def forward(
        self,
        t5_input_ids: torch.Tensor,
        t5_attention_mask: torch.Tensor,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t5 = self.t5_embed(t5_input_ids) + self.t5_pos[:, : t5_input_ids.shape[1]]
        t5_padding = ~t5_attention_mask.bool()
        for block in self.t5_blocks:
            t5 = block(t5, t5_padding)
        prompt_embeds = self.prompt_out(self.t5_norm(t5))

        clip = self.clip_embed(clip_input_ids) + self.clip_pos[:, : clip_input_ids.shape[1]]
        clip_padding = ~clip_attention_mask.bool()
        for block in self.clip_blocks:
            clip = block(clip, clip_padding)
        clip = self.clip_norm(clip)
        mask = clip_attention_mask.float().unsqueeze(-1)
        pooled = (clip * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return prompt_embeds, self.pooled_out(pooled)


def tokenize(pipe, prompts: list[str], max_sequence_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t5 = pipe.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    clip = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return (
        t5.input_ids.to(device),
        t5.attention_mask.to(device),
        clip.input_ids.to(device),
        clip.attention_mask.to(device),
    )


def save_checkpoint(output_dir: Path, model: TinyFluxPromptEncoder, config: TinyFluxPromptEncoderConfig, step: int, loss: float, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_kind": "agentkernel_lite_flux_tiny_prompt_encoder_distill_checkpoint",
            "step": int(step),
            "loss": float(loss),
            "config": asdict(config),
            "model": model.state_dict(),
            "args": vars(args),
        },
        output_dir / "tiny_flux_prompt_encoder.pt",
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

    pipe = load_flux_text_pipeline(args)
    teacher_device = torch.device(args.teacher_device)

    config = TinyFluxPromptEncoderConfig(
        t5_vocab_size=len(pipe.tokenizer_2),
        clip_vocab_size=len(pipe.tokenizer),
        max_sequence_length=args.max_sequence_length,
        clip_sequence_length=pipe.tokenizer.model_max_length,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    )
    student_device = torch.device(args.student_device)
    model = TinyFluxPromptEncoder(config).to(student_device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        start_step = int(checkpoint.get("step", 0))
    else:
        start_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ledger_path = output_dir / "tiny_prompt_encoder_ledger.jsonl"

    step = start_step
    shard_index = start_step % max(len(shards), 1)
    while step < start_step + args.steps:
        shard_name = shards[shard_index % len(shards)]
        shard_index += 1
        shard_path = Path(hf_hub_download(args.dataset_repo, repo_type="dataset", filename=shard_name, local_dir=str(cache_dir), token=token))
        prompts = prompt_batch_from_zip(shard_path, args.max_items_per_shard)
        random.Random(args.seed + step + shard_index).shuffle(prompts)
        for offset in range(0, len(prompts), args.batch_size):
            if step >= start_step + args.steps:
                break
            batch_prompts = prompts[offset : offset + args.batch_size]
            if not batch_prompts:
                continue
            with torch.no_grad():
                target_prompt, target_pooled, _ = pipe.encode_prompt(
                    prompt=batch_prompts,
                    prompt_2=None,
                    device=teacher_device,
                    num_images_per_prompt=1,
                    max_sequence_length=args.max_sequence_length,
                )
                t5_ids, t5_mask, clip_ids, clip_mask = tokenize(pipe, batch_prompts, args.max_sequence_length, student_device)
            pred_prompt, pred_pooled = model(t5_ids, t5_mask, clip_ids, clip_mask)
            target_prompt = target_prompt.to(student_device, dtype=torch.float32)
            target_pooled = target_pooled.to(student_device, dtype=torch.float32)
            prompt_loss = F.smooth_l1_loss(pred_prompt.float(), target_prompt.float(), beta=args.huber_beta)
            pooled_loss = F.smooth_l1_loss(pred_pooled.float(), target_pooled.float(), beta=args.huber_beta)
            loss = prompt_loss + args.pooled_loss_weight * pooled_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "prompt_loss": float(prompt_loss.detach().item()),
                    "pooled_loss": float(pooled_loss.detach().item()),
                    "shard": shard_name,
                    "prompt": batch_prompts[:2],
                }
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
    parser = argparse.ArgumentParser(description="Distill FLUX text encoders into a compact browser prompt encoder.")
    parser.add_argument("--dataset-repo", default="poloclub/diffusiondb")
    parser.add_argument("--shard-prefix", default="diffusiondb-large-part-1/")
    parser.add_argument("--cache-dir", default="/dev/shm/diffusiondb_zip_cache")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_flux_tiny_prompt_encoder_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-items-per-shard", type=int, default=1000)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pooled-loss-weight", type=float, default=0.25)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
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
