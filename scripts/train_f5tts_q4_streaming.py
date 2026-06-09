#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from itertools import islice
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F


DEFAULT_CHECKPOINT = "/data/resumebot/checkpoints/final_finetuned_model.pt"
DEFAULT_VOCAB = "/data/resumebot/checkpoints/F5TTS_Base_vocab.txt"
DEFAULT_OUTPUT = "/data/agent_kernel_lite/artifacts/f5tts_peyton_q4_streaming"


class RowwiseQ4STE(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        original_dtype = weight.dtype
        flat = weight.float().reshape(weight.shape[0], -1)
        scale = flat.detach().abs().amax(dim=1).clamp_min(1e-8) / 7.0
        quantized = torch.round(flat / scale[:, None]).clamp(-8, 7)
        dequantized = (quantized * scale[:, None]).reshape_as(weight).to(original_dtype)
        return weight + (dequantized - weight).detach()


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_vocab(path: Path) -> tuple[dict[str, int], int]:
    vocab = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    vocab_char_map = {char: idx for idx, char in enumerate(vocab) if char}
    return vocab_char_map, len(vocab_char_map) + 1


def should_q4_module(name: str, module: nn.Module, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return False
    if include and not any(item in name for item in include):
        return False
    if exclude and any(item in name for item in exclude):
        return False
    return hasattr(module, "weight") and module.weight.ndim >= 2


def apply_q4_parametrizations(
    model: nn.Module,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> tuple[int, int]:
    modules = 0
    params = 0
    for name, module in model.named_modules():
        if should_q4_module(name, module, include, exclude):
            parametrize.register_parametrization(module, "weight", RowwiseQ4STE())
            modules += 1
            params += int(module.parametrizations.weight.original.numel())
    return modules, params


def configure_trainable_parameters(
    model: nn.Module,
    *,
    train_include: tuple[str, ...],
    train_exclude: tuple[str, ...],
) -> tuple[int, int]:
    if not train_include and not train_exclude:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tensors = sum(1 for p in model.parameters() if p.requires_grad)
        return tensors, int(params)

    tensors = 0
    params = 0
    for name, parameter in model.named_parameters():
        trainable = True
        if train_include:
            trainable = any(item in name for item in train_include)
        if train_exclude and any(item in name for item in train_exclude):
            trainable = False
        parameter.requires_grad_(trainable)
        if trainable:
            tensors += 1
            params += int(parameter.numel())
    return tensors, params


def build_model(vocab_path: Path, device: torch.device):
    from f5_tts.model import CFM, DiT

    vocab_char_map, vocab_size = load_vocab(vocab_path)
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    return model.to(device)


def load_checkpoint_state(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"missing": missing, "unexpected": unexpected}, indent=2))
    del checkpoint


def stream_hf_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {"split": args.split, "streaming": True}
    dataset = load_dataset(args.dataset, args.config or None, **kwargs)
    if args.audio_column:
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sample_rate))
    if int(args.shuffle_buffer) > 0:
        dataset = dataset.shuffle(buffer_size=int(args.shuffle_buffer), seed=int(args.seed))
    return dataset


def audio_array_to_item(
    array: Any,
    text: str,
    sample_rate: int,
    *,
    args: argparse.Namespace,
    model,
    device: torch.device,
) -> dict[str, Any] | None:
    if array is None:
        return None
    audio = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)
    duration = float(audio.numel()) / float(sample_rate)
    if duration < args.min_duration or duration > args.max_duration:
        return None
    if sample_rate != args.sample_rate:
        audio = F.interpolate(
            audio.view(1, 1, -1),
            size=int(round(audio.numel() * args.sample_rate / sample_rate)),
            mode="linear",
            align_corners=False,
        ).view(-1)
    with torch.no_grad():
        mel = model.mel_spec(audio.to(device).view(1, -1)).detach().cpu().squeeze(0)
    return {"mel_spec": mel, "text": str(text)}


def row_to_item(row: dict[str, Any], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    audio_obj = row.get(args.audio_column)
    text = row.get(args.text_column)
    if not audio_obj or text is None:
        return None
    return audio_array_to_item(
        audio_obj.get("array"),
        str(text),
        int(audio_obj.get("sampling_rate") or args.sample_rate),
        args=args,
        model=model,
        device=device,
    )


def load_local_samples(samples_path: str) -> list[tuple[Path, str]]:
    path = Path(samples_path)
    if not path.exists():
        return []
    rows: list[tuple[Path, str]] = []
    base_dir = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "|" not in line:
            continue
        audio_ref, text = line.split("|", 1)
        audio_path = Path(audio_ref)
        if not audio_path.exists():
            basename = PureWindowsPath(audio_ref).name if "\\" in audio_ref else audio_path.name
            audio_path = base_dir / basename
        if audio_path.exists():
            rows.append((audio_path, text.strip()))
    return rows


def local_row_to_item(row: tuple[Path, str], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    audio_path, text = row
    array, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
    return audio_array_to_item(array, text, int(sample_rate), args=args, model=model, device=device)


def make_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    from f5_tts.model.dataset import collate_fn

    return collate_fn(items)


def materialized_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".parametrizations.weight." in name:
            continue
        state[name] = tensor.detach().cpu()
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            key = f"{name}.weight" if name else "weight"
            state[key] = module.weight.detach().cpu()
    return state


def save_training_checkpoint(model: nn.Module, path: Path, *, step: int, args: argparse.Namespace) -> None:
    checkpoint = {
        "model_state_dict": materialized_state_dict(model),
        "step": step,
        "args": vars(args),
    }
    if args.save_qat_state:
        checkpoint["qat_state_dict"] = model.state_dict()
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(Path(args.vocab), device)
    load_checkpoint_state(model, Path(args.checkpoint))
    q4_modules, q4_params = apply_q4_parametrizations(
        model,
        include=split_csv(args.q4_include),
        exclude=split_csv(args.q4_exclude),
    )
    train_tensors, train_params = configure_trainable_parameters(
        model,
        train_include=split_csv(args.train_include),
        train_exclude=split_csv(args.train_exclude),
    )
    model.train()
    print(json.dumps({
        "q4_modules": q4_modules,
        "q4_params": q4_params,
        "train_tensors": train_tensors,
        "train_params": train_params,
        "device": str(device),
    }, indent=2))

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    local_rows = load_local_samples(args.local_samples)
    if local_rows:
        print(json.dumps({"local_samples": len(local_rows), "local_samples_path": args.local_samples}))
    use_hf_stream = bool(args.dataset) and float(args.local_sample_prob) < 1.0
    row_iter = iter(stream_hf_rows(args)) if use_hf_stream else iter(())
    pending: list[dict[str, Any]] = []
    step = 0
    while step < int(args.max_steps):
        use_local = bool(local_rows) and random.random() < float(args.local_sample_prob)
        if use_local:
            item = local_row_to_item(random.choice(local_rows), args=args, model=model, device=device)
        else:
            try:
                row = next(row_iter)
            except StopIteration:
                if local_rows:
                    item = local_row_to_item(random.choice(local_rows), args=args, model=model, device=device)
                    if item is not None:
                        pending.append(item)
                    continue
                break
            item = row_to_item(row, args=args, model=model, device=device)
        if item is None:
            continue
        pending.append(item)
        if len(pending) < int(args.batch_size):
            continue

        batch = make_batch(pending)
        pending = []
        mel = batch["mel"].permute(0, 2, 1).to(device)
        lens = batch["mel_lengths"].to(device)
        text = batch["text"]

        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(mel, text, lens=lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()

        step += 1
        if step % int(args.log_every) == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu())}))
        if step % int(args.save_every) == 0:
            save_path = output_dir / f"model_q4_step_{step}.pt"
            save_training_checkpoint(model, save_path, step=step, args=args)
            print(f"saved={save_path}")
        if step >= int(args.max_steps):
            break

    final_path = output_dir / "model_q4_last.pt"
    save_training_checkpoint(model, final_path, step=step, args=args)
    print(f"saved={final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream HF audio and fine-tune F5TTS with rowwise Q4 STE weights.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="train.100")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument("--local-samples", default="/data/resumebot/voice_profiles/Peyton/samples.txt")
    parser.add_argument(
        "--local-sample-prob",
        type=float,
        default=0.5,
        help="Probability of drawing a local voice-clone sample instead of the HF stream.",
    )
    parser.add_argument(
        "--train-include",
        default="transformer.transformer_blocks.21,transformer.norm_out",
        help="Comma-separated parameter-name substrings to train. Empty means all parameters.",
    )
    parser.add_argument("--train-exclude", default="")
    parser.add_argument("--save-qat-state", action="store_true", help="Also save parametrized QAT state_dict.")
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    train(args)


if __name__ == "__main__":
    main()
