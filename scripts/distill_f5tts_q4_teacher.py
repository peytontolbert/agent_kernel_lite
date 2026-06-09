#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from f5_tts.model.utils import lens_to_mask, list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_f5tts_q4_streaming import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    DEFAULT_VOCAB,
    apply_q4_parametrizations,
    audio_array_to_item,
    build_model,
    configure_trainable_parameters,
    load_checkpoint_state,
    load_local_samples,
    local_row_to_item,
    make_batch,
    materialized_state_dict,
    row_to_item,
    split_csv,
    stream_hf_rows,
)


def text_to_ids(model, text, device: torch.device) -> torch.Tensor:
    if isinstance(text, torch.Tensor):
        return text.to(device)
    if isinstance(text, list):
        if getattr(model, "vocab_char_map", None):
            return list_str_to_idx(text, model.vocab_char_map).to(device)
        return list_str_to_tensor(text).to(device)
    raise TypeError(f"unsupported text batch type: {type(text)!r}")


def make_teacher_batch(model, mel: torch.Tensor, text, lens: torch.Tensor, *, device: torch.device):
    batch, seq_len = mel.shape[:2]
    mask = lens_to_mask(lens, length=seq_len)
    frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*model.frac_lengths_mask)
    rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
    rand_span_mask &= mask

    x1 = mel
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=mel.dtype, device=device)
    t = time[:, None, None]
    phi = (1.0 - t) * x0 + t * x1
    flow = x1 - x0
    cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
    return phi, cond, text_to_ids(model, text, device), time, rand_span_mask, flow


def save_checkpoint(model, path: Path, *, step: int, args: argparse.Namespace) -> None:
    checkpoint = {
        "model_state_dict": materialized_state_dict(model),
        "step": step,
        "args": vars(args),
    }
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

    teacher = build_model(Path(args.vocab), device)
    student = build_model(Path(args.vocab), device)
    load_checkpoint_state(teacher, Path(args.checkpoint))
    load_checkpoint_state(student, Path(args.checkpoint))
    teacher.eval().requires_grad_(False)

    q4_modules, q4_params = apply_q4_parametrizations(
        student,
        include=split_csv(args.q4_include),
        exclude=split_csv(args.q4_exclude),
    )
    train_tensors, train_params = configure_trainable_parameters(
        student,
        train_include=split_csv(args.train_include),
        train_exclude=split_csv(args.train_exclude),
    )
    student.train()
    print(json.dumps({
        "q4_modules": q4_modules,
        "q4_params": q4_params,
        "train_tensors": train_tensors,
        "train_params": train_params,
        "device": str(device),
        "loss": "teacher_mse + flow_mse",
    }, indent=2))

    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))

    local_rows = load_local_samples(args.local_samples)
    if local_rows:
        print(json.dumps({"local_samples": len(local_rows), "local_samples_path": args.local_samples}))
    use_hf_stream = bool(args.dataset) and float(args.local_sample_prob) < 1.0
    row_iter = iter(stream_hf_rows(args)) if use_hf_stream else iter(())

    pending = []
    step = 0
    best_distill_loss = float("inf")
    while step < int(args.max_steps):
        use_local = bool(local_rows) and random.random() < float(args.local_sample_prob)
        if use_local:
            item = local_row_to_item(random.choice(local_rows), args=args, model=student, device=device)
        else:
            try:
                item = row_to_item(next(row_iter), args=args, model=student, device=device)
            except StopIteration:
                if not local_rows:
                    break
                item = local_row_to_item(random.choice(local_rows), args=args, model=student, device=device)
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
        phi, cond, text_ids, time, rand_span_mask, flow = make_teacher_batch(student, mel, text, lens, device=device)
        drop_audio_cond = random.random() < float(args.drop_audio_cond_prob)
        drop_text = random.random() < float(args.drop_text_prob)

        with torch.no_grad():
            teacher_pred = teacher.transformer(
                x=phi,
                cond=cond,
                text=text_ids,
                time=time,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
            )
        student_pred = student.transformer(
            x=phi,
            cond=cond,
            text=text_ids,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )
        distill_loss = F.mse_loss(student_pred[rand_span_mask], teacher_pred[rand_span_mask])
        flow_loss = F.mse_loss(student_pred[rand_span_mask], flow[rand_span_mask])
        loss = distill_loss + float(args.flow_loss_weight) * flow_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.max_grad_norm))
        optimizer.step()

        step += 1
        distill_value = float(distill_loss.detach().cpu())
        if distill_value < best_distill_loss:
            best_distill_loss = distill_value
            if bool(args.save_best):
                save_path = output_dir / "model_q4_distill_best.pt"
                save_checkpoint(student, save_path, step=step, args=args)
        if step % int(args.log_every) == 0:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "distill_loss": distill_value,
                "flow_loss": float(flow_loss.detach().cpu()),
                "best_distill_loss": best_distill_loss,
            }))
        if step % int(args.save_every) == 0:
            save_path = output_dir / f"model_q4_distill_step_{step}.pt"
            save_checkpoint(student, save_path, step=step, args=args)
            print(f"saved={save_path}")

    final_path = output_dir / "model_q4_distill_last.pt"
    save_checkpoint(student, final_path, step=step, args=args)
    print(f"saved={final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill an F5TTS Q4 STE student from the original FP teacher.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", default=str(Path(DEFAULT_OUTPUT).with_name("f5tts_peyton_q4_teacher_distill")))
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
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument("--local-samples", default="/data/resumebot/voice_profiles/Peyton/samples.txt")
    parser.add_argument("--local-sample-prob", type=float, default=0.25)
    parser.add_argument("--train-include", default="transformer.transformer_blocks.21,transformer.norm_out")
    parser.add_argument("--train-exclude", default="")
    parser.add_argument("--drop-audio-cond-prob", type=float, default=0.2)
    parser.add_argument("--drop-text-prob", type=float, default=0.1)
    parser.add_argument("--flow-loss-weight", type=float, default=0.1)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    train(args)


if __name__ == "__main__":
    main()
