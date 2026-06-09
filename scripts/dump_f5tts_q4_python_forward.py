#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_f5tts_q4_streaming import DEFAULT_CHECKPOINT, DEFAULT_VOCAB, build_model


def should_q4_simulate(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim < 2:
        return False
    if "text_embed.text_embed" in name or "mel_spec" in name:
        return False
    return torch.is_floating_point(tensor)


def q4_dequantize_rowwise(tensor: torch.Tensor) -> torch.Tensor:
    original_dtype = tensor.dtype
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 7.0
    quantized = torch.round(flat / scale[:, None]).clamp(-8, 7)
    return (quantized * scale[:, None]).reshape_as(tensor).to(original_dtype)


def apply_q4_state(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = dict(checkpoint["model_state_dict"])
    for name, tensor in list(state.items()):
        if torch.is_tensor(tensor) and should_q4_simulate(name, tensor):
            state[name] = q4_dequantize_rowwise(tensor)
    model.load_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump deterministic Python F5TTS Q4 forward fixture.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--out", default="/data/agent_kernel_lite/tmp/f5tts_q4_forward_fixture.npz")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--time", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = build_model(Path(args.vocab), torch.device("cpu"))
    apply_q4_state(model, Path(args.checkpoint))
    model.eval()

    seq_len = int(args.seq_len)
    mel_dim = 100
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((seq_len, mel_dim), dtype=np.float32) * 0.25
    cond = rng.standard_normal((seq_len, mel_dim), dtype=np.float32) * 0.25
    text_ids = np.full((seq_len,), -1, dtype=np.int32)
    text_ids[: min(seq_len, 16)] = np.arange(min(seq_len, 16), dtype=np.int32)
    time = torch.tensor(float(args.time), dtype=torch.float32)

    with torch.inference_mode():
        x_t = torch.from_numpy(x).unsqueeze(0)
        cond_t = torch.from_numpy(cond).unsqueeze(0)
        text_t = torch.from_numpy(text_ids).unsqueeze(0)
        t_emb = model.transformer.time_embed(time.repeat(1))
        text_emb = model.transformer.text_embed(text_t, seq_len, drop_text=False)
        hidden = model.transformer.input_embed(x_t, cond_t, text_emb, drop_audio_cond=False)
        rope = model.transformer.rotary_embed.forward_from_seq_len(seq_len)
        block_outputs = []
        for block_idx, block in enumerate(model.transformer.transformer_blocks):
            hidden = block(hidden, t_emb, mask=None, rope=rope)
            if block_idx in (0, 1, 21):
                block_outputs.append((f"block_{block_idx}", hidden.squeeze(0).contiguous().cpu().numpy().astype(np.float32)))
        final_norm = model.transformer.norm_out(hidden, t_emb)
        out = model.transformer.proj_out(final_norm).squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        t_np = t_emb.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        t_silu_np = torch.nn.functional.silu(t_emb).squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        text_np = text_emb.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        input_np = model.transformer.input_embed(x_t, cond_t, text_emb, drop_audio_cond=False).squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        final_norm_np = final_norm.squeeze(0).contiguous().cpu().numpy().astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        x=x,
        cond=cond,
        text_ids=text_ids,
        output=out,
        time_embedding=t_np,
        time_embedding_silu=t_silu_np,
        text_embedding=text_np,
        input_embedding=input_np,
        final_norm=final_norm_np,
        time=np.array([args.time], dtype=np.float32),
        **{name: value for name, value in block_outputs},
    )
    print(json.dumps({
        "out": str(out_path),
        "seq_len": seq_len,
        "output_shape": list(out.shape),
        "checksum": float(out.sum()),
        "mean": float(out.mean()),
        "std": float(out.std()),
    }, indent=2))


if __name__ == "__main__":
    main()
