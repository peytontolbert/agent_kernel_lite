#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RESUMEBOT_DIR = Path("/data/resumebot")
DEFAULT_CHECKPOINT = RESUMEBOT_DIR / "checkpoints" / "final_finetuned_model.pt"
DEFAULT_VOCAB = RESUMEBOT_DIR / "checkpoints" / "F5TTS_Base_vocab.txt"
DEFAULT_OUT = Path("/data/agent_kernel_lite/native-models/f5tts_peyton_coreml_int4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Peyton F5TTS DiT forward chunks as a Core ML int4 package.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seq-len", type=int, default=96, help="Fixed mel/text sequence length for one native chunk.")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip-palettize", action="store_true", help="Write an uncompressed Core ML package.")
    parser.add_argument("--palettize-mode", default="uniform", choices=["uniform", "kmeans"])
    parser.add_argument("--ios-target", default="iOS18")
    return parser.parse_args()


class F5DiTForwardWrapper:
    def __init__(self, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, cond, text, time):
                return self.inner(
                    x,
                    cond,
                    text,
                    time,
                    drop_audio_cond=False,
                    drop_text=False,
                    mask=None,
                )

        self.module = Wrapped(model).eval()


def load_dit(checkpoint_path: Path, vocab_path: Path):
    import torch
    from f5_tts.model import DiT

    vocab = [line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    vocab_size = len(vocab) + 1
    model = DiT(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        text_num_embeds=vocab_size,
        mel_dim=100,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    dit_state = {}
    for key, value in state.items():
        if key.startswith("transformer."):
            dit_state[key.removeprefix("transformer.")] = value
    missing, unexpected = model.load_state_dict(dit_state, strict=False)
    if missing:
      print(f"missing={len(missing)}", file=sys.stderr)
      print("\n".join(missing[:20]), file=sys.stderr)
    if unexpected:
      print(f"unexpected={len(unexpected)}", file=sys.stderr)
      print("\n".join(unexpected[:20]), file=sys.stderr)
    model.eval()
    return model, vocab_size


def coreml_target(name: str):
    import coremltools as ct

    target = getattr(ct.target, name, None)
    if target is None:
        raise ValueError(f"unknown Core ML target: {name}")
    return target


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, str(RESUMEBOT_DIR))

    import coremltools as ct
    from coremltools.optimize import coreml as cto
    import numpy as np
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    vocab = Path(args.vocab)
    seq_len = int(args.seq_len)

    print(f"loading DiT checkpoint: {checkpoint}")
    dit, vocab_size = load_dit(checkpoint, vocab)
    wrapped = F5DiTForwardWrapper(dit).module

    x = torch.randn(1, seq_len, 100, dtype=torch.float32)
    cond = torch.zeros(1, seq_len, 100, dtype=torch.float32)
    text = torch.zeros(1, seq_len, dtype=torch.int64)
    time = torch.zeros(1, dtype=torch.float32)

    print(f"tracing seq_len={seq_len} vocab_size={vocab_size}")
    traced = torch.jit.trace(wrapped, (x, cond, text, time), strict=False)
    traced = torch.jit.freeze(traced.eval())

    precision = ct.precision.FLOAT16 if args.compute_precision == "float16" else ct.precision.FLOAT32
    target = coreml_target(args.ios_target)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=precision,
        inputs=[
            ct.TensorType(name="x", shape=x.shape, dtype=np.float32),
            ct.TensorType(name="cond", shape=cond.shape, dtype=np.float32),
            ct.TensorType(name="text", shape=text.shape, dtype=np.int32),
            ct.TensorType(name="time", shape=time.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="pred", dtype=np.float32)],
    )

    package_path = out_dir / f"F5TTS_Peyton_DiT_seq{seq_len}.mlpackage"
    if args.skip_palettize:
        print(f"saving fp package: {package_path}")
        mlmodel.save(package_path)
    else:
        print(f"palettizing Core ML weights to int4 mode={args.palettize_mode}")
        config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                mode=args.palettize_mode,
                nbits=4,
                granularity="per_grouped_channel",
                group_size=32,
            )
        )
        compressed = cto.palettize_weights(mlmodel, config=config)
        print(f"saving int4 package: {package_path}")
        compressed.save(package_path)

    metadata = out_dir / f"F5TTS_Peyton_DiT_seq{seq_len}.json"
    metadata.write_text(
        "{\n"
        f'  "model": "F5TTS Peyton DiT forward",\n'
        f'  "format": "coreml-mlprogram",\n'
        f'  "quantization": "{("fp16" if args.skip_palettize else "int4-palettized")}",\n'
        f'  "seq_len": {seq_len},\n'
        f'  "mel_dim": 100,\n'
        f'  "vocab_size": {vocab_size},\n'
        f'  "ios_target": "{args.ios_target}"\n'
        "}\n",
        encoding="utf-8",
    )
    print(f"metadata={metadata}")


if __name__ == "__main__":
    main()
