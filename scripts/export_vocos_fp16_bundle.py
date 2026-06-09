#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SNAPSHOT_ROOT = Path("/data/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots")


def find_snapshot(root: Path) -> Path:
    refs = root.parent / "refs" / "main"
    if refs.exists():
        candidate = root / refs.read_text(encoding="utf-8").strip()
        if candidate.exists():
            return candidate
    snapshots = sorted(path for path in root.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"no Vocos snapshots under {root}")
    return snapshots[-1]


def write_tensor(handle, index: dict[str, dict[str, Any]], offset: int, name: str, tensor: torch.Tensor) -> int:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"unsupported non-floating tensor: {name} {tensor.dtype}")
    array = tensor.detach().contiguous().to(torch.float16).cpu().numpy()
    payload = array.tobytes(order="C")
    handle.write(payload)
    index[name] = {
        "dtype": "float16",
        "shape": list(array.shape),
        "offset": offset,
        "nbytes": len(payload),
    }
    return offset + len(payload)


def export(snapshot: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True)
    index: dict[str, dict[str, Any]] = {}
    offset = 0
    with (output_dir / "tensors.fp16.bin").open("wb") as handle:
        for name, tensor in state.items():
            offset = write_tensor(handle, index, offset, name, tensor)

    manifest = {
        "format": "vocos-mel-24khz-fp16-bundle-v0",
        "model_id": "charactr/vocos-mel-24khz",
        "source_snapshot": str(snapshot),
        "architecture": {
            "input_channels": 100,
            "dim": 512,
            "intermediate_dim": 1536,
            "num_layers": 8,
            "n_fft": 1024,
            "hop_length": 256,
            "padding": "center",
            "sample_rate": 24000,
        },
        "files": {
            "tensors": "tensors.fp16.bin",
            "index": "tensor_fp16_index.json",
        },
    }
    summary = {
        "manifest_path": str(output_dir / "manifest.json"),
        "tensor_count": len(index),
        "tensor_bytes": offset,
        "tensor_mib": round(offset / 1024 / 1024, 3),
    }
    (output_dir / "tensor_fp16_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached charactr/vocos-mel-24khz weights as an FP16 browser bundle.")
    parser.add_argument("--snapshot", default="", help="Optional Hugging Face snapshot directory.")
    parser.add_argument("--output-dir", default="/data/resumebot/checkpoints/vocos_mel_24khz_fp16_v0")
    args = parser.parse_args()
    snapshot = Path(args.snapshot) if args.snapshot else find_snapshot(DEFAULT_SNAPSHOT_ROOT)
    print(json.dumps(export(snapshot, Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
