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


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    return torch.is_floating_point(tensor) and tensor.ndim >= 2 and name.endswith(".weight")


def quantize_q4_rowwise(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    array = tensor.detach().contiguous().float().numpy()
    flat = array.reshape(array.shape[0], -1)
    max_abs = np.maximum(np.max(np.abs(flat), axis=1).astype(np.float32), 1e-8)
    scales = max_abs / 7.0
    quantized = np.rint(flat / scales[:, None]).clip(-8, 7).astype(np.int8)
    return quantized.reshape(array.shape), scales


def pack_int4_signed(values: np.ndarray) -> bytes:
    flat = values.reshape(-1).astype(np.int8, copy=False)
    encoded = (flat.astype(np.int16) & 0x0F).astype(np.uint8)
    if encoded.size % 2:
        encoded = np.pad(encoded, (0, 1), constant_values=0)
    packed = encoded[0::2] | (encoded[1::2] << 4)
    return packed.tobytes(order="C")


def write_dense(handle, index: dict[str, dict[str, Any]], offset: int, name: str, tensor: torch.Tensor) -> int:
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
    dense_index: dict[str, dict[str, Any]] = {}
    q4_index: dict[str, dict[str, Any]] = {}
    dense_offset = 0
    q4_offset = 0
    q4_params = 0
    dense_params = 0

    with (output_dir / "tensors.fp16.bin").open("wb") as dense_handle, (output_dir / "tensors.q4.bin").open("wb") as q4_handle:
        for name, tensor in state.items():
            if should_quantize(name, tensor):
                quantized, scales = quantize_q4_rowwise(tensor)
                packed_payload = pack_int4_signed(quantized)
                scale_payload = scales.astype(np.float16).tobytes(order="C")
                q4_handle.write(packed_payload)
                q4_handle.write(scale_payload)
                q4_index[name] = {
                    "dtype": "q4_symmetric_row_scale_f16",
                    "shape": list(tensor.shape),
                    "offset": q4_offset,
                    "nbytes": len(packed_payload),
                    "scale_offset": q4_offset + len(packed_payload),
                    "scale_nbytes": len(scale_payload),
                    "row_count": int(quantized.reshape(quantized.shape[0], -1).shape[0]),
                    "values_per_byte": 2,
                }
                q4_offset += len(packed_payload) + len(scale_payload)
                q4_params += int(tensor.numel())
            else:
                dense_offset = write_dense(dense_handle, dense_index, dense_offset, name, tensor)
                dense_params += int(tensor.numel())

    manifest = {
        "format": "vocos-mel-24khz-q4-bundle-v0",
        "model_id": "charactr/vocos-mel-24khz-q4",
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
        "quantization": {
            "scheme": "q4_symmetric_row_scale_f16",
            "q4_parameter_count": q4_params,
            "dense_parameter_count": dense_params,
        },
        "files": {
            "dense": "tensors.fp16.bin",
            "dense_index": "tensor_fp16_index.json",
            "q4": "tensors.q4.bin",
            "q4_index": "tensor_q4_index.json",
        },
    }
    summary = {
        "manifest_path": str(output_dir / "manifest.json"),
        "dense_tensors": len(dense_index),
        "q4_tensors": len(q4_index),
        "dense_bytes": dense_offset,
        "q4_bytes": q4_offset,
        "total_mib": round((dense_offset + q4_offset) / 1024 / 1024, 3),
    }
    (output_dir / "tensor_fp16_index.json").write_text(json.dumps(dense_index, indent=2), encoding="utf-8")
    (output_dir / "tensor_q4_index.json").write_text(json.dumps(q4_index, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached charactr/vocos-mel-24khz as FP16 dense + rowwise Q4 weights.")
    parser.add_argument("--snapshot", default="", help="Optional Hugging Face snapshot directory.")
    parser.add_argument("--output-dir", default="/data/resumebot/checkpoints/vocos_mel_24khz_q4_v0")
    args = parser.parse_args()
    snapshot = Path(args.snapshot) if args.snapshot else find_snapshot(DEFAULT_SNAPSHOT_ROOT)
    print(json.dumps(export(snapshot, Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
