#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_EXCLUDES = (
    "text_embed.text_embed",
    "mel_spec",
)


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GiB"


def load_state(checkpoint_path: Path, state_key: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    if state_key not in checkpoint:
        raise KeyError(f"{state_key!r} not found; available keys: {sorted(checkpoint.keys())}")
    state = checkpoint[state_key]
    if not isinstance(state, dict):
        raise ValueError(f"{state_key!r} must be a state dict")
    return dict(state)


def should_quantize(name: str, tensor: torch.Tensor, includes: tuple[str, ...], excludes: tuple[str, ...]) -> bool:
    if not torch.is_floating_point(tensor):
        return False
    if tensor.ndim < 2:
        return False
    if not name.endswith(".weight"):
        return False
    if includes and not any(item in name for item in includes):
        return False
    if excludes and any(item in name for item in excludes):
        return False
    return True


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


def write_dense_tensor(
    handle,
    index: dict[str, dict[str, Any]],
    offset: int,
    name: str,
    tensor: torch.Tensor,
    *,
    dense_float_dtype: str,
) -> int:
    if torch.is_floating_point(tensor):
        if dense_float_dtype == "float32":
            array = tensor.detach().contiguous().to(torch.float32).numpy()
            dtype = "float32"
        else:
            array = tensor.detach().contiguous().to(torch.float16).numpy()
            dtype = "float16"
    elif tensor.dtype == torch.bool:
        array = tensor.detach().contiguous().numpy().astype(np.uint8)
        dtype = "bool_u8"
    elif tensor.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        array = tensor.detach().contiguous().numpy()
        dtype = str(array.dtype)
    else:
        raise ValueError(f"unsupported dense tensor dtype for {name}: {tensor.dtype}")
    payload = array.tobytes(order="C")
    handle.write(payload)
    index[name] = {
        "dtype": dtype,
        "shape": list(array.shape),
        "offset": offset,
        "nbytes": len(payload),
    }
    return offset + len(payload)


def export_q4_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    state_key: str,
    model_id: str,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    dense_float_dtype: str,
    dry_run: bool,
) -> dict[str, Any]:
    state = load_state(checkpoint_path, state_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_index: dict[str, dict[str, Any]] = {}
    q4_index: dict[str, dict[str, Any]] = {}
    dense_offset = 0
    q4_offset = 0
    dense_params = 0
    q4_params = 0

    dense_path = output_dir / "tensors.fp16.bin"
    q4_path = output_dir / "tensors.q4.bin"
    dense_handle = None if dry_run else dense_path.open("wb")
    q4_handle = None if dry_run else q4_path.open("wb")
    try:
        for name, tensor in state.items():
            if should_quantize(name, tensor, include, exclude):
                quantized, scales = quantize_q4_rowwise(tensor)
                packed_payload = pack_int4_signed(quantized)
                scale_payload = scales.astype(np.float16).tobytes(order="C")
                if q4_handle is not None:
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
                continue

            if dense_handle is not None:
                dense_offset = write_dense_tensor(
                    dense_handle,
                    dense_index,
                    dense_offset,
                    name,
                    tensor,
                    dense_float_dtype=dense_float_dtype,
                )
            else:
                if torch.is_floating_point(tensor):
                    if dense_float_dtype == "float32":
                        nbytes = tensor.numel() * 4
                        dtype = "float32"
                    else:
                        nbytes = tensor.numel() * 2
                        dtype = "float16"
                elif tensor.dtype == torch.bool:
                    nbytes = tensor.numel()
                    dtype = "bool_u8"
                else:
                    nbytes = tensor.numel() * tensor.element_size()
                    dtype = str(tensor.detach().cpu().numpy().dtype)
                dense_index[name] = {
                    "dtype": dtype,
                    "shape": list(tensor.shape),
                    "offset": dense_offset,
                    "nbytes": int(nbytes),
                }
                dense_offset += int(nbytes)
            dense_params += int(tensor.numel())
    finally:
        if dense_handle is not None:
            dense_handle.close()
        if q4_handle is not None:
            q4_handle.close()

    manifest = {
        "format": "f5tts-q4-bundle-v0",
        "model_id": model_id,
        "source_checkpoint": str(checkpoint_path),
        "state_key": state_key,
        "architecture": {
            "name": "F5TTS CFM DiT",
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "conv_layers": 4,
            "mel_dim": 100,
            "sample_rate": 24000,
        },
        "quantization": {
            "scheme": "q4_symmetric_row_scale_f16",
            "packed_values": "signed int4 two values per byte, low nibble first",
            "dense_dtype": dense_float_dtype,
            "include": list(include),
            "exclude": list(exclude),
            "q4_parameter_count": q4_params,
            "dense_parameter_count": dense_params,
        },
        "files": {
            "dense": "tensors.fp16.bin",
            "dense_index": "tensor_fp16_index.json",
            "q4": "tensors.q4.bin",
            "q4_index": "tensor_q4_index.json",
        },
        "runtime_status": "weights_only_export; F5TTS graph/vocoder runtime still required",
    }
    if not dry_run:
        (output_dir / "tensor_fp16_index.json").write_text(json.dumps(dense_index, indent=2), encoding="utf-8")
        (output_dir / "tensor_q4_index.json").write_text(json.dumps(q4_index, indent=2), encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_bytes = dense_offset + q4_offset
    summary = {
        "dry_run": dry_run,
        "manifest_path": str(output_dir / "manifest.json"),
        "dense_tensors": len(dense_index),
        "q4_tensors": len(q4_index),
        "dense_parameter_count": dense_params,
        "q4_parameter_count": q4_params,
        "dense_bytes": dense_offset,
        "q4_bytes": q4_offset,
        "total_tensor_bytes": total_bytes,
        "total_tensor_human": human_bytes(total_bytes),
    }
    if not dry_run:
        (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an F5TTS checkpoint state as FP16 dense + rowwise packed Q4 weights.")
    parser.add_argument("--checkpoint", default="/data/resumebot/checkpoints/final_finetuned_model.pt")
    parser.add_argument("--output-dir", default="/data/resumebot/checkpoints/f5tts_peyton_q4_v0")
    parser.add_argument("--state-key", default="model_state_dict")
    parser.add_argument("--model-id", default="f5tts-peyton-q4-v0")
    parser.add_argument("--include", default="")
    parser.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDES))
    parser.add_argument("--dense-float-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = export_q4_bundle(
        Path(args.checkpoint),
        Path(args.output_dir),
        state_key=str(args.state_key),
        model_id=str(args.model_id),
        include=split_csv(args.include),
        exclude=split_csv(args.exclude),
        dense_float_dtype=str(args.dense_float_dtype),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
