#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def quantize_ternary_grouped(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    array = weight.detach().contiguous().float().numpy()
    if array.ndim != 2:
        raise ValueError("ternary export expects 2D weights")
    scales = np.maximum(np.mean(np.abs(array), axis=1).astype(np.float32), 1e-6)
    threshold = 0.7 * scales[:, None]
    quantized = np.where(array > threshold, 1, np.where(array < -threshold, -1, 0)).astype(np.int8)
    return quantized, scales


def should_export_ternary(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False
    if name.startswith("class_embed.") or name in {"pos"}:
        return False
    if ".norm" in name:
        return False
    return True


def export_checkpoint(checkpoint_path: Path, output_dir: Path, *, model_id: str, ternary: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors_path = output_dir / "tensors.f32.bin"
    index: dict[str, dict[str, object]] = {}
    ternary_path = output_dir / "tensors.ternary.i8.bin"
    ternary_index: dict[str, dict[str, object]] = {}
    offset = 0
    ternary_offset = 0
    with tensors_path.open("wb") as handle:
        ternary_handle = ternary_path.open("wb") if ternary else None
        state = checkpoint.get("ema") or checkpoint["model"]
        try:
            for name, tensor in state.items():
                if not torch.is_floating_point(tensor):
                    continue
                if ternary and should_export_ternary(name, tensor):
                    quantized, scales = quantize_ternary_grouped(tensor)
                    q_payload = quantized.tobytes(order="C")
                    s_payload = scales.tobytes(order="C")
                    assert ternary_handle is not None
                    ternary_handle.write(q_payload)
                    ternary_handle.write(s_payload)
                    ternary_index[name] = {
                        "dtype": "ternary_i8_row_scale_f32",
                        "shape": list(quantized.shape),
                        "offset": ternary_offset,
                        "nbytes": len(q_payload),
                        "scale_offset": ternary_offset + len(q_payload),
                        "scale_nbytes": len(s_payload),
                    }
                    ternary_offset += len(q_payload) + len(s_payload)
                    continue
                array = tensor.detach().contiguous().float().numpy()
                payload = array.tobytes(order="C")
                handle.write(payload)
                index[name] = {
                    "dtype": "float32",
                    "shape": list(array.shape),
                    "offset": offset,
                    "nbytes": len(payload),
                }
                offset += len(payload)
        finally:
            if ternary_handle is not None:
                ternary_handle.close()
    (output_dir / "tensor_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    if ternary:
        (output_dir / "tensor_ternary_index.json").write_text(json.dumps(ternary_index, indent=2), encoding="utf-8")
    manifest = {
        "format": "agentkernel-lite-image-bitdit-browser",
        "model_id": model_id,
        "architecture": checkpoint.get("architecture") or "bitdit-ddpm-pixel-v0",
        "source_checkpoint": str(checkpoint_path),
        "training_step": int(checkpoint.get("step") or 0),
        "training_loss": float(checkpoint.get("loss") or 0.0),
        "config": checkpoint["config"],
        "classes": checkpoint.get("classes") or [],
        "tensors": {
            "data": "tensors.f32.bin",
            "index": "tensor_index.json",
            "format": "flat-f32-state-dict-v0",
        },
        "ternary_tensors": {
            "data": "tensors.ternary.i8.bin",
            "index": "tensor_ternary_index.json",
            "format": "row-scale-ternary-i8-v0",
            "enabled": bool(ternary),
        },
        "runtime": {
            "status": "pending",
            "target": "browser-webgpu-wasm",
            "entry": "runtime/bitdit_runtime.js",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an AgentKernel Lite BitDiT checkpoint as a browser bundle scaffold.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--ternary", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    export_checkpoint(checkpoint, output_dir, model_id=args.model_id or output_dir.name, ternary=args.ternary)


if __name__ == "__main__":
    main()
