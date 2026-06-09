#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def quantize_ternary_rowwise(weight: torch.Tensor, threshold_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = weight.detach().contiguous().float().numpy()
    if array.ndim != 2:
        raise ValueError("ternary export expects a 2D linear weight")
    scales = np.maximum(np.mean(np.abs(array), axis=1).astype(np.float32), 1e-6)
    threshold = threshold_ratio * scales[:, None]
    ternary = np.where(array > threshold, 1, np.where(array < -threshold, -1, 0)).astype(np.int8)
    codes = np.where(ternary < 0, 0, np.where(ternary > 0, 2, 1)).astype(np.uint8)
    density = (ternary != 0).mean(axis=1).astype(np.float32)
    return codes, scales, density


def pack_2bit(codes: np.ndarray) -> np.ndarray:
    flat = codes.reshape(-1)
    padded = np.zeros(((flat.size + 3) // 4) * 4, dtype=np.uint8)
    padded[: flat.size] = flat
    packed = (
        padded[0::4]
        | (padded[1::4] << 2)
        | (padded[2::4] << 4)
        | (padded[3::4] << 6)
    )
    return packed.astype(np.uint8, copy=False)


def should_export_ternary(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False
    if ".norm" in name:
        return False
    return True


def write_float_tensor(handle, index: dict[str, dict[str, object]], offset: int, name: str, tensor: torch.Tensor) -> int:
    array = tensor.detach().contiguous().float().numpy()
    payload = array.tobytes(order="C")
    handle.write(payload)
    index[name] = {
        "dtype": "float32",
        "shape": list(array.shape),
        "offset": offset,
        "nbytes": len(payload),
    }
    return offset + len(payload)


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    ternary: bool,
    threshold_ratio: float,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("student_materialized") or checkpoint.get("student")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a FLUX student state dict")

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_index: dict[str, dict[str, object]] = {}
    ternary_index: dict[str, dict[str, object]] = {}
    float_offset = 0
    ternary_offset = 0

    float_path = output_dir / "tensors.f32.bin"
    ternary_path = output_dir / "tensors.ternary.t2.bin"
    with float_path.open("wb") as float_handle:
        ternary_handle = ternary_path.open("wb") if ternary else None
        try:
            for name, tensor in state.items():
                if not torch.is_floating_point(tensor):
                    continue
                if ternary and should_export_ternary(name, tensor):
                    codes, scales, density = quantize_ternary_rowwise(tensor, threshold_ratio)
                    packed = pack_2bit(codes)
                    scale_payload = scales.tobytes(order="C")
                    density_payload = density.tobytes(order="C")
                    assert ternary_handle is not None
                    ternary_handle.write(packed.tobytes(order="C"))
                    ternary_handle.write(scale_payload)
                    ternary_handle.write(density_payload)
                    ternary_index[name] = {
                        "dtype": "ternary_packed_2bit_row_scale_f32",
                        "shape": list(codes.shape),
                        "offset": ternary_offset,
                        "nbytes": int(packed.nbytes),
                        "scale_offset": ternary_offset + int(packed.nbytes),
                        "scale_nbytes": len(scale_payload),
                        "density_offset": ternary_offset + int(packed.nbytes) + len(scale_payload),
                        "density_nbytes": len(density_payload),
                        "codebook": {"0": -1, "1": 0, "2": 1, "3": 0},
                        "threshold_ratio": float(threshold_ratio),
                    }
                    ternary_offset += int(packed.nbytes) + len(scale_payload) + len(density_payload)
                else:
                    float_offset = write_float_tensor(float_handle, tensor_index, float_offset, name, tensor)
        finally:
            if ternary_handle is not None:
                ternary_handle.close()

    (output_dir / "tensor_index.json").write_text(json.dumps(tensor_index, indent=2), encoding="utf-8")
    if ternary:
        (output_dir / "tensor_ternary_index.json").write_text(json.dumps(ternary_index, indent=2), encoding="utf-8")

    config = checkpoint.get("config") or {}
    manifest = {
        "format": "agentkernel-lite-image-flux-packed-browser",
        "model_id": model_id,
        "architecture": checkpoint.get("architecture") or "agentkernel-lite-flux-packed-flow-student-v0",
        "source_checkpoint": str(checkpoint_path),
        "training_step": int(checkpoint.get("step") or 0),
        "training_loss": float(checkpoint.get("loss") or 0.0),
        "bitnet_qat": bool(checkpoint.get("bitnet_qat", False)),
        "config": config,
        "tensors": {
            "data": "tensors.f32.bin",
            "index": "tensor_index.json",
            "format": "flat-f32-state-dict-v0",
        },
        "ternary_tensors": {
            "data": "tensors.ternary.t2.bin",
            "index": "tensor_ternary_index.json",
            "format": "packed-2bit-row-scale-ternary-v0",
            "enabled": bool(ternary),
            "threshold_ratio": float(threshold_ratio),
        },
        "runtime": {
            "status": "pending-runtime-kernel",
            "target": "browser-wasm-webgpu",
            "entry": "runtime/flux_packed_runtime.js",
            "sample_steps": 64,
            "requires": ["prompt_encoder", "flux_latent_decoder"],
        },
        "quality_notes": {
            "deployment_gate": "Only publish after held-out random prompt eval is coherent and QAT samples match dense samples.",
            "training_corpus": "general_hq_mix streamed Hugging Face prompts with FLUX teacher flow targets",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "float_bytes": float_offset, "ternary_bytes": ternary_offset}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a FLUX packed-latent student checkpoint as a browser bundle scaffold.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--ternary", action="store_true")
    parser.add_argument("--threshold-ratio", type=float, default=0.7)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    export_checkpoint(
        Path(args.checkpoint),
        output_dir,
        model_id=args.model_id or output_dir.name,
        ternary=bool(args.ternary),
        threshold_ratio=float(args.threshold_ratio),
    )


if __name__ == "__main__":
    main()
