#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def quantize_ternary_rowwise(weight: torch.Tensor, threshold_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    array = weight.detach().contiguous().float().numpy()
    if array.ndim < 2:
        raise ValueError("ternary export expects a weight with at least 2 dimensions")
    flat = array.reshape(array.shape[0], -1)
    scales = np.maximum(np.mean(np.abs(flat), axis=1).astype(np.float32), 1e-6)
    threshold = float(threshold_ratio) * scales[:, None]
    quantized = np.where(flat > threshold, 1, np.where(flat < -threshold, -1, 0)).astype(np.int8)
    return quantized.reshape(array.shape), scales


def tensor_state(checkpoint: dict[str, Any], *, materialized: bool) -> dict[str, torch.Tensor]:
    if materialized and checkpoint.get("student_materialized"):
        return checkpoint["student_materialized"]
    return checkpoint["student"]


def qatable_weight(name: str, tensor: torch.Tensor, includes: tuple[str, ...], excludes: tuple[str, ...]) -> bool:
    if not name.endswith(".weight") or tensor.ndim < 2:
        return False
    if includes and not any(item in name for item in includes):
        return False
    if excludes and any(item in name for item in excludes):
        return False
    if any(part in name for part in (".norm", "norm_out", "pos_embed")):
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
    materialized: bool,
    threshold_ratio: float,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = tensor_state(checkpoint, materialized=materialized)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors_path = output_dir / "tensors.f32.bin"
    ternary_path = output_dir / "tensors.ternary.i8.bin"
    tensor_index: dict[str, dict[str, object]] = {}
    ternary_index: dict[str, dict[str, object]] = {}
    offset = 0
    ternary_offset = 0
    exported_float_params = 0
    exported_ternary_params = 0

    with tensors_path.open("wb") as float_handle:
        ternary_handle = ternary_path.open("wb") if ternary else None
        try:
            for name, tensor in state.items():
                if not torch.is_floating_point(tensor):
                    continue
                if name.endswith(".weight_scale"):
                    offset = write_float_tensor(float_handle, tensor_index, offset, name, tensor)
                    exported_float_params += int(tensor.numel())
                    continue
                if ternary and qatable_weight(name, tensor, include, exclude):
                    quantized, scales = quantize_ternary_rowwise(tensor, threshold_ratio)
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
                    exported_ternary_params += int(tensor.numel())
                    continue
                offset = write_float_tensor(float_handle, tensor_index, offset, name, tensor)
                exported_float_params += int(tensor.numel())
        finally:
            if ternary_handle is not None:
                ternary_handle.close()

    (output_dir / "tensor_index.json").write_text(json.dumps(tensor_index, indent=2), encoding="utf-8")
    if ternary:
        (output_dir / "tensor_ternary_index.json").write_text(json.dumps(ternary_index, indent=2), encoding="utf-8")

    config = dict(checkpoint.get("config") or {})
    manifest = {
        "format": "agentkernel-lite-image-sana-browser",
        "model_id": model_id,
        "architecture": checkpoint.get("architecture") or "agentkernel-lite-sana-latent-distill-v0",
        "student_architecture": checkpoint.get("student_architecture") or "sana_transformer",
        "source_checkpoint": str(checkpoint_path),
        "training_step": int(checkpoint.get("step") or 0),
        "training_loss": float(checkpoint.get("loss") or 0.0),
        "teacher_model": checkpoint.get("teacher_model") or "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers",
        "quality_tier": "dense-student-browser-export" if not ternary else "staged-bitnet-qat-browser-export",
        "config": {
            **config,
            "sample_steps": 50,
            "sample_guidance": 4.5,
            "resolution": int(config.get("resolution") or 512),
        },
        "tensors": {
            "data": "tensors.f32.bin",
            "index": "tensor_index.json",
            "format": "flat-f32-state-dict-v0",
            "parameter_count": exported_float_params,
        },
        "ternary_tensors": {
            "data": "tensors.ternary.i8.bin",
            "index": "tensor_ternary_index.json",
            "format": "row-scale-ternary-i8-v0",
            "enabled": bool(ternary),
            "parameter_count": exported_ternary_params,
            "threshold_ratio": float(threshold_ratio),
            "include": list(include),
            "exclude": list(exclude),
        },
        "runtime": {
            "status": "pending_sana_wasm_runtime",
            "target": "browser-wasm-webgpu",
            "entry": "js/image-worker.js",
            "required_components": [
                "Sana text encoder prompt embeddings",
                "Sana latent transformer forward pass",
                "Sana scheduler step",
                "Sana VAE decoder",
            ],
        },
        "quality_notes": {
            "dense_reference_eval": "checkpoints/agentkernel_lite_image_sana_300m_broad_v6/eval_step3000_broad_random_50steps/contact_sheet.png",
            "bitnet_reference_eval": "checkpoints/agentkernel_lite_image_sana_300m_bitnet_block12_13ff_recover_v10b/eval_best_broad_random_50steps/contact_sheet.png",
            "browser_status": "Model weights are exported for a real browser runtime; this bundle is not a sample-artifact fallback.",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "manifest_path": str(output_dir / "manifest.json"),
        "model_id": model_id,
        "float_parameter_count": exported_float_params,
        "ternary_parameter_count": exported_ternary_params,
        "size_bytes": sum(path.stat().st_size for path in output_dir.glob("*") if path.is_file()),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Sana latent student checkpoint as a browser model bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--ternary", action="store_true")
    parser.add_argument("--materialized", action="store_true")
    parser.add_argument("--threshold-ratio", type=float, default=0.5)
    parser.add_argument("--include", default="")
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    summary = export_checkpoint(
        Path(args.checkpoint),
        output_dir,
        model_id=args.model_id or output_dir.name,
        ternary=bool(args.ternary),
        materialized=bool(args.materialized),
        threshold_ratio=float(args.threshold_ratio),
        include=split_csv(args.include),
        exclude=split_csv(args.exclude),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
