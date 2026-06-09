#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import load_teacher
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from train_agentkernel_lite_image_flux_flow_distill import (
    FluxPackedStudent,
    FluxPackedStudentConfig,
    seed_everything,
    unpack_packed_latent_grid,
)
from train_agentkernel_lite_image_flux_flow_rollout_distill import packed_lowfreq_loss, teacher_step_delta
from train_agentkernel_lite_image_flux_live_teacher_trajectory_reuse import (
    DEFAULT_FLUX_TEACHER,
    PromptMixer,
    build_teacher_trajectory,
    clone_state_dict,
    decode_flux_latents_tensor,
    load_prompt_condition_aliases,
    load_prompt_negative_aliases,
    load_prompt_seed_aliases,
    optimizer_checkpoint_path,
    parse_prompt_mix,
    save_optimizer_checkpoint,
)
from sample_agentkernel_lite_image_flux_flow_distill import load_final_latent_refiner


def direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = F.normalize(pred.float().reshape(pred.shape[0], -1), dim=-1)
    target_flat = F.normalize(target.float().reshape(target.shape[0], -1), dim=-1)
    return (1.0 - (pred_flat * target_flat).sum(dim=-1)).mean()


def norm_ratio_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = pred.float().flatten(1).norm(dim=-1)
    target_norm = target.float().flatten(1).norm(dim=-1).clamp_min(1e-6)
    return F.mse_loss(pred_norm / target_norm, torch.ones_like(target_norm))


def log_norm_ratio_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = pred.float().flatten(1).norm(dim=-1).clamp_min(1e-8)
    target_norm = target.float().flatten(1).norm(dim=-1).clamp_min(1e-8)
    return F.mse_loss(torch.log(pred_norm), torch.log(target_norm))


def orthogonal_delta_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.float().flatten(1)
    target_flat = target.float().flatten(1)
    target_norm = target_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target_unit = target_flat / target_norm
    pred_parallel = (pred_flat * target_unit).sum(dim=-1, keepdim=True) * target_unit
    pred_perp = pred_flat - pred_parallel
    return (pred_perp.norm(dim=-1) / target_norm.squeeze(-1)).pow(2).mean()


def rms_normalized_delta_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_float = pred.float()
    target_float = target.float()
    pred_rms = pred_float.flatten(1).pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    target_rms = target_float.flatten(1).pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    view_shape = [pred.shape[0], *([1] * (pred.ndim - 1))]
    pred_unit = pred_float / pred_rms.view(*view_shape)
    target_unit = target_float / target_rms.view(*view_shape)
    return F.mse_loss(pred_unit, target_unit)


def packed_lowfreq(value: torch.Tensor, pool: int) -> torch.Tensor:
    tokens = value.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return value.float()
    pool = max(int(pool), 1)
    grid = value.float().transpose(1, 2).reshape(value.shape[0], value.shape[2], side, side)
    low = F.avg_pool2d(grid, kernel_size=pool, stride=pool)
    return low.flatten(2).transpose(1, 2)


def packed_lowfreq_direction_loss(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    return direction_loss(packed_lowfreq(pred, pool), packed_lowfreq(target, pool))


def packed_lowfreq_norm_ratio_loss(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    return norm_ratio_loss(packed_lowfreq(pred, pool), packed_lowfreq(target, pool))


def packed_lowfreq_log_norm_ratio_loss(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    return log_norm_ratio_loss(packed_lowfreq(pred, pool), packed_lowfreq(target, pool))


def packed_lowfreq_orthogonal_delta_loss(pred: torch.Tensor, target: torch.Tensor, pool: int) -> torch.Tensor:
    return orthogonal_delta_loss(packed_lowfreq(pred, pool), packed_lowfreq(target, pool))


def packed_parity_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    pred_grid = unpack_packed_latent_grid(pred.float())
    target_grid = unpack_packed_latent_grid(target.float())
    losses: list[torch.Tensor] = []
    for y_offset in range(2):
        for x_offset in range(2):
            pred_parity = pred_grid[:, :, y_offset::2, x_offset::2]
            target_parity = target_grid[:, :, y_offset::2, x_offset::2]
            if int(size) > 0:
                pred_parity = F.interpolate(pred_parity, size=(int(size), int(size)), mode="area")
                target_parity = F.interpolate(target_parity, size=(int(size), int(size)), mode="area")
            losses.append(F.l1_loss(pred_parity, target_parity))
            losses.append(F.l1_loss(pred_parity.mean(dim=(-2, -1)), target_parity.mean(dim=(-2, -1))))
    return torch.stack(losses).mean()


def image_parity_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for y_offset in range(2):
        for x_offset in range(2):
            pred_parity = pred.float()[:, :, y_offset::2, x_offset::2]
            target_parity = target.float()[:, :, y_offset::2, x_offset::2]
            if int(size) > 0:
                pred_parity = F.interpolate(pred_parity, size=(int(size), int(size)), mode="area")
                target_parity = F.interpolate(target_parity, size=(int(size), int(size)), mode="area")
            losses.append(F.l1_loss(pred_parity, target_parity))
            losses.append(F.l1_loss(pred_parity.mean(dim=(-2, -1)), target_parity.mean(dim=(-2, -1))))
    pred_checker_h = pred.float()[:, :, :, 0::2].mean() - pred.float()[:, :, :, 1::2].mean()
    target_checker_h = target.float()[:, :, :, 0::2].mean() - target.float()[:, :, :, 1::2].mean()
    pred_checker_v = pred.float()[:, :, 0::2, :].mean() - pred.float()[:, :, 1::2, :].mean()
    target_checker_v = target.float()[:, :, 0::2, :].mean() - target.float()[:, :, 1::2, :].mean()
    losses.append((pred_checker_h - target_checker_h).abs())
    losses.append((pred_checker_v - target_checker_v).abs())
    return torch.stack(losses).mean()


def image_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_float = pred.float()
    target_float = target.float()
    pred_dx = pred_float[:, :, :, 1:] - pred_float[:, :, :, :-1]
    target_dx = target_float[:, :, :, 1:] - target_float[:, :, :, :-1]
    pred_dy = pred_float[:, :, 1:, :] - pred_float[:, :, :-1, :]
    target_dy = target_float[:, :, 1:, :] - target_float[:, :, :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))


def image_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    pred_float = pred.float()
    target_float = target.float()
    size = max(int(size), 4)
    pred_low = F.interpolate(F.interpolate(pred_float, size=(size, size), mode="area"), size=pred_float.shape[-2:], mode="bilinear", align_corners=False)
    target_low = F.interpolate(F.interpolate(target_float, size=(size, size), mode="area"), size=target_float.shape[-2:], mode="bilinear", align_corners=False)
    return F.l1_loss(pred_float - pred_low, target_float - target_low)


def tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().pow(2).mean().sqrt()


def read_prompt_filter(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_prompt_loss_weights(path_value: str) -> dict[str, float]:
    if not path_value:
        return {}
    path = Path(path_value)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--prompt-loss-weight-file must be a JSON object mapping prompt to weight")
    weights: dict[str, float] = {}
    for prompt, value in raw.items():
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"prompt loss weights must be positive, got {weight} for {prompt!r}")
        weights[str(prompt)] = weight
    return weights


def parse_stage_target_indices(value: str, final_target_index: int) -> list[int]:
    if not value.strip():
        return [int(final_target_index)]
    stage_indices: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        stage_indices.append(int(part))
    if not stage_indices:
        return [int(final_target_index)]
    previous = 0
    for stage_index in stage_indices:
        if stage_index <= previous:
            raise ValueError("--stage-target-indices must be strictly increasing and greater than 0")
        previous = stage_index
    if stage_indices[-1] != int(final_target_index):
        raise ValueError("--stage-target-indices must end at --target-index")
    return stage_indices


def load_stage_bridge_bank(path: str) -> dict[int, dict[str, object]]:
    if not path:
        return {}
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = manifest.get("bridges", manifest.get("stages", []))
    bank: dict[int, dict[str, object]] = {}
    for entry in entries:
        target_index = int(entry["target_index"])
        bank[target_index] = {
            "checkpoint": Path(entry["checkpoint"]),
            "weights": str(entry.get("weights", "raw")),
            "output_mode": str(entry.get("output_mode", manifest.get("output_mode", "delta"))),
            "scale": float(entry.get("scale", manifest.get("scale", 1.0))),
        }
    return bank


def select_checkpoint_state(checkpoint: dict[str, object], weights: str) -> dict[str, torch.Tensor]:
    if weights == "ema" and checkpoint.get("ema_student"):
        return checkpoint["ema_student"]  # type: ignore[return-value]
    if weights == "materialized" and checkpoint.get("materialized_student"):
        return checkpoint["materialized_student"]  # type: ignore[return-value]
    state = checkpoint.get("student") or checkpoint.get("model") or checkpoint
    return state  # type: ignore[return-value]


class CachedBridgePairs:
    def __init__(
        self,
        target_dirs: list[str],
        source_index: int,
        target_index: int,
        prompt_filter: set[str] | None = None,
    ) -> None:
        self.pairs: list[dict[str, object]] = []
        prompt_filter = prompt_filter or set()
        for target_dir_value in target_dirs:
            target_dir = Path(target_dir_value)
            metadata_path = target_dir / "metadata.jsonl"
            if not metadata_path.exists():
                raise FileNotFoundError(metadata_path)
            grouped: dict[tuple[str, int], dict[int, dict[str, object]]] = {}
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    continue
                if prompt_filter and prompt not in prompt_filter:
                    continue
                seed = int(row.get("seed", 0) or 0)
                timestep_index = int(row.get("timestep_index", 0) or 0)
                grouped.setdefault((prompt, seed), {})[timestep_index] = row
            for (prompt, seed), rows in grouped.items():
                if int(source_index) not in rows or int(target_index) not in rows:
                    continue
                first = rows[int(source_index)]
                target = rows[int(target_index)]
                continuation_rows = [rows[index] for index in sorted(rows) if index >= int(target_index)]
                self.pairs.append(
                    {
                        "target_dir": target_dir,
                        "prompt": prompt,
                        "seed": seed,
                        "source_name": target_dir.name,
                        "initial_target_path": first["target_path"],
                        "target_path": target["target_path"],
                        "target_paths_by_index": {
                            str(index): row["target_path"] for index, row in sorted(rows.items())
                        },
                        "embedding_path": first["embedding_path"],
                        "continuation_target_paths": [row["target_path"] for row in continuation_rows],
                    }
                )
        if not self.pairs:
            raise ValueError(
                f"no cached t{source_index:03d}/t{target_index:03d} bridge pairs found in {target_dirs}"
            )

    def next(self) -> dict[str, object]:
        return random.choice(self.pairs)


def load_student(path: Path, device: torch.device) -> tuple[FluxPackedStudent, FluxPackedStudentConfig, int]:
    checkpoint = torch.load(path, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    student = FluxPackedStudent(config).to(device)
    state = select_checkpoint_state(checkpoint, "raw")
    student.load_state_dict(state, strict=False)
    return student, config, int(checkpoint.get("step") or 0)


def load_student_for_weights(
    path: Path,
    device: torch.device,
    weights: str,
) -> tuple[FluxPackedStudent, FluxPackedStudentConfig, int]:
    checkpoint = torch.load(path, map_location="cpu")
    config = FluxPackedStudentConfig(**checkpoint["config"])
    student = FluxPackedStudent(config).to(device)
    state = select_checkpoint_state(checkpoint, weights)
    student.load_state_dict(state, strict=False)
    return student, config, int(checkpoint.get("step") or 0)


def save_checkpoint(
    path: Path,
    student: FluxPackedStudent,
    config: FluxPackedStudentConfig,
    step: int,
    ema_state: dict[str, torch.Tensor] | None,
) -> None:
    payload: dict[str, object] = {
        "step": int(step),
        "config": config.__dict__,
        "student": {key: value.detach().cpu() for key, value in student.state_dict().items()},
        "mode": "flux_timestep_bridge_initializer",
    }
    if ema_state is not None:
        payload["student_ema"] = ema_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def update_ema(ema_state: dict[str, torch.Tensor], student: FluxPackedStudent, decay: float) -> None:
    with torch.no_grad():
        for key, value in student.state_dict().items():
            ema_state[key].mul_(float(decay)).add_(value.detach().cpu(), alpha=1.0 - float(decay))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct FLUX fresh-latent to timestep bridge initializer.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--target-dir", action="append", default=[])
    parser.add_argument("--extra-target-dir", action="append", default=[])
    parser.add_argument(
        "--distill-target-refiner",
        default="",
        help="Optional frozen latent refiner whose output from the current source latent replaces the final stage target.",
    )
    parser.add_argument("--cached-prompt-file", default="")
    parser.add_argument("--prompt-loss-weight-file", default="")
    parser.add_argument("--prompt-mix", action="append", default=[])
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize-transformer-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-device", default="")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=15)
    parser.add_argument(
        "--stage-target-indices",
        default="",
        help="Optional comma-separated increasing bridge waypoints ending at --target-index, e.g. 5,10,15.",
    )
    parser.add_argument(
        "--source-bridge-bank-manifest",
        default="",
        help="Optional bridge-bank manifest used to synthesize the source latent from t0 before training this bridge.",
    )
    parser.add_argument(
        "--source-bridge-stage-indices",
        default="",
        help="Comma-separated upstream stage indices to apply from --source-bridge-bank-manifest, ending at --source-index.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=96)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--bridge-output-mode", choices=("delta", "absolute"), default="delta")
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=999_999_999)
    parser.add_argument("--randomize-seeds", action="store_true")
    parser.add_argument("--prompt-seed-alias-file", default="")
    parser.add_argument("--prompt-condition-alias-file", default="")
    parser.add_argument("--prompt-negative-alias-file", default="")
    parser.add_argument("--trainable-name-regex", default="")
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--output-scale-lr", type=float, default=0.0)
    parser.add_argument("--output-log-scale-min", type=float, default=-20.0)
    parser.add_argument("--output-log-scale-max", type=float, default=20.0)
    parser.add_argument("--prompt-output-scale-clip", type=float, default=0.0)
    parser.add_argument("--output-scale-mode", choices=("output", "delta_from_input"), default="output")
    parser.add_argument("--continuation-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.004)
    parser.add_argument("--continuation-weight-decay", type=float, default=-1.0)
    parser.add_argument("--grad-clip", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=80.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--orthogonal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--normalized-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-delta-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-delta-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-delta-log-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-delta-orthogonal-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-delta-pool", type=int, default=4)
    parser.add_argument("--terminal-normalized-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-log-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-orthogonal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-lowfreq-delta-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-lowfreq-delta-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-lowfreq-delta-log-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-lowfreq-delta-orthogonal-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-normalized-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-lowfreq-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--recovery-error-scale", type=float, default=0.5)
    parser.add_argument("--recovery-noise-std", type=float, default=0.0)
    parser.add_argument("--prompt-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-delta-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--prompt-delta-norm-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--live-same-seed-prompt-delta",
        action="store_true",
        help="For cached rows, build the contrast prompt teacher trajectory with the current row seed so prompt-delta uses the same initial noise.",
    )
    parser.add_argument("--absolute-latent-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-delta-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-lowfreq-pool", type=int, default=4)
    parser.add_argument("--future-head-lr", type=float, default=0.0)
    parser.add_argument("--decoded-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--decoded-loss-final-stage-only",
        action="store_true",
        help="For staged bridge training, apply decoded bridge losses only at the final stage to reduce memory.",
    )
    parser.add_argument("--decoded-lowfreq-size", type=int, default=32)
    parser.add_argument("--decoded-parity-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-highfreq-size", type=int, default=48)
    parser.add_argument("--decoded-parity-size", type=int, default=64)
    parser.add_argument("--parity-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--parity-lowfreq-size", type=int, default=32)
    parser.add_argument("--continuation-checkpoint", default="")
    parser.add_argument("--train-continuation", action="store_true")
    parser.add_argument("--continuation-trainable-name-regex", default="")
    parser.add_argument("--continuation-rollout-len", type=int, default=0)
    parser.add_argument("--continuation-step-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-step-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-step-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-step-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-normalized-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-decoded-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-decoded-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-decoded-parity-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-decoded-gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--continuation-decoded-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=352)
    args = parser.parse_args()

    seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "bridge_initializer_ledger.jsonl"
    stage_target_indices = parse_stage_target_indices(str(args.stage_target_indices), int(args.target_index))
    source_index = int(args.source_index)
    if source_index < 0:
        raise ValueError("--source-index must be >= 0")
    if any(index <= source_index for index in stage_target_indices):
        raise ValueError("--stage-target-indices must all be greater than --source-index")
    source_bridge_bank = load_stage_bridge_bank(str(args.source_bridge_bank_manifest))
    source_bridge_stage_indices = (
        parse_stage_target_indices(str(args.source_bridge_stage_indices), source_index)
        if str(args.source_bridge_stage_indices).strip()
        else []
    )
    if source_bridge_stage_indices and not source_bridge_bank:
        raise ValueError("--source-bridge-stage-indices requires --source-bridge-bank-manifest")

    cached_target_dirs = [*args.target_dir, *args.extra_target_dir]
    pipe = None
    teacher_device = torch.device(args.device)
    need_pipe = (
        (not cached_target_dirs)
        or float(args.decoded_loss_weight) > 0.0
        or float(args.decoded_lowfreq_loss_weight) > 0.0
        or float(args.decoded_parity_loss_weight) > 0.0
        or float(args.decoded_gradient_loss_weight) > 0.0
        or float(args.decoded_highfreq_loss_weight) > 0.0
        or float(args.continuation_decoded_loss_weight) > 0.0
        or float(args.continuation_decoded_lowfreq_loss_weight) > 0.0
        or float(args.continuation_decoded_parity_loss_weight) > 0.0
        or float(args.continuation_decoded_gradient_loss_weight) > 0.0
        or float(args.continuation_decoded_highfreq_loss_weight) > 0.0
        or bool(args.live_same_seed_prompt_delta)
    )
    if need_pipe:
        teacher_args = argparse.Namespace(
            teacher_family="flux",
            teacher_model=args.teacher_model,
            dtype=args.teacher_dtype,
            variant="",
            local_files_only=args.local_files_only,
            quantize_transformer_4bit=args.quantize_transformer_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            cpu_offload=args.cpu_offload,
            gpu_id=args.gpu_id,
            device=args.device,
        )
        pipe = load_teacher(teacher_args)
        teacher_device = torch.device(getattr(pipe, "_execution_device", args.device))
        if cached_target_dirs and not bool(args.live_same_seed_prompt_delta):
            pipe.transformer = None
            pipe.vae.to(student_device if "student_device" in locals() else teacher_device)
    student_device = torch.device(args.student_device or str(teacher_device))
    student, config, start_step = load_student(Path(args.resume), student_device)
    student.config.prompt_output_scale_clip = float(args.prompt_output_scale_clip)
    student.config.output_scale_mode = str(args.output_scale_mode)
    student.train()
    use_future_aux = (
        float(args.future_latent_loss_weight) > 0.0
        or float(args.future_delta_direction_loss_weight) > 0.0
    )
    future_head = nn.Linear(int(config.dim), int(config.latent_channels)).to(student_device) if use_future_aux else None
    distill_target_refiner = load_final_latent_refiner(str(args.distill_target_refiner), student_device)
    if distill_target_refiner is not None:
        distill_target_refiner.requires_grad_(False)
        distill_target_refiner.eval()
    continuation_student = None
    if args.continuation_checkpoint:
        continuation_student, continuation_config, _ = load_student(Path(args.continuation_checkpoint), student_device)
        if continuation_config.__dict__ != config.__dict__:
            raise ValueError("--continuation-checkpoint config does not match bridge student config")
        if args.train_continuation:
            continuation_student.train()
            if args.continuation_trainable_name_regex:
                import re

                continuation_pattern = re.compile(args.continuation_trainable_name_regex)
                for name, parameter in continuation_student.named_parameters():
                    parameter.requires_grad_(bool(continuation_pattern.search(name)))
            else:
                continuation_student.requires_grad_(True)
        else:
            continuation_student.requires_grad_(False)
            continuation_student.eval()
    if args.trainable_name_regex:
        import re

        pattern = re.compile(args.trainable_name_regex)
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(bool(pattern.search(name)))
    bridge_parameters = []
    output_scale_parameters = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if (name == "output_log_scale" or name.startswith("prompt_output_scale.")) and float(args.output_scale_lr) > 0:
            output_scale_parameters.append(parameter)
        else:
            bridge_parameters.append(parameter)
    optimizer_parameters = []
    if bridge_parameters:
        optimizer_parameters.append(
            {
                "params": bridge_parameters,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            }
        )
    if output_scale_parameters:
        optimizer_parameters.append(
            {
                "params": output_scale_parameters,
                "lr": float(args.output_scale_lr),
                "weight_decay": 0.0,
            }
        )
    if future_head is not None:
        optimizer_parameters.append(
            {
                "params": future_head.parameters(),
                "lr": float(args.future_head_lr) if float(args.future_head_lr) > 0 else float(args.lr),
                "weight_decay": float(args.weight_decay),
            }
        )
    if continuation_student is not None and args.train_continuation:
        continuation_parameters = [
            parameter for parameter in continuation_student.parameters() if parameter.requires_grad
        ]
        continuation_lr = float(args.continuation_lr) if float(args.continuation_lr) > 0 else float(args.lr)
        continuation_weight_decay = (
            float(args.continuation_weight_decay)
            if float(args.continuation_weight_decay) >= 0
            else float(args.weight_decay)
        )
        optimizer_parameters.append(
            {
                "params": continuation_parameters,
                "lr": continuation_lr,
                "weight_decay": continuation_weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(args.lr),
    )
    ema_state = clone_state_dict(student) if args.ema_decay > 0 else None
    cached_prompt_filter = read_prompt_filter(args.cached_prompt_file)
    prompt_loss_weights = read_prompt_loss_weights(args.prompt_loss_weight_file)
    cached_pairs = (
        CachedBridgePairs(cached_target_dirs, source_index, int(args.target_index), cached_prompt_filter)
        if cached_target_dirs
        else None
    )
    prompts = PromptMixer(parse_prompt_mix(args), None) if cached_pairs is None else None
    seed_aliases = load_prompt_seed_aliases(args.prompt_seed_alias_file)
    condition_aliases = load_prompt_condition_aliases(args.prompt_condition_alias_file)
    negative_aliases = load_prompt_negative_aliases(args.prompt_negative_alias_file)
    manifest = {
        "mode": "flux_timestep_bridge_initializer",
        "resume": str(args.resume),
        "source_index": int(source_index),
        "target_index": int(args.target_index),
        "stage_target_indices": stage_target_indices,
        "source_bridge_bank_manifest": str(args.source_bridge_bank_manifest),
        "source_bridge_stage_indices": source_bridge_stage_indices,
        "distill_target_refiner": str(args.distill_target_refiner),
        "bridge_output_mode": str(args.bridge_output_mode),
        "target_dirs": cached_target_dirs,
        "cached_prompt_file": str(args.cached_prompt_file),
        "prompt_loss_weight_file": str(args.prompt_loss_weight_file),
        "prompt_loss_weights": prompt_loss_weights,
        "prompt_mix": args.prompt_mix,
        "decoded_lowfreq_cached_pipe": bool(cached_target_dirs and pipe is not None),
        "continuation_checkpoint": str(args.continuation_checkpoint),
        "train_continuation": bool(args.train_continuation),
        "continuation_lr": float(args.continuation_lr),
        "continuation_weight_decay": float(args.continuation_weight_decay),
        "continuation_trainable_name_regex": str(args.continuation_trainable_name_regex),
        "continuation_rollout_len": int(args.continuation_rollout_len),
        "args": vars(args),
        "config": config.__dict__,
    }
    (output_dir / "bridge_initializer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    step = start_step
    while step < start_step + int(args.steps):
        target_index = int(args.target_index)
        if cached_pairs is not None:
            cached_row = cached_pairs.next()
            target_dir = Path(cached_row["target_dir"])
            prompt = str(cached_row["prompt"])
            source_name = str(cached_row["source_name"])
            seed = int(cached_row["seed"])
            embeds = torch.load(target_dir / str(cached_row["embedding_path"]), map_location="cpu")
            initial_target = torch.load(target_dir / str(cached_row["initial_target_path"]), map_location="cpu")
            target_target = torch.load(target_dir / str(cached_row["target_path"]), map_location="cpu")
            target_paths_by_index = {
                int(index): path for index, path in dict(cached_row["target_paths_by_index"]).items()
            }
            prompt_embeds = embeds["prompt_embeds"].to(device=student_device)
            pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(device=student_device)
            initial_latents = initial_target["latents"].to(device=student_device)
            target_latents = target_target["latents"].to(device=student_device)
            timestep = initial_target.get("timestep", torch.tensor(1000.0)).to(device=student_device)
            timesteps_by_index = {}
            stage_targets = {source_index: initial_target}
            required_stage_indices = sorted(set([0, source_index, *source_bridge_stage_indices, *stage_target_indices]))
            for stage_index in required_stage_indices:
                if stage_index == source_index:
                    continue
                if stage_index not in target_paths_by_index:
                    raise ValueError(f"cached row is missing stage target index {stage_index}")
                stage_targets[stage_index] = torch.load(
                    target_dir / str(target_paths_by_index[stage_index]),
                    map_location="cpu",
                )
            for stage_index in required_stage_indices:
                stage_timestep = stage_targets[stage_index].get("timestep", torch.tensor(1000.0))
                timesteps_by_index[stage_index] = stage_timestep.to(device=student_device).reshape(1)
            continuation_targets = [
                torch.load(target_dir / str(path), map_location="cpu")
                for path in cached_row.get("continuation_target_paths", [])
            ]
            contrast_prompt_embeds = None
            contrast_pooled_prompt_embeds = None
            contrast_stage_targets: dict[int, dict[str, torch.Tensor]] = {}
            use_prompt_delta = (
                float(args.prompt_delta_loss_weight) > 0
                or float(args.prompt_delta_direction_loss_weight) > 0
                or float(args.prompt_delta_norm_loss_weight) > 0
            )
            if use_prompt_delta and cached_pairs is not None:
                contrast_row = cached_pairs.next()
                for _ in range(8):
                    if str(contrast_row["prompt"]) != prompt:
                        break
                    contrast_row = cached_pairs.next()
                contrast_target_dir = Path(contrast_row["target_dir"])
                contrast_embeds = torch.load(
                    contrast_target_dir / str(contrast_row["embedding_path"]),
                    map_location="cpu",
                )
                contrast_prompt_embeds = contrast_embeds["prompt_embeds"].to(device=student_device)
                contrast_pooled_prompt_embeds = contrast_embeds["pooled_prompt_embeds"].to(device=student_device)
                if args.live_same_seed_prompt_delta:
                    if pipe is None:
                        raise ValueError("--live-same-seed-prompt-delta requires a loaded teacher pipe")
                    teacher_ns = argparse.Namespace(
                        teacher_steps=int(args.teacher_steps),
                        width=int(args.width),
                        height=int(args.height),
                        max_sequence_length=int(args.max_sequence_length),
                        guidance=float(args.guidance),
                    )
                    contrast_condition_prompt = condition_aliases.get(str(contrast_row["prompt"]), str(contrast_row["prompt"]))
                    contrast_trajectory = build_teacher_trajectory(
                        pipe,
                        contrast_condition_prompt,
                        int(seed),
                        teacher_ns,
                        teacher_device,
                    )
                    contrast_prompt_embeds = contrast_trajectory["prompt_embeds"].to(
                        device=student_device,
                        dtype=initial_latents.dtype,
                    )
                    contrast_pooled_prompt_embeds = contrast_trajectory["pooled_prompt_embeds"].to(
                        device=student_device,
                        dtype=initial_latents.dtype,
                    )
                    contrast_traj = contrast_trajectory["trajectory"]
                    for stage_index in required_stage_indices:
                        if stage_index >= len(contrast_traj):
                            raise ValueError(f"contrast live trajectory missing stage target index {stage_index}")
                        contrast_stage_targets[stage_index] = {
                            "latents": contrast_traj[stage_index]["latents"],
                        }
                else:
                    contrast_target_paths_by_index = {
                        int(index): path for index, path in dict(contrast_row["target_paths_by_index"]).items()
                    }
                    for stage_index in required_stage_indices:
                        if stage_index not in contrast_target_paths_by_index:
                            raise ValueError(f"contrast cached row is missing stage target index {stage_index}")
                        contrast_stage_targets[stage_index] = torch.load(
                            contrast_target_dir / str(contrast_target_paths_by_index[stage_index]),
                            map_location="cpu",
                        )
        else:
            assert pipe is not None and prompts is not None
            prompt, source_name = prompts.next()
            condition_prompt = condition_aliases.get(prompt, prompt)
            seed_prompt = seed_aliases.get(prompt, prompt)
            if args.randomize_seeds:
                seed = torch.randint(int(args.seed_min), int(args.seed_max) + 1, ()).item()
            else:
                from sample_agentkernel_lite_image_flux_flow_distill import stable_prompt_seed

                seed = stable_prompt_seed(seed_prompt, int(args.seed_min), int(args.seed_max))
            teacher_ns = argparse.Namespace(
                teacher_steps=int(args.teacher_steps),
                width=int(args.width),
                height=int(args.height),
                max_sequence_length=int(args.max_sequence_length),
                guidance=float(args.guidance),
            )
            trajectory = build_teacher_trajectory(pipe, condition_prompt, int(seed), teacher_ns, teacher_device)
            teacher_traj = trajectory["trajectory"]
            if target_index >= len(teacher_traj):
                raise ValueError(f"target index {target_index} >= trajectory length {len(teacher_traj)}")
            prompt_embeds = trajectory["prompt_embeds"].to(device=student_device, dtype=pipe.dtype)
            pooled_prompt_embeds = trajectory["pooled_prompt_embeds"].to(device=student_device, dtype=pipe.dtype)
            if source_index >= len(teacher_traj):
                raise ValueError(f"source index {source_index} >= trajectory length {len(teacher_traj)}")
            initial_latents = teacher_traj[source_index]["latents"].to(device=student_device, dtype=pipe.dtype)
            target_latents = teacher_traj[target_index]["latents"].to(device=student_device, dtype=pipe.dtype)
            timesteps, _ = flux_timesteps(pipe, initial_latents, int(args.teacher_steps), student_device)
            required_stage_indices = sorted(set([0, source_index, *source_bridge_stage_indices, *stage_target_indices]))
            timesteps_by_index = {index: timesteps[index].reshape(1) for index in required_stage_indices}
            stage_targets = {
                stage_index: {
                    "latents": teacher_traj[stage_index]["latents"].to(device=student_device, dtype=pipe.dtype)
                }
                for stage_index in required_stage_indices
            }
            continuation_targets = []
            contrast_prompt_embeds = None
            contrast_pooled_prompt_embeds = None
            contrast_stage_targets = {}
        guidance = torch.full([initial_latents.shape[0]], float(args.guidance), device=student_device, dtype=torch.float32)
        current_bridge_latents = initial_latents
        stage_start_index = source_index
        if source_bridge_stage_indices:
            current_bridge_latents = stage_targets[0]["latents"].to(device=student_device, dtype=initial_latents.dtype)
            upstream_start_index = 0
            for upstream_target_index in source_bridge_stage_indices:
                stage_entry = source_bridge_bank.get(int(upstream_target_index))
                if stage_entry is None:
                    raise ValueError(f"source bridge bank has no stage {upstream_target_index}")
                source_bridge, source_bridge_config, _ = load_student_for_weights(
                    Path(stage_entry["checkpoint"]),
                    student_device,
                    str(stage_entry["weights"]),
                )
                if source_bridge_config.__dict__ != config.__dict__:
                    raise ValueError(f"source bridge config does not match current student: {stage_entry['checkpoint']}")
                source_bridge.eval()
                with torch.no_grad():
                    upstream_timestep = timesteps_by_index[upstream_start_index]
                    upstream_output = source_bridge(
                        current_bridge_latents.float(),
                        upstream_timestep.expand(current_bridge_latents.shape[0]).float(),
                        prompt_embeds.float(),
                        pooled_prompt_embeds.float(),
                        guidance,
                    ).to(current_bridge_latents.dtype)
                    source_output_mode = str(stage_entry["output_mode"])
                    source_scale = float(stage_entry["scale"])
                    if source_output_mode == "absolute":
                        current_bridge_latents = upstream_output * source_scale
                    else:
                        current_bridge_latents = current_bridge_latents + upstream_output * source_scale
                del source_bridge
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                upstream_start_index = int(upstream_target_index)
            initial_latents = current_bridge_latents
        if distill_target_refiner is not None:
            final_stage_index = stage_target_indices[-1]
            with torch.no_grad():
                refined_target_latents = distill_target_refiner(current_bridge_latents.float()).to(
                    device=student_device,
                    dtype=initial_latents.dtype,
                )
            stage_targets[final_stage_index] = {"latents": refined_target_latents}
            target_latents = refined_target_latents
        stage_delta_losses: list[torch.Tensor] = []
        stage_latent_losses: list[torch.Tensor] = []
        stage_direction_losses: list[torch.Tensor] = []
        stage_norm_losses: list[torch.Tensor] = []
        stage_log_norm_losses: list[torch.Tensor] = []
        stage_orthogonal_losses: list[torch.Tensor] = []
        stage_normalized_delta_losses: list[torch.Tensor] = []
        stage_lowfreq_direction_losses: list[torch.Tensor] = []
        stage_lowfreq_norm_losses: list[torch.Tensor] = []
        stage_lowfreq_log_norm_losses: list[torch.Tensor] = []
        stage_lowfreq_orthogonal_losses: list[torch.Tensor] = []
        stage_prompt_delta_losses: list[torch.Tensor] = []
        stage_prompt_delta_direction_losses: list[torch.Tensor] = []
        stage_prompt_delta_norm_losses: list[torch.Tensor] = []
        stage_recovery_losses: list[torch.Tensor] = []
        stage_recovery_direction_losses: list[torch.Tensor] = []
        stage_recovery_norm_losses: list[torch.Tensor] = []
        stage_recovery_normalized_delta_losses: list[torch.Tensor] = []
        stage_recovery_lowfreq_direction_losses: list[torch.Tensor] = []
        stage_absolute_norm_losses: list[torch.Tensor] = []
        stage_parity_losses: list[torch.Tensor] = []
        stage_future_latent_losses: list[torch.Tensor] = []
        stage_future_delta_direction_losses: list[torch.Tensor] = []
        decoded_losses: list[torch.Tensor] = []
        decoded_lowfreq_losses: list[torch.Tensor] = []
        decoded_parity_losses: list[torch.Tensor] = []
        decoded_gradient_losses: list[torch.Tensor] = []
        decoded_highfreq_losses: list[torch.Tensor] = []
        pred_delta = torch.zeros_like(initial_latents)
        target_delta = target_latents - initial_latents
        for stage_target_index in stage_target_indices:
            stage_target_latents = stage_targets[stage_target_index]["latents"].to(device=student_device).to(initial_latents.dtype)
            stage_timestep_tensor = timesteps_by_index[stage_start_index]
            if future_head is not None:
                bridge_output_raw, bridge_hidden = student(
                    current_bridge_latents.float(),
                    stage_timestep_tensor.expand(current_bridge_latents.shape[0]).float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                    return_hidden=True,
                )
                bridge_output = bridge_output_raw.to(current_bridge_latents.dtype)
            else:
                bridge_output = student(
                    current_bridge_latents.float(),
                    stage_timestep_tensor.expand(current_bridge_latents.shape[0]).float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                ).to(current_bridge_latents.dtype)
                bridge_hidden = None
            stage_target_delta = stage_target_latents - current_bridge_latents
            if args.bridge_output_mode == "absolute":
                stage_pred_latents = bridge_output
                stage_pred_delta = stage_pred_latents - current_bridge_latents
                bridge_target = stage_target_latents
            else:
                stage_pred_delta = bridge_output
                stage_pred_latents = current_bridge_latents + stage_pred_delta
                bridge_target = stage_target_delta
            stage_delta_losses.append(F.huber_loss(bridge_output.float(), bridge_target.float(), delta=0.08))
            stage_latent_losses.append(F.mse_loss(stage_pred_latents.float(), stage_target_latents.float()))
            if args.direction_loss_weight > 0:
                stage_direction_losses.append(direction_loss(stage_pred_delta, stage_target_delta))
            if args.norm_loss_weight > 0:
                stage_norm_losses.append(norm_ratio_loss(stage_pred_delta, stage_target_delta))
            if args.log_norm_loss_weight > 0:
                stage_log_norm_losses.append(log_norm_ratio_loss(stage_pred_delta, stage_target_delta))
            if args.orthogonal_delta_loss_weight > 0:
                stage_orthogonal_losses.append(orthogonal_delta_loss(stage_pred_delta, stage_target_delta))
            if args.normalized_delta_loss_weight > 0:
                stage_normalized_delta_losses.append(rms_normalized_delta_loss(stage_pred_delta, stage_target_delta))
            if args.lowfreq_delta_direction_loss_weight > 0:
                stage_lowfreq_direction_losses.append(
                    packed_lowfreq_direction_loss(stage_pred_delta, stage_target_delta, args.lowfreq_delta_pool)
                )
            if args.lowfreq_delta_norm_loss_weight > 0:
                stage_lowfreq_norm_losses.append(
                    packed_lowfreq_norm_ratio_loss(stage_pred_delta, stage_target_delta, args.lowfreq_delta_pool)
                )
            if args.lowfreq_delta_log_norm_loss_weight > 0:
                stage_lowfreq_log_norm_losses.append(
                    packed_lowfreq_log_norm_ratio_loss(stage_pred_delta, stage_target_delta, args.lowfreq_delta_pool)
                )
            if args.lowfreq_delta_orthogonal_loss_weight > 0:
                stage_lowfreq_orthogonal_losses.append(
                    packed_lowfreq_orthogonal_delta_loss(stage_pred_delta, stage_target_delta, args.lowfreq_delta_pool)
                )
            use_recovery = (
                float(args.recovery_loss_weight) > 0
                or float(args.recovery_direction_loss_weight) > 0
                or float(args.recovery_norm_loss_weight) > 0
                or float(args.recovery_normalized_delta_loss_weight) > 0
                or float(args.recovery_lowfreq_direction_loss_weight) > 0
            )
            if use_recovery:
                with torch.no_grad():
                    recovery_source_latents = current_bridge_latents.float()
                    if float(args.recovery_error_scale) != 0:
                        recovery_source_latents = recovery_source_latents + float(args.recovery_error_scale) * (
                            stage_pred_latents.float().detach() - stage_target_latents.float()
                        )
                    if float(args.recovery_noise_std) > 0:
                        target_rms = stage_target_delta.float().flatten(1).pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
                        view_shape = [stage_target_delta.shape[0], *([1] * (stage_target_delta.ndim - 1))]
                        recovery_source_latents = recovery_source_latents + torch.randn_like(recovery_source_latents) * (
                            float(args.recovery_noise_std) * target_rms.view(*view_shape)
                        )
                recovery_output = student(
                    recovery_source_latents.float(),
                    stage_timestep_tensor.expand(recovery_source_latents.shape[0]).float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                ).to(current_bridge_latents.dtype)
                if args.bridge_output_mode == "absolute":
                    recovery_pred_latents = recovery_output
                    recovery_pred_delta = recovery_pred_latents - recovery_source_latents.to(recovery_output.dtype)
                    recovery_target = stage_target_latents
                else:
                    recovery_pred_delta = recovery_output
                    recovery_pred_latents = recovery_source_latents.to(recovery_output.dtype) + recovery_pred_delta
                    recovery_target = stage_target_latents - recovery_source_latents.to(stage_target_latents.dtype)
                recovery_target_delta = stage_target_latents - recovery_source_latents.to(stage_target_latents.dtype)
                if args.recovery_loss_weight > 0:
                    stage_recovery_losses.append(F.huber_loss(recovery_output.float(), recovery_target.float(), delta=0.08))
                if args.recovery_direction_loss_weight > 0:
                    stage_recovery_direction_losses.append(direction_loss(recovery_pred_delta, recovery_target_delta))
                if args.recovery_norm_loss_weight > 0:
                    stage_recovery_norm_losses.append(norm_ratio_loss(recovery_pred_delta, recovery_target_delta))
                if args.recovery_normalized_delta_loss_weight > 0:
                    stage_recovery_normalized_delta_losses.append(
                        rms_normalized_delta_loss(recovery_pred_delta, recovery_target_delta)
                    )
                if args.recovery_lowfreq_direction_loss_weight > 0:
                    stage_recovery_lowfreq_direction_losses.append(
                        packed_lowfreq_direction_loss(
                            recovery_pred_delta,
                            recovery_target_delta,
                            args.lowfreq_delta_pool,
                        )
                    )
            if (
                contrast_prompt_embeds is not None
                and contrast_pooled_prompt_embeds is not None
                and (
                    float(args.prompt_delta_loss_weight) > 0
                    or float(args.prompt_delta_direction_loss_weight) > 0
                    or float(args.prompt_delta_norm_loss_weight) > 0
                )
            ):
                contrast_output = student(
                    current_bridge_latents.float(),
                    stage_timestep_tensor.expand(current_bridge_latents.shape[0]).float(),
                    contrast_prompt_embeds.float(),
                    contrast_pooled_prompt_embeds.float(),
                    guidance,
                ).to(current_bridge_latents.dtype)
                contrast_source_latents = contrast_stage_targets[stage_start_index]["latents"].to(
                    device=student_device,
                    dtype=current_bridge_latents.dtype,
                )
                contrast_target_latents = contrast_stage_targets[stage_target_index]["latents"].to(
                    device=student_device,
                    dtype=current_bridge_latents.dtype,
                )
                contrast_teacher_delta = contrast_target_latents - contrast_source_latents
                if args.bridge_output_mode == "absolute":
                    contrast_pred_delta = contrast_output - current_bridge_latents.to(contrast_output.dtype)
                else:
                    contrast_pred_delta = contrast_output
                prompt_delta_pred = stage_pred_delta - contrast_pred_delta
                prompt_delta_target = stage_target_delta - contrast_teacher_delta
                if args.prompt_delta_loss_weight > 0:
                    stage_prompt_delta_losses.append(
                        rms_normalized_delta_loss(prompt_delta_pred, prompt_delta_target)
                    )
                if args.prompt_delta_direction_loss_weight > 0:
                    stage_prompt_delta_direction_losses.append(
                        direction_loss(prompt_delta_pred, prompt_delta_target)
                    )
                if args.prompt_delta_norm_loss_weight > 0:
                    stage_prompt_delta_norm_losses.append(norm_ratio_loss(prompt_delta_pred, prompt_delta_target))
            if args.absolute_latent_norm_loss_weight > 0:
                stage_absolute_norm_losses.append(norm_ratio_loss(stage_pred_latents, stage_target_latents))
            if args.parity_latent_loss_weight > 0:
                stage_parity_losses.append(
                    packed_parity_lowfreq_loss(
                        stage_pred_latents.float(),
                        stage_target_latents.float(),
                        int(args.parity_lowfreq_size),
                    )
                )
            if future_head is not None and bridge_hidden is not None:
                future_pred_latents = future_head(bridge_hidden.float()).to(stage_target_latents.dtype)
                future_pool = int(args.future_lowfreq_pool)
                if args.future_latent_loss_weight > 0:
                    stage_future_latent_losses.append(
                        F.mse_loss(
                            packed_lowfreq(future_pred_latents, future_pool),
                            packed_lowfreq(stage_target_latents, future_pool),
                        )
                    )
                if args.future_delta_direction_loss_weight > 0:
                    stage_future_delta_direction_losses.append(
                        direction_loss(
                            packed_lowfreq(future_pred_latents - current_bridge_latents, future_pool),
                            packed_lowfreq(stage_target_latents - current_bridge_latents, future_pool),
                        )
                    )
            use_decoded_stage_loss = (
                (
                    args.decoded_loss_weight > 0
                    or args.decoded_lowfreq_loss_weight > 0
                    or args.decoded_parity_loss_weight > 0
                    or args.decoded_gradient_loss_weight > 0
                    or args.decoded_highfreq_loss_weight > 0
                )
                and (not args.decoded_loss_final_stage_only or stage_target_index == stage_target_indices[-1])
            )
            if use_decoded_stage_loss:
                if pipe is None:
                    raise ValueError("decoded losses require a loaded FLUX VAE")
                pred_tensor = decode_flux_latents_tensor(pipe, stage_pred_latents, args.height, args.width).float()
                with torch.no_grad():
                    target_tensor = decode_flux_latents_tensor(pipe, stage_target_latents, args.height, args.width).float()
                if args.decoded_loss_weight > 0:
                    decoded_losses.append(F.l1_loss(pred_tensor, target_tensor))
                if args.decoded_lowfreq_loss_weight > 0:
                    pred_low = F.interpolate(
                        pred_tensor,
                        size=(args.decoded_lowfreq_size, args.decoded_lowfreq_size),
                        mode="area",
                    )
                    target_low = F.interpolate(
                        target_tensor,
                        size=(args.decoded_lowfreq_size, args.decoded_lowfreq_size),
                        mode="area",
                    )
                    decoded_lowfreq_losses.append(F.l1_loss(pred_low, target_low))
                if args.decoded_parity_loss_weight > 0:
                    decoded_parity_losses.append(
                        image_parity_lowfreq_loss(pred_tensor, target_tensor, int(args.decoded_parity_size))
                    )
                if args.decoded_gradient_loss_weight > 0:
                    decoded_gradient_losses.append(image_gradient_loss(pred_tensor, target_tensor))
                if args.decoded_highfreq_loss_weight > 0:
                    decoded_highfreq_losses.append(
                        image_highfreq_loss(pred_tensor, target_tensor, int(args.decoded_highfreq_size))
                    )
            current_bridge_latents = stage_pred_latents
            stage_start_index = stage_target_index
            pred_delta = stage_pred_delta
        pred_latents = current_bridge_latents
        target_latents = stage_targets[stage_target_indices[-1]]["latents"].to(device=student_device).to(initial_latents.dtype)
        target_delta = target_latents - initial_latents
        delta_loss = torch.stack(stage_delta_losses).mean()
        latent_loss = torch.stack(stage_latent_losses).mean()
        direction_value = (
            torch.stack(stage_direction_losses).mean()
            if stage_direction_losses
            else torch.zeros((), device=student_device)
        )
        norm_value = (
            torch.stack(stage_norm_losses).mean()
            if stage_norm_losses
            else torch.zeros((), device=student_device)
        )
        log_norm_value = (
            torch.stack(stage_log_norm_losses).mean()
            if stage_log_norm_losses
            else torch.zeros((), device=student_device)
        )
        orthogonal_value = (
            torch.stack(stage_orthogonal_losses).mean()
            if stage_orthogonal_losses
            else torch.zeros((), device=student_device)
        )
        normalized_delta_value = (
            torch.stack(stage_normalized_delta_losses).mean()
            if stage_normalized_delta_losses
            else torch.zeros((), device=student_device)
        )
        lowfreq_direction_value = (
            torch.stack(stage_lowfreq_direction_losses).mean()
            if stage_lowfreq_direction_losses
            else torch.zeros((), device=student_device)
        )
        lowfreq_norm_value = (
            torch.stack(stage_lowfreq_norm_losses).mean()
            if stage_lowfreq_norm_losses
            else torch.zeros((), device=student_device)
        )
        lowfreq_log_norm_value = (
            torch.stack(stage_lowfreq_log_norm_losses).mean()
            if stage_lowfreq_log_norm_losses
            else torch.zeros((), device=student_device)
        )
        lowfreq_orthogonal_value = (
            torch.stack(stage_lowfreq_orthogonal_losses).mean()
            if stage_lowfreq_orthogonal_losses
            else torch.zeros((), device=student_device)
        )
        prompt_delta_value = (
            torch.stack(stage_prompt_delta_losses).mean()
            if stage_prompt_delta_losses
            else torch.zeros((), device=student_device)
        )
        prompt_delta_direction_value = (
            torch.stack(stage_prompt_delta_direction_losses).mean()
            if stage_prompt_delta_direction_losses
            else torch.zeros((), device=student_device)
        )
        prompt_delta_norm_value = (
            torch.stack(stage_prompt_delta_norm_losses).mean()
            if stage_prompt_delta_norm_losses
            else torch.zeros((), device=student_device)
        )
        recovery_value = (
            torch.stack(stage_recovery_losses).mean()
            if stage_recovery_losses
            else torch.zeros((), device=student_device)
        )
        recovery_direction_value = (
            torch.stack(stage_recovery_direction_losses).mean()
            if stage_recovery_direction_losses
            else torch.zeros((), device=student_device)
        )
        recovery_norm_value = (
            torch.stack(stage_recovery_norm_losses).mean()
            if stage_recovery_norm_losses
            else torch.zeros((), device=student_device)
        )
        recovery_normalized_delta_value = (
            torch.stack(stage_recovery_normalized_delta_losses).mean()
            if stage_recovery_normalized_delta_losses
            else torch.zeros((), device=student_device)
        )
        recovery_lowfreq_direction_value = (
            torch.stack(stage_recovery_lowfreq_direction_losses).mean()
            if stage_recovery_lowfreq_direction_losses
            else torch.zeros((), device=student_device)
        )
        terminal_pred_delta = pred_latents - initial_latents
        terminal_normalized_delta_value = (
            rms_normalized_delta_loss(terminal_pred_delta, target_delta)
            if args.terminal_normalized_delta_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_direction_value = (
            direction_loss(terminal_pred_delta, target_delta)
            if args.terminal_direction_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_norm_value = (
            norm_ratio_loss(terminal_pred_delta, target_delta)
            if args.terminal_norm_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_log_norm_value = (
            log_norm_ratio_loss(terminal_pred_delta, target_delta)
            if args.terminal_log_norm_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_orthogonal_value = (
            orthogonal_delta_loss(terminal_pred_delta, target_delta)
            if args.terminal_orthogonal_delta_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_lowfreq_direction_value = (
            packed_lowfreq_direction_loss(terminal_pred_delta, target_delta, args.lowfreq_delta_pool)
            if args.terminal_lowfreq_delta_direction_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_lowfreq_norm_value = (
            packed_lowfreq_norm_ratio_loss(terminal_pred_delta, target_delta, args.lowfreq_delta_pool)
            if args.terminal_lowfreq_delta_norm_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_lowfreq_log_norm_value = (
            packed_lowfreq_log_norm_ratio_loss(terminal_pred_delta, target_delta, args.lowfreq_delta_pool)
            if args.terminal_lowfreq_delta_log_norm_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        terminal_lowfreq_orthogonal_value = (
            packed_lowfreq_orthogonal_delta_loss(terminal_pred_delta, target_delta, args.lowfreq_delta_pool)
            if args.terminal_lowfreq_delta_orthogonal_loss_weight > 0
            else torch.zeros((), device=student_device)
        )
        absolute_latent_norm_value = (
            torch.stack(stage_absolute_norm_losses).mean()
            if stage_absolute_norm_losses
            else torch.zeros((), device=student_device)
        )
        parity_latent_value = (
            torch.stack(stage_parity_losses).mean()
            if stage_parity_losses
            else torch.zeros((), device=student_device)
        )
        future_latent_value = (
            torch.stack(stage_future_latent_losses).mean()
            if stage_future_latent_losses
            else torch.zeros((), device=student_device)
        )
        future_delta_direction_value = (
            torch.stack(stage_future_delta_direction_losses).mean()
            if stage_future_delta_direction_losses
            else torch.zeros((), device=student_device)
        )
        decoded_loss = torch.zeros((), device=student_device)
        decoded_lowfreq_loss = torch.zeros((), device=student_device)
        decoded_parity_loss = torch.zeros((), device=student_device)
        decoded_gradient_loss = torch.zeros((), device=student_device)
        decoded_highfreq_loss = torch.zeros((), device=student_device)
        if decoded_losses:
            decoded_loss = torch.stack(decoded_losses).mean()
        if decoded_lowfreq_losses:
            decoded_lowfreq_loss = torch.stack(decoded_lowfreq_losses).mean()
        if decoded_parity_losses:
            decoded_parity_loss = torch.stack(decoded_parity_losses).mean()
        if decoded_gradient_losses:
            decoded_gradient_loss = torch.stack(decoded_gradient_losses).mean()
        if decoded_highfreq_losses:
            decoded_highfreq_loss = torch.stack(decoded_highfreq_losses).mean()
        continuation_latent_loss = torch.zeros((), device=student_device)
        continuation_lowfreq_loss = torch.zeros((), device=student_device)
        continuation_step_latent_loss = torch.zeros((), device=student_device)
        continuation_step_lowfreq_loss = torch.zeros((), device=student_device)
        continuation_step_direction_loss = torch.zeros((), device=student_device)
        continuation_step_norm_loss = torch.zeros((), device=student_device)
        continuation_direction_loss = torch.zeros((), device=student_device)
        continuation_norm_loss = torch.zeros((), device=student_device)
        continuation_normalized_delta_loss = torch.zeros((), device=student_device)
        continuation_decoded_loss = torch.zeros((), device=student_device)
        continuation_decoded_lowfreq_loss = torch.zeros((), device=student_device)
        continuation_decoded_parity_loss = torch.zeros((), device=student_device)
        continuation_decoded_gradient_loss = torch.zeros((), device=student_device)
        continuation_decoded_highfreq_loss = torch.zeros((), device=student_device)
        if continuation_student is not None and int(args.continuation_rollout_len) > 0:
            if cached_pairs is None:
                raise ValueError("--continuation-checkpoint currently requires cached target dirs")
            if len(continuation_targets) < 2:
                raise ValueError("cached bridge row does not include enough continuation targets")
            rollout_targets = continuation_targets[: int(args.continuation_rollout_len) + 1]
            current_rollout_latents = pred_latents.float()
            rollout_step_count = 0
            for current_row, next_row in zip(rollout_targets[:-1], rollout_targets[1:]):
                current_timestep = current_row["timestep"].to(device=student_device).reshape(1)
                current_teacher_target = current_row["teacher_target"].to(device=student_device).float()
                current_teacher_latents = current_row["latents"].to(device=student_device).float()
                next_teacher_latents = next_row["latents"].to(device=student_device).float()
                previous_rollout_latents = current_rollout_latents
                continuation_pred = continuation_student(
                    current_rollout_latents.float(),
                    current_timestep.expand(current_rollout_latents.shape[0]).float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance.float(),
                )
                step_delta = teacher_step_delta(
                    current_teacher_latents,
                    next_teacher_latents,
                    current_teacher_target,
                )
                current_rollout_latents = current_rollout_latents + step_delta * continuation_pred.float()
                predicted_step_delta = current_rollout_latents.float() - previous_rollout_latents.float()
                target_step_delta = next_teacher_latents.float() - previous_rollout_latents.float()
                if args.continuation_step_latent_loss_weight > 0:
                    continuation_step_latent_loss = continuation_step_latent_loss + F.mse_loss(
                        current_rollout_latents.float(),
                        next_teacher_latents.float(),
                    )
                if args.continuation_step_lowfreq_loss_weight > 0:
                    continuation_step_lowfreq_loss = continuation_step_lowfreq_loss + packed_lowfreq_loss(
                        current_rollout_latents.float(),
                        next_teacher_latents.float(),
                        pool=4,
                    )
                if args.continuation_step_direction_loss_weight > 0:
                    continuation_step_direction_loss = continuation_step_direction_loss + direction_loss(
                        predicted_step_delta,
                        target_step_delta,
                    )
                if args.continuation_step_norm_loss_weight > 0:
                    continuation_step_norm_loss = continuation_step_norm_loss + norm_ratio_loss(
                        predicted_step_delta,
                        target_step_delta,
                    )
                rollout_step_count += 1
            if rollout_step_count > 0:
                continuation_step_latent_loss = continuation_step_latent_loss / float(rollout_step_count)
                continuation_step_lowfreq_loss = continuation_step_lowfreq_loss / float(rollout_step_count)
                continuation_step_direction_loss = continuation_step_direction_loss / float(rollout_step_count)
                continuation_step_norm_loss = continuation_step_norm_loss / float(rollout_step_count)
            continuation_target_latents = rollout_targets[-1]["latents"].to(device=student_device).float()
            continuation_latent_loss = F.mse_loss(current_rollout_latents.float(), continuation_target_latents.float())
            continuation_lowfreq_loss = packed_lowfreq_loss(
                current_rollout_latents.float(),
                continuation_target_latents.float(),
                pool=4,
            )
            continuation_pred_delta = current_rollout_latents.float() - pred_latents.float()
            continuation_target_delta = continuation_target_latents.float() - pred_latents.float()
            if args.continuation_direction_loss_weight > 0:
                continuation_direction_loss = direction_loss(continuation_pred_delta, continuation_target_delta)
            if args.continuation_norm_loss_weight > 0:
                continuation_norm_loss = norm_ratio_loss(continuation_pred_delta, continuation_target_delta)
            if args.continuation_normalized_delta_loss_weight > 0:
                continuation_normalized_delta_loss = rms_normalized_delta_loss(
                    continuation_pred_delta,
                    continuation_target_delta,
                )
            if (
                args.continuation_decoded_loss_weight > 0
                or args.continuation_decoded_lowfreq_loss_weight > 0
                or args.continuation_decoded_parity_loss_weight > 0
                or args.continuation_decoded_gradient_loss_weight > 0
                or args.continuation_decoded_highfreq_loss_weight > 0
            ):
                if pipe is None:
                    raise ValueError("continuation decoded losses require a loaded FLUX VAE")
                pred_continuation_tensor = decode_flux_latents_tensor(
                    pipe,
                    current_rollout_latents,
                    args.height,
                    args.width,
                ).float()
                with torch.no_grad():
                    target_continuation_tensor = decode_flux_latents_tensor(
                        pipe,
                        continuation_target_latents,
                        args.height,
                        args.width,
                    ).float()
                if args.continuation_decoded_loss_weight > 0:
                    continuation_decoded_loss = F.l1_loss(
                        pred_continuation_tensor,
                        target_continuation_tensor,
                    )
                if args.continuation_decoded_lowfreq_loss_weight > 0:
                    pred_continuation_low = F.interpolate(
                        pred_continuation_tensor,
                        size=(args.decoded_lowfreq_size, args.decoded_lowfreq_size),
                        mode="area",
                    )
                    target_continuation_low = F.interpolate(
                        target_continuation_tensor,
                        size=(args.decoded_lowfreq_size, args.decoded_lowfreq_size),
                        mode="area",
                    )
                    continuation_decoded_lowfreq_loss = F.l1_loss(
                        pred_continuation_low,
                        target_continuation_low,
                    )
                if args.continuation_decoded_parity_loss_weight > 0:
                    continuation_decoded_parity_loss = image_parity_lowfreq_loss(
                        pred_continuation_tensor,
                        target_continuation_tensor,
                        int(args.decoded_parity_size),
                    )
                if args.continuation_decoded_gradient_loss_weight > 0:
                    continuation_decoded_gradient_loss = image_gradient_loss(
                        pred_continuation_tensor,
                        target_continuation_tensor,
                    )
                if args.continuation_decoded_highfreq_loss_weight > 0:
                    continuation_decoded_highfreq_loss = image_highfreq_loss(
                        pred_continuation_tensor,
                        target_continuation_tensor,
                        int(args.decoded_highfreq_size),
                    )
        loss = (
            float(args.delta_loss_weight) * delta_loss
            + float(args.latent_loss_weight) * latent_loss
            + float(args.direction_loss_weight) * direction_value
            + float(args.norm_loss_weight) * norm_value
            + float(args.log_norm_loss_weight) * log_norm_value
            + float(args.orthogonal_delta_loss_weight) * orthogonal_value
            + float(args.normalized_delta_loss_weight) * normalized_delta_value
            + float(args.lowfreq_delta_direction_loss_weight) * lowfreq_direction_value
            + float(args.lowfreq_delta_norm_loss_weight) * lowfreq_norm_value
            + float(args.lowfreq_delta_log_norm_loss_weight) * lowfreq_log_norm_value
            + float(args.lowfreq_delta_orthogonal_loss_weight) * lowfreq_orthogonal_value
            + float(args.prompt_delta_loss_weight) * prompt_delta_value
            + float(args.prompt_delta_direction_loss_weight) * prompt_delta_direction_value
            + float(args.prompt_delta_norm_loss_weight) * prompt_delta_norm_value
            + float(args.recovery_loss_weight) * recovery_value
            + float(args.recovery_direction_loss_weight) * recovery_direction_value
            + float(args.recovery_norm_loss_weight) * recovery_norm_value
            + float(args.recovery_normalized_delta_loss_weight) * recovery_normalized_delta_value
            + float(args.recovery_lowfreq_direction_loss_weight) * recovery_lowfreq_direction_value
            + float(args.terminal_normalized_delta_loss_weight) * terminal_normalized_delta_value
            + float(args.terminal_direction_loss_weight) * terminal_direction_value
            + float(args.terminal_norm_loss_weight) * terminal_norm_value
            + float(args.terminal_log_norm_loss_weight) * terminal_log_norm_value
            + float(args.terminal_orthogonal_delta_loss_weight) * terminal_orthogonal_value
            + float(args.terminal_lowfreq_delta_direction_loss_weight) * terminal_lowfreq_direction_value
            + float(args.terminal_lowfreq_delta_norm_loss_weight) * terminal_lowfreq_norm_value
            + float(args.terminal_lowfreq_delta_log_norm_loss_weight) * terminal_lowfreq_log_norm_value
            + float(args.terminal_lowfreq_delta_orthogonal_loss_weight) * terminal_lowfreq_orthogonal_value
            + float(args.absolute_latent_norm_loss_weight) * absolute_latent_norm_value
            + float(args.parity_latent_loss_weight) * parity_latent_value
            + float(args.future_latent_loss_weight) * future_latent_value
            + float(args.future_delta_direction_loss_weight) * future_delta_direction_value
            + float(args.decoded_loss_weight) * decoded_loss
            + float(args.decoded_lowfreq_loss_weight) * decoded_lowfreq_loss
            + float(args.decoded_parity_loss_weight) * decoded_parity_loss
            + float(args.decoded_gradient_loss_weight) * decoded_gradient_loss
            + float(args.decoded_highfreq_loss_weight) * decoded_highfreq_loss
            + float(args.continuation_step_latent_loss_weight) * continuation_step_latent_loss
            + float(args.continuation_step_lowfreq_loss_weight) * continuation_step_lowfreq_loss
            + float(args.continuation_step_direction_loss_weight) * continuation_step_direction_loss
            + float(args.continuation_step_norm_loss_weight) * continuation_step_norm_loss
            + float(args.continuation_latent_loss_weight) * continuation_latent_loss
            + float(args.continuation_lowfreq_loss_weight) * continuation_lowfreq_loss
            + float(args.continuation_direction_loss_weight) * continuation_direction_loss
            + float(args.continuation_norm_loss_weight) * continuation_norm_loss
            + float(args.continuation_normalized_delta_loss_weight) * continuation_normalized_delta_loss
            + float(args.continuation_decoded_loss_weight) * continuation_decoded_loss
            + float(args.continuation_decoded_lowfreq_loss_weight) * continuation_decoded_lowfreq_loss
            + float(args.continuation_decoded_parity_loss_weight) * continuation_decoded_parity_loss
            + float(args.continuation_decoded_gradient_loss_weight) * continuation_decoded_gradient_loss
            + float(args.continuation_decoded_highfreq_loss_weight) * continuation_decoded_highfreq_loss
        )
        prompt_loss_weight = float(prompt_loss_weights.get(prompt, 1.0))
        if prompt_loss_weight != 1.0:
            loss = prompt_loss_weight * loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_parameters = [p for p in student.parameters() if p.requires_grad]
        if future_head is not None:
            clip_parameters.extend(future_head.parameters())
        if continuation_student is not None and args.train_continuation:
            clip_parameters.extend(p for p in continuation_student.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_(clip_parameters, float(args.grad_clip))
        optimizer.step()
        if hasattr(student, "output_log_scale"):
            with torch.no_grad():
                student.output_log_scale.clamp_(
                    min=float(args.output_log_scale_min),
                    max=float(args.output_log_scale_max),
                )
        step += 1
        if ema_state is not None:
            update_ema(ema_state, student, float(args.ema_decay))
        row = {
            "step": int(step),
            "prompt": prompt,
            "source_name": source_name,
            "seed": int(seed),
            "target_index": int(target_index),
            "stage_target_indices": stage_target_indices,
            "bridge_output_mode": str(args.bridge_output_mode),
            "loss": float(loss.detach().item()),
            "prompt_loss_weight": float(prompt_loss_weight),
            "delta_loss": float(delta_loss.detach().item()),
            "latent_loss": float(latent_loss.detach().item()),
            "direction_loss": float(direction_value.detach().item()),
            "norm_loss": float(norm_value.detach().item()),
            "log_norm_loss": float(log_norm_value.detach().item()),
            "orthogonal_delta_loss": float(orthogonal_value.detach().item()),
            "normalized_delta_loss": float(normalized_delta_value.detach().item()),
            "lowfreq_delta_direction_loss": float(lowfreq_direction_value.detach().item()),
            "lowfreq_delta_norm_loss": float(lowfreq_norm_value.detach().item()),
            "lowfreq_delta_log_norm_loss": float(lowfreq_log_norm_value.detach().item()),
            "lowfreq_delta_orthogonal_loss": float(lowfreq_orthogonal_value.detach().item()),
            "prompt_delta_loss": float(prompt_delta_value.detach().item()),
            "prompt_delta_direction_loss": float(prompt_delta_direction_value.detach().item()),
            "prompt_delta_norm_loss": float(prompt_delta_norm_value.detach().item()),
            "recovery_loss": float(recovery_value.detach().item()),
            "recovery_direction_loss": float(recovery_direction_value.detach().item()),
            "recovery_norm_loss": float(recovery_norm_value.detach().item()),
            "recovery_normalized_delta_loss": float(recovery_normalized_delta_value.detach().item()),
            "recovery_lowfreq_direction_loss": float(recovery_lowfreq_direction_value.detach().item()),
            "terminal_normalized_delta_loss": float(terminal_normalized_delta_value.detach().item()),
            "terminal_direction_loss": float(terminal_direction_value.detach().item()),
            "terminal_norm_loss": float(terminal_norm_value.detach().item()),
            "terminal_log_norm_loss": float(terminal_log_norm_value.detach().item()),
            "terminal_orthogonal_delta_loss": float(terminal_orthogonal_value.detach().item()),
            "terminal_lowfreq_delta_direction_loss": float(terminal_lowfreq_direction_value.detach().item()),
            "terminal_lowfreq_delta_norm_loss": float(terminal_lowfreq_norm_value.detach().item()),
            "terminal_lowfreq_delta_log_norm_loss": float(terminal_lowfreq_log_norm_value.detach().item()),
            "terminal_lowfreq_delta_orthogonal_loss": float(terminal_lowfreq_orthogonal_value.detach().item()),
            "absolute_latent_norm_loss": float(absolute_latent_norm_value.detach().item()),
            "parity_latent_loss": float(parity_latent_value.detach().item()),
            "future_latent_loss": float(future_latent_value.detach().item()),
            "future_delta_direction_loss": float(future_delta_direction_value.detach().item()),
            "decoded_loss": float(decoded_loss.detach().item()),
            "decoded_lowfreq_loss": float(decoded_lowfreq_loss.detach().item()),
            "decoded_parity_loss": float(decoded_parity_loss.detach().item()),
            "decoded_gradient_loss": float(decoded_gradient_loss.detach().item()),
            "decoded_highfreq_loss": float(decoded_highfreq_loss.detach().item()),
            "continuation_step_latent_loss": float(continuation_step_latent_loss.detach().item()),
            "continuation_step_lowfreq_loss": float(continuation_step_lowfreq_loss.detach().item()),
            "continuation_step_direction_loss": float(continuation_step_direction_loss.detach().item()),
            "continuation_step_norm_loss": float(continuation_step_norm_loss.detach().item()),
            "continuation_latent_loss": float(continuation_latent_loss.detach().item()),
            "continuation_lowfreq_loss": float(continuation_lowfreq_loss.detach().item()),
            "continuation_direction_loss": float(continuation_direction_loss.detach().item()),
            "continuation_norm_loss": float(continuation_norm_loss.detach().item()),
            "continuation_normalized_delta_loss": float(continuation_normalized_delta_loss.detach().item()),
            "continuation_decoded_loss": float(continuation_decoded_loss.detach().item()),
            "continuation_decoded_lowfreq_loss": float(continuation_decoded_lowfreq_loss.detach().item()),
            "continuation_decoded_parity_loss": float(continuation_decoded_parity_loss.detach().item()),
            "continuation_decoded_gradient_loss": float(continuation_decoded_gradient_loss.detach().item()),
            "continuation_decoded_highfreq_loss": float(continuation_decoded_highfreq_loss.detach().item()),
            "pred_delta_rms": float(tensor_rms(pred_delta).detach().item()),
            "target_delta_rms": float(tensor_rms(target_delta).detach().item()),
            "pred_latent_rms": float(tensor_rms(pred_latents).detach().item()),
            "target_latent_rms": float(tensor_rms(target_latents).detach().item()),
            "output_log_scale": float(student.output_log_scale.detach().float().item())
            if hasattr(student, "output_log_scale")
            else 0.0,
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if step % int(args.log_every) == 0 or step == start_step + 1:
            print(json.dumps(row), flush=True)
        if step % int(args.checkpoint_every) == 0:
            save_checkpoint(output_dir / "flux_packed_student.pt", student, config, step, ema_state)
            if continuation_student is not None and args.train_continuation:
                save_checkpoint(output_dir / "flux_packed_continuation.pt", continuation_student, config, step, None)
            save_optimizer_checkpoint(output_dir, step, optimizer)
    save_checkpoint(output_dir / "flux_packed_student.pt", student, config, step, ema_state)
    if continuation_student is not None and args.train_continuation:
        save_checkpoint(output_dir / "flux_packed_continuation.pt", continuation_student, config, step, None)
    save_optimizer_checkpoint(output_dir, step, optimizer)


if __name__ == "__main__":
    main()
