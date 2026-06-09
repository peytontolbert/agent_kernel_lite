#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from train_agentkernel_lite_image_flux_flow_distill import (
    FluxPackedStudent,
    FluxPackedStudentConfig,
)
from train_agentkernel_lite_image_flux_flow_rollout_distill import (
    build_sequences,
    load_embedding,
    load_target,
    packed_lowfreq_loss,
    sequence_key,
    teacher_step_delta,
)


def load_student(checkpoint_path: Path, device: torch.device) -> tuple[FluxPackedStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"{checkpoint_path} does not contain a student config")
    config_defaults = asdict(FluxPackedStudentConfig())
    config_defaults.update(config_payload)
    config = FluxPackedStudentConfig(**config_defaults)
    student = FluxPackedStudent(config).to(device)
    state = checkpoint.get("student_ema") if checkpoint.get("student_ema") and checkpoint.get("use_ema_for_trace") else None
    if state is None:
        state = checkpoint.get("student", checkpoint)
    missing, unexpected = student.load_state_dict(state, strict=False)
    student.eval()
    return student, {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step") or 0),
        "loss": float(checkpoint.get("loss") or 0.0),
        "missing": missing,
        "unexpected": unexpected,
        "config": asdict(config),
    }


def tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten(1)
    bf = b.float().flatten(1)
    return float(F.cosine_similarity(af, bf, dim=1).mean().detach().cpu().item())


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    an = a.float().flatten(1).norm(dim=1)
    bn = b.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    return float((an / bn).mean().detach().cpu().item())


def rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt().detach().cpu().item())


def highfreq_mse(a: torch.Tensor, b: torch.Tensor, pool: int) -> float:
    tokens = a.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return float(F.mse_loss(a.float(), b.float()).detach().cpu().item())
    pool = max(int(pool), 1)
    ag = a.float().transpose(1, 2).reshape(a.shape[0], a.shape[2], side, side)
    bg = b.float().transpose(1, 2).reshape(b.shape[0], b.shape[2], side, side)
    al = F.interpolate(F.avg_pool2d(ag, kernel_size=pool, stride=pool), size=(side, side), mode="nearest")
    bl = F.interpolate(F.avg_pool2d(bg, kernel_size=pool, stride=pool), size=(side, side), mode="nearest")
    return float(F.mse_loss(ag - al, bg - bl).detach().cpu().item())


def packed_parity_mse(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    tokens = a.shape[1]
    side = int(tokens**0.5)
    if side * side != tokens:
        return {}
    ag = a.float().reshape(a.shape[0], side, side, a.shape[2])
    bg = b.float().reshape(b.shape[0], side, side, b.shape[2])
    out: dict[str, float] = {}
    for y in range(2):
        for x in range(2):
            out[f"p{y}{x}"] = float(F.mse_loss(ag[:, y::2, x::2, :], bg[:, y::2, x::2, :]).detach().cpu().item())
    return out


@torch.no_grad()
def trace_sequence(
    student: FluxPackedStudent,
    sequence: list[dict[str, Any]],
    *,
    device: torch.device,
    rollout_len: int,
    lowfreq_pool: int,
    latent_divergence_threshold: float,
    lowfreq_divergence_threshold: float,
) -> dict[str, Any]:
    prompt_embeds, pooled_prompt_embeds = load_embedding(sequence[0], device)
    current = load_target(sequence[0], device)["latents"].float()
    guidance = load_target(sequence[0], device)["guidance"].reshape(1)
    max_len = min(int(rollout_len), len(sequence) - 1)
    step_rows: list[dict[str, Any]] = []
    first_divergence_index: int | None = None
    endpoint_target = None
    for idx in range(max_len):
        row = sequence[idx]
        next_row = sequence[idx + 1]
        teacher_current = load_target(row, device)
        teacher_next = load_target(next_row, device)
        endpoint_target = teacher_next["latents"].float()
        timestep = teacher_current["timestep"].reshape(1).float()
        teacher_flow = teacher_current["teacher_target"].float()
        pred = student(current, timestep, prompt_embeds.float(), pooled_prompt_embeds.float(), guidance.float()).float()
        delta = teacher_step_delta(teacher_current["latents"], teacher_next["latents"], teacher_flow)
        student_next = current + delta * pred
        latent_mse = float(F.mse_loss(student_next.float(), teacher_next["latents"].float()).detach().cpu().item())
        lowfreq_mse = float(packed_lowfreq_loss(student_next, teacher_next["latents"], lowfreq_pool).detach().cpu().item())
        row_out = {
            "index": idx,
            "timestep_index": int(row.get("timestep_index", idx)),
            "timestep": float(teacher_current["timestep"].detach().cpu().item()),
            "flow_mse": float(F.mse_loss(pred, teacher_flow).detach().cpu().item()),
            "flow_cosine": tensor_cosine(pred, teacher_flow),
            "flow_norm_ratio": norm_ratio(pred, teacher_flow),
            "student_latent_rms": rms(student_next),
            "teacher_latent_rms": rms(teacher_next["latents"]),
            "pred_flow_rms": rms(pred),
            "teacher_flow_rms": rms(teacher_flow),
            "latent_mse_to_teacher_next": latent_mse,
            "lowfreq_latent_mse_to_teacher_next": lowfreq_mse,
            "highfreq_latent_mse_to_teacher_next": highfreq_mse(student_next, teacher_next["latents"], lowfreq_pool),
            "packed_parity_mse": packed_parity_mse(student_next, teacher_next["latents"]),
        }
        if first_divergence_index is None and (
            latent_mse >= float(latent_divergence_threshold) or lowfreq_mse >= float(lowfreq_divergence_threshold)
        ):
            first_divergence_index = idx
        step_rows.append(row_out)
        current = student_next
    if endpoint_target is None:
        endpoint_target = current
    endpoint_mse = float(F.mse_loss(current, endpoint_target.float()).detach().cpu().item())
    endpoint_lowfreq_mse = float(packed_lowfreq_loss(current, endpoint_target, lowfreq_pool).detach().cpu().item())
    prompt = str(sequence[0].get("prompt", "")).strip()
    return {
        "sequence_key": sequence_key(sequence[0]),
        "prompt": prompt,
        "seed": int(sequence[0].get("seed", 0) or 0),
        "source": Path(str(sequence[0].get("_target_dir", ""))).name,
        "steps_traced": max_len,
        "first_divergence_index": first_divergence_index,
        "endpoint_mse": endpoint_mse,
        "endpoint_lowfreq_mse": endpoint_lowfreq_mse,
        "endpoint_highfreq_mse": highfreq_mse(current, endpoint_target, lowfreq_pool),
        "endpoint_packed_parity_mse": packed_parity_mse(current, endpoint_target),
        "steps": step_rows,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    endpoint = torch.tensor([float(row["endpoint_mse"]) for row in results])
    endpoint_low = torch.tensor([float(row["endpoint_lowfreq_mse"]) for row in results])
    first = [row["first_divergence_index"] for row in results if row["first_divergence_index"] is not None]
    step_buckets: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        for step in result["steps"]:
            step_buckets.setdefault(int(step["index"]), []).append(step)
    per_step = []
    for idx, rows in sorted(step_buckets.items()):
        flow_norm_ratio_value = float(torch.tensor([r["flow_norm_ratio"] for r in rows]).mean().item())
        per_step.append(
            {
                "index": idx,
                "flow_mse": float(torch.tensor([r["flow_mse"] for r in rows]).mean().item()),
                "flow_cosine": float(torch.tensor([r["flow_cosine"] for r in rows]).mean().item()),
                "flow_norm_ratio": flow_norm_ratio_value,
                "renorm_flow_scale_to_teacher": 1.0 / max(flow_norm_ratio_value, 1e-8),
                "latent_mse": float(torch.tensor([r["latent_mse_to_teacher_next"] for r in rows]).mean().item()),
                "lowfreq_latent_mse": float(torch.tensor([r["lowfreq_latent_mse_to_teacher_next"] for r in rows]).mean().item()),
                "highfreq_latent_mse": float(torch.tensor([r["highfreq_latent_mse_to_teacher_next"] for r in rows]).mean().item()),
            }
        )
    worst = sorted(results, key=lambda row: float(row["endpoint_lowfreq_mse"]), reverse=True)
    early_divergence_rate = sum(1 for idx in first if idx <= 2) / max(len(results), 1)
    low_to_high = float(endpoint_low.mean().item()) / max(float(torch.tensor([r["endpoint_highfreq_mse"] for r in results]).mean().item()), 1e-8)
    if early_divergence_rate >= 0.5:
        focus = "first_steps"
    elif per_step and max(per_step, key=lambda row: row["lowfreq_latent_mse"])["index"] >= max(1, len(per_step) // 2):
        focus = "mid_late_rollout"
    else:
        focus = "full_rollout"
    return {
        "endpoint_mse_mean": float(endpoint.mean().item()),
        "endpoint_mse_max": float(endpoint.max().item()),
        "endpoint_lowfreq_mse_mean": float(endpoint_low.mean().item()),
        "endpoint_lowfreq_mse_max": float(endpoint_low.max().item()),
        "first_divergence_index_mean": float(torch.tensor(first, dtype=torch.float32).mean().item()) if first else None,
        "early_divergence_rate": early_divergence_rate,
        "lowfreq_to_highfreq_endpoint_ratio": low_to_high,
        "controller_recommendation": {
            "focus": focus,
            "front_start_prob": 1.0 if focus == "first_steps" else 0.75,
            "rollout_len": len(per_step),
            "increase_lowfreq_endpoint": bool(low_to_high >= 1.0),
            "increase_unpacked_artifact_loss": bool(low_to_high < 1.0),
            "overweight_sequence_key_file": "worst_sequence_keys.txt",
            "overweight_prompt_file": "worst_prompts.txt",
        },
        "per_step_mean": per_step,
        "worst_sequences": [
            {
                "sequence_key": row["sequence_key"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "endpoint_mse": row["endpoint_mse"],
                "endpoint_lowfreq_mse": row["endpoint_lowfreq_mse"],
                "first_divergence_index": row["first_divergence_index"],
            }
            for row in worst[: min(16, len(worst))]
        ],
    }


def build_renormalization_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    per_step = list(summary.get("per_step_mean") or [])
    worst = list(summary.get("worst_sequences") or [])
    highfreq_mean = 0.0
    lowfreq_mean = 0.0
    flow_scale_by_step = []
    for row in per_step:
        index = int(row["index"])
        scale = float(row.get("renorm_flow_scale_to_teacher", 1.0))
        flow_scale_by_step.append({"step": index, "scale": scale})
        highfreq_mean += float(row.get("highfreq_latent_mse", 0.0))
        lowfreq_mean += float(row.get("lowfreq_latent_mse", 0.0))
    denom = max(len(per_step), 1)
    highfreq_mean /= denom
    lowfreq_mean /= denom
    controller = dict(summary.get("controller_recommendation") or {})
    validity = {
        "cached_or_target_trajectory": {
            "endpoint_mse_mean": summary.get("endpoint_mse_mean"),
            "endpoint_lowfreq_mse_mean": summary.get("endpoint_lowfreq_mse_mean"),
            "first_divergence_index_mean": summary.get("first_divergence_index_mean"),
        },
        "fresh_seed": "run sampler/probe with fresh latents and compare against cached target; not inferred from this trace alone",
        "off_policy": "requires student-rollout corruption targets; not inferred from this trace alone",
    }
    if highfreq_mean > lowfreq_mean:
        controller["renormalization_focus"] = "packed_highfreq_or_parity_map"
    else:
        controller["renormalization_focus"] = "lowfreq_latent_map"
    return {
        "artifact_kind": "agentkernel_lite_flux_student_renormalization_report_v0",
        "formal_condition": "R_z(teacher_step(z_t)) ~= student_step(R_z(z_t))",
        "distribution": {
            "sources": sorted({str(row.get("source", "")) for row in results}),
            "sequences": len(results),
            "seeds": sorted({int(row.get("seed", 0) or 0) for row in results}),
        },
        "maps": {
            "latent_map": {"type": "identity", "reason": "teacher and student both operate in FLUX packed latent coordinates"},
            "prompt_map": {"type": "cached_flux_prompt_embedding", "status": "student is sensitive to cached-vs-live trajectory distribution"},
            "timestep_map": {"type": "raw_flux_timestep", "scale": 1.0},
            "output_flow_map": {
                "type": "per_timestep_scalar_norm_renormalization",
                "scale_to_teacher_by_step": flow_scale_by_step,
            },
            "packed_latent_channel_map": {
                "type": "pending_low_rank_or_parity_map",
                "trigger": "highfreq/parity error exceeds lowfreq error" if highfreq_mean > lowfreq_mean else "lowfreq endpoint still dominates",
            },
        },
        "commutation_error": {
            "endpoint_mse_mean": summary.get("endpoint_mse_mean"),
            "endpoint_lowfreq_mse_mean": summary.get("endpoint_lowfreq_mse_mean"),
            "endpoint_highfreq_mse_mean": highfreq_mean,
            "first_divergence_index_mean": summary.get("first_divergence_index_mean"),
            "worst_sequences": worst,
        },
        "controller": controller,
        "validity_report": validity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace student rollout divergence on cached FLUX target trajectories.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--rollout-len", type=int, default=23)
    parser.add_argument("--lowfreq-pool", type=int, default=4)
    parser.add_argument("--latent-divergence-threshold", type=float, default=0.15)
    parser.add_argument("--lowfreq-divergence-threshold", type=float, default=0.08)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    student, checkpoint_info = load_student(Path(args.checkpoint), device)
    sequences = build_sequences([Path(value) for value in args.target_dir])
    sequences = sequences[: max(int(args.max_sequences), 1)]
    results = [
        trace_sequence(
            student,
            sequence,
            device=device,
            rollout_len=args.rollout_len,
            lowfreq_pool=args.lowfreq_pool,
            latent_divergence_threshold=args.latent_divergence_threshold,
            lowfreq_divergence_threshold=args.lowfreq_divergence_threshold,
        )
        for sequence in sequences
    ]
    summary = summarize(results)
    renormalization_report = build_renormalization_report(summary, results)
    payload = {"checkpoint": checkpoint_info, "targets": [str(value) for value in args.target_dir], "summary": summary, "sequences": results}
    (output_dir / "trace.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "renormalization_report.json").write_text(json.dumps(renormalization_report, indent=2), encoding="utf-8")
    worst = summary.get("worst_sequences", [])
    (output_dir / "worst_sequence_keys.txt").write_text("\n".join(str(row["sequence_key"]) for row in worst) + "\n", encoding="utf-8")
    (output_dir / "worst_prompts.txt").write_text("\n".join(str(row["prompt"]) for row in worst) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
