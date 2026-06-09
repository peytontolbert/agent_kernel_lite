#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

import torch


def read_rows(log_paths: list[Path], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in log_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = line.strip()
                if not text.startswith("{"):
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if row.get("mode") == mode and "step" in row and "loss" in row:
                    rows.append(row)
    rows.sort(key=lambda row: int(row.get("step", 0)))
    return rows


def read_rows_after_last_resume(log_paths: list[Path], mode: str) -> list[dict[str, Any]]:
    segment: list[dict[str, Any]] = []
    saw_resume = False
    for path in log_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                text = line.strip()
                if not text.startswith("{"):
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if "resume_optimizer_state" in row:
                    segment = []
                    saw_resume = True
                    continue
                if saw_resume and row.get("mode") == mode and "step" in row and "loss" in row:
                    segment.append(row)
    segment.sort(key=lambda row: int(row.get("step", 0)))
    return segment


def bucket_name(timestep_index: int) -> str:
    if timestep_index < 16:
        return "early"
    if timestep_index < 48:
        return "mid"
    return "late"


def fmean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_window(rows: list[dict[str, Any]], count: int) -> dict[str, Any]:
    window = rows[-count:] if count > 0 else rows
    losses: list[float] = []
    buckets: dict[str, list[float]] = defaultdict(list)
    components: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sources: dict[str, int] = defaultdict(int)
    prompts: set[str] = set()
    final_timestep_count = 0
    contrast_nonzero = 0
    multistep_nonzero = 0
    latent_adv_nonzero = 0
    latent_disc_nonzero = 0
    latent_adv_active_values: list[float] = []
    latent_disc_active_values: list[float] = []
    on_policy_nonzero = 0
    on_policy_active_values: list[float] = []
    decoded_nonzero = 0
    decoded_active_values: list[float] = []
    decoded_clip_nonzero = 0
    decoded_clip_active_values: list[float] = []
    decoded_dino_nonzero = 0
    decoded_dino_active_values: list[float] = []
    decoded_edge_nonzero = 0
    decoded_edge_active_values: list[float] = []
    decoded_adv_nonzero = 0
    decoded_adv_active_values: list[float] = []
    decoded_disc_nonzero = 0
    decoded_disc_active_values: list[float] = []
    for row in window:
        loss = float(row["loss"])
        losses.append(loss)
        index = int(row.get("start_timestep_index", 0))
        bucket = bucket_name(index)
        buckets[bucket].append(loss)
        if index >= 63:
            final_timestep_count += 1
        row_sources = row.get("source_name", "unknown")
        if isinstance(row_sources, list):
            for source in row_sources:
                sources[str(source)] += 1
        else:
            sources[str(row_sources)] += 1
        row_prompts = row.get("prompt", "")
        if isinstance(row_prompts, list):
            for prompt in row_prompts:
                prompts.add(str(prompt))
        else:
            prompts.add(str(row_prompts))
        if float(row.get("prompt_contrast_loss", 0.0) or 0.0) > 0:
            contrast_nonzero += 1
        if float(row.get("multistep_consistency_loss", 0.0) or 0.0) > 0:
            multistep_nonzero += 1
        latent_adv_loss = float(row.get("latent_adv_loss", 0.0) or 0.0)
        latent_disc_loss = float(row.get("latent_disc_loss", 0.0) or 0.0)
        on_policy_loss = float(row.get("on_policy_teacher_loss", 0.0) or 0.0)
        decoded_loss = float(row.get("decoded_image_loss", 0.0) or 0.0)
        decoded_clip_loss = float(row.get("decoded_clip_loss", 0.0) or 0.0)
        decoded_dino_loss = float(row.get("decoded_dino_loss", 0.0) or 0.0)
        decoded_edge_loss = float(row.get("decoded_edge_loss", 0.0) or 0.0)
        decoded_adv_loss = float(row.get("decoded_adv_loss", 0.0) or 0.0)
        decoded_disc_loss = float(row.get("decoded_disc_loss", 0.0) or 0.0)
        if latent_adv_loss > 0:
            latent_adv_nonzero += 1
            latent_adv_active_values.append(latent_adv_loss)
        if latent_disc_loss > 0:
            latent_disc_nonzero += 1
            latent_disc_active_values.append(latent_disc_loss)
        if on_policy_loss > 0:
            on_policy_nonzero += 1
            on_policy_active_values.append(on_policy_loss)
        if decoded_loss > 0:
            decoded_nonzero += 1
            decoded_active_values.append(decoded_loss)
        if decoded_clip_loss > 0:
            decoded_clip_nonzero += 1
            decoded_clip_active_values.append(decoded_clip_loss)
        if decoded_dino_loss > 0:
            decoded_dino_nonzero += 1
            decoded_dino_active_values.append(decoded_dino_loss)
        if decoded_edge_loss > 0:
            decoded_edge_nonzero += 1
            decoded_edge_active_values.append(decoded_edge_loss)
        if decoded_adv_loss > 0:
            decoded_adv_nonzero += 1
            decoded_adv_active_values.append(decoded_adv_loss)
        if decoded_disc_loss > 0:
            decoded_disc_nonzero += 1
            decoded_disc_active_values.append(decoded_disc_loss)
        for key in (
            "flow_loss",
            "latent_loss",
            "direction_loss",
            "norm_loss",
            "spatial_loss",
            "prompt_contrast_loss",
            "endpoint_loss",
            "multistep_consistency_loss",
            "latent_adv_loss",
            "latent_disc_loss",
            "on_policy_teacher_loss",
            "decoded_image_loss",
            "decoded_clip_loss",
            "decoded_dino_loss",
            "decoded_edge_loss",
            "decoded_adv_loss",
            "decoded_disc_loss",
        ):
            if key in row:
                components[bucket][key].append(float(row[key]))
    return {
        "count": len(window),
        "step_start": int(window[0]["step"]) if window else None,
        "step_end": int(window[-1]["step"]) if window else None,
        "mean_loss": fmean(losses),
        "bucket_loss": {key: fmean(value) for key, value in buckets.items()},
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "component_means": {
            bucket: {key: fmean(value) for key, value in values.items()}
            for bucket, values in components.items()
        },
        "source_counts": dict(sources),
        "unique_prompts": len(prompts),
        "final_timestep_count": final_timestep_count,
        "prompt_contrast_nonzero_count": contrast_nonzero,
        "multistep_consistency_nonzero_count": multistep_nonzero,
        "latent_adv_nonzero_count": latent_adv_nonzero,
        "latent_disc_nonzero_count": latent_disc_nonzero,
        "latent_adv_rate": latent_adv_nonzero / len(window) if window else None,
        "latent_disc_rate": latent_disc_nonzero / len(window) if window else None,
        "latent_adv_active_mean": fmean(latent_adv_active_values),
        "latent_disc_active_mean": fmean(latent_disc_active_values),
        "on_policy_teacher_nonzero_count": on_policy_nonzero,
        "on_policy_teacher_rate": on_policy_nonzero / len(window) if window else None,
        "on_policy_teacher_active_mean": fmean(on_policy_active_values),
        "decoded_image_nonzero_count": decoded_nonzero,
        "decoded_image_rate": decoded_nonzero / len(window) if window else None,
        "decoded_image_active_mean": fmean(decoded_active_values),
        "decoded_clip_nonzero_count": decoded_clip_nonzero,
        "decoded_clip_rate": decoded_clip_nonzero / len(window) if window else None,
        "decoded_clip_active_mean": fmean(decoded_clip_active_values),
        "decoded_dino_nonzero_count": decoded_dino_nonzero,
        "decoded_dino_rate": decoded_dino_nonzero / len(window) if window else None,
        "decoded_dino_active_mean": fmean(decoded_dino_active_values),
        "decoded_edge_nonzero_count": decoded_edge_nonzero,
        "decoded_edge_rate": decoded_edge_nonzero / len(window) if window else None,
        "decoded_edge_active_mean": fmean(decoded_edge_active_values),
        "decoded_adv_nonzero_count": decoded_adv_nonzero,
        "decoded_adv_rate": decoded_adv_nonzero / len(window) if window else None,
        "decoded_adv_active_mean": fmean(decoded_adv_active_values),
        "decoded_disc_nonzero_count": decoded_disc_nonzero,
        "decoded_disc_rate": decoded_disc_nonzero / len(window) if window else None,
        "decoded_disc_active_mean": fmean(decoded_disc_active_values),
    }


def trend(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "unknown"
    delta = current - previous
    if delta < -0.02:
        return "improving"
    if delta > 0.02:
        return "regressing"
    return "flat"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report FLUX image-student training progress.")
    parser.add_argument("--log", action="append", type=Path, default=[Path("logs/flux_live_teacher_300m_trajectory_reuse_v1.log")])
    parser.add_argument("--mode", default="live_teacher_flux_trajectory_reuse")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/agentkernel_lite_image_flux_prompt_image_300m_scratch_v1/flux_packed_student.pt"))
    parser.add_argument("--optimizer", type=Path, default=Path("checkpoints/agentkernel_lite_image_flux_prompt_image_300m_scratch_v1/flux_trajectory_optimizer.pt"))
    parser.add_argument("--short-window", type=int, default=500)
    parser.add_argument("--long-window", type=int, default=2000)
    parser.add_argument("--after-last-resume", action="store_true")
    args = parser.parse_args()

    rows = read_rows_after_last_resume(args.log, args.mode) if args.after_last_resume else read_rows(args.log, args.mode)
    short = summarize_window(rows, args.short_window)
    long = summarize_window(rows, args.long_window)
    checkpoint_info: dict[str, Any] = {"exists": args.checkpoint.exists()}
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_info.update({"step": checkpoint.get("step"), "loss": checkpoint.get("loss")})
    optimizer_info: dict[str, Any] = {"exists": args.optimizer.exists()}
    if args.optimizer.exists():
        optimizer = torch.load(args.optimizer, map_location="cpu")
        optimizer_info.update({"step": optimizer.get("step")})

    latest_step = int(rows[-1]["step"]) if rows else 0
    short_late = (short.get("bucket_loss") or {}).get("late")
    long_late = (long.get("bucket_loss") or {}).get("late")
    sample_recommendation = "wait"
    if checkpoint_info.get("step") and int(checkpoint_info["step"]) >= latest_step - 200:
        if trend(short_late, long_late) in {"improving", "flat"} and latest_step >= 10000:
            sample_recommendation = "sample_checkpoint"

    report = {
        "latest_step": latest_step,
        "rows": len(rows),
        "checkpoint": checkpoint_info,
        "optimizer": optimizer_info,
        "short_window": short,
        "long_window": long,
        "late_loss_trend": trend(short_late, long_late),
        "criteria": {
            "field_learning": "bucket losses falling, especially late timesteps",
            "scale_learning": "unique prompts and source mix keep increasing without anchor dominance",
            "coherence_gate": "fresh-noise samples show stable large shapes before detail quality is judged",
            "next_phase": "short rollout refinement after one-step late loss is consistently low",
        },
        "sample_recommendation": sample_recommendation,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
