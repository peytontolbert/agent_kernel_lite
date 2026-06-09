#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from generate_agentkernel_lite_image_teacher_corpus import DEFAULT_FLUX_TEACHER, import_flux_pipeline
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from train_agentkernel_lite_image_flux_flow_rollout_distill import build_sequences


def load_flux_scheduler(args: argparse.Namespace) -> Any:
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    _bits_config, FluxPipeline, _transformer_cls = import_flux_pipeline()
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "transformer": None,
        "text_encoder": None,
        "text_encoder_2": None,
        "tokenizer": None,
        "tokenizer_2": None,
        "local_files_only": bool(args.local_files_only),
    }
    pipe = FluxPipeline.from_pretrained(args.teacher_model, **kwargs)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Append final post-scheduler FLUX latents to packed flow target sequences.")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_FLUX_TEACHER)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    metadata_path = target_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    pipe = load_flux_scheduler(args)
    device = torch.device(args.device)
    sequences = build_sequences([target_dir])
    existing_ids = {json.loads(line)["target_id"] for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    appended = 0
    with metadata_path.open("a", encoding="utf-8") as metadata:
        for sequence in sequences:
            last_row = sequence[-1]
            final_index = int(last_row.get("timestep_index", len(sequence) - 1)) + 1
            final_id = f'{str(last_row["target_id"]).rsplit("_t", 1)[0]}_t{final_index:03d}'
            if final_id in existing_ids:
                continue
            last_target_path = target_dir / last_row["target_path"]
            last_target = torch.load(last_target_path, map_location="cpu")
            latents = last_target["latents"].to(device)
            teacher_target = last_target["teacher_target"].to(device)
            timestep = last_target["timestep"].to(device)
            flux_timesteps(pipe, latents, int(last_row.get("steps", len(sequence))), device)
            next_latents = pipe.scheduler.step(teacher_target, timestep, latents, return_dict=False)[0]
            final_target_path = target_dir / "targets" / f"{final_id}.pt"
            torch.save(
                {
                    "latents": next_latents.detach().to("cpu", dtype=torch.float16),
                    "timestep": torch.zeros((), dtype=torch.float32),
                    "teacher_target": torch.zeros_like(last_target["teacher_target"], dtype=torch.float16),
                    "latent_image_ids": last_target["latent_image_ids"],
                    "guidance": float(last_target.get("guidance", last_row.get("guidance", 3.5))),
                    "prompt": str(last_row.get("prompt", last_target.get("prompt", ""))),
                    "embedding_path": last_row["embedding_path"],
                    "latent_format": "flux_packed_latents",
                    "final_post_scheduler": True,
                },
                final_target_path,
            )
            row = dict(last_row)
            row.update(
                {
                    "target_id": final_id,
                    "target_path": str(final_target_path.relative_to(target_dir)),
                    "timestep_index": final_index,
                    "final_post_scheduler": True,
                }
            )
            metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing_ids.add(final_id)
            appended += 1
            print(json.dumps({"appended": final_id, "prompt": row.get("prompt", ""), "timestep_index": final_index}), flush=True)
    print(json.dumps({"target_dir": str(target_dir), "appended": appended}), flush=True)


if __name__ == "__main__":
    main()
