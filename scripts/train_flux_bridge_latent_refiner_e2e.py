#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from generate_agentkernel_lite_image_teacher_corpus import load_teacher, read_prompts
from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
from sample_agentkernel_lite_image_flux_flow_distill import (
    load_cached_prompt_embeddings,
    load_final_latent_refiner,
    load_student,
    prompt_seed_cache_key,
)
from train_flux_bridge_latent_refiner import parse_indices, rows_by_prompt_seed_timestep
from train_flux_final_latent_refiner import (
    decode_flux_latents_tensor,
    flux_unpacked_block_loss,
    flux_unpacked_spatial_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end train a bridge refiner through a frozen continuation student.")
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--continuation-checkpoint", required=True)
    parser.add_argument("--resume-refiner", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--seed-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--stage-indices", default="5,10,15")
    parser.add_argument("--bridge-output-mode", choices=("delta", "absolute"), default="absolute")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--direct-latent-loss-weight", type=float, default=8.0)
    parser.add_argument("--direct-spatial-loss-weight", type=float, default=0.25)
    parser.add_argument("--direct-block-loss-weight", type=float, default=0.25)
    parser.add_argument("--endpoint-loss-weight", type=float, default=12.0)
    parser.add_argument("--endpoint-lowfreq-loss-weight", type=float, default=20.0)
    parser.add_argument("--decoded-loss-weight", type=float, default=2.0)
    parser.add_argument("--decoded-lowfreq-loss-weight", type=float, default=8.0)
    parser.add_argument("--decoded-loss-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = Path(args.target_dir)
    stage_indices = parse_indices(args.stage_indices)
    target_index = stage_indices[-1]
    prompts = read_prompts(Path(args.prompts), int(args.limit))
    seeds = [int(item) for item in Path(args.seed_csv).read_text(encoding="utf-8").strip().split(",") if item.strip()]
    if len(seeds) < len(prompts):
        raise ValueError("seed csv has fewer seeds than prompts")

    teacher_args = argparse.Namespace(
        teacher_family="flux",
        teacher_model=args.teacher_model,
        dtype=args.dtype,
        variant="",
        local_files_only=args.local_files_only,
        quantize_transformer_4bit=False,
        bnb_4bit_quant_type="nf4",
        cpu_offload=args.cpu_offload,
        gpu_id=args.gpu_id,
        device=args.device,
    )
    pipe = load_teacher(teacher_args)
    device = torch.device(getattr(pipe, "_execution_device", args.device))
    bridge, _bridge_config = load_student(Path(args.bridge_checkpoint), device, "raw")
    bridge.requires_grad_(False)
    bridge.eval()
    continuation, _continuation_config = load_student(Path(args.continuation_checkpoint), device, "raw")
    continuation.requires_grad_(False)
    continuation.eval()
    refiner = load_final_latent_refiner(args.resume_refiner, device)
    if refiner is None:
        raise ValueError("failed to load resume refiner")
    refiner.train()

    embedding_cache = load_cached_prompt_embeddings(target_dir)
    target_rows = rows_by_prompt_seed_timestep(target_dir)
    examples = []
    with torch.no_grad():
        for prompt, seed in zip(prompts, seeds):
            cache_key = prompt_seed_cache_key(prompt, seed)
            embeds = embedding_cache[cache_key]
            prompt_embeds = embeds["prompt_embeds"].to(device=device, dtype=pipe.dtype)
            pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(device=device, dtype=pipe.dtype)
            latents = embeds["initial_latents"].to(device=device, dtype=pipe.dtype)
            timesteps, _ = flux_timesteps(pipe, latents, int(args.steps), device)
            guidance = torch.full([1], float(args.guidance), device=device, dtype=torch.float32)
            start_index = 0
            for stage_index in stage_indices:
                output = bridge(
                    latents.float(),
                    timesteps[start_index].expand(latents.shape[0]).float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                ).to(latents.dtype)
                latents = output if args.bridge_output_mode == "absolute" else latents + output
                start_index = stage_index

            t15_payload = torch.load(target_dir / str(target_rows[(prompt, seed, target_index)]["target_path"]), map_location="cpu")
            final_row = target_rows[(prompt, seed, int(args.steps) - 1)]
            final_payload = torch.load(target_dir / str(final_row["target_path"]), map_location="cpu")
            final_latents = final_payload["latents"].to(device=device, dtype=torch.float32)
            final_timestep = final_payload["timestep"].to(device=device, dtype=torch.float32).reshape(-1, 1, 1)
            clean_target = final_latents - (final_timestep / 1000.0) * final_payload["teacher_target"].to(device=device, dtype=torch.float32)
            target_pixels = decode_flux_latents_tensor(
                pipe,
                clean_target.to(device=device, dtype=next(pipe.vae.parameters()).dtype),
                int(args.height),
                int(args.width),
            ).float()
            examples.append(
                {
                    "prompt": prompt,
                    "source": latents.float().detach().clone(),
                    "target_t15": t15_payload["latents"].to(device=device, dtype=torch.float32).detach().clone(),
                    "clean_target": clean_target.detach().clone(),
                    "target_pixels": target_pixels.detach().clone(),
                    "prompt_embeds": prompt_embeds.detach().clone(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.detach().clone(),
                    "timesteps": timesteps.detach().clone(),
                }
            )
    print(json.dumps({"examples": len(examples), "target_index": target_index}), flush=True)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=float(args.lr), weight_decay=0.01)
    for step in range(1, int(args.train_steps) + 1):
        example = examples[(step - 1) % len(examples)]
        source = example["source"].to(device)
        target_t15 = example["target_t15"].to(device)
        clean_target = example["clean_target"].to(device)
        prompt_embeds = example["prompt_embeds"].to(device)
        pooled_prompt_embeds = example["pooled_prompt_embeds"].to(device)
        timesteps = example["timesteps"].to(device)
        guidance = torch.full([source.shape[0]], float(args.guidance), device=device, dtype=torch.float32)

        refined = refiner(source)
        direct_latent_loss = F.mse_loss(refined.float(), target_t15.float())
        direct_spatial_loss = flux_unpacked_spatial_loss(refined, target_t15)
        direct_block_loss = flux_unpacked_block_loss(refined, target_t15)

        rollout = refined.to(dtype=prompt_embeds.dtype)
        if hasattr(pipe.scheduler, "_step_index"):
            pipe.scheduler._step_index = None
        for timestep_value in timesteps[target_index:]:
            timestep = timestep_value.expand(rollout.shape[0]).to(device)
            pred = continuation(
                rollout.float(),
                timestep.float(),
                prompt_embeds.float(),
                pooled_prompt_embeds.float(),
                guidance,
            ).to(rollout.dtype)
            rollout = pipe.scheduler.step(pred, timestep_value, rollout, return_dict=False)[0]

        endpoint_loss = F.mse_loss(rollout.float(), clean_target.float())
        endpoint_lowfreq_loss = F.mse_loss(
            torch.nn.functional.avg_pool1d(rollout.float().transpose(1, 2), kernel_size=8, stride=8),
            torch.nn.functional.avg_pool1d(clean_target.float().transpose(1, 2), kernel_size=8, stride=8),
        )
        decoded_loss = source.new_zeros(())
        decoded_lowfreq_loss = source.new_zeros(())
        if float(args.decoded_loss_weight) > 0 or float(args.decoded_lowfreq_loss_weight) > 0:
            decoded = decode_flux_latents_tensor(
                pipe,
                rollout.to(dtype=next(pipe.vae.parameters()).dtype),
                int(args.height),
                int(args.width),
            ).float()
            target_pixels = example["target_pixels"].to(device).float()
            if int(args.decoded_loss_size) > 0 and int(args.decoded_loss_size) != int(args.height):
                size = (int(args.decoded_loss_size), int(args.decoded_loss_size))
                decoded = F.interpolate(decoded, size=size, mode="bilinear", align_corners=False)
                target_pixels = F.interpolate(target_pixels, size=size, mode="bilinear", align_corners=False)
            decoded_loss = F.l1_loss(decoded, target_pixels)
            decoded_lowfreq_loss = F.l1_loss(
                F.interpolate(decoded, size=(32, 32), mode="area"),
                F.interpolate(target_pixels, size=(32, 32), mode="area"),
            )
        loss = (
            float(args.direct_latent_loss_weight) * direct_latent_loss
            + float(args.direct_spatial_loss_weight) * direct_spatial_loss
            + float(args.direct_block_loss_weight) * direct_block_loss
            + float(args.endpoint_loss_weight) * endpoint_loss
            + float(args.endpoint_lowfreq_loss_weight) * endpoint_lowfreq_loss
            + float(args.decoded_loss_weight) * decoded_loss
            + float(args.decoded_lowfreq_loss_weight) * decoded_lowfreq_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "prompt": str(example["prompt"]),
                        "loss": float(loss.detach().item()),
                        "direct_latent_loss": float(direct_latent_loss.detach().item()),
                        "endpoint_loss": float(endpoint_loss.detach().item()),
                        "endpoint_lowfreq_loss": float(endpoint_lowfreq_loss.detach().item()),
                        "decoded_loss": float(decoded_loss.detach().item()),
                        "decoded_lowfreq_loss": float(decoded_lowfreq_loss.detach().item()),
                    }
                ),
                flush=True,
            )
    payload = torch.load(args.resume_refiner, map_location="cpu")
    payload["model"] = refiner.state_dict()
    payload["e2e_continuation_checkpoint"] = str(args.continuation_checkpoint)
    payload["e2e_resume_refiner"] = str(args.resume_refiner)
    torch.save(payload, output_dir / "flux_bridge_latent_refiner.pt")
    print(json.dumps({"refiner": str(output_dir / "flux_bridge_latent_refiner.pt")}), flush=True)


if __name__ == "__main__":
    main()
