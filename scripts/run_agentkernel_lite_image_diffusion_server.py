#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import os
from pathlib import Path
import sys
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision.transforms.functional import to_pil_image


def _install_local_diffusers() -> None:
    candidates = [
        os.environ.get("DIFFUSERS_SRC", ""),
        "/data/webgl-game/repos/diffusers/src",
        "/data/repositories/diffusers/src",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


_install_local_diffusers()

try:
    from diffusers import AutoPipelineForText2Image
except Exception as exc:  # pragma: no cover - environment dependent.
    AutoPipelineForText2Image = None
    DIFFUSERS_IMPORT_ERROR = exc
else:
    DIFFUSERS_IMPORT_ERROR = None


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance: float = 0.0
    seed: int | None = None


class DiffusionRuntime:
    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self.pipe = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.pipe is not None:
            return
        if AutoPipelineForText2Image is None:
            raise RuntimeError(f"diffusers import failed: {DIFFUSERS_IMPORT_ERROR}")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
        pipe = pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        self.load()
        assert self.pipe is not None
        width = min(max(int(request.width), 256), 1024)
        height = min(max(int(request.height), 256), 1024)
        steps = min(max(int(request.steps), 1), 50)
        seed = int(request.seed if request.seed is not None else torch.seed() % (2**31 - 1))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": steps,
            "guidance_scale": float(request.guidance),
            "width": width,
            "height": height,
            "generator": generator,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        with self.lock:
            image = self.pipe(**kwargs).images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "model": self.model_id,
            "backend": "diffusers",
            "width": image.width,
            "height": image.height,
            "seed": seed,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }


class SanaStudentRuntime:
    def __init__(self, checkpoint_path: str, device: str, teacher_model: str, dtype_name: str = "bfloat16") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.teacher_model = teacher_model
        self.dtype_name = dtype_name
        self.teacher = None
        self.student = None
        self.args = None
        self.step = 0
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.teacher is not None and self.student is not None:
            return
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from train_agentkernel_lite_image_sana_latent_distill import (
            SanaLatentStudentConfig,
            encode_prompts,
            load_teacher,
            make_student,
            student_predict_cfg,
        )

        self.encode_prompts = encode_prompts
        self.student_predict_cfg = student_predict_cfg
        args = argparse.Namespace(
            teacher_model=self.teacher_model,
            teacher_device=self.device,
            teacher_dtype=self.dtype_name,
            local_files_only=True,
            max_sequence_length=300,
            sample_guidance=4.5,
            student_device=self.device,
            student_architecture="sana_transformer",
            patch_size=1,
            sana_num_layers=28,
            sana_num_attention_heads=36,
            sana_attention_head_dim=32,
            sana_num_cross_attention_heads=16,
            sana_cross_attention_head_dim=72,
            sana_mlp_ratio=2.5,
            sana_qk_norm="rms_norm_across_heads",
            disable_resolution_binning=False,
        )
        teacher = load_teacher(args)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        config = SanaLatentStudentConfig(**{**checkpoint.get("config", {}), "patch_size": args.patch_size})
        student = make_student(config, args)
        state = checkpoint.get("student_materialized") or checkpoint["student"]
        student.load_state_dict(state, strict=True)
        self.step = int(checkpoint.get("step") or 0)
        del checkpoint, state
        dtype = torch.float16 if self.dtype_name == "float16" else torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float32
        student = student.to(device=self.device, dtype=dtype).eval()
        self.teacher = teacher
        self.student = student
        self.args = args

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        self.load()
        assert self.teacher is not None and self.student is not None and self.args is not None
        prompt = request.prompt.strip()
        steps = min(max(int(request.steps), 4), 80)
        seed = int(request.seed if request.seed is not None else torch.seed() % (2**31 - 1))
        device = torch.device(self.device)
        with self.lock:
            prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.encode_prompts(self.teacher, [prompt], self.args)
            prompt_embeds = prompt_embeds.to(device)
            prompt_mask = prompt_mask.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_prompt_mask = negative_prompt_mask.to(device)
            generator = torch.Generator(device=device).manual_seed(seed)
            latent_channels = self.student.config.in_channels
            latent_size = self.student.config.sample_size
            latents = torch.randn(1, latent_channels, latent_size, latent_size, generator=generator, device=device)
            scheduler = self.teacher.scheduler
            scheduler.set_timesteps(steps, device=device)
            for timestep_value in scheduler.timesteps:
                timestep = timestep_value.expand(latents.shape[0]).to(device)
                guidance = float(request.guidance) if float(request.guidance) > 0.0 else float(self.args.sample_guidance)
                pred = self.student_predict_cfg(
                    self.student,
                    latents,
                    timestep,
                    prompt_embeds,
                    prompt_mask,
                    negative_prompt_embeds,
                    negative_prompt_mask,
                    guidance,
                    self.args,
                )
                latents = scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
            image_latents = latents.to(self.teacher.vae.device, dtype=self.teacher.vae.dtype)
            decoded = self.teacher.vae.decode(image_latents / self.teacher.vae.config.scaling_factor, return_dict=False)[0]
            image_tensor = self.teacher.image_processor.postprocess(decoded, output_type="pt")[0].detach().cpu().float().clamp(0, 1)
        image = to_pil_image(image_tensor)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "model": "agentkernel_lite_image_sana_student_live_v0",
            "backend": "sana-student-local-dev",
            "checkpoint": self.checkpoint_path,
            "training_step": self.step,
            "width": image.width,
            "height": image.height,
            "seed": seed,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }


class FluxStudentRuntime:
    def __init__(self, checkpoint_path: str, device: str, teacher_model: str, dtype_name: str = "bfloat16") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.teacher_model = teacher_model
        self.dtype_name = dtype_name
        self.teacher = None
        self.student = None
        self.step = 0
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.teacher is not None and self.student is not None:
            return
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from generate_agentkernel_lite_image_teacher_corpus import decode_flux_latents, load_teacher
        from generate_agentkernel_lite_flux_flow_targets import flux_timesteps
        from train_agentkernel_lite_image_flux_flow_distill import FluxPackedStudent, FluxPackedStudentConfig

        self.decode_flux_latents = decode_flux_latents
        self.flux_timesteps = flux_timesteps
        args = argparse.Namespace(
            teacher_family="flux",
            teacher_model=self.teacher_model,
            dtype=self.dtype_name,
            variant="",
            local_files_only=True,
            quantize_transformer_4bit=True,
            bnb_4bit_quant_type="nf4",
            cpu_offload=True,
            gpu_id=0,
            device=self.device,
        )
        teacher = load_teacher(args)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        config = FluxPackedStudentConfig(**checkpoint["config"])
        student = FluxPackedStudent(config)
        state = checkpoint.get("student_materialized") or checkpoint["student"]
        student.load_state_dict(state, strict=True)
        self.step = int(checkpoint.get("step") or 0)
        del checkpoint, state
        student = student.to(device=self.device, dtype=torch.float32).eval()
        self.teacher = teacher
        self.student = student

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        self.load()
        assert self.teacher is not None and self.student is not None
        prompt = request.prompt.strip()
        width = min(max(int(request.width), 512), 512)
        height = min(max(int(request.height), 512), 512)
        steps = min(max(int(request.steps), 8), 48)
        guidance_value = float(request.guidance) if float(request.guidance) > 0.0 else 3.5
        seed = int(request.seed if request.seed is not None else torch.seed() % (2**31 - 1))
        device = torch.device(self.device)
        with self.lock:
            prompt_embeds, pooled_prompt_embeds, _text_ids = self.teacher.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            generator = torch.Generator(device=device).manual_seed(seed)
            num_channels_latents = self.teacher.transformer.config.in_channels // 4
            latents, _latent_image_ids = self.teacher.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                None,
            )
            timesteps, _ = self.flux_timesteps(self.teacher, latents, steps, device)
            guidance = torch.full([latents.shape[0]], guidance_value, device=device, dtype=torch.float32)
            for timestep_value in timesteps:
                timestep = timestep_value.expand(latents.shape[0]).to(device)
                pred = self.student(
                    latents.float(),
                    timestep.float(),
                    prompt_embeds.float(),
                    pooled_prompt_embeds.float(),
                    guidance,
                ).to(latents.dtype)
                latents = self.teacher.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
            image = self.decode_flux_latents(self.teacher, latents, height, width)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "model": "agentkernel_lite_image_flux_student_dense_dev_v0",
            "backend": "flux-student-local-dev",
            "checkpoint": self.checkpoint_path,
            "training_step": self.step,
            "width": image.width,
            "height": image.height,
            "seed": seed,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }


def create_app(runtime: DiffusionRuntime) -> FastAPI:
    app = FastAPI(title="AgentKernel Lite Image Diffusion Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:4173",
            "http://localhost:4173",
            "http://127.0.0.1:8797",
            "http://localhost:8797",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model": getattr(runtime, "model_id", getattr(runtime, "checkpoint_path", "")),
            "device": runtime.device,
            "loaded": getattr(runtime, "pipe", None) is not None or getattr(runtime, "student", None) is not None,
            "diffusers": AutoPipelineForText2Image is not None,
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        try:
            return runtime.generate(request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentKernel Lite high-quality local diffusion backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument("--model", default=os.environ.get("AGENTKERNEL_IMAGE_MODEL", "stabilityai/sd-turbo"))
    parser.add_argument("--sana-student-checkpoint", default=os.environ.get("AGENTKERNEL_SANA_STUDENT_CHECKPOINT", ""))
    parser.add_argument("--flux-student-checkpoint", default=os.environ.get("AGENTKERNEL_FLUX_STUDENT_CHECKPOINT", ""))
    parser.add_argument("--teacher-model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers")
    parser.add_argument("--flux-teacher-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.flux_student_checkpoint:
        runtime = FluxStudentRuntime(args.flux_student_checkpoint, args.device, args.flux_teacher_model, args.dtype)
    elif args.sana_student_checkpoint:
        runtime = SanaStudentRuntime(args.sana_student_checkpoint, args.device, args.teacher_model, args.dtype)
    else:
        runtime = DiffusionRuntime(args.model, args.device)
    app = create_app(runtime)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
