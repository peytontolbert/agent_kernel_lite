# Agent Kernel Lite Image Generation Plan

The 32x32 BitDiT checkpoint is not a viable product model. It only proves that a custom browser/WASM image runtime can execute ternary transformer weights. The next model should be trained in latent space and only quantized after dense samples are recognizable.

## Architecture

- Use Sana-Sprint 0.6B as the offline teacher.
- Keep the student in Sana's native latent space: 32 latent channels, 32x compression, 16x16 latent grid at 512px.
- Use the teacher Gemma prompt embeddings during dense distillation. Hashed prompt tokens are only acceptable for later browser text-conditioner distillation, not for the dense quality phase.
- First train a dense student until sampled images are coherent.
- Only after dense samples pass, train a small browser text conditioner, then run ternary/BitNet QAT.
- Export the student flow transformer plus a browser-compatible decoder bundle. Do not switch the app default away from the placeholder until this quality gate passes.

## Why This Direction

Recent diffusion work points away from tiny RGB-pixel DDPMs:

- SANA uses efficient latent diffusion transformers and aggressive compression for high-resolution synthesis.
- Stable Diffusion 3 shows rectified-flow transformers are a stronger high-resolution text-to-image formulation than older DDPM-style training.
- SD3.5-Flash and LCM-style work show the mobile path is few-step distillation, not many-step denoising.
- Q-DiT shows DiT quantization is sensitive; quantization should be staged carefully after the dense model works.

## New Scripts

Build prompts from a Hugging Face dataset:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/build_agentkernel_lite_image_prompt_corpus.py \
  --preset journeydb \
  --streaming \
  --limit 100000 \
  --output data/vision/prompts/journeydb_prompts_100k.jsonl
```

Train directly from streamed prompts and Sana final latents:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_sana_latent_distill.py \
  --output-dir checkpoints/agentkernel_lite_image_sana_final_latent_flow_v0 \
  --prompt-dataset poloclub/diffusiondb \
  --prompt-config 2m_text_only \
  --prompt-trust-remote-code \
  --teacher-model Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers \
  --resolution 512 \
  --final-latent-flow \
  --teacher-steps 4 \
  --teacher-guidance 1.0 \
  --dim 512 \
  --depth 10 \
  --heads 8 \
  --steps 100000
```

Teacher trajectory distillation is also supported:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_sana_latent_distill.py \
  --output-dir checkpoints/agentkernel_lite_image_sana_latent_distill_v0 \
  --prompt-dataset poloclub/diffusiondb \
  --prompt-config 2m_text_only \
  --prompt-trust-remote-code \
  --resolution 512 \
  --trajectory-steps 8 \
  --dim 512 \
  --depth 10 \
  --heads 8 \
  --steps 100000
```

Overfit sanity check:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_sana_latent_distill.py \
  --output-dir checkpoints/agentkernel_lite_image_sana_final_latent_flow_overfit \
  --prompt-file data/vision/prompts/sana_overfit_prompt.txt \
  --final-latent-flow \
  --fixed-teacher-seed 1234 \
  --fixed-training-noise \
  --teacher-steps 4 \
  --steps 500
```

The overfit check currently passes: the student reproduces the teacher latent image for the fixed prompt/seed. The general corpus model still produces abstract texture at the latest checked samples, so it is not ready for QAT or browser export.

## Current Training Result

- `checkpoints/agentkernel_lite_image_sana_final_latent_flow_overfit/` confirms the Sana latent flow path can reproduce a fixed teacher latent.
- `data/vision/sana_latent_cache_v0/` contains 500 compact teacher latents, about 10MB total.
- `checkpoints/agentkernel_lite_image_sana_final_latent_flow_cache_replay_v0/` trained a 512-dim, 10-layer dense student for 3000 replay steps with 8 flow updates per latent.
- Latest checked sample: `sana_latent_student_step_003000.png`.
- Result: not browser-ready. It has color/texture and weak global layout, but no reliable prompt-following objects.
- `data/vision/sana_latent_cache_2k_v0/` contains 2500 compact teacher latents, about 50MB total.
- `checkpoints/agentkernel_lite_image_sana_final_latent_flow_160m_cache2k_v0/` trained a larger 768-dim, 12-layer dense student. It still produced texture-only samples by the checked checkpoints.
- `checkpoints/agentkernel_lite_image_sana_direct_latent_160m_cache2k_v0/` tested deterministic direct latent prediction. One-prompt overfit works, but the general cache run collapses to a smooth latent mean by step 2000.
- `checkpoints/agentkernel_lite_image_sana_teacher_noise_160m_cache2k_v0/` added teacher-noise distillation on noised cached latents. It overfits one prompt and trains with healthier loss than the earlier flow objective, but cache samples remain abstract by the checked checkpoints.
- `checkpoints/agentkernel_lite_image_sana_transformer_teacher_noise_cache2k_v0/` switched the student from the custom transformer to a small `SanaTransformer2DModel`-style student, about 20.7M dense params. This produced the clearest broad composition so far but still no reliable prompt-following objects by step 4500.
- `checkpoints/agentkernel_lite_image_sana_transformer85m_teacher_noise_cache2k_v0/` scaled the Sana-style student to about 85M dense params. It did not materially beat the smaller Sana-style student on the 2.5k cache by step 2500.

Do not run BitNet QAT or browser export from these checkpoints. The current student class can memorize a fixed teacher latent, but it does not yet learn a useful conditional image distribution from the cached corpus. The next quality step is a stronger student formulation, not quantization.

Estimate older custom-autoencoder student size:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/estimate_agentkernel_lite_image_student_params.py \
  --image-size 256 \
  --downsample 16 \
  --latent-channels 16 \
  --dim 640 \
  --depth 12 \
  --heads 8
```

Export a browser bundle:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/export_agentkernel_lite_image_latent_flow_browser.py \
  --checkpoint checkpoints/agentkernel_lite_image_latent_flow_256_sana_sprint_v0/latent_flow.pt \
  --output-dir web/models/agentkernel_lite_image_latent_flow_256_sana_sprint_v0 \
  --model-id agentkernel_lite_image_latent_flow_256_sana_sprint_v0 \
  --ternary
```

## Dataset Requirement

This repository currently does not include generated teacher images. Do not train this on CIFAR again except for smoke tests. The minimum useful target is a generated or curated 256x256 image student corpus. The intended first real teacher is `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers`; generate teacher samples into `data/vision/teacher_sana_sprint_0_6b_1024px_v0`, then train the custom latent model from that `metadata.jsonl`.
