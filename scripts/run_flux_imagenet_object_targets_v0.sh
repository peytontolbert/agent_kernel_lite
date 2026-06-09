#!/usr/bin/env bash
set -euo pipefail

cd /data/agent_kernel_lite

PROMPTS="${PROMPTS:-data/vision/prompts/imagenet_object_photo_12k_v0.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/vision/flux1_dev_flow_targets_imagenet_object_photo_512p_seed70700000_stride1_v0}"
LOG_PATH="${LOG_PATH:-logs/flux_imagenet_object_targets_v0.log}"
LIMIT="${LIMIT:-4096}"
PROMPT_SKIP="${PROMPT_SKIP:-0}"
SEEDS_PER_PROMPT="${SEEDS_PER_PROMPT:-2}"
STEPS="${STEPS:-24}"
DEVICE="${DEVICE:-cuda:1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}"
SEED="${SEED:-70700000}"

if [[ ! -f "${PROMPTS}" ]]; then
  /home/peyton/miniconda3/envs/ai/bin/python scripts/build_agentkernel_lite_imagenet_object_prompts.py \
    --output "${PROMPTS}" \
    --templates-per-class 12 \
    --include-scenes
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
  DEVICE="cuda:0"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peyton/miniconda3/envs/ai/bin/python scripts/generate_agentkernel_lite_flux_flow_targets.py \
  --prompts "${PROMPTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --teacher-model black-forest-labs/FLUX.1-dev \
  --device "${DEVICE}" \
  --dtype bfloat16 \
  --width 512 \
  --height 512 \
  --steps "${STEPS}" \
  --target-stride 1 \
  --guidance 3.5 \
  --max-sequence-length 512 \
  --seed "${SEED}" \
  --prompt-skip "${PROMPT_SKIP}" \
  --seeds-per-prompt "${SEEDS_PER_PROMPT}" \
  --limit "${LIMIT}" \
  --resume \
  --quantize-transformer-4bit \
  2>&1 | tee "${LOG_PATH}"
