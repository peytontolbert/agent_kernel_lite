#!/usr/bin/env bash
set -euo pipefail

cd /data/agent_kernel_lite

BASE_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_stride1_v0"
SEED2_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_seed30300000_stride1_v0"
SEED3_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_seed40400000_stride1_v0"
OBJECT_TARGET="${OBJECT_TARGET:-data/vision/flux1_dev_flow_targets_imagenet_object_photo_512p_seed70700000_stride1_v0}"
HF_OBJECT_TARGET="data/vision/flux1_dev_flow_targets_general_object_caption_clean_512p_seed60600000_stride1_v0"
RESUME_CKPT="${RESUME_CKPT:-checkpoints/agentkernel_lite_image_flux_flow_student_anchor_only_150m_dense_rollout_strong_v1/flux_packed_student.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/agentkernel_lite_image_flux_flow_student_150m_object_photo_rollout_v0}"
LOG_PATH="${LOG_PATH:-logs/flux_150m_object_photo_rollout_v0.log}"
DEVICE="${DEVICE:-cuda:1}"
STEPS="${STEPS:-120000}"

if [[ ! -f "${OBJECT_TARGET}/metadata.jsonl" ]]; then
  echo "missing object target metadata: ${OBJECT_TARGET}/metadata.jsonl" >&2
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_flux_flow_rollout_distill.py \
  --target-dir "${BASE_TARGET}" \
  --extra-target-dir "${SEED2_TARGET}" \
  --extra-target-dir "${SEED3_TARGET}" \
  --extra-target-dir "${HF_OBJECT_TARGET}" \
  --extra-target-dir "${OBJECT_TARGET}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume "${RESUME_CKPT}" \
  --steps "${STEPS}" \
  --grad-accum-steps 1 \
  --rollout-len 16 \
  --detach-rollout \
  --include-terminal-flow-loss \
  --front-start-prob 0.45 \
  --dim 512 \
  --depth 28 \
  --heads 8 \
  --mlp-ratio 3 \
  --dropout 0.05 \
  --pos2d-scale 0.05 \
  --timestep-scale 1.0 \
  --local-mixer-scale 0.10 \
  --latent-noise-std 0.035 \
  --latent-noise-timestep-scale \
  --prompt-dropout 0.05 \
  --loss-type huber \
  --huber-delta 0.1 \
  --snr-weighting \
  --snr-gamma 5.0 \
  --min-snr-weight 0.05 \
  --max-snr-weight 1.0 \
  --lr 2e-6 \
  --weight-decay 0.02 \
  --ema-decay 0.999 \
  --flow-loss-weight 1.0 \
  --latent-loss-weight 500.0 \
  --direction-loss-weight 0.1 \
  --norm-loss-weight 0.02 \
  --spatial-loss-weight 0.05 \
  --log-every 10 \
  --checkpoint-every 500 \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_PATH}"
