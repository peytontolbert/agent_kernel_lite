#!/usr/bin/env bash
set -euo pipefail

cd /data/agent_kernel_lite

BASE_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_stride1_v0"
SEED2_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_seed30300000_stride1_v0"
SEED2_METADATA="${SEED2_TARGET}/metadata.jsonl"
RESUME_CKPT="checkpoints/agentkernel_lite_image_flux_flow_student_anchor_only_150m_dense_rollout_strong_v1/flux_packed_student.pt"
OUTPUT_DIR="checkpoints/agentkernel_lite_image_flux_flow_student_anchor_2seed_150m_dense_rollout_v0"
LOG_PATH="logs/flux_150m_anchor_2seed_dense_rollout_v0.log"
READY_ROWS=6144

while true; do
  rows=0
  if [[ -f "${SEED2_METADATA}" ]]; then
    rows=$(wc -l < "${SEED2_METADATA}")
  fi
  echo "{\"seed2_rows\": ${rows}, \"ready_rows\": ${READY_ROWS}}"
  if [[ "${rows}" -ge "${READY_ROWS}" ]]; then
    break
  fi
  sleep 300
done

tmux kill-session -t akl_flux_150m_rollout_strong_v1 2>/dev/null || true

tmux new-session -d -s akl_flux_150m_2seed_rollout_v0 "\
  cd /data/agent_kernel_lite && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_flux_flow_rollout_distill.py \
    --target-dir ${BASE_TARGET} \
    --extra-target-dir ${SEED2_TARGET} \
    --output-dir ${OUTPUT_DIR} \
    --resume ${RESUME_CKPT} \
    --steps 40000 \
    --grad-accum-steps 1 \
    --rollout-len 10 \
    --detach-rollout \
    --front-start-prob 1.0 \
    --dim 512 \
    --depth 28 \
    --heads 8 \
    --mlp-ratio 3 \
    --lr 5e-6 \
    --weight-decay 0.01 \
    --ema-decay 0.0 \
    --flow-loss-weight 1.0 \
    --latent-loss-weight 1000.0 \
    --log-every 10 \
    --checkpoint-every 250 \
    --device cuda:1 \
    2>&1 | tee ${LOG_PATH}"

echo "{\"started\": \"akl_flux_150m_2seed_rollout_v0\", \"output_dir\": \"${OUTPUT_DIR}\"}"
