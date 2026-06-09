#!/usr/bin/env bash
set -euo pipefail

cd /data/agent_kernel_lite

BASE_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_stride1_v0"
SEED2_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_seed30300000_stride1_v0"
SEED3_TARGET="data/vision/flux1_dev_flow_targets_coherence_anchor_256p_seed40400000_stride1_v0"
DIVERSE_TARGET="data/vision/flux1_dev_flow_targets_general_hq_mix_clean_512p_seed50500000_stride1_v0"
DIVERSE_METADATA="${DIVERSE_TARGET}/metadata.jsonl"
OBJECT_TARGET="data/vision/flux1_dev_flow_targets_general_object_caption_clean_512p_seed60600000_stride1_v0"
OBJECT_METADATA="${OBJECT_TARGET}/metadata.jsonl"
RESUME_CKPT="checkpoints/agentkernel_lite_image_flux_flow_student_anchor_3seed_150m_dense_rollout_enh_v2/flux_packed_student.pt"
OUTPUT_DIR="checkpoints/agentkernel_lite_image_flux_flow_student_anchor3_diverse512_object512_150m_dense_rollout_enh_v4"
LOG_PATH="logs/flux_150m_anchor3_diverse512_object512_dense_rollout_enh_v4.log"
READY_ROWS=6144

while true; do
  diverse_rows=0
  object_rows=0
  if [[ -f "${DIVERSE_METADATA}" ]]; then
    diverse_rows=$(wc -l < "${DIVERSE_METADATA}")
  fi
  if [[ -f "${OBJECT_METADATA}" ]]; then
    object_rows=$(wc -l < "${OBJECT_METADATA}")
  fi
  echo "{\"diverse_rows\": ${diverse_rows}, \"object_rows\": ${object_rows}, \"ready_rows\": ${READY_ROWS}}"
  if [[ "${diverse_rows}" -ge "${READY_ROWS}" && "${object_rows}" -ge "${READY_ROWS}" && -f "${RESUME_CKPT}" ]]; then
    break
  fi
  sleep 300
done

tmux kill-session -t akl_flux_150m_3seed_rollout_enh_v2 2>/dev/null || true
tmux kill-session -t akl_flux_diverse512_targets 2>/dev/null || true
tmux kill-session -t akl_flux_object512_targets 2>/dev/null || true

tmux new-session -d -s akl_flux_150m_anchor3_diverse512_object512_rollout_enh_v4 "\
  cd /data/agent_kernel_lite && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_flux_flow_rollout_distill.py \
    --target-dir ${BASE_TARGET} \
    --extra-target-dir ${SEED2_TARGET} \
    --extra-target-dir ${SEED3_TARGET} \
    --extra-target-dir ${DIVERSE_TARGET} \
    --extra-target-dir ${OBJECT_TARGET} \
    --output-dir ${OUTPUT_DIR} \
    --resume ${RESUME_CKPT} \
    --steps 70000 \
    --grad-accum-steps 1 \
    --rollout-len 12 \
    --detach-rollout \
    --front-start-prob 0.35 \
    --dim 512 \
    --depth 28 \
    --heads 8 \
    --mlp-ratio 3 \
    --dropout 0.05 \
    --pos2d-scale 0.05 \
    --timestep-scale 1.0 \
    --local-mixer-scale 0.10 \
    --latent-noise-std 0.03 \
    --latent-noise-timestep-scale \
    --prompt-dropout 0.02 \
    --lr 2e-6 \
    --weight-decay 0.02 \
    --ema-decay 0.0 \
    --flow-loss-weight 1.0 \
    --latent-loss-weight 500.0 \
    --direction-loss-weight 0.1 \
    --log-every 10 \
    --checkpoint-every 250 \
    --device cuda:1 \
    2>&1 | tee ${LOG_PATH}"

echo "{\"started\": \"akl_flux_150m_anchor3_diverse512_object512_rollout_enh_v4\", \"output_dir\": \"${OUTPUT_DIR}\"}"
