#!/usr/bin/env bash
set -euo pipefail

cd /data/agent_kernel_lite

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/data/repositories/diffusers/src \
HF_HOME=/data/.cache/huggingface \
TRANSFORMERS_CACHE=/data/.cache/huggingface \
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_image_sana_latent_distill.py \
  --output-dir checkpoints/agentkernel_lite_image_sana_sprint_0_6b_512_v045_flickr30k_realphoto_texture_polish \
  --resume checkpoints/agentkernel_lite_image_sana_sprint_0_6b_512_v043_flickr30k_broadreplay_detail128/sana_latent_student.pt \
  --prompt-dataset nlphuji/flickr30k \
  --prompt-config TEST \
  --prompt-split test \
  --prompt-columns caption \
  --stream-image-column image \
  --replay-prompt-file data/vision/prompts/sana_teacher_good_guided_v029_shape_detail_only.jsonl \
  --replay-probability 0.18 \
  --min-prompt-words 4 \
  --max-prompt-chars 220 \
  --max-nsfw-score 0.15 \
  --teacher-model Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers \
  --teacher-device cuda:0 \
  --student-device cuda:0 \
  --teacher-dtype bfloat16 \
  --resolution 512 \
  --max-sequence-length 300 \
  --steps 300 \
  --batch-size 1 \
  --teacher-steps 4 \
  --teacher-guidance 1.0 \
  --sample-steps 12 \
  --sample-guidance 2.0 \
  --trajectory-steps 4 \
  --trajectory-rollout-loss-weight 0.065 \
  --decoded-rollout-loss-weight 0.050 \
  --decoded-rollout-lowfreq-loss-weight 0.085 \
  --decoded-rollout-moment-loss-weight 0.025 \
  --decoded-rollout-highfreq-loss-weight 0.030 \
  --decoded-rollout-gradient-loss-weight 0.020 \
  --decoded-rollout-size 128 \
  --decoded-rollout-highfreq-kernel-size 9 \
  --use-teacher-ref-decoded-targets \
  --direction-loss-weight 0.16 \
  --norm-loss-weight 0.06 \
  --normalized-target-loss-weight 0.09 \
  --lowfreq-target-loss-weight 0.08 \
  --lowfreq-target-kernel-size 7 \
  --lr 1.2e-5 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-every 10 \
  --sample-every 150 \
  --checkpoint-every 150 \
  --student-architecture sana_transformer \
  --sana-num-layers 16 \
  --sana-num-attention-heads 36 \
  --sana-attention-head-dim 32 \
  --sana-num-cross-attention-heads 16 \
  --sana-cross-attention-head-dim 72 \
  --sana-mlp-ratio 2.5 \
  --sana-qk-norm rms_norm_across_heads \
  --local-files-only
