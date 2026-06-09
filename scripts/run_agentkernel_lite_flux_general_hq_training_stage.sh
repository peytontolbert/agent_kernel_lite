#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
TARGET_DIR="${TARGET_DIR:-data/vision/flux1_dev_flow_targets_general_hq_mix_1024p_2seed_v0}"
DENSE_DIR="${DENSE_DIR:-checkpoints/agentkernel_lite_image_flux_flow_student_general_hq_28l_dense_v0}"
QAT_DIR="${QAT_DIR:-checkpoints/agentkernel_lite_image_flux_flow_student_general_hq_28l_bitnet_qat_v0}"
BROWSER_DIR="${BROWSER_DIR:-web/models/agentkernel_lite_image_flux_general_hq_28l_bitnet_v0}"
PROMPTS="${PROMPTS:-data/vision/prompts/flux_general_hq_holdout_eval_v0.txt}"
DEVICE="${DEVICE:-cuda:0}"
GPU_ID="${GPU_ID:-0}"
STAGE1_ROWS="${STAGE1_ROWS:-2048}"
FULL_ROWS="${FULL_ROWS:-12000}"

rows() {
  if [ -f "$TARGET_DIR/metadata.jsonl" ]; then
    wc -l < "$TARGET_DIR/metadata.jsonl"
  else
    echo 0
  fi
}

wait_for_rows() {
  local label="$1"
  local threshold="$2"
  while [ "$(rows)" -lt "$threshold" ]; do
    echo "$(date -Is) waiting_for_${label}_rows rows=$(rows) threshold=${threshold}"
    sleep 120
  done
}

sample_checkpoint() {
  local checkpoint="$1"
  local output_dir="$2"
  local steps="$3"
  "$PYTHON_BIN" scripts/sample_agentkernel_lite_image_flux_flow_distill.py \
    --checkpoint "$checkpoint" \
    --prompts "$PROMPTS" \
    --output-dir "$output_dir" \
    --limit 16 \
    --steps "$steps" \
    --guidance 3.5 \
    --width 512 \
    --height 512 \
    --local-files-only \
    --cpu-offload \
    --quantize-transformer-4bit \
    --device "$DEVICE" \
    --gpu-id "$GPU_ID"
}

wait_for_rows "stage1" "$STAGE1_ROWS"
"$PYTHON_BIN" scripts/train_agentkernel_lite_image_flux_flow_distill.py \
  --target-dir "$TARGET_DIR" \
  --output-dir "$DENSE_DIR" \
  --steps 10000 \
  --batch-size 1 \
  --dim 384 \
  --depth 28 \
  --heads 8 \
  --mlp-ratio 3 \
  --lr 6e-5 \
  --weight-decay 0.01 \
  --log-every 100 \
  --checkpoint-every 2000 \
  --device "$DEVICE"

sample_checkpoint "$DENSE_DIR/flux_packed_student.pt" "$DENSE_DIR/eval_holdout_stage1" 24

wait_for_rows "full_target" "$FULL_ROWS"
"$PYTHON_BIN" scripts/train_agentkernel_lite_image_flux_flow_distill.py \
  --target-dir "$TARGET_DIR" \
  --output-dir "$DENSE_DIR" \
  --resume "$DENSE_DIR/flux_packed_student.pt" \
  --steps 30000 \
  --batch-size 1 \
  --dim 384 \
  --depth 28 \
  --heads 8 \
  --mlp-ratio 3 \
  --lr 4e-5 \
  --weight-decay 0.01 \
  --log-every 100 \
  --checkpoint-every 5000 \
  --device "$DEVICE"

sample_checkpoint "$DENSE_DIR/flux_packed_student.pt" "$DENSE_DIR/eval_holdout_full_dense" 24

"$PYTHON_BIN" scripts/train_agentkernel_lite_image_flux_flow_distill.py \
  --target-dir "$TARGET_DIR" \
  --output-dir "$QAT_DIR" \
  --resume "$DENSE_DIR/flux_packed_student.pt" \
  --steps 10000 \
  --batch-size 1 \
  --dim 384 \
  --depth 28 \
  --heads 8 \
  --mlp-ratio 3 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --log-every 100 \
  --checkpoint-every 2000 \
  --device "$DEVICE" \
  --bitnet-qat \
  --bitnet-qat-learned-scale \
  --save-materialized-bitnet

sample_checkpoint "$QAT_DIR/flux_packed_student.pt" "$QAT_DIR/eval_holdout_bitnet_qat" 24

"$PYTHON_BIN" scripts/export_agentkernel_lite_image_flux_browser.py \
  --checkpoint "$QAT_DIR/flux_packed_student.pt" \
  --output-dir "$BROWSER_DIR" \
  --model-id "$(basename "$BROWSER_DIR")" \
  --ternary
