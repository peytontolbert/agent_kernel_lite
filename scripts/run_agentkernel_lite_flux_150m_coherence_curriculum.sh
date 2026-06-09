#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
PROMPTS="${PROMPTS:-data/vision/prompts/general_hq_mix_clean_50k_v1.jsonl}"
EVAL_PROMPTS="${EVAL_PROMPTS:-data/vision/prompts/flux_general_hq_holdout_eval_v0.txt}"
ANCHOR_DIR="${ANCHOR_DIR:-data/vision/flux1_dev_flow_targets_coherence_anchor_256p_stride1_v0}"
SEED_DIVERSE_DIR="${SEED_DIVERSE_DIR:-data/vision/flux1_dev_flow_targets_general_hq_mix_clean_512p_stride1_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/agentkernel_lite_image_flux_flow_student_coherence_150m_dense_v0}"
CHUNK_ROOT="${CHUNK_ROOT:-data/vision/flux1_dev_flow_targets_general_hq_10k_chunks_v0}"
DEVICE="${DEVICE:-cuda:0}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:1}"
GPU_ID="${GPU_ID:-1}"
STEPS="${STEPS:-24}"
TARGET_STRIDE="${TARGET_STRIDE:-1}"
ANCHOR_ROWS="${ANCHOR_ROWS:-6144}"
BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-50000}"
CHUNK_STEPS="${CHUNK_STEPS:-20000}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
START_SKIP="${START_SKIP:-1024}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-10000}"
SAMPLE_EVERY_CHUNKS="${SAMPLE_EVERY_CHUNKS:-2}"
DELETE_TRAINED_CHUNKS="${DELETE_TRAINED_CHUNKS:-1}"

rows() {
  local dir="$1"
  if [ -f "$dir/metadata.jsonl" ]; then
    wc -l < "$dir/metadata.jsonl"
  else
    echo 0
  fi
}

wait_for_rows() {
  local dir="$1"
  local threshold="$2"
  local label="$3"
  while [ "$(rows "$dir")" -lt "$threshold" ]; do
    echo "$(date -Is) waiting_for_${label}_rows rows=$(rows "$dir") threshold=${threshold}"
    sleep 120
  done
}

train_chunk() {
  local target_dir="$1"
  local steps="$2"
  local resume_args=()
  if [ -f "$OUTPUT_DIR/flux_packed_student.pt" ]; then
    resume_args=(--resume "$OUTPUT_DIR/flux_packed_student.pt")
  fi
  "$PYTHON_BIN" scripts/train_agentkernel_lite_image_flux_flow_distill.py \
    --target-dir "$target_dir" \
    --extra-target-dir "$ANCHOR_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "${resume_args[@]}" \
    --steps "$steps" \
    --batch-size 1 \
    --dim 512 \
    --depth 28 \
    --heads 8 \
    --mlp-ratio 3 \
    --lr 4e-5 \
    --weight-decay 0.01 \
    --log-every 100 \
    --checkpoint-every 5000 \
    --device "$DEVICE"
}

sample_checkpoint() {
  local output_dir="$1"
  "$PYTHON_BIN" scripts/sample_agentkernel_lite_image_flux_flow_distill.py \
    --checkpoint "$OUTPUT_DIR/flux_packed_student.pt" \
    --prompts "$EVAL_PROMPTS" \
    --output-dir "$output_dir" \
    --limit 16 \
    --steps "$STEPS" \
    --guidance 3.5 \
    --width 512 \
    --height 512 \
    --local-files-only \
    --cpu-offload \
    --quantize-transformer-4bit \
    --device "$DEVICE" \
    --gpu-id 0
}

extract_chunk() {
  local skip="$1"
  local limit="$2"
  local target_dir="$3"
  local target_rows=$((limit * STEPS / TARGET_STRIDE))
  OUTPUT_DIR="$target_dir" \
  PROMPTS="$PROMPTS" \
  LIMIT="$limit" \
  PROMPT_SKIP="$skip" \
  SEEDS_PER_PROMPT=1 \
  TARGET_ROWS="$target_rows" \
  STEPS="$STEPS" \
  TARGET_STRIDE="$TARGET_STRIDE" \
  DEVICE="$TEACHER_DEVICE" \
  GPU_ID="$GPU_ID" \
    scripts/run_agentkernel_lite_flux_target_extractor_until_done.sh
}

mkdir -p "$OUTPUT_DIR" "$CHUNK_ROOT"

wait_for_rows "$ANCHOR_DIR" "$ANCHOR_ROWS" "anchor"

echo "$(date -Is) bootstrap_train seed_diverse_dir=$SEED_DIVERSE_DIR anchor_dir=$ANCHOR_DIR"
train_chunk "$SEED_DIVERSE_DIR" "$BOOTSTRAP_STEPS"
sample_checkpoint "$OUTPUT_DIR/eval_after_bootstrap"

if [ "$DELETE_TRAINED_CHUNKS" = "1" ] && [ -d "$SEED_DIVERSE_DIR" ]; then
  echo "$(date -Is) deleting_seed_diverse_dir $SEED_DIVERSE_DIR"
  rm -rf "$SEED_DIVERSE_DIR"
fi

chunk_index=0
skip="$START_SKIP"
while [ "$skip" -lt "$TOTAL_PROMPTS" ]; do
  remaining=$((TOTAL_PROMPTS - skip))
  limit="$CHUNK_SIZE"
  if [ "$remaining" -lt "$limit" ]; then
    limit="$remaining"
  fi
  chunk_dir="$CHUNK_ROOT/chunk_$(printf '%05d' "$skip")_$(printf '%05d' $((skip + limit)))"
  echo "$(date -Is) extract_chunk index=$chunk_index skip=$skip limit=$limit dir=$chunk_dir"
  extract_chunk "$skip" "$limit" "$chunk_dir"
  echo "$(date -Is) train_chunk index=$chunk_index rows=$(rows "$chunk_dir")"
  train_chunk "$chunk_dir" "$CHUNK_STEPS"
  if [ $((chunk_index % SAMPLE_EVERY_CHUNKS)) -eq 0 ]; then
    sample_checkpoint "$OUTPUT_DIR/eval_after_chunk_${chunk_index}"
  fi
  if [ "$DELETE_TRAINED_CHUNKS" = "1" ]; then
    echo "$(date -Is) deleting_chunk $chunk_dir"
    rm -rf "$chunk_dir"
  fi
  skip=$((skip + limit))
  chunk_index=$((chunk_index + 1))
done

sample_checkpoint "$OUTPUT_DIR/eval_final_dense"
echo "$(date -Is) coherence_curriculum_complete output_dir=$OUTPUT_DIR"
