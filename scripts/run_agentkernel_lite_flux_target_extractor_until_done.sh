#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/vision/flux1_dev_flow_targets_general_hq_mix_clean_2048p_1seed_v1}"
PROMPTS="${PROMPTS:-data/vision/prompts/general_hq_mix_clean_50k_v1.jsonl}"
LIMIT="${LIMIT:-2048}"
PROMPT_SKIP="${PROMPT_SKIP:-0}"
SEEDS_PER_PROMPT="${SEEDS_PER_PROMPT:-1}"
STEPS="${STEPS:-24}"
TARGET_STRIDE="${TARGET_STRIDE:-4}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
GUIDANCE="${GUIDANCE:-3.5}"
DEVICE="${DEVICE:-cuda:1}"
GPU_ID="${GPU_ID:-1}"
TARGET_ROWS="${TARGET_ROWS:-12288}"

rows() {
  if [ -f "$OUTPUT_DIR/metadata.jsonl" ]; then
    wc -l < "$OUTPUT_DIR/metadata.jsonl"
  else
    echo 0
  fi
}

attempt=0
while [ "$(rows)" -lt "$TARGET_ROWS" ]; do
  attempt=$((attempt + 1))
  echo "$(date -Is) extractor_attempt=${attempt} rows=$(rows) target_rows=${TARGET_ROWS}"
  set +e
  "$PYTHON_BIN" scripts/generate_agentkernel_lite_flux_flow_targets.py \
    --output-dir "$OUTPUT_DIR" \
    --prompts "$PROMPTS" \
    --prompt-skip "$PROMPT_SKIP" \
    --limit "$LIMIT" \
    --seeds-per-prompt "$SEEDS_PER_PROMPT" \
    --steps "$STEPS" \
    --target-stride "$TARGET_STRIDE" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --guidance "$GUIDANCE" \
    --local-files-only \
    --cpu-offload \
    --quantize-transformer-4bit \
    --device "$DEVICE" \
    --gpu-id "$GPU_ID" \
    --resume
  status=$?
  set -e
  echo "$(date -Is) extractor_exit_status=${status} rows=$(rows)"
  if [ "$(rows)" -lt "$TARGET_ROWS" ]; then
    sleep 30
  fi
done

echo "$(date -Is) extractor_complete rows=$(rows)"
