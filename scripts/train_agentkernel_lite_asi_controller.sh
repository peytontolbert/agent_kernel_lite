#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

DATASET_MANIFEST="${1:-artifacts/agentkernel_lite_encdec/controller_trace_x5_plus_retrieval_chatmix_v1/agentkernel_lite_encdec_dataset_manifest.json}"
INIT_CHECKPOINT="${2:-}"
OUTPUT_DIR="${3:-artifacts/agentkernel_lite_encdec/asi_controller_trace_retrieval_stage1}"
TOKENIZER_SOURCE_DIR="${TOKENIZER_SOURCE_DIR:-}"
MAX_STEPS="${MAX_STEPS:-3000}"

cd "$ROOT"

ARGS=(
  scripts/train_agentkernel_lite_encdec.py
  --dataset-manifest "$DATASET_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --preset agentkernel-lite-100m
  --tokenizer-kind agentkernel-bpe
  --agentkernel-special-tokens 1
  --retrieval-head-dim 256
  --agent-policy-heads 1
  --policy-head-loss-weight 0.12
  --retrieval-contrastive-weight 0.08
  --retrieval-temperature 0.05
  --decoder-loss-weight 1.0
  --max-encoder-tokens 1024
  --max-decoder-tokens 512
  --max-retrieval-query-tokens 96
  --max-retrieval-doc-tokens 256
  --batch-size 4
  --eval-batch-size 4
  --max-steps "$MAX_STEPS"
  --learning-rate 2e-5
  --weight-decay 0.01
  --clip-grad-norm 1.0
  --log-every 25
  --eval-every 250
  --max-eval-batches 64
  --checkpoint-every 500
  --dry-run 0
  --export-browser-bitnet 0
)

if [[ -n "$INIT_CHECKPOINT" ]]; then
  ARGS+=(--init-from-checkpoint "$INIT_CHECKPOINT" --checkpoint-vocab-mismatch expand)
fi
if [[ -n "$TOKENIZER_SOURCE_DIR" ]]; then
  ARGS+=(--tokenizer-source-dir "$TOKENIZER_SOURCE_DIR")
fi

"$PYTHON_BIN" "${ARGS[@]}"
