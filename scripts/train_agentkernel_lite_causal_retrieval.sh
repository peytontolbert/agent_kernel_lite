#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATASET_MANIFEST="${1:-artifacts/agentkernel_lite_encdec/research_retrieval_curriculum_1m_abs_200k_fulltext_parquet/agentkernel_lite_encdec_dataset_manifest.json}"
INIT_CHECKPOINT="${2:-}"
OUTPUT_DIR="${3:-artifacts/agentkernel_lite_encdec/causal_retrieval_dense_stage1}"
TOKENIZER_SOURCE_DIR="${TOKENIZER_SOURCE_DIR:-}"

cd "$ROOT"

ARGS=(
  scripts/train_agentkernel_lite_encdec.py
  --dataset-manifest "$DATASET_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --preset agentkernel-lite-100m
  --tokenizer-kind agentkernel-bpe
  --agentkernel-special-tokens 1
  --retrieval-head-dim 256
  --retrieval-contrastive-weight 0.08
  --retrieval-temperature 0.05
  --decoder-loss-weight 1.0
  --max-encoder-tokens 1024
  --max-decoder-tokens 512
  --max-retrieval-query-tokens 96
  --max-retrieval-doc-tokens 256
  --batch-size 8
  --eval-batch-size 8
  --max-steps 20000
  --learning-rate 3e-5
  --weight-decay 0.01
  --clip-grad-norm 1.0
  --log-every 50
  --eval-every 500
  --max-eval-batches 64
  --checkpoint-every 1000
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
