#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"

DATASET_MANIFEST="${1:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/overall_research_assistant_balanced_v1/agentkernel_lite_encdec_dataset_manifest.json}"
OUTPUT_DIR="${2:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/overall_research_assistant_balanced_v1_train}"
INIT_CHECKPOINT="${3:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/chatfirst_current_loop_line_100m_from_mixed7000_train_05000/checkpoints/step_00005000.pt}"
TOKENIZER_DIR="${4:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/chatfirst_current_loop_line_100m_from_mixed7000_train_05000/tokenizer}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/train_agentkernel_lite_encdec.py" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --preset agentkernel-lite-100m \
  --tokenizer-kind agentkernel-bpe \
  --tokenizer-source-dir "$TOKENIZER_DIR" \
  --agentkernel-special-tokens 1 \
  --init-from-checkpoint "$INIT_CHECKPOINT" \
  --checkpoint-vocab-mismatch strict \
  --decoder-loss-weight "${DECODER_LOSS_WEIGHT:-1}" \
  --retrieval-contrastive-weight "${RETRIEVAL_CONTRASTIVE_WEIGHT:-0}" \
  --max-encoder-tokens "${MAX_ENCODER_TOKENS:-768}" \
  --max-decoder-tokens "${MAX_DECODER_TOKENS:-320}" \
  --max-retrieval-query-tokens "${MAX_RETRIEVAL_QUERY_TOKENS:-96}" \
  --max-retrieval-doc-tokens "${MAX_RETRIEVAL_DOC_TOKENS:-256}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --max-steps "${MAX_STEPS:-6000}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --log-every "${LOG_EVERY:-25}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --max-eval-batches "${MAX_EVAL_BATCHES:-64}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-1000}" \
  --dry-run 0 \
  --export-browser-bitnet "${EXPORT_BROWSER_BITNET:-0}" \
  --device "${DEVICE:-cuda}"
