#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peyton/miniconda3/envs/ai/bin/python}"

DATASET_MANIFEST="${1:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/current_loop_plus_1mabs_200kfulltext_parquet/agentkernel_lite_encdec_dataset_manifest.json}"
OUTPUT_DIR="${2:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/encoder_retrieval_ternary_aware_from_1mabs_train_01000}"
INIT_CHECKPOINT="${3:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/encoder_retrieval_1mabs_from_gooddecoder_freezeemb_train_02000/checkpoints/step_00002000.pt}"
TOKENIZER_DIR="${4:-$ROOT_DIR/artifacts/agentkernel_lite_encdec/encoder_retrieval_1mabs_from_gooddecoder_freezeemb_train_02000/tokenizer}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/train_agentkernel_lite_encdec.py" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --preset agentkernel-lite-100m \
  --tokenizer-kind agentkernel-bpe \
  --tokenizer-source-dir "$TOKENIZER_DIR" \
  --agentkernel-special-tokens 1 \
  --init-from-checkpoint "$INIT_CHECKPOINT" \
  --checkpoint-vocab-mismatch strict \
  --encoder-position-embeddings 1 \
  --retrieval-contrastive-weight "${RETRIEVAL_CONTRASTIVE_WEIGHT:-0.08}" \
  --retrieval-ternary-aware-weight "${RETRIEVAL_TERNARY_AWARE_WEIGHT:-0.12}" \
  --retrieval-ternary-teacher-distill-weight "${RETRIEVAL_TERNARY_TEACHER_DISTILL_WEIGHT:-0.2}" \
  --retrieval-ternary-teacher-temperature "${RETRIEVAL_TERNARY_TEACHER_TEMPERATURE:-0.05}" \
  --retrieval-ternary-reconstruction-weight "${RETRIEVAL_TERNARY_RECONSTRUCTION_WEIGHT:-0.05}" \
  --retrieval-hard-negative-weight "${RETRIEVAL_HARD_NEGATIVE_WEIGHT:-0.0}" \
  --retrieval-hard-negative-ternary "${RETRIEVAL_HARD_NEGATIVE_TERNARY:-1}" \
  --retrieval-temperature "${RETRIEVAL_TEMPERATURE:-0.05}" \
  --retrieval-ternary-threshold-ratio "${RETRIEVAL_TERNARY_THRESHOLD_RATIO:-0.20}" \
  --retrieval-ternary-group-size "${RETRIEVAL_TERNARY_GROUP_SIZE:-16}" \
  --retrieval-ternary-residual-dims "${RETRIEVAL_TERNARY_RESIDUAL_DIMS:-64}" \
  --decoder-loss-weight "${DECODER_LOSS_WEIGHT:-0}" \
  --parquet-require-retrieval-pair 1 \
  --max-encoder-tokens "${MAX_ENCODER_TOKENS:-256}" \
  --max-decoder-tokens "${MAX_DECODER_TOKENS:-64}" \
  --max-retrieval-query-tokens "${MAX_RETRIEVAL_QUERY_TOKENS:-96}" \
  --max-retrieval-doc-tokens "${MAX_RETRIEVAL_DOC_TOKENS:-256}" \
  --max-retrieval-negatives "${MAX_RETRIEVAL_NEGATIVES:-8}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --max-steps "${MAX_STEPS:-1000}" \
  --learning-rate "${LEARNING_RATE:-5e-5}" \
  --log-every "${LOG_EVERY:-10}" \
  --eval-every "${EVAL_EVERY:-100}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-250}" \
  --dry-run 0 \
  --export-browser-bitnet "${EXPORT_BROWSER_BITNET:-0}" \
  --device "${DEVICE:-cuda}"
