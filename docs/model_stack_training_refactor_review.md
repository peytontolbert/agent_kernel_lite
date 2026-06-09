# Model-Stack Training Refactor Review

Date: 2026-05-22

## Cleanup Kept

Current useful local artifacts kept:

- `artifacts/pocketpal_controller_100m_v268a_v264_direct_intent_calibrated`
- `artifacts/pocketpal_controller_100m_v280a_akv1_300_failure_refine_from_v279`
- `tmp/pocketpal_stage67_structured_copy_decoder_v172d_akv1`
- `tmp/pocketpal_stage69_akv1_broad_failure_curriculum`
- `tmp/pocketpal_stage70_akv1_weak_bucket_curriculum`
- `tmp/pocketpal_stage71_akv1_300_failure_refine`

Deleted old superseded seq2seq artifacts are logged in:

- `artifacts/cleanup_old_seq2seq_artifacts_20260522.txt`

## Existing Model-Stack Training Code

Useful existing modules under `/data/agentkernel/other_repos/model-stack`:

- `train/trainer.py`
  - `TrainConfig`
  - `Trainer`
  - AdamW param groups with norm/bias weight-decay exemption
  - cosine/linear warmup schedules
  - AMP support
  - gradient accumulation
  - gradient clipping
  - EMA state support
  - checkpoint save/resume
  - validation hooks and early stopping

- `train/run.py`
  - distributed wrapper entry point for DDP/FSDP/DeepSpeed via `DistributedEngine`
  - loader wrapping
  - optional validation dataset support

- `compress/distill.py`
  - `kd_loss`
  - `mse_match`
  - `DistillHooks`

- `data/tokenizer.py`
  - HF and local tokenizer wrappers
  - pure local LLaMA tokenizer path

- `runtime/seq2seq.py`
  - already provides `EncoderDecoderLM`
  - retrieval query/doc heads
  - policy heads
  - intent head

## What Should Move Into Model-Stack

These pieces are generic enough to move out of `scripts/train_agentkernel_lite_encdec.py`:

1. Checkpoint compatibility loader
   - Current local code supports loading `.pt`, `.safetensors`, and model directories.
   - It also handles vocab expansion for embedding/lm-head rows and tolerates training-only aux heads.
   - This belongs in `model-stack/train/checkpoint_compat.py` or `model-stack/model/checkpoint.py`.

2. Tokenizer special-token expansion utilities
   - The AK special token row initialization is currently PocketPal-flavored, but the mechanism is generic:
     seed new special-token embeddings from related existing tokens.
   - Move the generic “expand vocab and initialize new rows from token aliases” logic to `model-stack/data/tokenizer.py`.
   - Keep the AK token alias table local.

3. Auxiliary loss scheduling
   - `_scheduled_aux_weight`, warmup/cosine decay, and aux gradient budget are generic.
   - Move into `model-stack/train/loss_schedules.py`.

4. Future/target sketch losses
   - The future-sketch and target-sketch losses are generic ultra-small-model objectives.
   - Move their core functions to `model-stack/train/aux_losses.py`.
   - Keep PocketPal-specific bucket settings and dataset wiring local.

5. Train-state JSON logging
   - The current script emits structured events directly.
   - Model-stack trainer already has logging hooks but not our JSON event style.
   - Add a small event logger interface to model-stack so PocketPal and image/TTS distillation can share it.

6. Eval/checkpoint ledger support
   - `scripts/record_pocketpal_training_state.py` is mostly application-specific, but the capability-vector pattern is reusable.
   - Move a generic record builder to `model-stack/train/ledger.py`; keep PocketPal task labels local.

## What Should Stay Local

These are application-specific and should remain in this repo:

- AK structured token protocol:
  - `scripts/pocketpal_structured_decode.py`
  - `docs/pocketpal_ak_token_protocol_v1.json`

- PocketPal content operators:
  - `scripts/pocketpal_content_operators.py`
  - browser-side `activeAgentApplyContentOperators(...)`

- PocketPal dataset builders:
  - structured decoder dataset conversion
  - direct failure replay curriculum
  - gate-specific repair datasets

- PocketPal eval gates:
  - runtime gate
  - direct active-agent prompt gate
  - browser parity gates

## Recommended Next Refactor Order

1. Move checkpoint compatibility loading first.
   - Lowest risk.
   - Removes repeated checkpoint/vocab expansion code.
   - Immediately useful for seq2seq, image, and TTS training.

2. Move aux schedule helpers and aux gradient budget.
   - Clean separation between training mechanics and task-specific losses.

3. Move future/target sketch loss cores.
   - Keep local configs and dataset fields in PocketPal.

4. Add a model-stack event logger / ledger helper.
   - Makes long training runs easier to compare without duplicating JSON logging.

5. Only after those, consider wrapping PocketPal training with `train.Trainer`.
   - The current trainer is LM-oriented and expects causal batches.
   - PocketPal seq2seq has custom encoder/decoder batches, retrieval heads, policy heads, intent heads, negative losses, sketch losses, and anchor penalties.
   - It is better to extract reusable pieces first, then adapt `Trainer` to accept a custom `step_fn`.

## Key Constraint

Do not push the AK app protocol into model-stack. Model-stack should own reusable training/runtime machinery. PocketPal should own product protocol, app orchestration, structured action tokens, and runtime content materialization.
