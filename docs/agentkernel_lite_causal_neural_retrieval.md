# AgentKernel Lite Causal Neural Retrieval

## Goal

AgentKernel Lite should not treat retrieval as a prompt preamble. The release target is a WASM-first browser research assistant for iPhone/Safari where the BitNet encoder learns to query paper memory, the browser scans compact local vectors, and retrieved paper evidence is fed back into the decoder as causal context.

The initial release uses semi-neural Level 4 retrieval:

1. The BitNet encoder produces a latent query vector from the current turn.
2. The browser scans compact paper memory vectors.
3. Retrieved memory is compiled into memory slots/evidence blocks.
4. The same BitNet encoder-decoder generates the response from those memory slots.
5. The older paper embedding model can act as a teacher while the BitNet retrieval heads improve.

This Lite scope is intentionally paper-only. Code development actions, shell actions, file edits, and repository tools stay outside this browser assistant until the paper retrieval and grounded response loop is solid.

## Implemented Surfaces

### Model

`other_repos/model-stack/runtime/seq2seq.py` now supports optional encoder retrieval heads:

- `retrieval_query_head`
- `retrieval_doc_head`
- `retrieval_query_embedding(...)`
- `retrieval_doc_embedding(...)`

The heads are enabled by `ModelConfig.retrieval_head_dim`.

It also supports optional paper-controller policy heads via `ModelConfig.agent_policy_heads`:

- `query_confidence`
- `retrieval_coverage`
- `ood_query`
- `ood_evidence`
- `answer_confidence`
- `needs_verification`
- `paper_action_validity`

These heads are not a hardcoded browser rule system. They are trainable calibration/control outputs so the loop can learn when a paper query is understood, whether evidence is sufficient, when a selected paper should be read more deeply, and when the final answer needs more verification.

### Training

`scripts/train_agentkernel_lite_encdec.py` now routes retrieval contrastive loss through the optional retrieval heads when present.

Use:

```bash
scripts/train_agentkernel_lite_causal_retrieval.sh \
  artifacts/agentkernel_lite_encdec/research_retrieval_curriculum_1m_abs_200k_fulltext_parquet/agentkernel_lite_encdec_dataset_manifest.json \
  artifacts/agentkernel_lite_encdec/dense_qwen35_9b_teacher_30000_from_targeted52000_train_56000/checkpoints/step_00056000.pt \
  artifacts/agentkernel_lite_encdec/causal_retrieval_dense_stage1
```

This trains:

- decoder answer/action loss,
- encoder query/doc contrastive retrieval loss,
- optional paper-controller policy head loss,
- 256-dimensional retrieval heads suitable for compact browser memory vectors.

Add policy heads with:

```bash
--agent-policy-heads 1 \
--policy-head-loss-weight 0.1
```

The current paper curriculum emits targets for query confidence, retrieval coverage, OOD query/evidence, answer confidence, verification need, and paper action validity. Merged parquet datasets preserve those targets.

### Browser Runtime

`other_repos/model-stack/browser/bitnet/encdec_runtime.js` now exposes:

- `retrievalQueryEmbedding(inputIds)`
- `retrievalDocEmbedding(inputIds)`

These work in the same model-stack browser runtime as generation. The app worker exposes this as an `embed` message.

### Memory Pack Export

`scripts/export_agentkernel_lite_neural_memory_pack.py` exports browser memory packs:

- `vectors.i8.bin`: int8 per-vector quantized document vectors,
- `scales.f32.bin`: per-vector scales,
- `vectors.t2.bin`: optional packed ternary document vectors, four dimensions per byte,
- `ternary_scales.f32.bin`: optional per-vector ternary magnitude scales,
- `ternary_density.f32.bin`: optional nonzero density diagnostics for ternary rows,
- `vectors.t2.grouped.bin`: optional packed ternary vectors with local group scales,
- `ternary_group_scales.f32.bin`: optional per-row, per-group magnitude scales,
- `ternary_group_density.f32.bin`: optional per-row, per-group nonzero density diagnostics,
- `vectors.t2.grouped_signed.bin`: optional packed ternary vectors with separate positive/negative group scales,
- `ternary_group_signed_scales.f32.bin`: optional per-row, per-group positive and negative magnitude scales,
- `ternary_group_signed_density.f32.bin`: optional per-row, per-group nonzero density diagnostics,
- `vectors.t2.grouped_signed_residual.bin`: optional signed grouped ternary vectors with sparse residual correction,
- `ternary_group_signed_residual_scales.f16.bin`: optional fp16 positive/negative group scales,
- `ternary_residual_indices.u16.bin`: optional sparse residual dimension ids,
- `ternary_residual_values.f16.bin`: optional sparse residual values,
- `metadata.jsonl`: paper ids, titles, abstracts, categories,
- `memory_manifest.json`: browser load contract.

Example:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/export_agentkernel_lite_neural_memory_pack.py \
  --bundle-dir artifacts/agentkernel_lite_encdec/causal_retrieval_dense_stage1 \
  --paper-root /arxiv/huggingface/paper_text_1m_dedup_v1 \
  --output-dir artifacts/agentkernel_lite_memory/papers_10k \
  --max-rows 10000 \
  --vector-format ternary_grouped_signed_residual \
  --ternary-threshold-ratio 0.20 \
  --ternary-group-size 16 \
  --ternary-residual-dims 64
```

The manifest can expose `int8`, `ternary`, `ternary_grouped`, `ternary_grouped_signed`, `ternary_grouped_signed_residual`, or `both`. Int8 remains the default quality baseline because it preserves more continuous embedding magnitude. Simple ternary keeps one scalar per vector; grouped ternary keeps one scalar per small dimension block; signed grouped ternary keeps separate positive and negative scales per block; signed grouped residual adds a small sparse correction for dimensions whose magnitude was least representable by ternary. Signed grouped residual is the most likely iPhone default because it crossed 0.90 float top-10 overlap in the current eval while still being reducible below int8 size with fp16 scales. Ternary scans are not inherently worse, but post-hoc ternarizing embeddings trained for cosine/int8 can lose recall. The current iPhone-target setting is threshold `0.20`, group size `16`, residual dims `64`, fp16 scales/residuals.

Before switching the browser default, compare formats against the float retrieval-head baseline:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/evaluate_agentkernel_lite_memory_quantization.py \
  --bundle-dir artifacts/agentkernel_lite_encdec/causal_retrieval_dense_stage1 \
  --paper-root /arxiv/huggingface/paper_text_1m_dedup_v1 \
  --max-rows 512 \
  --top-k 10
```

This reports title-to-paper recall plus top-k agreement with the float score matrix for `int8`, `ternary`, `ternary_grouped`, `ternary_grouped_signed`, and optional residual formats. The release target is for signed grouped residual ternary to approach int8's float agreement while reducing bytes scanned on Safari. Current measured result after hard-negative continuation: threshold `0.20`, group size `16`, residual dims `64` reached `0.90039` float top-10 overlap on the 256-row quantization eval, versus `0.99531` for int8.

### Ternary-Aware Training

The retrieval model can now be continued with a ternary-aware retrieval loss:

```bash
scripts/train_agentkernel_lite_ternary_retrieval.sh
```

The launcher resumes from the current 1M-abstract retrieval checkpoint and adds an auxiliary contrastive loss where document embeddings are passed through the signed grouped residual ternary approximation:

```text
query embedding
  @ ternary_grouped_signed_residual(doc embedding)
  -> contrastive ranking loss
```

Default layout:

- threshold ratio: `0.20`,
- group size: `16`,
- sparse residual dims: `64`,
- decoder loss disabled for encoder-focused retrieval continuation.

This makes the model learn embeddings that survive the browser memory layout instead of relying only on post-hoc compression.

The first hard-negative training pass is now implemented:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/build_agentkernel_lite_hard_negative_retrieval_dataset.py \
  --bundle-dir artifacts/agentkernel_lite_encdec/encoder_retrieval_1mabs_from_gooddecoder_freezeemb_train_02000 \
  --output-dir artifacts/agentkernel_lite_encdec/hard_negative_retrieval_4096_dataset \
  --max-rows 4096 \
  --negative-count 8
```

Then continue retrieval training with:

```bash
RETRIEVAL_HARD_NEGATIVE_WEIGHT=0.12 \
RETRIEVAL_CONTRASTIVE_WEIGHT=0.02 \
RETRIEVAL_TERNARY_AWARE_WEIGHT=0.03 \
MAX_RETRIEVAL_NEGATIVES=4 \
scripts/train_agentkernel_lite_ternary_retrieval.sh \
  artifacts/agentkernel_lite_encdec/hard_negative_retrieval_4096_dataset/dataset_manifest.json \
  artifacts/agentkernel_lite_encdec/encoder_retrieval_hardneg4096_ternary_from_1mabs_train_00200
```

This pass reduced held-out hard-negative eval loss to `0.00562` at step 1000 and moved signed grouped residual top-10 overlap from `0.88398` to `0.90039` when evaluated with threshold `0.20`, group size `16`, and residual dims `64`.

### Web App

The app can opt into neural memory search with:

```text
?neuralMemory=1&neuralMemoryPack=<url-to-memory_manifest.json>
```

When enabled, retrieval uses the loaded AgentKernel BitNet model to embed the query instead of the separate ONNX embedding model. The old retrieval model remains useful as a teacher and fallback.

## Release Shape

For iPhone/Safari, the release-friendly stack is:

```text
BitNet WASM encoder-decoder
  + retrieval query/doc heads
  + int8 or signed grouped residual ternary compact memory pack
  + local vector scan
  + active selected-paper context
  + paper-controller confidence/OOD heads
  + explicit evidence rendering
```

This avoids shipping a second embedding model as the required path. It also avoids trying to memorize the 1M paper corpus inside 100M parameters.

WebGPU remains useful for desktop experiments, but it is not the production assumption. The iPhone path should optimize WASM decode kernels, typed-array cache reuse, decoder KV cache reuse, compact int8/ternary memory scans, IndexedDB artifact/model caching, and short responsive generation chunks.

## Next Optimization Targets

1. Add fused WASM decode kernels so norm/attention/MLP avoid repeated JS/WASM boundary crossings.
2. Cache query embeddings and memory search results in IndexedDB.
3. Add tiled SIMD dot-product kernels for int8 memory scan.
4. Shard memory packs by topic/year/source so iPhone users can download selectively.
5. Add compact full-text span packs separate from abstract packs.
6. Scale hard-negative mining beyond 4k rows and include teacher-ranked negatives from the existing paper embedding model.
7. Evaluate int8 vs packed ternary recall, latency, battery, and storage on Safari/iPhone-scale packs.
8. Continue ternary-aware hard-negative retrieval-head training until residual-64 overlap is stable above 0.90 on larger held-out paper subsets.
9. Add step-boundary interleaved retrieval during generation:
   - initial answer plan,
   - uncertainty trigger,
   - source/citation trigger,
   - selected-paper followup trigger.
10. Export policy heads into the browser runtime so the UI can expose learned confidence and coverage without relying on regex fallback behavior.

## Important Constraint

The model should learn the retrieval policy, query vectors, and evidence sufficiency signals. The browser runtime should still expose provenance and exact source ids. Neural retrieval should influence decoding causally, but it should not hide where evidence came from.
