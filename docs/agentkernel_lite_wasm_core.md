# AgentKernel Lite WASM Core

For the full staged migration plan, see
[`agent_kernel_rust_wasm_migration.md`](agent_kernel_rust_wasm_migration.md).

AgentKernel Lite should use a partial Rust migration, not a full kernel rewrite.
The browser target cannot own the same authority as the Python runtime because it
does not execute shell commands, mutate repositories, run verifiers, or manage
artifact promotion. It should own the deterministic agent core that still matters
on device:

- task turn state
- step budget and stop state
- retrieved-evidence packing
- context/prompt compilation
- mode-specific chat/code instruction surfaces
- message and turn history snapshots
- browser-safe runtime assertions, especially `code_execution=disabled`

The JavaScript layer remains an adapter boundary:

- UI rendering
- Hugging Face dataset and model downloads
- Cache API and IndexedDB storage
- model-stack browser-bitnet/WebGPU inference
- optional retrieval fetches from Hugging Face

This matches the main kernel architecture: adapters can be disconnected, while
the core remains deterministic and testable.

## Current Implementation

The static export has a Rust/WASM core at:

```text
/data/repository_library/exports/agent_kernel/wasm/agent_kernel_lite_core
```

The compiled web package is:

```text
/data/repository_library/exports/agent_kernel/wasm/agent_kernel_lite_core/pkg
```

`AgentLiteCore` currently exposes:

- `start_turn(user_text, context_rows_json, language, max_new_tokens)`
- `finish_turn(assistant_text)`
- `reset()`
- `set_mode(mode)`
- `snapshot_json()`
- `step_count()`
- `can_continue()`

`start_turn` returns a JSON packet containing the compiled prompt, step index,
evidence records, and continuation state. The app imports this package from
`js/agent-kernel-app.js` and falls back to the old JavaScript prompt compiler if
WASM initialization fails.

## Migration Answer

We do need a Rust migration for AgentKernel Lite, but only for the portable core.
We should not port the full Python kernel into the browser. The full kernel has
server/workspace authority; Lite is an on-device chat/code generation runtime
with no command execution.

The right end state is:

```text
Rust/WASM core:
  TaskSpecLite -> AgentStateLite -> ContextPacketLite -> PromptPacket -> TurnRecord

JS adapters:
  UI, HF datasets, model-stack browser-bitnet inference, storage, network retrieval

Python kernel:
  full benchmark/runtime authority, sandbox, verifier, artifact promotion
```

## Training Path

AgentKernel Lite should train as an encoder-decoder model over the actual kernel
loop surface, not as a generic chat model. The current training entry points are:

```text
scripts/build_agentkernel_lite_encdec_dataset.py
scripts/train_agentkernel_lite_encdec.py
```

The dataset builder now pulls from three local evidence streams:

- `trajectories/episodes/*.json`: full AgentKernel task episodes with prompts,
  actions, verifier feedback, and step outputs.
- `trajectories/improvement/candidates/qwen_adapter/**/qwen_sft_*.jsonl`:
  existing Qwen adapter supervised traces.
- `docs/*.md`: low-weight implementation-grounded explanatory examples.

The trainer uses model-stack's `runtime.seq2seq.EncoderDecoderLM` and an
AgentKernel byte-level BPE tokenizer by default. The production preset is:

```text
d_model=576
n_heads=9
n_layers=5
d_ff=2048
vocab_size=32768
```

With tied encoder/decoder embeddings this is `104,193,600` parameters. The dense
checkpoint is saved first, then model-stack exports a browser bundle via:

```text
ExportConfig(target="browser-bitnet", quantize="bitnet")
```

That bundle is the Safari/WebGPU artifact. It contains `manifest.json`,
BitNet-packed linear tensors, dense embedding/norm tensors, `encdec_runtime.js`,
`bitnet_webgpu.js`, and WGSL kernels.

Use a working PyTorch environment for training. On this machine the base conda
`torch` package resolves as a namespace-only install, while
`/home/peyton/miniconda3/envs/ai/bin/python` has working PyTorch.

## C++/WebGPU Decision

C++ WebGPU is possible through Emscripten, but it should not be the first browser
target. Safari's WebGPU path is most direct as JavaScript plus WGSL, and
model-stack already exports that runtime for encoder-decoder BitNet bundles.
Rust/WASM should own the deterministic agent loop; WebGPU should own tensor
execution. A C++ runtime only becomes worth adding if profiling proves the
JavaScript/WGSL orchestration overhead dominates after the BitNet bundle is
trained and loaded.

## Next Required Core Functions

To make Lite closer to the full loop without giving it unsafe authority, add:

- `ingest_pack_manifest(manifest_json)` for deterministic pack metadata tracking
- `rank_evidence(query, candidate_rows_json, limit)` for small candidate batches
- `compile_context_packet(task_json, evidence_json, history_json)`
- `parse_model_reply(text)` for structured `respond` / `propose_code` decisions
- `runtime_attestation()` for model id, device, pack id, storage mode, and core
  version
- `export_episode_json()` matching the main kernel episode shape where possible

Avoid moving whole 1M-row scans into WASM by passing massive JSON strings. The
large corpus should stay in IndexedDB/cache-backed adapters or prebuilt shards;
the Rust core should rank small candidate windows and compile the final context.
