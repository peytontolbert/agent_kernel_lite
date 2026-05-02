# AgentKernel Rust/WASM Migration

This is the structured migration path for turning AgentKernel Lite into a
browser-native on-device agent kernel. The browser can own persistent memory,
retrieval packs, artifacts, chat state, and user-approved extension actions. The
model execution path can use WebGPU, but the agent loop itself should remain a
deterministic Rust/WASM core.

## Target Architecture

```text
Rust/WASM AgentKernel core
  schemas, task state, memory ledger, context compiler, action parser,
  action governance, artifact graph, episode export, runtime attestation

Browser adapters
  UI, IndexedDB/OPFS/Cache API, Hugging Face downloads, retrieval shard loading,
  extension execution, user approval prompts, model-stack WebGPU runtime

WebGPU model runtime
  AgentKernel Lite encoder-decoder generation, embeddings, rerankers

Python AgentKernel
  benchmark authority, shell sandbox, verifiers, artifact promotion,
  unattended campaigns, training data generation
```

The Rust/WASM core should never silently acquire authority from the browser. It
may propose actions. Adapters execute actions only after user policy permits
them.

## Authority Model

Browser AgentKernel can support:

- chat memory and episodic memory
- local artifact stores and import/export bundles
- retrieval over cached papers, repositories, traces, and user artifacts
- code and patch generation without implicit execution
- extension requests for user-approved capabilities
- user-saveable workspaces and evidence bundles

Browser AgentKernel cannot directly own:

- unrestricted shell execution
- Docker or system package management
- arbitrary filesystem access outside browser grants
- benchmark verifiers that require system processes
- unattended background daemon behavior outside browser/PWA constraints

Those stay in Python/native AgentKernel or in explicit extensions.

## Repository Layout

Make the Rust core repo-owned, then export it into the static app.

```text
agent_kernel_rust_wasm/
  Cargo.toml
  src/
    lib.rs
    schemas.rs
    state.rs
    memory.rs
    artifacts.rs
    retrieval.rs
    context.rs
    policy.rs
    actions.rs
    extensions.rs
    attestation.rs
    episode.rs
    bindings.rs
  tests/
    fixtures/

/data/repository_library/exports/agent_kernel/wasm/
  agent_kernel_lite_core/     # generated/synced browser export target
```

The current static-export crate at
`/data/repository_library/exports/agent_kernel/wasm/agent_kernel_lite_core`
should become an export target, not the only source of truth.

## Python To Rust Surface Map

| Python Surface | Rust/WASM Module | Browser Scope |
| --- | --- | --- |
| `schemas.py::TaskSpec` | `schemas::TaskSpecLite` | import task/user request contracts |
| `schemas.py::ActionDecision` | `schemas::ActionDecisionLite` | parse model decisions and extension proposals |
| `schemas.py::StepRecord` | `episode::StepRecordLite` | persist browser episode trace |
| `schemas.py::EpisodeRecord` | `episode::EpisodeRecordLite` | export/import memory bundles |
| `state.py::AgentState` | `state::AgentStateLite` | turn state, plans, failure counters, active subgoal |
| `memory.py::EpisodeMemory` | `memory::EpisodeStore` | IndexedDB/OPFS-backed episode metadata |
| `memory.py::GraphMemory` | `memory::GraphRecall` | local semantic recall over small candidate windows |
| `policy.py::ContextProvider` | `context::ContextCompiler` | compile prompt/context packet from retrieval + memory |
| `policy.py::LLMDecisionPolicy` | `policy::DecisionPolicyLite` | deterministic fallback, action parsing, governance |
| `actions.py` | `actions::ActionKind` | respond, propose artifact, extension request |
| `sandbox.py` | `extensions::ActionBus` | browser-mediated user-approved actions only |
| `verifier.py` | `episode::LiteVerifier` | artifact/content checks that do not require shell |
| `config.py::KernelConfig` | `schemas::RuntimeConfigLite` | model/storage/extension settings |

## Core API Contract

The WASM boundary should use JSON strings for browser compatibility, but the
Rust side should deserialize into typed structs immediately.

Required exported methods:

```text
new(session_id, config_json) -> AgentKernelWasm
ingest_pack_manifest(manifest_json) -> ReceiptJson
ingest_memory_bundle(bundle_json) -> ReceiptJson
start_turn(user_text, retrieval_candidates_json, options_json) -> TurnPacketJson
finish_model_reply(model_text) -> DecisionPacketJson
record_extension_result(action_id, result_json) -> StepPacketJson
record_user_artifact(path, content, metadata_json) -> ArtifactReceiptJson
rank_evidence(query, candidate_rows_json, limit) -> EvidenceJson
compile_context_packet(task_json, evidence_json, history_json) -> ContextPacketJson
export_episode_json() -> EpisodeJson
snapshot_json() -> SnapshotJson
runtime_attestation() -> AttestationJson
reset()
```

The current static core already covers `start_turn`, `finish_turn`, `reset`,
`set_mode`, `snapshot_json`, `step_count`, and `can_continue`. The migration
extends that surface instead of replacing it all at once.

## Action Contract

Initial browser-safe actions:

```text
respond
propose_code
propose_artifact_write
retrieve
save_memory
extension_request
ask_user
done
```

Every action becomes an append-only ledger record:

```json
{
  "action_id": "act_...",
  "kind": "extension_request",
  "requested_capability": "github.read_repo",
  "input": {},
  "requires_user_approval": true,
  "policy_state": "pending",
  "created_at": "...",
  "model_trace_id": "..."
}
```

Adapters return receipts, not hidden side effects:

```json
{
  "action_id": "act_...",
  "status": "approved_executed",
  "output": {},
  "artifact_refs": [],
  "error": ""
}
```

## Storage Contract

Use browser adapters for persistence:

- IndexedDB: episode metadata, message history, small evidence records,
  extension receipts, model/runtime manifests.
- OPFS: larger artifacts, retrieval shard files, user workspaces, generated
  bundles.
- Cache API: Hugging Face model files, dataset shards, static runtime files.
- User import/export: zipped or JSONL bundles for memory, artifacts, and
  retrieval packs.

Rust/WASM should receive handles/metadata and bounded candidate windows. It
should not be passed full million-paper JSON payloads.

## Migration Phases

### Phase 0: Baseline And Ownership

Goal: make the current WASM core reproducible from the AgentKernel repo.

Tasks:

- create repo-owned `agent_kernel_rust_wasm/` crate
- copy current static-export core into that crate
- add a script to build and sync `pkg/` into the static export
- preserve the existing browser app fallback behavior
- keep static export loading from the generated package path

Acceptance:

- `cargo test` passes in `agent_kernel_rust_wasm`
- `wasm-pack build --target web --release` succeeds
- static app still shows `core ready`

### Phase 1: Schema Parity

Goal: encode Python kernel contracts in Rust without behavior drift.

Tasks:

- implement `TaskSpecLite`, `ActionDecisionLite`, `StepRecordLite`,
  `EpisodeRecordLite`, `RuntimeConfigLite`
- add JSON roundtrip tests using fixtures from existing episode JSON
- add normalization rules matching `TaskSpec.__post_init__`
- define compatibility version fields

Acceptance:

- Rust fixtures parse current `trajectories/episodes/*.json`
- exported episode JSON remains compatible with Python training dataset builder

### Phase 2: State And Memory

Goal: persistent browser sessions behave like a small AgentKernel runtime.

Tasks:

- port `AgentState` counters and compact history behavior
- add memory bundle import/export
- add artifact graph records with content hash, path, MIME/type, source, created
  turn, and parent refs
- add storage adapter receipts for IndexedDB/OPFS writes

Acceptance:

- browser can save/load a session with chat history, artifacts, and evidence
- exported episode includes all turns and action receipts

### Phase 3: Retrieval And Context Compilation

Goal: context compilation becomes deterministic and testable.

Tasks:

- implement `rank_evidence` for bounded candidate windows
- support paper/repository/trace/artifact evidence types
- add source attribution and arXiv PDF normalization
- implement context budget selection and evidence dedupe
- compile a stable model input packet for the encoder-decoder model

Acceptance:

- same query/candidates produce stable ranked evidence
- context packet contains only bounded selected evidence
- app can show evidence cards and open paper PDFs when IDs are present

### Phase 4: Model Decision Bridge

Goal: local model outputs become structured kernel decisions.

Tasks:

- connect model-stack browser-bitnet encoder-decoder worker
- parse generated text into `ActionDecisionLite`
- keep fallback plain text as `respond`
- add action confidence and parse failure receipts
- train/evaluate against AgentKernel trace dataset

Acceptance:

- chat/code replies are generated by the AgentKernel Lite model bundle
- malformed model output does not break the loop
- model decisions roundtrip into episode records for future distillation

### Phase 5: Extension Action Bus

Goal: user-controlled browser extensions expand actions safely.

Tasks:

- define extension manifest schema: id, name, capabilities, scopes, approval
  policy, input/output schema
- add `register_extension_manifest`
- add `propose_extension_action`
- add approval modes: always ask, remember for session, disabled, trusted local
- add receipts for approved, denied, failed, and timed-out actions

Acceptance:

- core can propose an extension action without executing it
- UI can approve/deny and return a receipt
- denied extension actions are preserved in memory as feedback

### Phase 6: Lite Verification

Goal: browser artifacts can be checked without shell authority.

Tasks:

- port expected artifact/content checks from `Verifier`
- add output substring checks for model/extension receipts
- add deterministic artifact hash checks
- expose `verify_artifact_contract`

Acceptance:

- browser tasks can verify generated artifacts that live in OPFS/IndexedDB
- verification results use the same `passed/reasons/failure_codes` shape as
  Python where possible

### Phase 7: PWA And Sync

Goal: make AgentKernel Lite durable on device.

Tasks:

- add PWA manifest and storage persistence flow
- add workspace export/import
- add optional Hugging Face bundle sync
- add model/runtime version attestation
- add storage pressure reporting

Acceptance:

- user can reload the browser and keep memory/artifacts
- user can export a full workspace bundle
- runtime attestation identifies model, core, packs, and storage mode

### Phase 8: Native/C++ Runtime Option

Goal: keep C++ available for native or future browser tensor acceleration.

Tasks:

- keep model-stack C++/CUDA as native authority for server/H100
- benchmark browser JS/WGSL model-stack runtime first
- only pursue C++/Emscripten/WebGPU if orchestration dominates runtime

Acceptance:

- decision is based on measured browser profile, not speculation

## First Implementation Slice

The next concrete slice should be small:

1. Create repo-owned `agent_kernel_rust_wasm/` from the current static-export
   Rust core.
2. Add `schemas.rs` with `TaskSpecLite`, `ActionDecisionLite`, and
   `EpisodeRecordLite`.
3. Add `extensions.rs` with extension manifest and action receipt structs only.
4. Add tests for JSON roundtrip and action proposal receipts.
5. Add a sync/build script that refreshes the static export package.

This gives us a stable foundation for memory/artifacts/extensions without
rewriting the model runtime or UI in the same step.

## Current Implementation Status

Started on 2026-04-28:

- repo-owned crate exists at `agent_kernel_rust_wasm/`
- current static-export core was copied in as the baseline
- typed Rust modules now exist for:
  - `actions.rs`: browser-safe action kinds and append-only action ledger records
  - `schemas.rs`: `TaskSpecLite`, `ActionDecisionLite`, `StepRecordLite`,
    `EpisodeRecordLite`, `RuntimeConfigLite`, evidence and turn records
  - `extensions.rs`: extension manifests, capability declarations, approval
    policy, and action receipts
- `AgentLiteCore` now exposes WASM methods for:
  - `register_extension_manifest`
  - `propose_extension_action`
  - `record_extension_result`
  - `rank_evidence`
  - `compile_context_packet`
  - `start_turn_with_context`
  - `parse_model_decision`
  - `finish_model_reply`
- the Rust context compiler now emits the Python-shaped packet keys
  `task`, `control`, `tolbert`, `retrieval`, and `verifier_contract`
- the browser worker has a model-stack browser-bitnet encoder-decoder bridge
  behind `modelstack:<manifest-url>` / `?modelStackManifest=...`
  - `export_episode_json`
  - `runtime_attestation`
  - `finish_model_reply`
- `scripts/build_agent_kernel_rust_wasm.py` builds/syncs the repo-owned crate
  into `/data/repository_library/exports/agent_kernel/wasm/agent_kernel_lite_core`
  without deleting existing export files.
- `scripts/build_agentkernel_lite_encdec_dataset.py` now supports a chat-first
  `--objective chat` mode for browser AgentKernel Lite distillation.
- `scripts/train_agentkernel_lite_encdec.py` now emits a chat bundle manifest
  whose primary action is `respond` and whose code execution authority is false.

## Chat-First Distillation Target

AgentKernel Lite should first be a strong local chat kernel. Code generation is
an extension capability, not the default browser action surface. That keeps the
on-device loop simple enough for a 100M-class encoder-decoder model while still
preserving a path to user-approved coding tools later.

The browser runtime now exposes three response-depth modes over the same chat
surface:

| Mode | Runtime Intent | Retrieval Shape |
| --- | --- | --- |
| `chat` | Fast grounded answer. | 5 selected evidence items, compact excerpts, medium generation. |
| `think` | Semantic synthesis before answering. | 8 selected evidence items, larger excerpts, lower temperature. |
| `deep_research` | Heaviest evidence-by-evidence analysis. | 14 selected evidence items, largest excerpts, lowest temperature. |

These are not tool-authority modes. All three still keep code execution
disabled. They only change retrieval breadth, context chunking, prompt policy,
and generation settings. The intended loop is:

```text
user message -> research/library retrieval -> mode-aware context compiler
-> 100M chat model -> grounded answer
```

For model training, `chat` is the current supervised target. `think` and
`deep_research` should be trained next with research-library grounded examples:
question plus selected paper/repo evidence to semantic synthesis, and
question plus larger evidence set to evidence-by-evidence final answer.

Dataset builder contract:

```text
python scripts/build_agentkernel_lite_encdec_dataset.py \
  --objective chat \
  --code-trace-mode explain
```

The chat objective converts every target into a compact structured JSON decision
with `action=respond`. Normal response traces become direct chat targets. Code
or shell traces are not trained as executable actions; they are translated into
lower-weight chat explanations with `extension_capability=code.generate_draft`.
`--code-trace-mode skip` drops those traces, and `--code-trace-mode raw` keeps
their raw content only as extension-conditioned chat data.

The trainer default tokenizer is currently `byte` because the browser
model-stack bridge can encode/decode byte tokens today. The AgentKernel BPE path
remains available for a better model later, but the browser worker needs BPE
tokenizer support before BPE-trained bundles are the default runtime target.

The `agentkernel-lite-100m` preset is currently:

```text
d_model=640
n_heads=10
n_layers=6
d_ff=2048
vocab_size=260 byte tokens
parameter_count=100,713,728
```

The trainer materializes model-stack lazy decoder self-attention modules before
counting parameters or constructing the optimizer. Without that step, decoder
self-attention parameters exist during the first forward pass but are missing
from optimizer state.

Training command shape:

```text
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_encdec.py \
  --dataset-manifest artifacts/agentkernel_lite_encdec/dataset/agentkernel_lite_encdec_dataset_manifest.json \
  --output-dir artifacts/agentkernel_lite_encdec/run_100m_chat \
  --preset agentkernel-lite-100m \
  --tokenizer-kind byte \
  --device cuda:1 \
  --dry-run 0 \
  --checkpoint-every 100 \
  --eval-every 100 \
  --max-eval-batches 8 \
  --export-browser-bitnet 1
```

Use the `ai` conda environment for this path. The base environment currently
resolves `torch` as a namespace package without `torch.nn`, so it cannot run the
trainer. If vLLM or another model server is occupying the GPU, CPU smoke tests
will pass but CUDA training will fail at optimizer allocation until GPU memory is
freed. On 2026-04-28, GPU 0 was occupied by `VLLM::EngineCore`, while GPU 1 was
free and successfully ran the 100M CUDA smoke.

Resume command shape:

```text
/home/peyton/miniconda3/envs/ai/bin/python scripts/train_agentkernel_lite_encdec.py \
  --dataset-manifest artifacts/agentkernel_lite_encdec/chat_train_dataset/agentkernel_lite_encdec_dataset_manifest.json \
  --output-dir artifacts/agentkernel_lite_encdec/run_100m_chat \
  --preset agentkernel-lite-100m \
  --tokenizer-kind byte \
  --device cuda:1 \
  --dry-run 0 \
  --resume-latest 1
```

Validation completed:

- `cargo test` in `agent_kernel_rust_wasm`: 12 passed
- `wasm-pack build --target web --release` in `agent_kernel_rust_wasm`: passed
- sync script with `--no-build`: passed
- `cargo test` in the synced static export crate: 12 passed
- `node --check` for the static app JS: passed
- dataset smoke build with `--objective chat --code-trace-mode explain`: passed
- trainer dry-run with `--preset tiny --tokenizer-kind byte`: passed in the
  `ai` conda environment
- full chat training dataset build: 4,914 examples, all target `respond`
- 100M dry-run: 100,713,728 parameters
- 100M one-step CPU train smoke with 64-token encoder/decoder windows: passed
- checkpoint/eval/resume tiny smoke: passed
- 100M one-step CUDA train smoke on `cuda:1` with eval: passed
- 100M CUDA on default `cuda` was blocked by occupied GPU 0 memory, not by
  trainer code

## Guardrails

- Keep adapters detachable; the core must run with no extensions registered.
- Never let a model output execute an extension directly.
- Store every proposed, approved, denied, or failed action as an episode event.
- Do not pass massive datasets into WASM; pass selected candidates and manifests.
- Preserve Python kernel compatibility for training data and benchmark traces.
- Keep browser verification separate from benchmark-grade Python verification.
