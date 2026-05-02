# Agent Kernel Lite Static Export

Deploy this folder at:

```text
https://peytontolbert.com/agent_kernel/
```

The app is a browser-only chat demo. It does not require signups, accounts,
code execution, or server-side inference. The browser downloads:

- an on-device BitNet model manifest, tensors, and tokenizer from Hugging Face
- app-hosted Model Stack BitNet runtime modules from `vendor/model-stack-bitnet/`
- selected paper metadata packs from `PeytonT/paper_universe_interactive`
- the distilled M1 paper embedding model from
  `PeytonT/1m-paper-embedding-model-lite-onnx`
- row-aligned M1 semantic vectors from
  `PeytonT/paper_universe_interactive/semantic_m1`
- paper record details from the Hugging Face Dataset Viewer API for
  `PeytonT/1m_papers_text`

## Current Scope

This is intentionally a lite demo:

- simple chat interface
- three response-depth modes over the same chat surface:
  - `Chat`: fast grounded answer over a compact evidence window
  - `Think`: semantic synthesis over a wider evidence window
  - `Deep`: heaviest evidence-by-evidence research pass
- local browser model execution
- code drafting is modeled as an extension capability rather than a primary UI
  mode, and generated code is not run by the app
- local hybrid retrieval over downloaded paper packs: M1 semantic ranking first,
  keyword fallback when the embedding path is unavailable, and evidence cards
  with abstracts plus arXiv PDF links when paper identifiers are available
- chat-side paper selection: clicking **Load Full Text** fetches the selected
  row from `PeytonT/1m_papers_text`, displays a local full-text preview, and
  keeps the paper in subsequent browser-side context
- Hugging Face Dataset Viewer fallback when no pack is loaded
- best-effort browser Cache API and IndexedDB storage
- optional persistent storage request

The production model source is:

```text
https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet/resolve/main/manifest.json
```

The browser does not import executable JavaScript modules from Hugging Face.
Following the research-library export pattern, executable runtime code is served
with the app and Hugging Face is used for model/data assets.

## WASM Agent Core

The browser app includes only the built Rust/WASM package at:

```text
wasm/agent_kernel_lite_core/pkg/
```

The core owns the browser-safe agent loop pieces: turn state, step budget,
retrieved-evidence ranking, Python-shaped context packet compilation,
prompt/context compilation, model decision parsing, message history, extension
action ledgers, and chat state. JavaScript remains the adapter for UI, Hugging
Face downloads, storage, and model-stack browser-bitnet inference. If the WASM
core cannot load, the app falls back to the previous JavaScript prompt compiler.

By default the browser worker loads the model-stack browser BitNet
encoder-decoder bundle from:

```text
https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet/resolve/main/manifest.json
```

For development only, the worker can load a different model-stack browser BitNet
encoder-decoder bundle when the page is opened with:

```text
?modelStackManifest=https://.../manifest.json
```

That path expects a model-stack `browser-bitnet` manifest for tensor paths and
tokenizer metadata. The worker ignores remote executable runtime paths and loads
the app-hosted runtime from `vendor/model-stack-bitnet/` to avoid strict browser
module MIME checks on Hugging Face `resolve` URLs. The Rust core still parses
and governs the generated text through `finish_model_reply`; code execution
remains disabled in the browser. The app registers `code.generate_draft` as an
extension capability so code generation can be enabled later through the
extension action bus.

## Browser Runtime Notes

Older temporary Transformers.js/ONNX demos may print a warning like:

```text
Some nodes were not assigned to the preferred execution providers
```

That warning is expected for some ONNX graphs and is not the target deployment
path. The AgentKernel Lite target is a model-stack browser-bitnet encoder-decoder
bundle using app-hosted JavaScript/WASM orchestration plus WGSL/WebGPU kernels
when available.
