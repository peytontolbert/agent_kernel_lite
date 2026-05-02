# Agent Kernel Lite

Standalone browser-first Agent Kernel Lite workspace.

This repository was split out of `/data/agentkernel` so the lite research assistant,
custom BitNet browser runtime, and Rust/WASM core can evolve independently from the
larger Agent Kernel project.

## Live App

The public browser deployment lives at:

```text
https://peytontolbert.com/agent_kernel/
```

The current production model bundle is hosted on Hugging Face:

```text
https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet
```

Direct model-stack manifest URL:

```text
https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet/resolve/main/manifest.json
```

The app serves executable JavaScript/WASM runtime files from this repository.
Hugging Face is used for model tensors, tokenizers, paper metadata, embeddings,
and full-text paper rows.

## Hugging Face Assets

Runtime model:

- `PeytonT/agentkernel-lite-100m-bitnet` - Agent Kernel Lite 100M BitNet encoder-decoder model bundle.

Browser research retrieval:

- `PeytonT/paper_universe_interactive` - paper metadata packs used by the browser.
- `PeytonT/paper_universe_interactive/semantic_m1` - row-aligned semantic vector packs.
- `PeytonT/1m-paper-embedding-model-lite-onnx` - lite paper embedding model used for browser-side ranking.
- `PeytonT/1m_papers_text` - 1M full-text paper dataset accessed through the Hugging Face Dataset Viewer API.

Research-library training/rebuild assets referenced by docs and scripts:

- `PeytonT/repo_graph` - repository graph dataset for code/repository retrieval experiments.
- `PeytonT/1m-papers-abstract-keywords` - paper keyword extraction/refresh dataset referenced by integration docs.

## Browser Decode Speed

Latest local browser WASM benchmark after the custom BitNet decoder-kernel
optimizations:

| Encoder context | Total decode speed | Steady decode speed |
| --- | ---: | ---: |
| 66 tokens | ~368 tok/s | ~408 tok/s |
| 130 tokens | ~360 tok/s | ~413 tok/s |
| 258 tokens | ~275 tok/s | ~334 tok/s |
| 514 tokens | ~176 tok/s | ~226 tok/s |

Full browser-worker path for a 64-token generation measured about `500ms` in
the local test harness. These numbers were measured on the local development
machine in browser WASM; iPhone/Safari numbers should be benchmarked separately
as the production mobile target.

## Layout

- `web/` - runnable browser app export, including local model runtime assets.
- `web/models/agentkernel_lite_100m_bitnet_v11/` - current local model bundle.
- `web/vendor/model-stack-bitnet/` - browser runtime used by the app.
- `wasm/agent_kernel_lite_core/` - Rust/WASM agent-kernel-lite core.
- `model-stack/browser/bitnet/` - model-stack browser BitNet JS runtime source.
- `model-stack/browser/bitnet_wasm/` - custom Rust/WASM BitNet kernels.
- `scripts/` - lite training, export, evaluation, and dataset utilities.
- `docs/` - architecture and training notes for Agent Kernel Lite.
- `tests/` - targeted lite tests copied from the parent workspace.

## Run Locally

```bash
cd agent_kernel_lite/web
python3 -m http.server 8797 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8797/?localModel=1
```

## Rebuild WASM BitNet Kernel

```bash
cd agent_kernel_lite/model-stack/browser/bitnet_wasm
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release --out-dir ../bitnet/pkg
cp ../bitnet/pkg/model_stack_bitnet_wasm.js ../../../../web/vendor/model-stack-bitnet/
cp ../bitnet/pkg/model_stack_bitnet_wasm_bg.wasm ../../../../web/vendor/model-stack-bitnet/
cp ../bitnet/pkg/model_stack_bitnet_wasm.js ../../../../web/models/agentkernel_lite_100m_bitnet_v11/runtime/
cp ../bitnet/pkg/model_stack_bitnet_wasm_bg.wasm ../../../../web/models/agentkernel_lite_100m_bitnet_v11/runtime/
```

## Rebuild Rust Agent Core WASM

```bash
cd agent_kernel_lite/wasm/agent_kernel_lite_core
wasm-pack build --target web --release
```

The browser app expects the built package under:

```text
web/wasm/agent_kernel_lite_core/pkg/
```

## Notes

Related public repositories and datasets:

- `model-stack`: https://github.com/peytontolbert/model-stack
- `Research_Library`: https://github.com/peytontolbert/Research_Library
- `PeytonT/1m_papers_text`: https://huggingface.co/datasets/PeytonT/1m_papers_text

Large research datasets and model checkpoints are intentionally not vendored
directly into this repository.
