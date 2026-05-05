# Agent Kernel Lite

Agent Kernel Lite is a browser-first local research assistant focused on the
BitNet/model-stack runtime, paper retrieval, and a small Rust/WASM agent core.
It pairs browser-side model execution with local research retrieval, portable
session backup, and an external verifier extension for checking the app assets
users are running.

This repository was split out of
`https://github.com/peytontolbert/agent_kernel` so the lite browser app,
model-stack runtime, and Rust/WASM core can evolve independently from the larger
Agent Kernel project.

## Live App

Public deployment:

```text
https://peytontolbert.com/agent_kernel/
```

Current release:

```text
https://github.com/peytontolbert/agent_kernel_lite/releases/tag/v6
```

The app shell is intended to be served from the website while release assets are
pinned to GitHub Releases. The app does not load executable JavaScript from
Hugging Face. Hugging Face is used for model/data assets such as model tensors,
tokenizers, paper packs, embeddings, and full-text paper rows.

## v6 Scope

The v6 shell includes:

- browser chat UI with `Chat`, `Think`, and `Deep` modes
- Rust/WASM core for turn state, context packets, model decision parsing,
  extension manifests, extension action proposals, receipts, and snapshots
- browser-side BitNet/model-stack runtime integration
- local paper retrieval over downloaded paper metadata/vector packs
- decoder-free quick paper search over the loaded research pack
- selected-paper context persistence inside the session
- extension menu with **Installed** and **Available** sections
- release-only custom extension manifest install
- localhost development exception for extension testing
- portable session export/import for restoring local focus sessions
- app hash display in Status
- separate browser extension verifier package

No extensions are default-installed in v6. The app shows available official
extension manifests from `web/extensions/catalog.json`; the catalog is empty
until an Agent Kernel Lite extension is release-ready.

Users must click **Install**, then explicitly enable an installed extension
before it can act. Image Generation is kept out of the official v6 catalog
until it is release-ready.

Mobile Computer Use is now a separate application. Use the standalone project
for mobile terminal and Codex control:

```text
https://github.com/peytontolbert/mobile-computer-use
```

## Verification

The app Status panel computes an app hash from the served shell assets:

- `index.html`
- `js/agent-kernel-app.js`
- `wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core.js`
- `wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core_bg.wasm`

Release assets include `SHA256SUMS` so users can verify the website against the
release.

For stronger verification, install the separate browser extension in:

```text
browser-extension-verifier/
```

The verifier runs outside the web app, fetches `SHA256SUMS` directly from the
selected GitHub Release, hashes the live assets from the active tab, and reports
pass/fail. The v6 release includes:

```text
agent-kernel-lite-verifier-v6.tar.gz
verifier-SHA256SUMS
```

## Extensions

Agent Kernel Lite has two extension concepts:

- **In-app Agent Kernel extensions**: browser app capabilities represented by
  manifests, installed into the local app, then enabled/disabled by the user.
- **Browser verifier extension**: an actual browser extension users install to
  verify the web app independently from the page.

In-app extension manifests are installed from:

- the app's available catalog at `web/extensions/catalog.json`
- custom GitHub Release asset URLs
- localhost URLs during development

Production custom installs reject mutable branch/latest URLs such as:

```text
raw.githubusercontent.com/.../main/...
github.com/.../blob/main/...
latest
```

Use immutable release assets such as:

```text
https://github.com/owner/repo/releases/download/v0.1.0/extension.json
```

An installed extension still starts disabled. Adapter code owns execution; model
output cannot execute extensions directly.

## Session Backup

The web app can export/import a portable JSON session bundle. The bundle stores:

- UI settings: theme, mode, token length, selected model/device, paper pack size
- installed extension manifests and enabled/disabled state
- chat messages and selected paper context rows
- app-scoped `localStorage`
- small IndexedDB metadata records
- app integrity metadata when available

Large Cache API entries are intentionally not exported. Model weights, ONNX
files, paper packs, and vector indexes should be redownloaded from their
manifests. This keeps the backup practical for Safari and for future iOS/Android
imports.

## Release Assets

The v6 GitHub Release publishes individual shell assets and a tarball:

```text
index.html
agent_kernel.webmanifest
agent-kernel-app.js
llm-worker.js
model-stack-bitnet-vendor.tar.gz
agent_kernel_lite_core.js
agent_kernel_lite_core_bg.wasm
app-release-manifest.json
catalog.json
agent-kernel-lite-v6-shell.tar.gz
SHA256SUMS
agent-kernel-lite-verifier-v6.tar.gz
verifier-SHA256SUMS
```

`web/app-release-manifest.example.json` documents the release manifest shape for
future shell bootstrapping.

## Hugging Face Assets

Runtime model:

- `PeytonT/agentkernel-lite-100m-bitnet`

Direct model-stack manifest:

```text
https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet/resolve/main/manifest.json
```

Browser research retrieval:

- `PeytonT/paper_universe_interactive`
- `PeytonT/paper_universe_interactive/semantic_m1`
- `PeytonT/1m-paper-embedding-model-lite-onnx`
- `PeytonT/1m_papers_text`

Related training/rebuild assets referenced by docs/scripts:

- `PeytonT/repo_graph`
- `PeytonT/1m-papers-abstract-keywords`

Large research datasets and model checkpoints are intentionally not vendored
directly into this repository.

## iPhone / Android App

The native app scaffold lives in:

```text
apps/mobile/
```

The mobile app is a small Capacitor shell, not a desktop bridge controller. It
boots a bundled copy of the Agent Kernel Lite web app from `apps/mobile/www/app/`
and can cache versioned content from GitHub Releases and Hugging Face:

- GitHub Releases provide executable app assets, WASM bindings, manifests, and
  hashes.
- Hugging Face provides model tensors, tokenizers, paper packs, embeddings, and
  full-text paper data.

Refresh the bundled app after editing `web/`:

```bash
cd agent_kernel_lite/apps/mobile
npm run sync:web
```

Then generate/sync native projects with Capacitor:

```bash
npm install
npm run add:ios
npm run add:android
npm run sync
```

See `apps/mobile/README.md` and
`docs/agentkernel_lite_mobile_app.md` for the trust model and release-loading
rules.

## Layout

- `apps/mobile/`: Capacitor iOS/Android shell for the web app
- `web/`: runnable browser app export
- `web/extensions/`: available in-app extension catalog
- `web/wasm/agent_kernel_lite_core/pkg/`: browser-loaded Rust/WASM core package
- `wasm/agent_kernel_lite_core/`: Rust source for the agent core
- `web/vendor/model-stack-bitnet/`: app-hosted model-stack runtime
- `model-stack/browser/bitnet/`: model-stack browser BitNet JS runtime source
- `model-stack/browser/bitnet_wasm/`: custom Rust/WASM BitNet kernels
- `browser-extension-verifier/`: installable browser extension for release hash
  verification
- `scripts/`: training, export, evaluation, and utility scripts
- `docs/`: architecture, integration, and planning notes
- `tests/`: targeted lite tests

## Run Locally

```bash
cd agent_kernel_lite/web
python3 -m http.server 8797 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8797/
```

Localhost is allowed for development extension installs. Production installs are
release-only.

## Rebuild Rust Agent Core WASM

```bash
cd agent_kernel_lite/wasm/agent_kernel_lite_core
wasm-pack build --target web --release
rsync -a --delete pkg/ ../../web/wasm/agent_kernel_lite_core/pkg/
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

## Browser Decode Speed

Latest local browser WASM benchmark after custom BitNet decoder-kernel
optimizations:

| Encoder context | Total decode speed | Steady decode speed |
| --- | ---: | ---: |
| 66 tokens | ~368 tok/s | ~408 tok/s |
| 130 tokens | ~360 tok/s | ~413 tok/s |
| 258 tokens | ~275 tok/s | ~334 tok/s |
| 514 tokens | ~176 tok/s | ~226 tok/s |

Full browser-worker path for a 64-token generation measured about `500ms` in
the local test harness. iPhone/Safari numbers should be benchmarked separately.

## Related Projects

- `model-stack`: https://github.com/peytontolbert/model-stack
- `Research_Library`: https://github.com/peytontolbert/Research_Library
- `PeytonT/1m_papers_text`: https://huggingface.co/datasets/PeytonT/1m_papers_text
