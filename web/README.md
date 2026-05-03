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
- browser-installed extensions. The v6 shell ships with no default-installed
  extensions; users install release manifests before enabling capabilities.
- portable session export/import for restoring a private local focus session
  across Safari/browser installs.

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
remains disabled in the browser.

## Browser Extensions

Extensions live behind the **Extensions** menu. They are installed into the
browser adapter by manifest, then enabled or disabled independently from their
card. Clicking a card expands setup details, declared capabilities, scopes, and
install source. Installing an extension only makes its capabilities available
for approval; it does not enable execution. This keeps external surfaces such
as YouTube, Discord, X, or local user tools out of the minimal kernel unless the
user imports them and completes setup.

The v6 app install includes no default-installed extensions. The Extensions menu
does display the official extensions available from `web/extensions/catalog.json`
so users can click to install them. Installing only adds the manifest to the
installed surface; the user still has to enable the extension before it can act.
For v6, the official catalog exposes Computer Use only. Image Generation remains
a development/future extension and is not advertised by the official catalog.
The app shell also constrains fetched catalog entries to the pinned release allowlist
so a stale release catalog cannot re-advertise development extensions.
For hosted mobile use, Computer Use can point at a relay bridge URL under
`/agent_kernel/api/relay/bridge/route_...`; the desktop bridge connects outbound
to the relay and all post-pairing bridge messages remain encrypted end to end.
The Computer Use manifest shape is:

```json
{
  "id": "computer_use",
  "name": "Computer Use",
  "source": "official",
  "default_enabled": false,
  "approval_policy": "trusted_local",
  "capabilities": [
    {
      "id": "computer.session.start",
      "description": "Launch and manage a paired local computer agent terminal.",
      "scopes": ["workspace.read", "computer.session.write"]
    }
  ]
}
```

Custom extensions use the same shape. A user extension should declare a stable
`id`, user-facing `name`, capability IDs, scope strings, and adapter metadata:

```json
{
  "id": "youtube_tools",
  "name": "YouTube Tools",
  "version": "0.1.0",
  "source": "user",
  "approval_policy": "always_ask",
  "capabilities": [
    {
      "id": "youtube.search",
      "description": "Search YouTube through a user-provided adapter.",
      "scopes": ["network.youtube", "results.read"]
    }
  ],
  "metadata": {
    "adapter": "browser",
    "homepage": "https://github.com/example/youtube_tools",
    "adapter_url": "https://github.com/example/youtube_tools/releases/download/v0.1.0/adapter.js"
  }
}
```

Extension installs are release-only. The browser accepts manifest URLs from
GitHub Release assets, for example:

```text
https://github.com/owner/repo/releases/download/v0.1.0/extension.json
```

It rejects branch URLs such as `raw.githubusercontent.com/.../main/...`,
`github.com/.../blob/main/...`, and `latest` URLs. If a manifest points to
adapter code, worker code, or a module URL, those code assets must also be
GitHub Release asset URLs. This keeps the app pinned to immutable release
artifacts instead of whatever the latest commit on `main` happens to be.

When the app is loaded, browser code can install and manage manifests through:

```js
await window.AgentKernelExtensions.installFromUrl(
  'https://github.com/owner/repo/releases/download/v0.1.0/extension.json'
);
window.AgentKernelExtensions.enable('youtube_tools');
window.AgentKernelExtensions.disable('youtube_tools');
window.AgentKernelExtensions.list();
```

The core records proposed extension actions and receipts, but adapter code owns
actual execution. A model output cannot execute an extension directly.

## Release-Only App Loading

Production should treat `peytontolbert.com/agent_kernel/` as a small stable
shell. The shell should load app code, WASM bindings, WASM bytes, and extension
manifests from tagged GitHub Release assets, not from `main` or raw branch URLs.
The example release manifest in `app-release-manifest.example.json` captures the
intended shape.

That split keeps the hosted shell small while allowing the release assets to be
cached independently. It also means an extension is not part of the installed
surface until the user installs a release manifest. Localhost development URLs
are still allowed by the browser installer so extension work can be tested before
publishing a release.

The Status panel shows an app hash. It is a SHA-256 digest derived from the
served `index.html`, app JavaScript, WASM binding JavaScript, and WASM bytes.
The full per-asset hashes are included in exported session bundles under
`app.integrity`. Release automation should publish the same hashes beside the
GitHub Release assets so users can compare the running app with the release they
intended to install.

For manipulation-resistant verification, users can install the separate
`browser-extension-verifier` package from this repository. That extension runs
outside the web app, fetches release `SHA256SUMS` directly from GitHub, hashes
the live app assets from the active tab, and reports whether the served app
matches the selected release.

## Session Backup

The browser can export a portable JSON bundle from **Export Session** and restore
it with **Import Session**. The bundle contains:

- UI settings: theme, mode, token length, selected model/device, selected paper
  pack size, and extension mode state
- chat messages and selected paper context rows
- installed extension manifests and enabled/disabled state
- app-scoped `localStorage` keys
- small IndexedDB metadata records used by the browser adapter

The export intentionally does not include large Cache API entries such as model
weights, downloaded paper packs, ONNX files, or vector indexes. Those assets are
restored by URL/manifests and can be downloaded again after import. This keeps
the file small enough for iPhone Safari sharing and gives a native iOS/Android
app a stable bridge format for importing browser sessions later.

## Native App Direction

The iOS and Android app should treat this JSON session file as the first import
format. The native shell can keep the same kernel concepts while replacing
browser-limited adapters with stronger platform capabilities:

- app-owned private document storage instead of Safari best-effort storage
- background downloads for model/data packs
- file-provider/document-picker import and export
- optional native extension bridges for contacts, share sheets, notifications,
  local files, and platform-secure credentials
- the same install/enable/receipt extension contract used by the browser

That keeps Safari, iPhone, and Android sessions interoperable while letting the
native apps add capabilities that browsers restrict.

## Browser Runtime Notes

Older temporary Transformers.js/ONNX demos may print a warning like:

```text
Some nodes were not assigned to the preferred execution providers
```

That warning is expected for some ONNX graphs and is not the target deployment
path. The AgentKernel Lite target is a model-stack browser-bitnet encoder-decoder
bundle using app-hosted JavaScript/WASM orchestration plus WGSL/WebGPU kernels
when available.
