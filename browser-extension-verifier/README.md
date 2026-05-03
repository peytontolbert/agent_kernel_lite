# Agent Kernel Lite Verifier Extension

This browser extension verifies the live Agent Kernel Lite web app against a
GitHub Release. It runs outside the page JavaScript context, fetches
`SHA256SUMS` directly from the selected release, hashes the served app assets,
and reports whether they match.

## What It Checks

For the active tab, the verifier hashes:

- `index.html`
- `js/agent-kernel-app.js`
- `wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core.js`
- `wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core_bg.wasm`

It compares those hashes with the release's `SHA256SUMS` asset, for example:

```text
https://github.com/peytontolbert/agent_kernel_lite/releases/download/v2/SHA256SUMS
```

The extension rejects mutable release names such as `latest`, `main`, and
`master`. Use a concrete release tag such as `v2`.

## Install For Local Testing

Chrome / Edge:

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Choose **Load unpacked**.
4. Select this `browser-extension-verifier` directory.
5. Open Agent Kernel Lite.
6. Click the extension icon and press **Verify**.

Firefox:

1. Open `about:debugging#/runtime/this-firefox`.
2. Choose **Load Temporary Add-on**.
3. Select `manifest.json` in this directory.
4. Open Agent Kernel Lite and run verification from the extension icon.

## Security Model

The verifier is intentionally separate from Agent Kernel Lite's in-app
Extensions menu. The web app cannot mark itself as verified. The browser
extension performs its own network fetches and hashing from the extension
context.

This verifies the static app shell assets against a release. It does not verify:

- browser extensions installed outside this verifier
- model weights or paper packs downloaded from Hugging Face
- private user data in IndexedDB/localStorage
- runtime behavior after a verified asset is executed by the browser

Those are separate trust surfaces.
