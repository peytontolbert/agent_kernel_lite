# Agent Kernel Lite Mobile App

Agent Kernel Lite mobile should be a native shell for the web runtime, not a
desktop bridge controller.

## Shape

```text
iPhone / Android app
  -> bundled Capacitor boot shell
  -> bundled Agent Kernel Lite fallback app
  -> GitHub Release manifest loader
  -> Hugging Face model and paper pack downloads
  -> local WebView execution
```

The bundled app must be useful offline for Search mode and any already cached
packs. New model/data packs can be downloaded after explicit user action.

## Trust Boundaries

- GitHub Releases are the source for executable app assets and extension
  manifests.
- Hugging Face is the source for model/data assets only.
- Branch URLs, `latest` release URLs, and raw mutable URLs are rejected.
- The app does not navigate the WebView to arbitrary user-entered URLs.
- Cached assets are verified against SHA-256 hashes when the manifest provides
  them.

## Why This Differs From Mobile Computer Use

Mobile Computer Use is a companion controller. Its native app collects a bridge
URL, checks `/health`, and opens the bridge-served `/mobile` page.

Agent Kernel Lite is the app. It should boot locally and fetch versioned content
from GitHub/Hugging Face. That keeps the app review surface, offline behavior,
and user trust model separate from terminal-control workflows.

## First Release Priorities

- Capacitor app scaffold with iOS/Android projects generated from `apps/mobile`.
- `www/app/` generated from the repo `web/` folder.
- Search mode as the first reliable mobile workflow.
- Cache manager for release manifests and large Hugging Face packs.
- Clear model/data download size prompts before downloading large assets.
- WASM CPU fallback as the baseline; WebGPU should be an optimization, not the
  minimum viable path.
