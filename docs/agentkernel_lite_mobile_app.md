# Agent Kernel Lite Mobile App

Agent Kernel Lite mobile should be a native shell for the web runtime, not a
desktop bridge controller.

## Shape

```text
iPhone / Android app
  -> Capacitor WebView
  -> bundled Agent Kernel Lite app
  -> bundled BitNet model
  -> bundled 50k paper metadata pack
  -> local WebView execution
```

The bundled app must be useful on first launch. Search mode should work from the
packaged 50k paper pack, and the default BitNet runtime should load from local
app assets.

## Trust Boundaries

- GitHub Actions/TestFlight/App Store are the delivery path for native app
  updates.
- Hugging Face is the upstream source for refreshed or larger model/data assets.
- The app does not navigate the WebView to arbitrary user-entered URLs.
- Remote model/data downloads should be explicit, size-disclosed, and cached.

## Why This Differs From Mobile Computer Use

Mobile Computer Use is a companion controller. Its native app collects a bridge
URL, checks `/health`, and opens the bridge-served `/mobile` page.

Agent Kernel Lite is the app. It should boot locally from the app bundle. That
keeps the app review surface, offline behavior, and user trust model separate
from terminal-control workflows.

## First Release Priorities

- Capacitor app scaffold with iOS/Android projects generated from `apps/mobile`.
- `www/app/` generated from the repo `web/` folder.
- Bundled `agentkernel_lite_100m_bitnet_12000` model.
- Bundled 50k paper metadata pack prepared from Hugging Face in CI.
- Search mode as the first reliable mobile workflow.
- Clear model/data download size prompts before downloading optional larger
  assets.
- WASM CPU fallback as the baseline; WebGPU should be an optimization, not the
  minimum viable path.
