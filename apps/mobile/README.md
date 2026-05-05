# Agent Kernel Lite Mobile

This is the iOS/Android shell for Agent Kernel Lite.

The app is intentionally different from Mobile Computer Use:

- Mobile Computer Use opens a trusted desktop bridge URL.
- Agent Kernel Lite opens a bundled local app and downloads versioned content.

The native shell keeps a small boot UI in `www/`, copies the browser app into
`www/app/`, and treats GitHub Releases plus Hugging Face as content sources:

- GitHub Releases: app shell assets, WASM bindings, extension manifests, hashes.
- Hugging Face: model tensors, tokenizers, paper packs, embeddings, full-text rows.

Executable app assets must come from immutable GitHub Release URLs. Hugging Face
is for model and data files, not remote JavaScript execution.

## Setup

```bash
cd apps/mobile
npm install
npm run sync:web
```

Open the bundled shell in a browser:

```bash
cd www
python3 -m http.server 8799 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8799/
```

## Native Projects

After installing dependencies:

```bash
npm run add:ios
npm run add:android
npm run sync
```

The generated native projects should stay thin. Keep the Agent Kernel Lite UI
in `web/`, then run `npm run sync:web` to refresh `www/app/`.

Android debug builds need a JDK plus `ANDROID_HOME` pointing at an installed
Android SDK:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export ANDROID_HOME=/path/to/android/sdk
npm run build:android:debug
```

iOS builds require macOS with Xcode and CocoaPods. The project can be generated
on other platforms, but `pod install` and `xcodebuild` will be skipped.

## Release Loading

The boot shell defaults to:

```text
https://github.com/peytontolbert/agent_kernel_lite/releases/download/v6/app-release-manifest.json
```

It accepts only immutable GitHub release manifest URLs. Listed executable assets
must also be immutable GitHub release assets. Non-executable model/data assets
may be Hugging Face `/resolve/` URLs.
