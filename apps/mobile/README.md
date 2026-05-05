# Agent Kernel Lite Mobile

This is the iOS/Android shell for Agent Kernel Lite.

The app is intentionally different from Mobile Computer Use:

- Mobile Computer Use opens a trusted desktop bridge URL.
- Agent Kernel Lite opens directly into a bundled local app.

The native package copies the browser app into `www/app/` and bundles the
default runtime assets:

- `web/models/agentkernel_lite_100m_bitnet_12000/`
- the 50k paper metadata pack from Hugging Face

GitHub Actions/TestFlight/App Store deliver native app updates. Hugging Face
remains the upstream source for refreshed or larger model/data packs, not a
remote application shell for the native app.

## Setup

```bash
cd apps/mobile
npm install
npm run prepare:assets
npm run sync:web
```

Open the bundled app in a browser:

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

The generated native projects should stay thin. Keep the Agent Kernel Lite UI in
`web/`, run `npm run prepare:assets` to fetch packaged data, then run
`npm run sync:web` to refresh `www/app/`.

Android debug builds need a JDK plus `ANDROID_HOME` pointing at an installed
Android SDK:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export ANDROID_HOME=/path/to/android/sdk
npm run build:android:debug
```

iOS builds require macOS with Xcode and CocoaPods. The project can be generated
on other platforms, but `pod install` and `xcodebuild` will be skipped.

## Packaged Assets

The native app opens:

```text
www/app/?native=1&autoload=1&autopack=1
```

That path loads the bundled BitNet model and bundled 50k paper pack first.
Larger paper packs and refreshed model/data assets can still use Hugging Face
URLs when the app adds an explicit update/download flow.
