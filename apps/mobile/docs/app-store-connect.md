# App Store Connect Setup

Use this checklist to create the Agent Kernel Lite iOS app record and prepare
TestFlight.

## App Record

- Platform: iOS
- Name: Agent Kernel Lite
- Bundle ID: `com.peytontolbert.agentkernellite`
- SKU: `agent-kernel-lite-ios`
- Primary category: Productivity
- Secondary category: Education
- Age rating: No objectionable content; no unrestricted web browsing.

## Build Upload

Recommended path:

1. Open `apps/mobile/ios/App/App.xcworkspace` on macOS.
2. Select the `App` scheme.
3. In Signing & Capabilities, choose the Apple Developer team.
4. Confirm bundle ID `com.peytontolbert.agentkernellite`.
5. Archive with `Any iOS Device`.
6. Distribute App > App Store Connect > Upload.
7. Wait for processing, then add the build to TestFlight.

Xcode Cloud path:

1. In App Store Connect, create the app record first.
2. In Xcode, configure Xcode Cloud for `apps/mobile/ios/App/App.xcworkspace`.
3. Use `ci_scripts/ci_post_clone.sh` as the post-clone setup script.
4. Set the workflow to archive the `App` scheme.
5. Enable TestFlight distribution after the first successful archive.

GitHub Actions path:

1. Add repository secrets:
   - `APPLE_TEAM_ID`
   - `APP_STORE_CONNECT_API_KEY_ID`
   - `APP_STORE_CONNECT_API_ISSUER_ID`
   - `APP_STORE_CONNECT_API_KEY_P8_BASE64`
2. Confirm the App Store Connect app record exists for
   `com.peytontolbert.agentkernellite`.
3. Run the manual workflow:

```text
.github/workflows/ios-testflight.yml
```

Apple build processing is not instant. After upload, wait for App Store Connect
to process the build before it appears in TestFlight.

## Required Local Tools

- macOS
- Xcode supported by App Store Connect
- CocoaPods
- Node.js
- Apple Developer Program membership

## App Review Positioning

Agent Kernel Lite is not a generic web browser. The app provides a local
research assistant shell, local paper search, local session storage, model/data
pack caching, and verified release loading from GitHub/Hugging Face.

Do not submit it as a website wrapper. Screenshots and review notes should show:

- Search mode returning papers.
- Bundled app opening offline.
- Release manifest check/caching screen.
- Hugging Face model/data pack download disclosure.
- No login requirement for the initial workflow.
