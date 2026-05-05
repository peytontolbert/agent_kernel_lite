# TestFlight Checklist

## Internal Testing

1. Create the app record in App Store Connect.
2. Upload the first archive from Xcode or Xcode Cloud.
3. Wait for build processing.
4. Add the build to Internal Testing.
5. Add internal testers.
6. Install from TestFlight and verify:
   - app opens directly into Agent Kernel Lite;
   - Search mode works;
   - bundled 50k paper pack loads without a network request;
   - bundled BitNet runtime starts;
   - app survives airplane mode after first launch.

## External Testing

Before external beta review:

- add screenshots showing the real app, not a placeholder Capacitor screen;
- add beta app description;
- add review notes from `app-review-notes.md`;
- confirm privacy answers from `privacy.md`;
- confirm optional remote model/data downloads are described accurately.

## Versioning

For each TestFlight upload:

- increment iOS `CURRENT_PROJECT_VERSION`;
- keep `MARKETING_VERSION` aligned with the public app version;
- increment Android `versionCode` when preparing a Play build;
- run `npm run prepare:assets` and `npm run sync:web` before archiving.
