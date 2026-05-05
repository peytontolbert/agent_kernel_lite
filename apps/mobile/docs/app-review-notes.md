# App Review Notes

Suggested review note:

```text
No account is required.

Agent Kernel Lite is a local research assistant for paper search and local
browser-side model execution. The app opens directly into the bundled Agent
Kernel Lite interface and includes the default BitNet model plus a 50k paper
metadata pack in the app bundle.

Review steps:
1. Open the app.
2. Use Search mode to run a paper query.
3. Load the runtime and submit a short local prompt.
4. Turn on airplane mode and confirm Search mode still returns bundled papers.

The app does not require user accounts, does not track users, and does not
provide unrestricted web browsing. GitHub Actions, TestFlight, and App Store
Connect deliver app updates. Hugging Face is used as the upstream source for
optional refreshed or larger model/data assets.
```

If review asks why this is not a website wrapper, emphasize:

- the app bundles the usable Agent Kernel Lite interface;
- the app has local storage and cache management;
- the default model and 50k paper pack are packaged with the app;
- optional Hugging Face assets are model/data packs, not remote executable
  JavaScript;
- Search mode works as the first native mobile workflow.
