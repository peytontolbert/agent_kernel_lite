# Privacy

Initial App Store privacy answers for Agent Kernel Lite:

- Tracking: No.
- Third-party advertising: No.
- Data linked to user: No.
- Data used for tracking: No.
- Account required: No.
- User-generated content publication: No.
- Export compliance: No non-exempt encryption.

Network requests:

- GitHub Releases for immutable app assets, WASM bindings, extension manifests,
  and checksums.
- Hugging Face for model tensors, tokenizers, paper packs, embeddings, and paper
  text rows.

Local data:

- chat/session state in app-local WebView storage;
- cached model and paper assets;
- imported/exported session JSON when the user explicitly uses that workflow.

The initial native app does not request camera, microphone, contacts, location,
Bluetooth, calendars, photos, or tracking permissions.

The app uses standard HTTPS/TLS through the platform networking stack to reach
GitHub Releases and Hugging Face, plus SHA-256 hashing for asset verification.
It does not implement custom encryption or encrypted messaging. The iOS
`Info.plist` sets `ITSAppUsesNonExemptEncryption` to `false`.

`PrivacyInfo.xcprivacy` currently declares no tracking, no collected data types,
and no required-reason accessed API categories. Revisit this file before
submission if native plugins are added for file picking, speech, notifications,
or platform credentials.
