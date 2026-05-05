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

- GitHub/App Store delivery for app updates.
- Hugging Face only for optional refreshed or larger model/data assets.

Local data:

- chat/session state in app-local WebView storage;
- bundled and cached model/paper assets;
- imported/exported session JSON when the user explicitly uses that workflow.

The initial native app does not request camera, microphone, contacts, location,
Bluetooth, calendars, photos, or tracking permissions.

The app uses standard HTTPS/TLS through the platform networking stack for
optional remote model/data downloads. It does not implement custom encryption or
encrypted messaging. The iOS `Info.plist` sets
`ITSAppUsesNonExemptEncryption` to `false`.

`PrivacyInfo.xcprivacy` currently declares no tracking, no collected data types,
and no required-reason accessed API categories. Revisit this file before
submission if native plugins are added for file picking, speech, notifications,
or platform credentials.
