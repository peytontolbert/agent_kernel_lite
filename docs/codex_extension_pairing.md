# Computer Use Extension Pairing

Agent Kernel Lite exposes local coding tools through a `computer_use` extension.
Codex is the completed first provider under that extension. Claude Code and
Cursor are represented as provider slots so Agent Kernel Desktop can orchestrate
them later without changing the browser trust boundary.

The browser app remains the UI and policy ledger. A local bridge running on the
user's working computer owns provider authentication, workspace access, process
control, and filesystem permissions.

## Goals

- Let the app launch, stream, cancel, and follow up on local provider sessions.
- Keep provider credentials out of the browser app.
- Require explicit pairing before the browser can talk to the local bridge.
- Encrypt all bridge traffic end to end at the extension layer.
- Limit provider access to user-approved workspace roots and actions.
- Preserve the existing extension install, enable, propose, and receipt flow.

## User Setup

When using Agent Kernel Desktop, the desktop shell should run the same bridge
internally and expose pairing from the desktop app. For local development, run:

```bash
python scripts/run_agentkernel_lite_computer_bridge.py --workspace /path/to/allowed/root
```

The legacy filename still works:

```bash
python scripts/run_agentkernel_lite_codex_bridge.py --workspace /path/to/allowed/root
```

The bridge path is an allowed root, not necessarily the exact repo for every
session. For example, starting the bridge with `--workspace /data` lets the app
start Codex in `/data/repo_a` or `/data/repo_b`. Starting the bridge with
`--workspace /data/repo_a` is tighter: the app can run in `/data/repo_a` or a
child directory, but not `/data` or `/data/repo_b`.

Workspace checks resolve `..` segments and symlinks before validation. A session
workspace is accepted only when it equals an allowed root or is inside one.

The bridge defaults to `--approval-policy never` and `--sandbox
danger-full-access`. Pairing is the user approval gate; after pairing, the phone
or browser UI can start sessions without another approval prompt. The no-sandbox
default avoids Codex CLI `bwrap` failures on hosts where bubblewrap cannot create
loopback networking. If the host supports Codex sandboxing, start the bridge with
`--sandbox workspace-write` or `--sandbox read-only`.

The bridge prints a six-digit pairing code whenever the browser starts a pairing
request. The user enters that code in the app, then the working computer must
explicitly approve the pairing before the bridge issues a grant. After pairing,
the app stores the browser-side pairing key in IndexedDB and excludes it from
session exports.

## Extension Manifest

The active manifest is `web/extensions/computer_use.json`.

Capabilities:

- `computer.session.start`
- `computer.session.send`
- `computer.session.status`
- `computer.session.cancel`
- `computer.diff.read`

Each encrypted session request includes a provider:

```json
{
  "type": "computer.session.start",
  "provider": "codex",
  "workspace": "/Users/alex/src/project",
  "prompt": "Find and fix the failing test."
}
```

Provider status is exposed through bridge health and encrypted session status.
Current provider support:

- `codex`: implemented through `codex exec --json`, `codex exec resume`, cancel,
  status polling, and git diff reads.
- `claude_code`: discovered as a provider slot; command contract still needs
  validation before enabling runs.
- `cursor`: discovered as a provider slot; command contract still needs
  validation before enabling runs.

## Pairing Protocol

Use an application-layer encrypted channel even on loopback. The current browser
implementation uses P-256 ECDH, HKDF-SHA256, and AES-256-GCM because those are
available through WebCrypto and Python `cryptography`.

1. The bridge starts on `127.0.0.1` with a persistent P-256 device identity.
2. The app checks `/health` on the configured bridge URL.
3. If unpaired, the app sends its browser public key to `/pairing/start`.
4. The bridge prints a short-lived six-digit code and browser key fingerprint in
   the terminal or desktop UI.
5. The app sends that code to `/pairing/confirm`.
6. The working computer displays the origin, code, and browser key fingerprint,
   then requires explicit local approval before granting access.
7. The bridge stores the browser public key, origin, grant id, and expiry.
8. Both sides derive an AES-GCM key with ECDH plus HKDF.
9. All post-pairing messages use encrypted `/v1/message` envelopes with a
   monotonic sequence number.

Pairing grants expire by default after 30 days and can be revoked by encrypted
`computer.grant.revoke`.

## Security Rules

- Bind the bridge only to `127.0.0.1`.
- Reject browser POSTs from unapproved origins before pairing or message
  handling.
- Require local approval on the paired computer before creating a pairing grant.
- Reject requests without a valid paired grant.
- Require encryption for every post-pairing message.
- Tie grants to browser origin and browser public key.
- Enforce a monotonic sequence number to block replay.
- Enforce a workspace-root allowlist selected on the computer; the app can pick
  only that root or descendants for individual sessions.
- Do not accept arbitrary shell commands from the browser.
- Do not let browser payloads choose Codex sandbox or approval policy; those are
  bridge startup options.
- Disable plaintext grant revocation; revocation is an encrypted paired command.
- Never send OpenAI API keys, Codex tokens, SSH keys, or Git credentials to the
  browser.
- Require a fresh local confirmation for adding a new workspace or enabling
  write/apply capabilities.

## Session Lifecycle

1. User installs and enables `Computer Use`.
2. Browser confirms the bridge is paired and healthy.
3. Browser proposes a `computer.session.start` action to the WASM core.
4. Browser sends the encrypted start message to the bridge.
5. Bridge validates grant, origin, workspace, provider, and action id.
6. Bridge launches the provider asynchronously and stores stdout/stderr events.
7. Browser polls encrypted `computer.session.status` until completion.
8. Browser reads encrypted `computer.diff.read` after successful runs.
9. Browser records `approved_executed`, `cancelled`, or `failed` with
   `recordExtensionResult`.

Follow-up prompts use `computer.session.send`; cancel uses
`computer.session.cancel`. `computer.patch.apply` remains intentionally absent
until a separate write approval gate exists.
