#!/bin/sh
set -eu

APP_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/../.." && pwd)"

echo "Xcode Cloud: preparing Agent Kernel Lite mobile app"
cd "$APP_ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "error: Node.js is required to build the Capacitor app." >&2
  exit 1
fi

npm ci
npm run sync:web
npx cap sync ios

echo "Xcode Cloud: web assets synced from $REPO_ROOT/web"
