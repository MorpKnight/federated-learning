#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m control_api.main --config "$ROOT_DIR/configs/control_api.yaml" >"$ROOT_DIR/control_api.log" 2>&1 &
CONTROL_PID=$!

python -m portal_web.main >"$ROOT_DIR/portal_web.log" 2>&1 &
PORTAL_PID=$!

python -m client_app.client_ui.main >"$ROOT_DIR/client_ui.log" 2>&1 &
CLIENT_PID=$!

cleanup() {
  kill "$CONTROL_PID" "$PORTAL_PID" "$CLIENT_PID" 2>/dev/null || true
}
trap cleanup EXIT

python "$ROOT_DIR/scripts/verify_connect.py"
