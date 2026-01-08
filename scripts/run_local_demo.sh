#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m control_api.main --config "$ROOT_DIR/configs/control_api.yaml" &
CONTROL_PID=$!

sleep 1

python -m fl_server.server --config "$ROOT_DIR/configs/server.yaml" &
SERVER_PID=$!

sleep 2

python -m fl_client.client --config "$ROOT_DIR/configs/client.yaml" --client-id client1 &
CLIENT1_PID=$!

python -m fl_client.client --config "$ROOT_DIR/configs/client.yaml" --client-id client2 &
CLIENT2_PID=$!

cleanup() {
  kill "$CONTROL_PID" "$SERVER_PID" "$CLIENT1_PID" "$CLIENT2_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait "$SERVER_PID" "$CLIENT1_PID" "$CLIENT2_PID"
