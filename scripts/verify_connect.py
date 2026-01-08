from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests
from pydantic import ValidationError

from shared.schemas import ConfigResponse, HeartbeatRequest


def wait_for_health(url: str, timeout_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Control API not healthy after {timeout_s}s: {url}")


def request_json(method: str, url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        resp = requests.request(method, url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned invalid JSON") from exc


def main() -> int:
    base_url = os.environ.get("CONTROL_API_URL", "http://127.0.0.1:8000").rstrip("/")
    base_url = "http://127.0.0.1:8000"
    try:
        wait_for_health(f"{base_url}/health", timeout_s=30)
        client_id = "demo-client"
        request_json("POST", f"{base_url}/register", {"client_id": client_id})
        config_payload = request_json("GET", f"{base_url}/config/{client_id}")
        try:
            ConfigResponse.model_validate(config_payload)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid config response: {exc}") from exc
        heartbeat_payload = HeartbeatRequest(
            client_id=client_id,
            status="online",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        request_json("POST", f"{base_url}/heartbeat", heartbeat_payload.model_dump())
        request_json("GET", f"{base_url}/config/{client_id}")
        request_json("POST", f"{base_url}/heartbeat", {"client_id": client_id})
    except Exception as exc:
        print(f"CONNECT_ERROR: {exc}", file=sys.stderr)
        return 1

    print("CONNECT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
