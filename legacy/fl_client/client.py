from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import flwr as fl
import numpy as np
import requests
import yaml

from .dataset import make_synthetic


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_headers(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"X-API-Token": token}


def merge_config(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], val)
        else:
            merged[key] = val
    return merged


def fetch_remote_config(api_url: str, token: Optional[str], client_id: str) -> Optional[Dict[str, Any]]:
    if not api_url:
        return None
    try:
        resp = requests.get(
            f"{api_url.rstrip('/')}/config/{client_id}",
            headers=build_headers(token),
            timeout=2,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def register_client(api_url: str, token: Optional[str], client_id: str) -> None:
    if not api_url:
        return
    try:
        requests.post(
            f"{api_url.rstrip('/')}/register",
            json={"client_id": client_id},
            headers=build_headers(token),
            timeout=2,
        )
    except Exception:
        return


class SimpleNumPyClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, train_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        self.client_id = client_id
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg

        self.x, self.y = make_synthetic(
            client_id,
            num_samples=int(data_cfg.get("num_samples", 200)),
            num_features=int(data_cfg.get("num_features", 5)),
        )
        self.w = np.zeros(self.x.shape[1], dtype=np.float64)
        self.b = 0.0

    def get_parameters(self, config):
        return [self.w, np.array([self.b], dtype=np.float64)]

    def set_parameters(self, parameters):
        self.w = parameters[0]
        self.b = float(parameters[1][0])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(self.train_cfg.get("epochs", 1))
        lr = float(self.train_cfg.get("lr", 0.05))

        n = self.x.shape[0]
        for _ in range(epochs):
            preds = self.x @ self.w + self.b
            err = preds - self.y
            grad_w = (2.0 / n) * (self.x.T @ err)
            grad_b = (2.0 / n) * np.sum(err)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

        loss = float(np.mean((self.x @ self.w + self.b - self.y) ** 2))
        return self.get_parameters({}), n, {"client_id": self.client_id, "loss": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.x @ self.w + self.b
        loss = float(np.mean((preds - self.y) ** 2))
        return loss, self.x.shape[0], {"client_id": self.client_id, "loss": loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/client.yaml")
    parser.add_argument("--client-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    client_id = args.client_id or cfg.get("client", {}).get("id", "client1")

    control_cfg = cfg.get("control", {})
    register_client(control_cfg.get("api_url", ""), control_cfg.get("api_token"), client_id)
    remote_cfg = fetch_remote_config(
        control_cfg.get("api_url", ""),
        control_cfg.get("api_token"),
        client_id,
    )
    merged_cfg = merge_config(cfg, remote_cfg)

    client_cfg = merged_cfg.get("client", {})
    train_cfg = merged_cfg.get("train", {})
    data_cfg = merged_cfg.get("data", {})

    fl.client.start_numpy_client(
        server_address=client_cfg.get("server_address", "127.0.0.1:8080"),
        client=SimpleNumPyClient(client_id, train_cfg, data_cfg),
    )


if __name__ == "__main__":
    main()
