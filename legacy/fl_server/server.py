from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import flwr as fl
import requests
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_headers(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"X-API-Token": token}


def weighted_average(metrics):
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total
    return {"loss": loss}


class ReportingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, control_api_url: str, api_token: Optional[str], **kwargs):
        super().__init__(**kwargs)
        self.control_api_url = control_api_url.rstrip("/") if control_api_url else ""
        self.api_token = api_token

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> None:
        if not self.control_api_url:
            return
        url = f"{self.control_api_url}{endpoint}"
        try:
            requests.post(
                url,
                json=payload,
                headers=build_headers(self.api_token),
                timeout=2,
            )
        except Exception:
            # Avoid crashing training if control plane is down
            return

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        for client, fit_res in results:
            metrics = fit_res.metrics or {}
            payload = {
                "round": rnd,
                "client_id": metrics.get("client_id") or client.cid,
                "stage": "fit",
                "loss": metrics.get("loss"),
                "num_examples": fit_res.num_examples,
            }
            self._post("/metrics", payload)
        if aggregated and aggregated[1]:
            self._post(
                "/metrics",
                {
                    "round": rnd,
                    "client_id": "server",
                    "stage": "fit_aggregate",
                    "loss": aggregated[1].get("loss"),
                    "num_examples": None,
                },
            )
        return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        for client, eval_res in results:
            payload = {
                "round": rnd,
                "client_id": (eval_res.metrics or {}).get("client_id") or client.cid,
                "stage": "eval",
                "loss": eval_res.loss,
                "num_examples": eval_res.num_examples,
            }
            self._post("/metrics", payload)
        if aggregated:
            loss, _metrics = aggregated
            self._post(
                "/metrics",
                {
                    "round": rnd,
                    "client_id": "server",
                    "stage": "eval_aggregate",
                    "loss": loss,
                    "num_examples": None,
                },
            )
        return aggregated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/server.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    server_cfg = cfg.get("server", {})
    control_cfg = cfg.get("control", {})

    strategy = ReportingFedAvg(
        control_api_url=control_cfg.get("api_url", ""),
        api_token=control_cfg.get("api_token"),
        min_fit_clients=server_cfg.get("min_fit_clients", 2),
        min_available_clients=server_cfg.get("min_available_clients", 2),
        min_evaluate_clients=server_cfg.get("min_fit_clients", 2),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address=server_cfg.get("address", "127.0.0.1:8080"),
        config=fl.server.ServerConfig(num_rounds=server_cfg.get("num_rounds", 3)),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
