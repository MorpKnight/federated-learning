from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl

from control_api.db import add_client_metrics, add_round_metrics


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}
    loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
    acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    return {"loss": loss, "accuracy": acc}


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        db_conn,
        logger: logging.Logger,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.db_conn = db_conn
        self.logger = logger

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        for client, fit_res in results:
            metrics = fit_res.metrics or {}
            add_client_metrics(
                self.db_conn,
                round_num=rnd,
                client_id=metrics.get("client_id", client.cid),
                loss=metrics.get("loss"),
                accuracy=metrics.get("accuracy"),
                num_examples=fit_res.num_examples,
            )
        return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        num_clients = len(results)
        loss: Optional[float] = None
        acc: Optional[float] = None
        if aggregated:
            loss, metrics = aggregated
            if metrics:
                acc = metrics.get("accuracy")
        add_round_metrics(
            self.db_conn,
            round_num=rnd,
            num_clients=num_clients,
            loss=loss,
            accuracy=acc,
        )
        self.logger.info(
            "round=%s clients=%s loss=%s acc=%s",
            rnd,
            num_clients,
            f"{loss:.4f}" if loss is not None else "n/a",
            f"{acc:.4f}" if acc is not None else "n/a",
        )
        return aggregated
