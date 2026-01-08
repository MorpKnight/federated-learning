from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict

import flwr as fl
import yaml

from control_api.db import init_db
from fl_server.strategy import LoggingFedAvg, weighted_average


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_level = os.getenv("FL_LOG_LEVEL", cfg.get("logging", {}).get("level", "INFO"))
    setup_logging(log_level)
    logger = logging.getLogger("fl_server")

    server_cfg = cfg.get("server", {})
    address = os.getenv("FL_SERVER_ADDRESS", server_cfg.get("address", "127.0.0.1:8080"))
    num_rounds = int(os.getenv("FL_NUM_ROUNDS", server_cfg.get("num_rounds", 3)))
    min_fit = int(os.getenv("FL_MIN_FIT_CLIENTS", server_cfg.get("min_fit_clients", 2)))
    min_avail = int(os.getenv("FL_MIN_AVAILABLE_CLIENTS", server_cfg.get("min_available_clients", 2)))

    db_path = os.getenv("FL_DB_PATH", cfg.get("db", {}).get("path", "data/metrics.db"))
    conn = init_db(db_path)

    strategy = LoggingFedAvg(
        db_conn=conn,
        logger=logger,
        min_fit_clients=min_fit,
        min_available_clients=min_avail,
        min_evaluate_clients=min_fit,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    logger.info("Starting Flower server at %s for %s rounds", address, num_rounds)
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
