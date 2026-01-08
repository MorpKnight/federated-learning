from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .db import get_client_metrics, get_round_metrics, init_db, upsert_client, update_heartbeat
from shared.schemas import (
    ConfigResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    Hyperparams,
    RegisterRequest,
    RegisterResponse,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def make_app(config_path: str) -> FastAPI:
    cfg = load_config(config_path)
    log_level = os.getenv("FL_LOG_LEVEL", cfg.get("logging", {}).get("level", "INFO"))
    setup_logging(log_level)
    logger = logging.getLogger("control_api")

    api_cfg = cfg.get("api", {})
    fl_cfg = cfg.get("fl_server", {})
    train_cfg = cfg.get("train", {})
    policy_cfg = cfg.get("policy", {})
    token_cfg = cfg.get("auth", {}).get("token")

    db_path = os.getenv("FL_DB_PATH", cfg.get("db", {}).get("path", "data/metrics.db"))
    conn = init_db(db_path)

    app = FastAPI(title="Control API")

    @app.post("/register")
    def register(req: RegisterRequest) -> RegisterResponse:
        logger.info("register client_id=%s", req.client_id)
        upsert_client(conn, req.client_id, token_cfg)
        return RegisterResponse(client_id=req.client_id, token=token_cfg)

    @app.get("/config/{client_id}")
    def get_config(client_id: str) -> ConfigResponse:
        logger.info("config request client_id=%s", client_id)
        hyperparams = Hyperparams(
            batch_size=train_cfg.get("batch_size", 32),
            epochs=train_cfg.get("epochs", 1),
            lr=train_cfg.get("lr", 0.01),
        )
        return ConfigResponse(
            fl_server_address=fl_cfg.get("address", "127.0.0.1:8080"),
            policy=policy_cfg,
            hyperparams=hyperparams,
        )

    @app.post("/heartbeat")
    def heartbeat(req: HeartbeatRequest, request: Request) -> HeartbeatResponse:
        logger.info("heartbeat client_id=%s ip=%s", req.client_id, request.client.host)
        update_heartbeat(conn, req.client_id)
        return HeartbeatResponse(status="ok", timestamp=datetime.utcnow().isoformat())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        rounds = get_round_metrics(conn, limit=50)
        clients = get_client_metrics(conn, limit=100)

        round_rows = "".join(
            "<tr>"
            f"<td>{r['round']}</td>"
            f"<td>{r['loss']}</td>"
            f"<td>{r['accuracy']}</td>"
            f"<td>{r['num_clients']}</td>"
            f"<td>{r['ts']}</td>"
            "</tr>"
            for r in rounds
        )
        client_rows = "".join(
            "<tr>"
            f"<td>{c['round']}</td>"
            f"<td>{c['client_id']}</td>"
            f"<td>{c['loss']}</td>"
            f"<td>{c['accuracy']}</td>"
            f"<td>{c['num_examples']}</td>"
            f"<td>{c['ts']}</td>"
            "</tr>"
            for c in clients
        )

        return f"""
        <html>
        <head><title>FL Dashboard</title></head>
        <body>
        <h1>Round Metrics</h1>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr>
            <th>round</th><th>loss</th><th>accuracy</th><th>num_clients</th><th>timestamp</th>
          </tr>
          {round_rows}
        </table>
        <h1>Client Metrics</h1>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr>
            <th>round</th><th>client_id</th><th>loss</th><th>accuracy</th>
            <th>num_examples</th><th>timestamp</th>
          </tr>
          {client_rows}
        </table>
        </body>
        </html>
        """

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/control_api.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_cfg = cfg.get("api", {})
    host = os.getenv("CONTROL_API_HOST", api_cfg.get("host", "127.0.0.1"))
    port = int(os.getenv("CONTROL_API_PORT", api_cfg.get("port", 8000)))

    import uvicorn

    uvicorn.run(make_app(args.config), host=host, port=port)


if __name__ == "__main__":
    main()
