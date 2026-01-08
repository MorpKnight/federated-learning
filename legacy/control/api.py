from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .db import (
    add_metric,
    get_client_config,
    get_metrics,
    init_db,
    upsert_client,
    upsert_client_config,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RegisterRequest(BaseModel):
    client_id: str


class MetricRequest(BaseModel):
    round: int
    client_id: str
    stage: str
    loss: Optional[float] = None
    num_examples: Optional[int] = None


def make_app(config_path: str) -> FastAPI:
    cfg = load_config(config_path)
    api_token = cfg.get("auth", {}).get("api_token")
    default_config = {
        "client": {"server_address": "127.0.0.1:8080"},
        "train": {"epochs": 1, "batch_size": 16, "lr": 0.05},
        "data": {"num_samples": 200, "num_features": 5},
    }

    db_path = cfg.get("db", {}).get("path", "data/control.db")
    conn = init_db(db_path)

    app = FastAPI(title="Control Plane")

    def require_token(request: Request):
        if not api_token:
            return
        token = request.headers.get("X-API-Token")
        if token != api_token:
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.get("/health")
    def health():
        return {"status": "ok", "ts": datetime.utcnow().isoformat()}

    @app.post("/register")
    def register(req: RegisterRequest, _=Depends(require_token)):
        upsert_client(conn, req.client_id)
        return {"status": "ok"}

    @app.get("/config/{client_id}")
    def get_config(client_id: str, _=Depends(require_token)):
        cfg_override = get_client_config(conn, client_id)
        if cfg_override:
            merged = {**default_config, **cfg_override}
            return merged
        return {**default_config, "client": {**default_config["client"], "id": client_id}}

    @app.post("/config/{client_id}")
    def set_config(client_id: str, request: Dict[str, Any], _=Depends(require_token)):
        upsert_client_config(conn, client_id, request)
        return {"status": "ok"}

    @app.post("/metrics")
    def metrics(req: MetricRequest, _=Depends(require_token)):
        add_metric(conn, req)
        return {"status": "ok"}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        rows = get_metrics(conn, limit=50)
        html_rows = "".join(
            f"<tr><td>{r['ts']}</td><td>{r['round']}</td><td>{r['client_id']}</td>"
            f"<td>{r['stage']}</td><td>{r['loss']}</td><td>{r['num_examples']}</td></tr>"
            for r in rows
        )
        return f"""
        <html>
        <head><title>FL Metrics</title></head>
        <body>
        <h1>FL Metrics (latest)</h1>
        <table border="1" cellpadding="4" cellspacing="0">
        <tr><th>ts</th><th>round</th><th>client</th><th>stage</th><th>loss</th><th>num_examples</th></tr>
        {html_rows}
        </table>
        </body>
        </html>
        """

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/control.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = cfg.get("server", {}).get("host", "127.0.0.1")
    port = int(cfg.get("server", {}).get("port", 8000))

    import uvicorn

    uvicorn.run(make_app(args.config), host=host, port=port)


if __name__ == "__main__":
    main()
