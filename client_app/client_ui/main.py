from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import requests
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from fl_client.autotune import detect_cpu_count, detect_gpu_available, detect_ram_gb
from fl_client.data import get_dataloaders

from .config_store import get_client_config_path, load_config, save_config
from .process_manager import ProcessManager


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_app() -> FastAPI:
    app = FastAPI(title="FL Client UI")
    logger = logging.getLogger("client_ui")
    root = repo_root()
    manager = ProcessManager(repo_root=root)

    static_dir = root / "client_app" / "client_ui" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(root / "client_app" / "client_ui" / "static" / "index.html")

    @app.get("/status")
    def status():
        cfg = load_config()
        device_info = {
            "cpu_count": detect_cpu_count(),
            "ram_gb": round(detect_ram_gb(), 1),
            "gpu_available": detect_gpu_available(),
        }
        dataset_size = None
        try:
            train_loader, _ = get_dataloaders(
                data_dir=cfg.get("data", {}).get("data_dir", "data"),
                batch_size=int(cfg.get("train", {}).get("batch_size", 32)),
                num_workers=int(cfg.get("data", {}).get("num_workers", 2)),
            )
            dataset_size = len(train_loader.dataset)
        except Exception:
            dataset_size = None

        return {
            "config": cfg,
            "device": device_info,
            "dataset": {"num_samples": dataset_size},
            "process": manager.status.__dict__,
        }

    @app.post("/config")
    def set_config(payload: Dict[str, Any]):
        cfg = load_config()
        cfg.update(payload)
        save_config(cfg)
        return {"status": "ok", "config": cfg}

    @app.post("/register")
    def register():
        cfg = load_config()
        control = cfg.get("control_api", {})
        url = control.get("url", "").rstrip("/")
        if not url:
            raise HTTPException(status_code=400, detail="control_api.url is required")
        client_id = cfg.get("client_id", "client1")
        try:
            resp = requests.post(f"{url}/register", json={"client_id": client_id}, timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        try:
            remote = requests.get(f"{url}/config/{client_id}", timeout=5)
            remote.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        remote_cfg = remote.json()
        fl_addr = remote_cfg.get("fl_server_address")
        if fl_addr:
            cfg.setdefault("fl_server", {})["address"] = fl_addr
        train_cfg = remote_cfg.get("train", {})
        if train_cfg:
            cfg.setdefault("train", {}).update(train_cfg)

        save_config(cfg)
        return {"status": "ok", "client_id": client_id, "remote": remote_cfg}

    @app.post("/start")
    def start_training():
        cfg = load_config()
        client_id = cfg.get("client_id", "client1")
        config_path = get_client_config_path()
        # write a client config compatible with fl_client.client
        client_cfg = {
            "client": {"server_address": cfg.get("fl_server", {}).get("address", "127.0.0.1:8080")},
            "control_api": {"url": cfg.get("control_api", {}).get("url", "")},
            "train": cfg.get("train", {}),
            "data": cfg.get("data", {}),
            "logging": cfg.get("logging", {"level": "INFO"}),
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(client_cfg, f, sort_keys=False)

        try:
            manager.start(config_path=config_path, client_id=client_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"status": "started", "pid": manager.status.pid}

    @app.post("/stop")
    def stop_training():
        manager.stop()
        return {"status": "stopped"}

    @app.get("/logs")
    def logs(since: int = Query(default=0)):
        return manager.get_logs(since=since)

    @app.get("/logs/stream")
    def logs_stream():
        return StreamingResponse(manager.stream(), media_type="text/event-stream")

    return app


app = make_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("CLIENT_UI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CLIENT_UI_PORT", "7000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
