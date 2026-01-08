from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def get_config_dir() -> Path:
    return Path.home() / ".fl_client_ui"


def get_ui_config_path() -> Path:
    return get_config_dir() / "config.yaml"


def get_client_config_path() -> Path:
    return get_config_dir() / "client.yaml"


def default_config() -> Dict[str, Any]:
    return {
        "client_id": "client1",
        "control_api": {"url": "http://127.0.0.1:8000", "token": ""},
        "fl_server": {"address": "127.0.0.1:8080"},
        "train": {
            "batch_size": 32,
            "epochs": 1,
            "lr": 0.01,
            "device": "auto",
            "auto": False,
        },
        "data": {"data_dir": "data", "num_workers": 2},
        "logging": {"level": "INFO"},
    }


def load_config() -> Dict[str, Any]:
    path = get_ui_config_path()
    if not path.exists():
        return default_config()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = default_config()
    merged.update(data)
    return merged


def save_config(cfg: Dict[str, Any]) -> None:
    path = get_ui_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
