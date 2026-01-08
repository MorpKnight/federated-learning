from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict

import flet as ft
import requests
import yaml

from fl_client.autotune import detect_cpu_count, detect_gpu_available, detect_ram_gb
from fl_client.data import get_dataloaders

from .config_store import get_client_config_path, load_config, save_config
from .process_manager import ProcessManager


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fetch_device_info() -> Dict[str, Any]:
    return {
        "cpu_count": detect_cpu_count(),
        "ram_gb": round(detect_ram_gb(), 1),
        "gpu_available": detect_gpu_available(),
    }


def fetch_dataset_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        train_loader, _ = get_dataloaders(
            data_dir=cfg.get("data", {}).get("data_dir", "data"),
            batch_size=int(cfg.get("train", {}).get("batch_size", 32)),
            num_workers=int(cfg.get("data", {}).get("num_workers", 2)),
        )
        return {"num_samples": len(train_loader.dataset)}
    except Exception:
        return {"num_samples": None}


def write_client_config(cfg: Dict[str, Any]) -> Path:
    config_path = get_client_config_path()
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
    return config_path


def register_and_fetch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    control = cfg.get("control_api", {})
    url = control.get("url", "").rstrip("/")
    if not url:
        raise RuntimeError("control_api.url is required")
    client_id = cfg.get("client_id", "client1")
    resp = requests.post(f"{url}/register", json={"client_id": client_id}, timeout=5)
    resp.raise_for_status()
    remote = requests.get(f"{url}/config/{client_id}", timeout=5)
    remote.raise_for_status()
    remote_cfg = remote.json()
    fl_addr = remote_cfg.get("fl_server_address")
    if fl_addr:
        cfg.setdefault("fl_server", {})["address"] = fl_addr
    train_cfg = remote_cfg.get("train", {})
    if train_cfg:
        cfg.setdefault("train", {}).update(train_cfg)
    save_config(cfg)
    return remote_cfg


def main(page: ft.Page) -> None:
    page.title = "FL Client Desktop"
    page.window_width = 960
    page.window_height = 720
    page.scroll = ft.ScrollMode.AUTO

    manager = ProcessManager(repo_root=repo_root())
    log_cursor = 0
    log_buffer: list[str] = []

    control_url = ft.TextField(label="Control API URL", width=420)
    server_addr = ft.TextField(label="FL Server Address", width=420)
    client_id = ft.TextField(label="Client ID", width=200)
    hp_mode = ft.Dropdown(
        label="Mode",
        options=[ft.dropdown.Option("manual"), ft.dropdown.Option("auto")],
        width=150,
        value="manual",
    )
    batch_size = ft.TextField(label="Batch Size", width=120)
    epochs = ft.TextField(label="Epochs", width=120)
    lr = ft.TextField(label="Learning Rate", width=120)

    connected = ft.Text(value="-")
    running = ft.Text(value="-")
    last_loss = ft.Text(value="-")
    last_acc = ft.Text(value="-")
    device_info = ft.Text(value="-")
    dataset_info = ft.Text(value="-")

    logs = ft.TextField(
        label="Logs",
        multiline=True,
        read_only=True,
        min_lines=10,
        max_lines=20,
        expand=True,
    )

    def load_from_config() -> Dict[str, Any]:
        cfg = load_config()
        control_url.value = cfg.get("control_api", {}).get("url", "")
        server_addr.value = cfg.get("fl_server", {}).get("address", "")
        client_id.value = cfg.get("client_id", "client1")
        batch_size.value = str(cfg.get("train", {}).get("batch_size", 32))
        epochs.value = str(cfg.get("train", {}).get("epochs", 1))
        lr.value = str(cfg.get("train", {}).get("lr", 0.01))
        hp_mode.value = "auto" if cfg.get("train", {}).get("auto", False) else "manual"
        return cfg

    def save_current_config() -> Dict[str, Any]:
        cfg = load_config()
        cfg["client_id"] = client_id.value or "client1"
        cfg["control_api"] = {"url": control_url.value or ""}
        cfg["fl_server"] = {"address": server_addr.value or ""}
        cfg.setdefault("train", {})
        cfg["train"]["batch_size"] = int(batch_size.value or 32)
        cfg["train"]["epochs"] = int(epochs.value or 1)
        cfg["train"]["lr"] = float(lr.value or 0.01)
        cfg["train"]["auto"] = hp_mode.value == "auto"
        save_config(cfg)
        return cfg

    def refresh_status() -> None:
        cfg = load_config()
        d = fetch_device_info()
        ds = fetch_dataset_info(cfg)
        connected.value = "yes" if manager.status.connected else "no"
        running.value = "yes" if manager.status.running else "no"
        last_loss.value = str(manager.status.last_loss) if manager.status.last_loss is not None else "-"
        last_acc.value = str(manager.status.last_acc) if manager.status.last_acc is not None else "-"
        device_info.value = f"{d['cpu_count']} CPU, {d['ram_gb']} GB, GPU={d['gpu_available']}"
        dataset_info.value = str(ds.get("num_samples"))
        page.update()

    def on_save(_):
        save_current_config()
        refresh_status()

    def on_register(_):
        try:
            cfg = save_current_config()
            register_and_fetch(cfg)
            page.snack_bar = ft.SnackBar(ft.Text("Registered and fetched config"))
        except Exception as exc:
            page.snack_bar = ft.SnackBar(ft.Text(f"Register failed: {exc}"))
        page.snack_bar.open = True
        refresh_status()

    def on_start(_):
        try:
            cfg = save_current_config()
            config_path = write_client_config(cfg)
            manager.start(config_path=config_path, client_id=cfg.get("client_id", "client1"))
            page.snack_bar = ft.SnackBar(ft.Text("Training started"))
        except Exception as exc:
            page.snack_bar = ft.SnackBar(ft.Text(f"Start failed: {exc}"))
        page.snack_bar.open = True
        refresh_status()

    def on_stop(_):
        manager.stop()
        page.snack_bar = ft.SnackBar(ft.Text("Training stopped"))
        page.snack_bar.open = True
        refresh_status()

    def poll_logs() -> None:
        nonlocal log_cursor
        while True:
            lines, log_cursor = manager.get_logs(since=log_cursor)
            if lines:
                log_buffer.extend(lines)
                log_text = "\n".join(log_buffer[-200:])
                page.call_from_thread(lambda: setattr(logs, "value", log_text))
                page.call_from_thread(page.update)
            time.sleep(0.5)

    def poll_status() -> None:
        while True:
            page.call_from_thread(refresh_status)
            time.sleep(2)

    load_from_config()

    page.add(
        ft.Column(
            [
                ft.Text("Federated Learning Client", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column(
                        [
                            control_url,
                            server_addr,
                            client_id,
                            ft.Row(
                                [
                                    ft.ElevatedButton("Register/Connect", on_click=on_register),
                                    ft.ElevatedButton("Save Config", on_click=on_save),
                                ]
                            ),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.colors.GREY_300),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([hp_mode, batch_size, epochs, lr]),
                            ft.Row(
                                [
                                    ft.ElevatedButton("Start Training", on_click=on_start),
                                    ft.ElevatedButton("Stop Training", on_click=on_stop),
                                ]
                            ),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.colors.GREY_300),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([ft.Text("Connected:"), connected, ft.Text("Running:"), running]),
                            ft.Row([ft.Text("Last Loss:"), last_loss, ft.Text("Last Acc:"), last_acc]),
                            ft.Row([ft.Text("Device:"), device_info]),
                            ft.Row([ft.Text("Dataset Samples:"), dataset_info]),
                        ]
                    ),
                    padding=10,
                    border=ft.border.all(1, ft.colors.GREY_300),
                ),
                logs,
            ],
            expand=True,
        )
    )

    threading.Thread(target=poll_logs, daemon=True).start()
    threading.Thread(target=poll_status, daemon=True).start()


if __name__ == "__main__":
    ft.app(target=main)
