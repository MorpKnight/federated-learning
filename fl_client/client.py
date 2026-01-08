from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from .autotune import tune_from_loader
from .data import get_dataloaders


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def register_client(api_url: str, client_id: str, logger: logging.Logger) -> str:
    try:
        resp = requests.post(
            f"{api_url}/register",
            json={"client_id": client_id},
            timeout=5,
        )
    except requests.RequestException as exc:
        logger.error("Control API not reachable at %s: %s", api_url, exc)
        raise SystemExit(1) from exc
    if resp.status_code != 200:
        logger.error("Control API register failed (%s): %s", resp.status_code, resp.text)
        raise SystemExit(1)
    return resp.json().get("client_id", client_id)


def fetch_remote_config(api_url: str, client_id: str, logger: logging.Logger) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{api_url}/config/{client_id}", timeout=5)
    except requests.RequestException as exc:
        logger.error("Control API not reachable at %s: %s", api_url, exc)
        raise SystemExit(1) from exc
    if resp.status_code != 200:
        logger.error("Control API config failed (%s): %s", resp.status_code, resp.text)
        raise SystemExit(1)
    return resp.json()


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lr: float,
) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int,
        lr: float,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.logger = logger

        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, Any]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = 0.0
        acc = 0.0
        for _ in range(self.epochs):
            loss, acc = train_one_epoch(self.model, self.train_loader, self.device, self.lr)
        num_examples = len(self.train_loader.dataset)
        self.logger.info("fit loss=%.4f acc=%.4f", loss, acc)
        return (
            get_parameters(self.model),
            num_examples,
            {"loss": loss, "accuracy": acc, "client_id": self.logger.name},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        self.logger.info("eval loss=%.4f acc=%.4f", loss, acc)
        return loss, num_examples, {"loss": loss, "accuracy": acc, "client_id": self.logger.name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/client.yaml")
    parser.add_argument("--client-id", default="client1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_level = os.getenv("FL_LOG_LEVEL", cfg.get("logging", {}).get("level", "INFO"))
    setup_logging(log_level)
    client_id = os.getenv("FL_CLIENT_ID", args.client_id)
    logger = logging.getLogger(f"fl_client.{client_id}")

    client_cfg = cfg.get("client", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    control_cfg = cfg.get("control_api", {})

    server_address = os.getenv("FL_SERVER_ADDRESS", client_cfg.get("server_address", "127.0.0.1:8080"))
    data_dir = os.getenv("FL_DATA_DIR", data_cfg.get("data_dir", "data"))
    num_workers = int(os.getenv("FL_NUM_WORKERS", data_cfg.get("num_workers", 2)))
    device_cfg = os.getenv("FL_DEVICE", train_cfg.get("device", "auto"))

    control_api_url = os.getenv("CONTROL_API_URL", control_cfg.get("url", "")).rstrip("/")
    if control_api_url:
        registered_id = register_client(control_api_url, client_id, logger)
        remote_cfg = fetch_remote_config(control_api_url, registered_id, logger)
        server_address = remote_cfg.get("fl_server_address", server_address)
        train_cfg = {**train_cfg, **remote_cfg.get("train", {})}

    batch_size = int(os.getenv("FL_BATCH_SIZE", train_cfg.get("batch_size", 32)))
    epochs = int(os.getenv("FL_EPOCHS", train_cfg.get("epochs", 1)))
    lr = float(os.getenv("FL_LR", train_cfg.get("lr", 0.01)))
    auto_tune = bool(train_cfg.get("auto", False))

    device = get_device(device_cfg)
    logger.info("Using device=%s", device.type)

    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if auto_tune:
        tune_result, meta = tune_from_loader(train_loader)
        batch_size = tune_result.batch_size
        epochs = tune_result.local_epochs
        lr = tune_result.lr
        logger.info(
            "auto_tune enabled: dataset=%s cpu=%s ram_gb=%.1f gpu=%s -> batch=%s epochs=%s lr=%s",
            meta["dataset_size"],
            meta["cpu_count"],
            meta["ram_gb"],
            meta["gpu_available"],
            batch_size,
            epochs,
            lr,
        )
        train_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    model = SimpleCNN()
    client = FlowerClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        logger=logger,
    )

    logger.info("Connecting to server at %s as %s", server_address, client_id)
    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except Exception as exc:
        logger.error("Failed to connect to server at %s: %s", server_address, exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

