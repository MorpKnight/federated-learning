from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    client_id: str


class RegisterResponse(BaseModel):
    client_id: str
    token: Optional[str] = None


class Hyperparams(BaseModel):
    batch_size: int = 32
    epochs: int = 1
    lr: float = 0.01


class ConfigResponse(BaseModel):
    fl_server_address: str
    policy: Dict[str, Any] = {}
    hyperparams: Hyperparams = Hyperparams()


class HeartbeatRequest(BaseModel):
    client_id: str
    status: str
    timestamp: str


class HeartbeatResponse(BaseModel):
    status: str
    timestamp: str
