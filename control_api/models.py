from __future__ import annotations

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    client_id: str


class HeartbeatRequest(BaseModel):
    client_id: str
