from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clients (
            client_id TEXT PRIMARY KEY,
            first_seen TEXT,
            last_seen TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS client_configs (
            client_id TEXT PRIMARY KEY,
            config_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            round INTEGER,
            client_id TEXT,
            stage TEXT,
            loss REAL,
            num_examples INTEGER
        )
        """
    )
    conn.commit()
    return conn


def upsert_client(conn: sqlite3.Connection, client_id: str) -> None:
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO clients (client_id, first_seen, last_seen)
        VALUES (?, ?, ?)
        ON CONFLICT(client_id) DO UPDATE SET last_seen=excluded.last_seen
        """,
        (client_id, now, now),
    )
    conn.commit()


def upsert_client_config(conn: sqlite3.Connection, client_id: str, config: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO client_configs (client_id, config_json)
        VALUES (?, ?)
        ON CONFLICT(client_id) DO UPDATE SET config_json=excluded.config_json
        """,
        (client_id, json.dumps(config)),
    )
    conn.commit()


def get_client_config(conn: sqlite3.Connection, client_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT config_json FROM client_configs WHERE client_id = ?",
        (client_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return json.loads(row["config_json"])


def add_metric(conn: sqlite3.Connection, req) -> None:
    conn.execute(
        """
        INSERT INTO metrics (ts, round, client_id, stage, loss, num_examples)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            req.round,
            req.client_id,
            req.stage,
            req.loss,
            req.num_examples,
        ),
    )
    conn.commit()


def get_metrics(conn: sqlite3.Connection, limit: int = 50):
    cur = conn.execute(
        """
        SELECT ts, round, client_id, stage, loss, num_examples
        FROM metrics
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cur.fetchall()
