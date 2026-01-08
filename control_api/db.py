from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS round_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            round INTEGER NOT NULL,
            num_clients INTEGER NOT NULL,
            loss REAL,
            accuracy REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS client_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            round INTEGER NOT NULL,
            client_id TEXT NOT NULL,
            loss REAL,
            accuracy REAL,
            num_examples INTEGER
        )
        """
    )
    conn.commit()
    return conn


def add_round_metrics(
    conn: sqlite3.Connection,
    round_num: int,
    num_clients: int,
    loss: Optional[float],
    accuracy: Optional[float],
) -> None:
    conn.execute(
        """
        INSERT INTO round_metrics (ts, round, num_clients, loss, accuracy)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(), round_num, num_clients, loss, accuracy),
    )
    conn.commit()


def add_client_metrics(
    conn: sqlite3.Connection,
    round_num: int,
    client_id: str,
    loss: Optional[float],
    accuracy: Optional[float],
    num_examples: int,
) -> None:
    conn.execute(
        """
        INSERT INTO client_metrics (ts, round, client_id, loss, accuracy, num_examples)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(), round_num, client_id, loss, accuracy, num_examples),
    )
    conn.commit()


def get_round_metrics(conn: sqlite3.Connection, limit: int = 100) -> List[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT id, ts, round, num_clients, loss, accuracy
        FROM round_metrics
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cur.fetchall()


def get_client_metrics(conn: sqlite3.Connection, limit: int = 200) -> List[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT id, ts, round, client_id, loss, accuracy, num_examples
        FROM client_metrics
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cur.fetchall()
