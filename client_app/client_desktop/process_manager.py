from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional


@dataclass
class ProcessStatus:
    running: bool = False
    pid: Optional[int] = None
    connected: bool = False
    last_loss: Optional[float] = None
    last_acc: Optional[float] = None
    last_log: str = ""


class ProcessManager:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.process: Optional[subprocess.Popen[str]] = None
        self.logs: Deque[str] = deque(maxlen=500)
        self.status = ProcessStatus()
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self, config_path: Path, client_id: str) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                raise RuntimeError("Training already running")
            cmd = [
                sys.executable,
                "-m",
                "fl_client.client",
                "--config",
                str(config_path),
                "--client-id",
                client_id,
            ]
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.status = ProcessStatus(running=True, pid=self.process.pid)
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self.process or self.process.poll() is not None:
                self.status.running = False
                return
            self.process.terminate()
        time.sleep(1)
        with self._lock:
            if self.process and self.process.poll() is None:
                self.process.kill()
            self.status.running = False

    def _reader_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip()
            self.logs.append(line)
            self.status.last_log = line
            if "Connecting to server" in line:
                self.status.connected = True
            if "Failed to connect" in line:
                self.status.connected = False
            if "fit loss=" in line or "eval loss=" in line:
                self._parse_metrics(line)
        with self._lock:
            if self.process:
                self.status.running = False

    def _parse_metrics(self, line: str) -> None:
        match = re.search(r"loss=([0-9.]+)\s+acc=([0-9.]+)", line)
        if not match:
            return
        try:
            self.status.last_loss = float(match.group(1))
            self.status.last_acc = float(match.group(2))
        except ValueError:
            return

    def get_logs(self, since: int = 0) -> tuple[list[str], int]:
        lines = list(self.logs)
        return lines[since:], len(lines)
