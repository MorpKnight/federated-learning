from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class TuneResult:
    batch_size: int
    local_epochs: int
    lr: float


def detect_cpu_count() -> int:
    count = os.cpu_count()
    return count if count is not None else 2


def detect_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return (page_size * phys_pages) / (1024**3)
        except Exception:
            pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024**3)
        except Exception:
            pass

    return 4.0


def detect_gpu_available() -> bool:
    return torch.cuda.is_available()


def recommend(
    dataset_size: int,
    cpu_count: int,
    ram_gb: float,
    gpu_available: bool,
) -> TuneResult:
    if gpu_available:
        batch_size = 64
        lr = 0.01
    else:
        batch_size = 32
        lr = 0.01

    if cpu_count >= 8 and ram_gb >= 16:
        batch_size *= 2
        lr = 0.02

    if dataset_size < 5000:
        local_epochs = 2
    elif dataset_size < 20000:
        local_epochs = 1
    else:
        local_epochs = 1

    batch_size = max(16, min(batch_size, 256))
    return TuneResult(batch_size=batch_size, local_epochs=local_epochs, lr=lr)


def tune_from_loader(train_loader) -> Tuple[TuneResult, dict]:
    dataset_size = len(train_loader.dataset)
    cpu_count = detect_cpu_count()
    ram_gb = detect_ram_gb()
    gpu_available = detect_gpu_available()
    result = recommend(dataset_size, cpu_count, ram_gb, gpu_available)
    meta = {
        "dataset_size": dataset_size,
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "gpu_available": gpu_available,
    }
    return result, meta
