from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    venv_dir = root / ".venv"
    python = sys.executable

    if not venv_dir.exists():
        print("Creating venv...")
        run([python, "-m", "venv", str(venv_dir)])

    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    print("Upgrading pip...")
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing requirements...")
    run([str(venv_python), "-m", "pip", "install", "-r", str(root / "requirements.txt")])

    print("Done. Activate with:")
    if os.name == "nt":
        print(r".\.venv\Scripts\Activate.ps1")
    else:
        print("source .venv/bin/activate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
