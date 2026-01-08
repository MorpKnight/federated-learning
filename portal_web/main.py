from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_app() -> FastAPI:
    app = FastAPI(title="FL Portal")
    root = repo_root()
    static_dir = root / "portal_web" / "static"
    releases_dir = root / "releases"

    app.mount("/releases", StaticFiles(directory=releases_dir), name="releases")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/download")
    def download():
        return FileResponse(static_dir / "download.html")

    return app


app = make_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("PORTAL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORTAL_PORT", "9000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
