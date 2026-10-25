#!/usr/bin/env python3
"""CLI entry point for the read-only explorer API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--layout-dir", type=Path)
    parser.add_argument("--density-dir", type=Path)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()

    mappings = {
        "EXPLORER_CACHE_DIR": args.cache_dir,
        "EXPLORER_LAYOUT_DIR": args.layout_dir,
        "EXPLORER_DENSITY_DIR": args.density_dir,
        "EXPLORER_API_CONTRACT": args.contract,
    }
    for name, value in mappings.items():
        if value is not None:
            os.environ[name] = str(value.resolve())
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False, access_log=True)


if __name__ == "__main__":
    main()
