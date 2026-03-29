#!/usr/bin/env python3
"""
Tool entrypoint for model template operations.

Usage:
  python tools.py list
  python tools.py fetch <key> <dest>

- list: prints available templates/snippets with keys and descriptions
- fetch: copies a directory template or a snippet file to <dest>

This is used by agents as a stable "tool-like" interface.
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.json"


def load_index():
    with open(INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_list():
    idx = load_index()
    for key, meta in idx.items():
        print(f"{key}\t{meta['type']}\t{meta.get('description','')}")


def copy_any(src: Path, dest: Path):
    if src.is_dir():
        if dest.exists() and not dest.is_dir():
            raise SystemExit(f"Destination exists and is not a directory: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest / src.name, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def cmd_fetch(key: str, dest: str):
    idx = load_index()
    if key not in idx:
        raise SystemExit(f"Unknown key: {key}")
    meta = idx[key]
    src = ROOT / meta["path"]
    dst = Path(dest)
    if not src.exists():
        raise SystemExit(f"Source does not exist: {src}")
    copy_any(src, dst)
    print(f"Copied {src} -> {dst}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "fetch":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python tools.py fetch <key> <dest>")
        cmd_fetch(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
