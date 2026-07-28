"""Inventory restricted inputs without exposing their content."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import sha256_file


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_inventory(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    manifest_path, embedding_path = Path(config["manifest_path"]), Path(config["embeddings_path"])
    manifest = pd.read_csv(manifest_path)
    vectors = np.load(embedding_path, mmap_mode="r")
    result = {
        "git_commit": git_value("rev-parse", "HEAD"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "rows": int(len(manifest)), "columns": list(manifest.columns)},
            "embeddings": {"path": str(embedding_path), "sha256": sha256_file(embedding_path), "shape": list(vectors.shape), "dtype": str(vectors.dtype)},
        },
        "privacy": "No manifest values, paths, names, URLs or embeddings are included in this inventory."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
