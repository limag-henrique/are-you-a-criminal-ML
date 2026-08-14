"""Shared deterministic and privacy-safe utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PUBLIC_COLUMNS = {"record_id", "group_id", "source", "quality", "embedding_index"}


def read_config(path: str | Path) -> dict[str, Any]:
    """Read JSON-compatible YAML configuration and validate the minimum contract."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"random_seed", "seeds", "k_values", "manifest_path", "embeddings_path", "output_root"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuration missing required keys: {sorted(missing)}")
    return config


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: object, salt: str, prefix: str = "rec") -> str:
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write only approved public fields with a stable order and atomic replacement."""
    from research_audit_v2.second_phase.src.io import atomic_write_csv

    atomic_write_csv(path, frame)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite, non-zero vectors.")
    return values / norms


def public_lineage(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["record_id", "group_id", "source", "quality", "embedding_index"] if column in frame]
    return frame.loc[:, columns].sort_values("record_id").reset_index(drop=True)
