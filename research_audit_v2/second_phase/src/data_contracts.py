"""Small dependency-free contracts for restricted audit inputs and public outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Raised before a non-conformant artifact can enter the analysis."""


REQUIRED_MANIFEST = {"embedding_index", "embedding_status", "quality"}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ContractError(f"{name} is missing required columns: {sorted(missing)}")


def validate_embeddings(values: np.ndarray, expected_rows: int | None = None) -> None:
    if not isinstance(values, np.ndarray) or values.ndim != 2 or not values.shape[1]:
        raise ContractError("Embeddings must be a non-empty two-dimensional array.")
    if expected_rows is not None and len(values) != expected_rows:
        raise ContractError(f"Embedding row count {len(values)} does not match expected count {expected_rows}.")
    if not np.isfinite(values).all():
        raise ContractError("Embeddings contain NaN or infinity.")
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ContractError("Embeddings contain a zero vector.")


def validate_manifest(frame: pd.DataFrame, embedding_count: int) -> None:
    require_columns(frame, REQUIRED_MANIFEST, "Embedding manifest")
    successful = frame.loc[(frame["embedding_status"] == "ok") & frame["embedding_index"].notna()].copy()
    if successful.empty:
        raise ContractError("Embedding manifest contains no successful embeddings.")
    indices = successful["embedding_index"].astype(int)
    if indices.duplicated().any():
        raise ContractError("Successful embedding indices must be unique.")
    if indices.min() < 0 or indices.max() >= embedding_count:
        raise ContractError("Embedding indices are outside the embedding matrix.")
    if len(indices) != embedding_count:
        raise ContractError("Successful manifest rows do not match the embedding matrix row count.")


def validate_groups(groups: pd.Series) -> None:
    if groups.isna().any() or not groups.astype(str).str.startswith("grp_").all():
        raise ContractError("Each record must have a non-sensitive probable-duplicate group ID.")
