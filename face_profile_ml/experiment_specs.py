"""Immutable, canonical specifications for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_id(prefix: str, value: Any) -> str:
    """Return a prefixed, short SHA-256 identity for a canonical value."""
    digest = sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _require_nonempty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class FitSpec:
    """Factors that determine a fold-level clustering fit."""

    encoder: str
    backend: str
    n_init: int
    k: int
    seed: int
    fold: int
    grouping_threshold: float | None

    def __post_init__(self) -> None:
        _require_nonempty("encoder", self.encoder)
        _require_nonempty("backend", self.backend)
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.fold < 0:
            raise ValueError("fold must be non-negative")

    @property
    def fit_id(self) -> str:
        return stable_id("fit", self)


@dataclass(frozen=True)
class AnalysisSpec:
    """Factors that determine rule-specific analysis of a reusable fit."""

    fit_id: str
    target_rule: str
    target_seed: int
    protocol_id: str

    def __post_init__(self) -> None:
        _require_nonempty("fit_id", self.fit_id)
        _require_nonempty("target_rule", self.target_rule)
        _require_nonempty("protocol_id", self.protocol_id)

    @property
    def spec_id(self) -> str:
        return stable_id("spec", self)
