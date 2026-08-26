"""Paired, reproducible bootstrap confidence intervals for OOF metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    n_bootstrap: int
    alpha: float
    valid_resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_bool, right_bool = left.astype(bool), right.astype(bool)
    union = np.logical_or(left_bool, right_bool).sum()
    return float(np.logical_and(left_bool, right_bool).sum() / union) if union else 1.0


def _metric_function(name: str, threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "auc": lambda y, s: float(roc_auc_score(y, s)),
        "pr_auc": lambda y, s: float(average_precision_score(y, s)),
        "brier": lambda y, s: float(brier_score_loss(y, s)),
        "balanced_accuracy": lambda y, s: float(balanced_accuracy_score(y, s >= threshold)),
        "jaccard": _jaccard,
    }
    try:
        return functions[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported bootstrap metric: {name}") from exc


def bootstrap_metric(
    y_true: np.ndarray,
    values: np.ndarray,
    *,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    threshold: float = 0.5,
) -> BootstrapResult:
    """Bootstrap paired observations and return a percentile interval."""
    first = np.asarray(y_true).reshape(-1)
    second = np.asarray(values).reshape(-1)
    if len(first) != len(second) or len(first) == 0:
        raise ValueError("Inputs must be non-empty and have equal length")
    if n_bootstrap < 1 or not 0 < alpha < 1:
        raise ValueError("n_bootstrap must be positive and alpha must be in (0, 1)")
    function = _metric_function(metric, threshold)
    point = function(first, second)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(first), size=len(first))
        try:
            estimate = function(first[indices], second[indices])
        except ValueError:
            continue
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        raise ValueError("No valid bootstrap resamples were produced")
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(point, float(lower), float(upper), n_bootstrap, alpha, len(estimates))


def bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    return bootstrap_metric(
        y_true, scores, metric="auc", n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
    )

