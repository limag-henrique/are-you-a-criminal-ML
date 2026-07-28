"""Reusable conditional resampling and multiple-comparison utilities."""
from __future__ import annotations

import numpy as np


def bootstrap_proportion(values: np.ndarray, iterations: int, seed: int, groups: np.ndarray | None = None) -> tuple[float, float]:
    """Return a 95% conditional resampling interval, by group when supplied."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if not len(values):
        return (np.nan, np.nan)
    if groups is None:
        samples = [rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)]
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        samples = []
        for _ in range(iterations):
            picked = rng.choice(unique, len(unique), replace=True)
            selected = np.concatenate([np.flatnonzero(groups == group) for group in picked])
            samples.append(values[selected].mean())
    return tuple(float(value) for value in np.quantile(samples, [.025, .975]))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(corrected)
    result[order] = np.clip(corrected, 0, 1)
    return result
