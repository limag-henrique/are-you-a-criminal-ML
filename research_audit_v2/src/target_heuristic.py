"""Explicit, testable reconstruction of the cluster-target selection rule."""
from __future__ import annotations

import numpy as np


def choose_target_cluster(labels: np.ndarray, rule: str = "largest_cluster") -> int:
    """Select the largest cluster, breaking ties by its numeric label.

    The original source rule was not preserved in the repository. This is a
    declared reconstruction for sensitivity analysis, not a recovered historic
    implementation.
    """
    if rule != "largest_cluster":
        raise ValueError(f"Unsupported declared target rule: {rule}")
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.flatnonzero(counts == counts.max())[0]])


def target_membership(labels: np.ndarray, target: int) -> np.ndarray:
    return np.asarray(labels) == target
