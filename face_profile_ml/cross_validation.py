"""Grouped cross-fitting for explicitly endogenous cluster targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import euclidean_distances

from .experiment_runner import _default_target_seed, run_specifications
from .experiment_specs import FitSpec


OOF_COLUMNS = [
    "sample_id", "group_id", "fold", "y_true", "score_raw",
    "prob_calibrated", "cluster_label", "distance_to_centroid",
    "seed", "k", "target_rule", "threshold",
]


def _scores(values: np.ndarray, centers: np.ndarray, target_cluster: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = euclidean_distances(values, centers)
    labels = np.argmin(distances, axis=1)
    target_distance = distances[:, target_cluster]
    other_distance = np.min(np.delete(distances, target_cluster, axis=1), axis=1)
    return labels, other_distance - target_distance, distances[np.arange(len(values)), labels]


def run_grouped_cluster_cv(
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_splits: int = 5,
    k: int = 64,
    seed: int = 42,
    target_rule: str = "largest",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Fit clustering and calibration on training folds and emit one row per held-out sample."""
    specs = [
        FitSpec("legacy", "minibatch", 3, k, seed, fold, None)
        for fold in range(n_splits)
    ]
    result = run_specifications(
        samples,
        embeddings,
        specs,
        [target_rule],
        "legacy-grouped-cluster-cv",
        target_seed=_default_target_seed(seed),
        target_seed_for_fold=lambda spec, _base_seed: spec.seed + spec.fold,
    )
    hard_failures = result.failures[
        result.failures["status"].astype(str).str.startswith("failed")
    ]
    if not hard_failures.empty:
        message = "; ".join(hard_failures["message"].astype(str))
        raise RuntimeError(f"grouped cluster CV failed: {message}")
    if len(result.specification_metrics) != 1:
        raise RuntimeError("legacy grouped cluster CV expected one pooled specification")
    oof = result.oof_predictions[OOF_COLUMNS].sort_values(
        "sample_id", ignore_index=True
    )
    y = oof["y_true"].to_numpy()
    probability = oof["prob_calibrated"].to_numpy()
    qualified = result.specification_metrics.iloc[0]
    metrics: dict[str, float | int] = {
        "n": len(oof),
        "auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(qualified["oof_brier"]),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
    }
    return oof, metrics
