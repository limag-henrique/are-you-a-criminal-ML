"""Grouped cross-fitting for explicitly endogenous cluster targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .target_rules import select_target_cluster


OOF_COLUMNS = [
    "sample_id", "group_id", "fold", "y_true", "score_raw",
    "prob_calibrated", "cluster_label", "distance_to_centroid",
    "seed", "k", "target_rule", "threshold",
]


def _scores(values: np.ndarray, centers: np.ndarray, target_cluster: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
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
    required = {"sample_id", "group_id"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"samples is missing columns: {missing}")
    values = np.asarray(embeddings, dtype=float)
    if len(samples) != len(values):
        raise ValueError("samples and embeddings must have equal length")
    if samples["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    if k < 2:
        raise ValueError("k must be at least 2")

    frames: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=n_splits)
    groups = samples["group_id"].astype(str).to_numpy()
    for fold, (train_index, test_index) in enumerate(splitter.split(values, groups=groups)):
        if k >= len(train_index):
            raise ValueError(f"k={k} must be smaller than each training fold")
        clusterer = MiniBatchKMeans(
            n_clusters=k, random_state=seed + fold, n_init=3, batch_size=min(1024, len(train_index))
        ).fit(values[train_index])
        train_labels = clusterer.labels_
        target_cluster = select_target_cluster(
            target_rule,
            train_labels,
            clusterer.cluster_centers_,
            values[train_index],
            seed=seed + fold,
        )
        _, train_scores, _ = _scores(values[train_index], clusterer.cluster_centers_, target_cluster)
        train_target = (train_labels == target_cluster).astype(int)
        calibrator = LogisticRegression(random_state=seed + fold).fit(
            train_scores.reshape(-1, 1), train_target
        )

        test_labels, test_scores, assigned_distance = _scores(
            values[test_index], clusterer.cluster_centers_, target_cluster
        )
        probabilities = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        target = (test_labels == target_cluster).astype(int)
        fold_frame = samples.iloc[test_index][["sample_id", "group_id"]].copy()
        fold_frame["fold"] = fold
        fold_frame["y_true"] = target
        fold_frame["score_raw"] = test_scores
        fold_frame["prob_calibrated"] = probabilities
        fold_frame["cluster_label"] = test_labels
        fold_frame["distance_to_centroid"] = assigned_distance
        fold_frame["seed"] = seed
        fold_frame["k"] = k
        fold_frame["target_rule"] = target_rule
        fold_frame["threshold"] = 0.5
        frames.append(fold_frame)

    oof = pd.concat(frames, ignore_index=True)[OOF_COLUMNS].sort_values("sample_id", ignore_index=True)
    y = oof["y_true"].to_numpy()
    probability = oof["prob_calibrated"].to_numpy()
    metrics: dict[str, float | int] = {
        "n": len(oof),
        "auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
    }
    return oof, metrics

