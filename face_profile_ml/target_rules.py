"""Auditable rules for selecting one cluster as the endogenous target."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _cluster_ids(labels: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(labels))
    if values.size == 0:
        raise ValueError("labels must contain at least one cluster")
    return values.astype(int)


def _centroid(cluster_id: int, centroids: np.ndarray) -> np.ndarray:
    if cluster_id < 0 or cluster_id >= len(centroids):
        raise ValueError(f"centroids does not contain cluster {cluster_id}")
    return np.asarray(centroids[cluster_id], dtype=float)


def largest_cluster(labels: np.ndarray, centroids: np.ndarray, X: np.ndarray) -> int:
    """Select the largest cluster; ties resolve to the smallest label."""
    del centroids, X
    ids, counts = np.unique(np.asarray(labels), return_counts=True)
    return int(ids[np.flatnonzero(counts == counts.max())[0]])


def most_compact_cluster(labels: np.ndarray, centroids: np.ndarray, X: np.ndarray) -> int:
    """Select the cluster with the smallest mean squared centroid distance."""
    labels_array = np.asarray(labels)
    values = np.asarray(X, dtype=float)
    scores = []
    for cluster_id in _cluster_ids(labels_array):
        delta = values[labels_array == cluster_id] - _centroid(cluster_id, centroids)
        scores.append((float(np.mean(np.sum(delta * delta, axis=1))), int(cluster_id)))
    return min(scores)[1]


def _global_centroid(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    if len(labels) != len(X):
        raise ValueError("labels and X must have the same number of rows")
    return np.asarray(X, dtype=float).mean(axis=0)


def most_separated_cluster(labels: np.ndarray, centroids: np.ndarray, X: np.ndarray) -> int:
    """Compatibility alias for :func:`isolated_cluster`."""
    return isolated_cluster(labels, centroids, X)


def random_cluster(
    labels: np.ndarray,
    centroids: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Select uniformly from observed cluster identifiers."""
    del centroids, X
    return int(rng.choice(_cluster_ids(labels)))


def central_cluster(labels: np.ndarray, centroids: np.ndarray, X: np.ndarray) -> int:
    """Select the centroid closest to the global sample centroid."""
    global_center = _global_centroid(labels, X)
    distances = [
        (float(np.linalg.norm(_centroid(cluster_id, centroids) - global_center)), int(cluster_id))
        for cluster_id in _cluster_ids(labels)
    ]
    return min(distances)[1]


def outlier_cluster(labels: np.ndarray, centroids: np.ndarray, X: np.ndarray) -> int:
    """Select the centroid farthest from the global sample centroid."""
    global_center = _global_centroid(labels, X)
    distances = [
        (float(np.linalg.norm(_centroid(cluster_id, centroids) - global_center)), -int(cluster_id))
        for cluster_id in _cluster_ids(labels)
    ]
    return -max(distances)[1]


def isolated_cluster(labels: np.ndarray, centroids: np.ndarray, X: np.ndarray) -> int:
    """Select the centroid with the most distant nearest observed neighbour."""
    del X
    ids = _cluster_ids(labels)
    observed_centroids = np.vstack(
        [_centroid(cluster_id, centroids) for cluster_id in ids]
    )
    pairwise = np.linalg.norm(
        observed_centroids[:, None, :] - observed_centroids[None, :, :], axis=2
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest_distances = pairwise.min(axis=1)
    return int(ids[np.argmax(nearest_distances)])


TARGET_RULES: dict[str, Callable[..., int]] = {
    "largest": largest_cluster,
    "compact": most_compact_cluster,
    "most_compact": most_compact_cluster,
    "separated": most_separated_cluster,
    "most_separated": most_separated_cluster,
    "isolated": isolated_cluster,
    "random": random_cluster,
    "central": central_cluster,
    "outlier": outlier_cluster,
}


def select_target_cluster(
    rule: str,
    labels: np.ndarray,
    centroids: np.ndarray,
    X: np.ndarray,
    *,
    seed: int = 0,
) -> int:
    """Dispatch a named rule using a deterministic RNG for the null rule."""
    try:
        function = TARGET_RULES[rule]
    except KeyError as exc:
        raise ValueError(f"Unknown target rule: {rule}") from exc
    if function is random_cluster:
        return random_cluster(labels, centroids, X, np.random.default_rng(seed))
    return function(labels, centroids, X)
