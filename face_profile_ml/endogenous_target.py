"""Constructive demonstration of separability without construct validity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .target_rules import largest_cluster


ArrayTransform = Callable[[np.ndarray], np.ndarray]


def construct_endogenous_pipeline(
    X: np.ndarray,
    f: ArrayTransform,
    h: ArrayTransform,
    g: ArrayTransform,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the declared flow ``Z=f(X), Y=h(Z), s=g(Z)``."""
    del seed
    values = np.asarray(X)
    z = np.asarray(f(values))
    y = np.asarray(h(z), dtype=int).reshape(-1)
    scores = np.asarray(g(z), dtype=float).reshape(-1)
    if len(z) != len(y) or len(y) != len(scores):
        raise ValueError("f, h and g must preserve the sample axis")
    return z, y, scores


@dataclass(frozen=True)
class SeparabilityValidityResult:
    auc_internal: float
    auc_vs_external: float


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        raise ValueError("AUC requires exactly two classes")
    return float(roc_auc_score(labels, scores))


def measure_separability_vs_validity(
    Y: np.ndarray,
    C_external: np.ndarray,
    scores: np.ndarray,
    X_test: np.ndarray | None = None,
) -> SeparabilityValidityResult:
    """Compare recovery of the endogenous target and an external criterion."""
    del X_test
    y = np.asarray(Y, dtype=int).reshape(-1)
    external = np.asarray(C_external, dtype=int).reshape(-1)
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    if not (len(y) == len(external) == len(score_values)):
        raise ValueError("Y, C_external and scores must have equal length")
    return SeparabilityValidityResult(
        auc_internal=_safe_auc(y, score_values),
        auc_vs_external=_safe_auc(external, score_values),
    )


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


@dataclass(frozen=True)
class PropositionResult:
    d: int
    n: int
    k: int
    seed: int
    auc_internal: float
    auc_vs_null: float
    jaccard_rerun: float
    calibration_brier: float


@dataclass(frozen=True)
class EndogenousProposition:
    """Parameterized synthetic witness for Proposition 1."""

    d: int = 512
    n: int = 500
    k: int = 4
    seed: int = 0

    def run(self) -> PropositionResult:
        if self.n <= self.k or self.k < 2 or self.d < 1:
            raise ValueError("Require d >= 1 and n > k >= 2")
        rng = np.random.default_rng(self.seed)
        latent = rng.normal(size=(self.n, self.d))
        model = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(latent)
        labels = model.labels_
        target_cluster = largest_cluster(labels, model.cluster_centers_, latent)
        target = labels == target_cluster
        all_distances = np.linalg.norm(
            latent[:, None, :] - model.cluster_centers_[None, :, :], axis=2
        )
        target_distance = all_distances[:, target_cluster]
        other_distance = np.min(
            np.delete(all_distances, target_cluster, axis=1), axis=1
        )
        score = other_distance - target_distance

        external = rng.integers(0, 2, size=self.n)
        if len(np.unique(external)) < 2:
            external[0], external[1] = 0, 1
        comparison = measure_separability_vs_validity(target, external, score)

        rerun_model = KMeans(n_clusters=self.k, n_init=1, random_state=self.seed + 10_000).fit(latent)
        rerun_cluster = largest_cluster(rerun_model.labels_, rerun_model.cluster_centers_, latent)
        rerun_target = rerun_model.labels_ == rerun_cluster

        calibrated = LogisticRegression(random_state=self.seed).fit(
            score.reshape(-1, 1), target.astype(int)
        ).predict_proba(score.reshape(-1, 1))[:, 1]
        return PropositionResult(
            d=self.d,
            n=self.n,
            k=self.k,
            seed=self.seed,
            auc_internal=comparison.auc_internal,
            auc_vs_null=comparison.auc_vs_external,
            jaccard_rerun=_jaccard(target, rerun_target),
            calibration_brier=float(brier_score_loss(target, calibrated)),
        )
