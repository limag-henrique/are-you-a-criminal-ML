"""Uniform clustering backends used by robustness experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class FittedClustering:
    """Common result of fitting a clustering backend.

    ``inertia`` is the native within-cluster sum of squares for K-Means
    backends. GMM and agglomerative clustering do not provide that metric, so
    it is ``NaN`` for those backends. ``objective`` carries the native
    objective when it differs from inertia (GMM's ``lower_bound_``).
    ``n_iter`` is ``NaN`` when the estimator does not expose an iteration
    count. Agglomerative clustering is retained for legacy ``fit_predict``
    use and consequently does not support prediction on new samples.
    """

    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: float | int
    objective: float = np.nan
    _estimator: Any = field(default=None, repr=False, compare=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to the fitted clusters when supported."""
        if self._estimator is None or not hasattr(self._estimator, "predict"):
            raise NotImplementedError(
                "This clustering backend only supports legacy fit_predict"
            )
        return np.asarray(self._estimator.predict(X))


class ClusteringBackend(Protocol):
    def fit(self, X: np.ndarray, seed: int) -> FittedClustering: ...

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray: ...


@dataclass(frozen=True)
class MiniBatchKMeansBackend:
    n_clusters: int
    n_init: int = 10
    batch_size: int = 1024
    max_iter: int = 100

    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        estimator = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=seed,
        ).fit(X)
        return FittedClustering(
            labels=np.asarray(estimator.labels_),
            centers=np.asarray(estimator.cluster_centers_),
            inertia=float(estimator.inertia_),
            n_iter=estimator.n_iter_,
            objective=float(estimator.inertia_),
            _estimator=estimator,
        )

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        return self.fit(X, seed).labels


@dataclass(frozen=True)
class KMeansBackend:
    n_clusters: int
    n_init: int = 10
    max_iter: int = 100

    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        estimator = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=seed,
        ).fit(X)
        return FittedClustering(
            labels=np.asarray(estimator.labels_),
            centers=np.asarray(estimator.cluster_centers_),
            inertia=float(estimator.inertia_),
            n_iter=estimator.n_iter_,
            objective=float(estimator.inertia_),
            _estimator=estimator,
        )

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        return self.fit(X, seed).labels


@dataclass(frozen=True)
class GMMBackend:
    n_clusters: int
    n_init: int = 10
    max_iter: int = 100

    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        estimator = GaussianMixture(
            n_components=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=seed,
        ).fit(X)
        lower_bound = float(estimator.lower_bound_)
        return FittedClustering(
            labels=np.asarray(estimator.predict(X)),
            centers=np.asarray(estimator.means_),
            inertia=float("nan"),
            n_iter=estimator.n_iter_,
            objective=lower_bound,
            _estimator=estimator,
        )

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        return self.fit(X, seed).labels


@dataclass(frozen=True)
class AgglomerativeBackend:
    n_clusters: int

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        del seed
        return AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(X)


def build_backend(
    name: str,
    *,
    n_clusters: int,
    n_init: int = 10,
    batch_size: int = 1024,
    max_iter: int = 100,
) -> ClusteringBackend:
    """Build a configured clustering backend by its stable experiment name."""
    backends = {
        "minibatch": MiniBatchKMeansBackend,
        "kmeans": KMeansBackend,
        "gmm": GMMBackend,
        "agglomerative": AgglomerativeBackend,
    }
    try:
        backend = backends[name]
    except KeyError as exc:
        raise ValueError(f"Unknown clustering backend: {name}") from exc
    if name == "minibatch":
        return backend(
            n_clusters=n_clusters,
            n_init=n_init,
            batch_size=batch_size,
            max_iter=max_iter,
        )
    if name == "agglomerative":
        return backend(n_clusters=n_clusters)
    return backend(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
