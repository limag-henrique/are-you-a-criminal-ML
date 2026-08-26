"""Uniform clustering backends used by robustness experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


class ClusteringBackend(Protocol):
    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray: ...


@dataclass(frozen=True)
class MiniBatchKMeansBackend:
    n_clusters: int
    batch_size: int = 1024

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        return MiniBatchKMeans(
            self.n_clusters, batch_size=self.batch_size, n_init=3, random_state=seed
        ).fit_predict(X)


@dataclass(frozen=True)
class KMeansBackend:
    n_clusters: int

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        return KMeans(self.n_clusters, n_init=10, random_state=seed).fit_predict(X)


@dataclass(frozen=True)
class GMMBackend:
    n_clusters: int

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        return GaussianMixture(self.n_clusters, random_state=seed).fit_predict(X)


@dataclass(frozen=True)
class AgglomerativeBackend:
    n_clusters: int

    def fit_predict(self, X: np.ndarray, seed: int) -> np.ndarray:
        del seed
        return AgglomerativeClustering(self.n_clusters).fit_predict(X)


def build_backend(name: str, *, n_clusters: int) -> ClusteringBackend:
    backends = {
        "minibatch": MiniBatchKMeansBackend,
        "kmeans": KMeansBackend,
        "gmm": GMMBackend,
        "agglomerative": AgglomerativeBackend,
    }
    try:
        return backends[name](n_clusters)
    except KeyError as exc:
        raise ValueError(f"Unknown clustering backend: {name}") from exc
