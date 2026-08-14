"""Auditable embedding representations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import PCA

from research_audit_v2.src.common import l2_normalize

from .io import atomic_write_json


@dataclass(frozen=True)
class PCA64Spec:
    name: str
    n_components: int
    svd_solver: str
    whiten: bool
    random_state: int
    centering: bool
    l2_before: bool
    l2_after: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PCA64Spec":
        required = {
            "name",
            "n_components",
            "svd_solver",
            "whiten",
            "random_state",
            "centering",
            "l2_before",
            "l2_after",
        }
        missing = required.difference(config)
        if missing:
            raise ValueError(f"PCA-64 configuration is missing: {sorted(missing)}")
        spec = cls(
            name=str(config["name"]),
            n_components=int(config["n_components"]),
            svd_solver=str(config["svd_solver"]),
            whiten=bool(config["whiten"]),
            random_state=int(config["random_state"]),
            centering=bool(config["centering"]),
            l2_before=bool(config["l2_before"]),
            l2_after=bool(config["l2_after"]),
        )
        if spec.name != "pca_64" or spec.n_components != 64:
            raise ValueError("This reconstructed representation must use exactly PCA-64.")
        if not spec.centering:
            raise ValueError("scikit-learn PCA centers inputs; centering must be declared true.")
        return spec

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_components": self.n_components,
            "svd_solver": self.svd_solver,
            "whiten": self.whiten,
            "random_state": self.random_state,
            "centering": self.centering,
            "l2_before": self.l2_before,
            "l2_after": self.l2_after,
        }


@dataclass(frozen=True)
class FittedRepresentation:
    train: np.ndarray
    test: np.ndarray
    specification: dict[str, object]
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    state_arrays: tuple[np.ndarray, ...]


def fit_representation(
    values: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: str | Mapping[str, Any],
    audit: Any,
) -> FittedRepresentation:
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    if isinstance(config, str):
        if config not in {"original", "original_l2", "l2_normalized_float32"}:
            raise ValueError(f"Unknown embedding representation: {config}")
        return FittedRepresentation(
            train=l2_normalize(np.asarray(values[train], dtype=np.float32)),
            test=l2_normalize(np.asarray(values[test], dtype=np.float32)),
            specification={
                "name": "original_l2",
                "n_components": int(values.shape[1]),
                "centering": False,
                "l2_before": True,
                "l2_after": False,
                "fit_scope": "row_local_no_fit",
            },
            explained_variance_ratio=np.array([], dtype=float),
            cumulative_explained_variance=np.array([], dtype=float),
            state_arrays=(),
        )

    spec = PCA64Spec.from_config(config)
    training = np.asarray(values[train], dtype=np.float32)
    testing = np.asarray(values[test], dtype=np.float32)
    if spec.l2_before:
        training = l2_normalize(training)
        testing = l2_normalize(testing)
    audit.record_fit("transformation", train)
    pca = PCA(
        n_components=spec.n_components,
        svd_solver=spec.svd_solver,
        whiten=spec.whiten,
        random_state=spec.random_state,
    )
    transformed_train = pca.fit_transform(training)
    transformed_test = pca.transform(testing)
    if spec.l2_after:
        transformed_train = l2_normalize(transformed_train)
        transformed_test = l2_normalize(transformed_test)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    return FittedRepresentation(
        train=np.asarray(transformed_train, dtype=np.float32),
        test=np.asarray(transformed_test, dtype=np.float32),
        specification={**spec.to_dict(), "fit_scope": "training_only"},
        explained_variance_ratio=explained,
        cumulative_explained_variance=np.cumsum(explained),
        state_arrays=(pca.components_, pca.mean_, pca.explained_variance_, explained),
    )


def write_pca_specification(path: str | Path, fitted: FittedRepresentation) -> None:
    specification = dict(fitted.specification)
    specification["representation"] = specification.pop("name")
    atomic_write_json(
        path,
        {
            **specification,
            "explained_variance_ratio": fitted.explained_variance_ratio.tolist(),
            "cumulative_explained_variance": fitted.cumulative_explained_variance.tolist(),
        },
    )


def fit_full_representation(
    values: np.ndarray, config: Mapping[str, Any]
) -> FittedRepresentation:
    spec = PCA64Spec.from_config(config)
    transformed_input = np.asarray(values, dtype=np.float32)
    if spec.l2_before:
        transformed_input = l2_normalize(transformed_input)
    pca = PCA(
        n_components=spec.n_components,
        svd_solver=spec.svd_solver,
        whiten=spec.whiten,
        random_state=spec.random_state,
    )
    transformed = pca.fit_transform(transformed_input)
    if spec.l2_after:
        transformed = l2_normalize(transformed)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    return FittedRepresentation(
        train=np.asarray(transformed, dtype=np.float32),
        test=np.empty((0, spec.n_components), dtype=np.float32),
        specification={
            **spec.to_dict(),
            "fit_scope": "all_records_representation_sensitivity",
            "historical_attribution": "reconstructed_new_method",
        },
        explained_variance_ratio=explained,
        cumulative_explained_variance=np.cumsum(explained),
        state_arrays=(pca.components_, pca.mean_, pca.explained_variance_, explained),
    )
