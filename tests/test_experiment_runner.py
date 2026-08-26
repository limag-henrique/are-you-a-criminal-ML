from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_profile_ml.clustering_backends import FittedClustering, build_backend
from face_profile_ml.experiment_runner import run_specifications
from face_profile_ml.experiment_specs import FitSpec


class CountingBackendFactory:
    def __init__(self) -> None:
        self.fit_count = 0

    def __call__(self, name: str, **kwargs):
        backend = build_backend(name, **kwargs)
        original_fit = backend.fit

        def counted_fit(X: np.ndarray, seed: int):
            self.fit_count += 1
            return original_fit(X, seed)

        object.__setattr__(backend, "fit", counted_fit)
        return backend


def _grouped_two_cluster_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(12)
    groups = [f"group-{index}" for index in range(20)]
    samples = pd.DataFrame(
        {
            "sample_id": [f"negative-{group}" for group in groups]
            + [f"positive-{group}" for group in groups],
            "group_id": groups + groups,
        }
    )
    embeddings = np.vstack(
        [rng.normal(-2, 0.15, size=(20, 3)), rng.normal(2, 0.15, size=(20, 3))]
    )
    return samples, embeddings


def test_rules_reuse_one_clustering_fit_per_fold() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    counting_backend = CountingBackendFactory()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest", "compact", "central"],
        "oof-v1",
        backend_factory=counting_backend,
    )

    assert counting_backend.fit_count == 4
    assert result.specification_metrics["target_rule"].nunique() == 3
    assert len(result.specification_metrics) == 3
    assert set(result.specification_metrics) >= {
        "oof_pooled_cluster_recovery_roc_auc",
        "oof_pooled_cluster_recovery_pr_auc",
        "oof_brier",
        "prevalence",
        "target_size",
        "eligible",
        "status",
    }
    assert result.specification_metrics["spec_id"].is_unique
    assert result.specification_metrics["eligible"].all()
    assert set(result.specification_metrics["status"]) == {"complete"}
    assert set(result.specification_metrics["target_seed"]) != {5}
    assert len(result.oof_predictions) == len(samples) * 3
    assert result.oof_predictions.groupby("spec_id")["fold"].nunique().eq(4).all()
    assert (
        result.oof_predictions["fold_target_seed"]
        == result.oof_predictions["target_seed"] + result.oof_predictions["fold"]
    ).all()
    assert result.failures.empty


class _AlwaysFirstCluster:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


class _SingleClassOOFBackend:
    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        del seed
        labels = np.arange(len(X)) % 2
        centers = np.vstack([X[labels == cluster].mean(axis=0) for cluster in (0, 1)])
        return FittedClustering(
            labels=labels,
            centers=centers,
            inertia=0.0,
            n_iter=1,
            _estimator=_AlwaysFirstCluster(),
        )


class _FailingPredictor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("declared fold prediction failed")


class _AlternatingPredictor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.arange(len(X)) % 2


class _ConstantMarginBackend:
    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        del seed
        return FittedClustering(
            labels=np.arange(len(X)) % 2,
            centers=np.zeros((2, X.shape[1])),
            inertia=0.0,
            n_iter=1,
            _estimator=_AlternatingPredictor(),
        )


class _OneFoldFailureBackend:
    def __init__(self, backend) -> None:
        self.backend = backend

    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        fitted = self.backend.fit(X, seed)
        if seed != 6:
            return fitted
        return FittedClustering(
            labels=fitted.labels,
            centers=fitted.centers,
            inertia=fitted.inertia,
            n_iter=fitted.n_iter,
            objective=fitted.objective,
            _estimator=_FailingPredictor(),
        )


class _FitFailureBackend:
    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        if seed == 6:
            raise RuntimeError("declared fold fit failed")
        labels = np.arange(len(X)) % 2
        centers = np.vstack([X[labels == cluster].mean(axis=0) for cluster in (0, 1)])
        return FittedClustering(
            labels=labels,
            centers=centers,
            inertia=0.0,
            n_iter=1,
            _estimator=_AlternatingPredictor(),
        )


class _OneFoldSingleClassBackend:
    def __init__(self, backend) -> None:
        self.backend = backend

    def fit(self, X: np.ndarray, seed: int) -> FittedClustering:
        fitted = self.backend.fit(X, seed)
        if seed != 6:
            return fitted
        return FittedClustering(
            labels=fitted.labels,
            centers=fitted.centers,
            inertia=fitted.inertia,
            n_iter=fitted.n_iter,
            objective=fitted.objective,
            _estimator=_AlwaysFirstCluster(),
        )


def test_single_class_oof_is_explicitly_ineligible() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest"],
        "oof-v1",
        backend_factory=lambda *args, **kwargs: _SingleClassOOFBackend(),
    )

    metrics = result.specification_metrics.iloc[0]
    assert not bool(metrics["eligible"])
    assert metrics["status"] == "ineligible_single_class"
    assert metrics["prevalence"] == 1.0
    assert metrics["target_size"] == len(samples)
    assert np.isfinite(metrics["oof_brier"])


def test_duplicate_rules_are_rejected_before_fitting() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    counting_backend = CountingBackendFactory()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    with pytest.raises(ValueError, match="rules must be unique"):
        run_specifications(
            samples,
            embeddings,
            specs,
            ["largest", "largest"],
            "oof-v1",
            backend_factory=counting_backend,
        )

    assert counting_backend.fit_count == 0


def test_failed_fold_marks_pooled_specification_partial() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    def backend_factory(name: str, **kwargs):
        return _OneFoldFailureBackend(build_backend(name, **kwargs))

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest"],
        "oof-v1",
        backend_factory=backend_factory,
    )

    metrics = result.specification_metrics.iloc[0]
    assert metrics["status"] == "partial_failed_folds"
    assert not bool(metrics["eligible"])
    assert metrics["expected_folds"] == 4
    assert metrics["completed_folds"] == 3
    assert set(result.failures["fold"]) == {1}


def test_fit_failure_marks_pooled_specification_partial() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest"],
        "oof-v1",
        backend_factory=lambda *args, **kwargs: _FitFailureBackend(),
    )

    metrics = result.specification_metrics.iloc[0]
    assert metrics["status"] == "partial_failed_folds"
    assert not bool(metrics["eligible"])
    assert metrics["expected_folds"] == 4
    assert metrics["completed_folds"] == 3
    assert set(result.failures["fold"]) == {1}


def test_missing_declared_fold_marks_pooled_specification_partial() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    specs = [
        FitSpec("arcface", "minibatch", 3, 2, 5, fold, None)
        for fold in (0, 1, 3)
    ]

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest"],
        "oof-v1",
    )

    metrics = result.specification_metrics.iloc[0]
    assert metrics["status"] == "partial_failed_folds"
    assert not bool(metrics["eligible"])
    assert metrics["expected_folds"] == 4
    assert metrics["completed_folds"] == 3


def test_constant_calibration_fails_positive_rank_equivalence() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest"],
        "oof-v1",
        backend_factory=lambda *args, **kwargs: _ConstantMarginBackend(),
    )

    metrics = result.specification_metrics.iloc[0]
    assert metrics["oof_pooled_cluster_recovery_roc_auc"] == 0.5
    assert metrics["status"] == "failed_rank_equivalence"
    assert not bool(metrics["eligible"])
    assert set(result.failures["status"]) == {"failed_rank_equivalence"}


def test_single_class_fold_is_audited_while_pooled_metric_remains_eligible() -> None:
    samples, embeddings = _grouped_two_cluster_data()
    specs = [FitSpec("arcface", "minibatch", 3, 2, 5, fold, None) for fold in range(4)]

    def backend_factory(name: str, **kwargs):
        return _OneFoldSingleClassBackend(build_backend(name, **kwargs))

    result = run_specifications(
        samples,
        embeddings,
        specs,
        ["largest"],
        "oof-v1",
        backend_factory=backend_factory,
    )

    metrics = result.specification_metrics.iloc[0]
    assert metrics["status"] == "complete"
    assert bool(metrics["eligible"])
    assert metrics["single_class_folds"] == 1
    diagnostic = result.failures.query("stage == 'fold_diagnostic'").iloc[0]
    assert diagnostic["status"] == "ineligible_single_class"
    assert diagnostic["fold"] == 1
