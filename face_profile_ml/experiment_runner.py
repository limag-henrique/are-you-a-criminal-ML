"""Reusable grouped-fold fits and rule-specific OOF analysis."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GroupKFold

from .clustering_backends import ClusteringBackend, FittedClustering, build_backend
from .experiment_specs import AnalysisSpec, FitSpec, stable_id
from .target_rules import select_target_cluster


@dataclass(frozen=True)
class FoldFit:
    """A fitted clustering model plus the data split needed for later analyses."""

    spec: FitSpec
    fitted: FittedClustering
    train_index: np.ndarray
    test_index: np.ndarray
    train_samples: pd.DataFrame
    test_samples: pd.DataFrame
    train_embeddings: np.ndarray
    test_embeddings: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    """Tabular outputs from a set of reusable fits and analytical rules."""

    oof_predictions: pd.DataFrame
    fit_index: pd.DataFrame
    specification_metrics: pd.DataFrame
    failures: pd.DataFrame


BackendFactory = Callable[..., ClusteringBackend]
TargetSeedFactory = Callable[[FitSpec, int], int]

FAILURE_COLUMNS = ["stage", "fit_id", "spec_id", "fold", "status", "message"]
METRIC_COLUMNS = [
    "spec_id",
    "encoder",
    "backend",
    "n_init",
    "k",
    "seed",
    "grouping_threshold",
    "target_rule",
    "target_seed",
    "protocol_id",
    "expected_folds",
    "completed_folds",
    "single_class_folds",
    "oof_pooled_cluster_recovery_roc_auc",
    "oof_pooled_cluster_recovery_pr_auc",
    "oof_brier",
    "prevalence",
    "target_size",
    "eligible",
    "status",
]


def _validate_inputs(samples: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    required = {"sample_id", "group_id"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"samples is missing columns: {missing}")
    values = np.asarray(embeddings, dtype=float)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(samples) != len(values):
        raise ValueError("samples and embeddings must have equal length")
    if samples["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    return values


def _fold_splits(samples: pd.DataFrame, values: np.ndarray, fit_specs: Sequence[FitSpec]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not fit_specs:
        raise ValueError("fit_specs must contain at least one specification")
    n_splits = max(spec.fold for spec in fit_specs) + 1
    if n_splits < 2:
        raise ValueError("fit_specs must declare at least two folds")
    groups = samples["group_id"].astype(str).to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(values, groups=groups))


def fit_grouped_folds(
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    fit_specs: Iterable[FitSpec],
    *,
    backend_factory: BackendFactory = build_backend,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, FoldFit]:
    """Fit every declared grouped fold exactly once, keyed by ``fit_id``."""
    values = _validate_inputs(samples, embeddings)
    specs = list(fit_specs)
    splits = _fold_splits(samples, values, specs)
    fits: dict[str, FoldFit] = {}
    for spec in specs:
        if spec.fit_id in fits:
            raise ValueError(f"duplicate fit specification: {spec.fit_id}")
        train_index, test_index = splits[spec.fold]
        if spec.k >= len(train_index):
            raise ValueError(f"k={spec.k} must be smaller than each training fold")
        try:
            backend = backend_factory(
                spec.backend,
                n_clusters=spec.k,
                n_init=spec.n_init,
                batch_size=min(1024, len(train_index)),
            )
            fitted = backend.fit(values[train_index], spec.seed + spec.fold)
        except Exception as exc:
            if failures is None:
                raise
            failures.append(
                {
                    "stage": "fit",
                    "fit_id": spec.fit_id,
                    "spec_id": "",
                    "fold": spec.fold,
                    "status": "failed_fit",
                    "message": str(exc),
                }
            )
            continue
        fits[spec.fit_id] = FoldFit(
            spec=spec,
            fitted=fitted,
            train_index=train_index,
            test_index=test_index,
            train_samples=samples.iloc[train_index].copy(),
            test_samples=samples.iloc[test_index].copy(),
            train_embeddings=values[train_index],
            test_embeddings=values[test_index],
        )
    return fits


def _margin_scores(values: np.ndarray, centers: np.ndarray, target_cluster: int) -> tuple[np.ndarray, np.ndarray]:
    distances = euclidean_distances(values, centers)
    target_distance = distances[:, target_cluster]
    other_distance = np.min(np.delete(distances, target_cluster, axis=1), axis=1)
    return other_distance - target_distance, distances


def analyze_fold_fit(fold_fit: FoldFit, rule: str, target_seed: int) -> pd.DataFrame:
    """Analyze one held-out fold for a target rule without refitting clustering."""
    fitted = fold_fit.fitted
    target_cluster = select_target_cluster(
        rule,
        fitted.labels,
        fitted.centers,
        fold_fit.train_embeddings,
        seed=target_seed,
    )
    train_scores, _ = _margin_scores(
        fold_fit.train_embeddings, fitted.centers, target_cluster
    )
    train_target = (np.asarray(fitted.labels) == target_cluster).astype(int)
    if np.unique(train_target).size < 2:
        raise ValueError("training target contains a single class")
    calibrator = LogisticRegression(random_state=target_seed).fit(
        train_scores.reshape(-1, 1), train_target
    )

    test_scores, test_distances = _margin_scores(
        fold_fit.test_embeddings, fitted.centers, target_cluster
    )
    test_labels = np.asarray(fitted.predict(fold_fit.test_embeddings), dtype=int)
    probabilities = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    assigned_distance = test_distances[np.arange(len(test_labels)), test_labels]
    output = fold_fit.test_samples[["sample_id", "group_id"]].copy()
    output["fold"] = fold_fit.spec.fold
    output["y_true"] = (test_labels == target_cluster).astype(int)
    output["score_raw"] = test_scores
    output["prob_calibrated"] = probabilities
    output["cluster_label"] = test_labels
    output["distance_to_centroid"] = assigned_distance
    output["seed"] = fold_fit.spec.seed
    output["k"] = fold_fit.spec.k
    output["target_rule"] = rule
    output["threshold"] = 0.5
    output["fit_id"] = fold_fit.spec.fit_id
    output["target_seed"] = target_seed
    output["target_cluster"] = target_cluster
    output["calibration_coefficient"] = float(calibrator.coef_[0, 0])
    return output


def _fit_family(spec: FitSpec) -> dict[str, Any]:
    factors = asdict(spec)
    factors.pop("fold")
    return factors


def _default_target_seed(clustering_seed: int) -> int:
    """Derive a deterministic analytical seed distinct from clustering."""
    return (clustering_seed + 1_000_003) % (2**32 - 1)


def _analysis_spec(
    spec: FitSpec, rule: str, protocol_id: str, target_seed: int
) -> AnalysisSpec:
    family_id = stable_id("fit-family", _fit_family(spec))
    return AnalysisSpec(family_id, rule, target_seed, protocol_id)


def _fit_index(fits: dict[str, FoldFit]) -> pd.DataFrame:
    rows = []
    for fit_id, fold_fit in fits.items():
        fitted = fold_fit.fitted
        rows.append(
            {
                "fit_id": fit_id,
                **asdict(fold_fit.spec),
                "status": "complete",
                "inertia": fitted.inertia,
                "n_iter": fitted.n_iter,
                "objective": fitted.objective,
            }
        )
    return pd.DataFrame(rows)


def _qualified_metrics(
    oof: pd.DataFrame, expected: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observed = (
        {
            spec_id: group
            for spec_id, group in oof.groupby("spec_id", sort=True)
        }
        if not oof.empty
        else {}
    )
    for spec_id, declaration in sorted(expected.items()):
        group = observed.get(spec_id, pd.DataFrame())
        expected_fold_ids = set(declaration["folds"])
        completed_fold_ids = set(group["fold"].unique()) if not group.empty else set()
        y_true = group["y_true"].to_numpy() if not group.empty else np.array([])
        raw = group["score_raw"].to_numpy() if not group.empty else np.array([])
        calibrated = (
            group["prob_calibrated"].to_numpy() if not group.empty else np.array([])
        )
        single_class_fold_ids = (
            [
                int(fold)
                for fold, fold_rows in group.groupby("fold", sort=True)
                if fold_rows["y_true"].nunique() < 2
            ]
            if not group.empty
            else []
        )
        for fold in single_class_fold_ids:
            fold_rows = group[group["fold"] == fold]
            failures.append(
                {
                    "stage": "fold_diagnostic",
                    "fit_id": str(fold_rows.iloc[0]["fit_id"]),
                    "spec_id": spec_id,
                    "fold": fold,
                    "status": "ineligible_single_class",
                    "message": "held-out fold contains a single target class",
                }
            )
        base = {
            "spec_id": spec_id,
            **declaration["factors"],
            "expected_folds": len(expected_fold_ids),
            "completed_folds": len(completed_fold_ids),
            "single_class_folds": len(single_class_fold_ids),
            "prevalence": float(np.mean(y_true)) if len(y_true) else np.nan,
            "target_size": int(np.sum(y_true)),
        }
        if completed_fold_ids != expected_fold_ids:
            rows.append(
                {
                    **base,
                    "oof_pooled_cluster_recovery_roc_auc": np.nan,
                    "oof_pooled_cluster_recovery_pr_auc": np.nan,
                    "oof_brier": np.nan,
                    "eligible": False,
                    "status": (
                        "partial_failed_folds"
                        if completed_fold_ids
                        else "failed_all_folds"
                    ),
                }
            )
            continue
        if np.unique(y_true).size < 2:
            rows.append(
                {
                    **base,
                    "oof_pooled_cluster_recovery_roc_auc": np.nan,
                    "oof_pooled_cluster_recovery_pr_auc": np.nan,
                    "oof_brier": float(brier_score_loss(y_true, calibrated)),
                    "eligible": False,
                    "status": "ineligible_single_class",
                }
            )
            continue
        raw_auc = float(roc_auc_score(y_true, raw))
        rank_equivalent = True
        rank_details = ""
        for fold, fold_rows in group.groupby("fold", sort=True):
            fold_y = fold_rows["y_true"].to_numpy()
            if np.unique(fold_y).size < 2:
                continue
            coefficients = fold_rows["calibration_coefficient"].unique()
            if (
                len(coefficients) != 1
                or not np.isfinite(coefficients[0])
                or coefficients[0] <= 0.0
            ):
                rank_equivalent = False
                rank_details = (
                    f"fold {fold} calibration coefficient must be strictly positive; "
                    f"observed {coefficients.tolist()}"
                )
                break
            fold_raw = fold_rows["score_raw"].to_numpy()
            fold_calibrated = fold_rows["prob_calibrated"].to_numpy()
            order = np.argsort(fold_raw, kind="stable")
            raw_differences = np.diff(fold_raw[order])
            calibrated_differences = np.diff(fold_calibrated[order])
            reverses_order = (raw_differences > 0.0) & (
                calibrated_differences <= 0.0
            )
            splits_tie = (raw_differences == 0.0) & (
                calibrated_differences != 0.0
            )
            if np.any(reverses_order | splits_tie):
                rank_equivalent = False
                rank_details = (
                    f"fold {fold} calibrated probabilities do not preserve "
                    "the raw-score ordering and ties"
                )
                break
        if not rank_equivalent:
            status = "failed_rank_equivalence"
            eligible = False
            failures.append(
                {
                    "stage": "metrics",
                    "fit_id": "",
                    "spec_id": spec_id,
                    "fold": int(fold),
                    "status": status,
                    "message": rank_details,
                }
            )
        else:
            status = "complete"
            eligible = True
        rows.append(
            {
                **base,
                "oof_pooled_cluster_recovery_roc_auc": raw_auc,
                "oof_pooled_cluster_recovery_pr_auc": float(
                    average_precision_score(y_true, raw)
                ),
                "oof_brier": float(brier_score_loss(y_true, calibrated)),
                "eligible": eligible,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS), failures


def run_specifications(
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    fit_specs: Iterable[FitSpec],
    rules: Iterable[str],
    protocol_id: str,
    *,
    target_seed: int | None = None,
    target_seed_for_fold: TargetSeedFactory | None = None,
    backend_factory: BackendFactory = build_backend,
) -> ExperimentResult:
    """Run grouped fits once and reuse them across all target rules."""
    specs = list(fit_specs)
    declared_rules = list(rules)
    if not declared_rules:
        raise ValueError("rules must contain at least one target rule")
    if len(declared_rules) != len(set(declared_rules)):
        raise ValueError("rules must be unique")
    fit_failures: list[dict[str, Any]] = []
    expected: dict[str, dict[str, Any]] = {}
    for spec in specs:
        declared_target_seed = (
            _default_target_seed(spec.seed) if target_seed is None else target_seed
        )
        for rule in declared_rules:
            analysis = _analysis_spec(spec, rule, protocol_id, declared_target_seed)
            declaration = expected.setdefault(
                analysis.spec_id,
                {
                    "folds": set(),
                    "factors": {
                        **_fit_family(spec),
                        "target_rule": rule,
                        "target_seed": analysis.target_seed,
                        "protocol_id": protocol_id,
                    },
                },
            )
            declaration["folds"].add(spec.fold)
    # ``GroupKFold`` is configured from the highest declared fold, so a
    # skipped fold is still part of the expected protocol and must not make a
    # pooled result look complete.
    for declaration in expected.values():
        highest_fold = max(declaration["folds"])
        declaration["folds"] = set(range(highest_fold + 1))
    fits = fit_grouped_folds(
        samples,
        embeddings,
        specs,
        backend_factory=backend_factory,
        failures=fit_failures,
    )
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = list(fit_failures)
    for spec in specs:
        fold_fit = fits.get(spec.fit_id)
        if fold_fit is None:
            continue
        for rule in declared_rules:
            declared_target_seed = (
                _default_target_seed(spec.seed) if target_seed is None else target_seed
            )
            analysis = _analysis_spec(
                spec, rule, protocol_id, declared_target_seed
            )
            if target_seed_for_fold is None:
                fold_target_seed = (analysis.target_seed + spec.fold) % (2**32 - 1)
            else:
                fold_target_seed = target_seed_for_fold(spec, analysis.target_seed)
            try:
                frame = analyze_fold_fit(fold_fit, rule, fold_target_seed)
            except Exception as exc:
                failures.append(
                    {
                        "stage": "analysis",
                        "fit_id": spec.fit_id,
                        "spec_id": analysis.spec_id,
                        "fold": spec.fold,
                        "status": "failed_analysis",
                        "message": str(exc),
                    }
                )
                continue
            frame["fold_target_seed"] = frame["target_seed"]
            frame["target_seed"] = analysis.target_seed
            frame["spec_id"] = analysis.spec_id
            frame["encoder"] = spec.encoder
            frame["backend"] = spec.backend
            frame["n_init"] = spec.n_init
            frame["grouping_threshold"] = spec.grouping_threshold
            frame["protocol_id"] = protocol_id
            frames.append(frame)
    oof = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not oof.empty:
        oof = oof.sort_values(["spec_id", "sample_id"], ignore_index=True)
    metrics, metric_failures = _qualified_metrics(oof, expected)
    failures.extend(metric_failures)
    return ExperimentResult(
        oof_predictions=oof,
        fit_index=_fit_index(fits),
        specification_metrics=metrics,
        failures=pd.DataFrame(failures, columns=FAILURE_COLUMNS),
    )
