"""Leakage-resistant grouped cross-fitting for a reconstructed synthetic target."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from research_audit_v2.src.common import l2_normalize, write_csv
from research_audit_v2.src.target_heuristic import choose_target_cluster

from .io import atomic_write_text
from .representation import fit_representation


class LeakageError(RuntimeError):
    """Raised whenever a fitted stage receives a test observation or group."""


class FitAuditTrail:
    """Fail-closed fold audit that exports counts, never individual indices."""

    def __init__(self, fold: int, train_indices: np.ndarray, test_indices: np.ndarray):
        self.fold = int(fold)
        self.train_indices = np.asarray(train_indices, dtype=int)
        self.test_indices = np.asarray(test_indices, dtype=int)
        if np.intersect1d(self.train_indices, self.test_indices).size:
            raise LeakageError(f"Fold {fold} has overlapping train/test indices.")
        self.events: list[dict[str, object]] = []

    def record_fit(self, stage: str, indices: np.ndarray) -> None:
        fitted = np.asarray(indices, dtype=int)
        overlap = int(np.intersect1d(fitted, self.test_indices).size)
        outside_train = int(np.setdiff1d(fitted, self.train_indices).size)
        if overlap or outside_train:
            raise LeakageError(
                f"{stage} in fold {self.fold} used records outside the training split "
                f"(test overlap={overlap}, outside train={outside_train})."
            )
        self.events.append(
            {
                "fold": self.fold,
                "stage": stage,
                "fit_n": int(len(fitted)),
                "train_n": int(len(self.train_indices)),
                "test_n": int(len(self.test_indices)),
                "test_overlap": overlap,
                "outside_train": outside_train,
                "fit_scope": "train_only",
            }
        )

    @staticmethod
    def assert_group_disjoint(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
        overlap = set(np.asarray(train_groups).tolist()).intersection(np.asarray(test_groups).tolist())
        if overlap:
            raise LeakageError(f"group_id overlap between train and test: {len(overlap)} group(s).")


@dataclass(frozen=True)
class FoldResult:
    metrics: dict[str, object]
    composition: list[dict[str, object]]
    audit_events: list[dict[str, object]]
    predictions: pd.DataFrame


def _fit_cluster(values: np.ndarray, config: dict[str, Any], seed: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=int(config["k"]),
        random_state=seed,
        batch_size=int(config["batch_size"]),
        max_iter=int(config["max_iter"]),
        n_init=int(config["n_init"]),
        reassignment_ratio=0.01,
    ).fit(values)


def _training_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) != 2:
        return float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if not len(thresholds):
        return float("nan")
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def _state_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=True).encode("utf-8"))
    return digest.hexdigest()


def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float, probabilistic: bool) -> dict[str, float]:
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    two_classes = len(np.unique(y_true)) == 2
    result = {
        "roc_auc": float(roc_auc_score(y_true, score)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, score)) if two_classes else float("nan"),
        "pr_auc_baseline": prevalence,
        "balanced_accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "brier": float(brier_score_loss(y_true, score)) if probabilistic and two_classes else float("nan"),
    }
    if two_classes and np.isfinite(threshold):
        predicted = score >= threshold
        result.update(
            {
                "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
                "recall": float(recall_score(y_true, predicted, zero_division=0)),
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
            }
        )
    return result


def run_fold(
    records: pd.DataFrame,
    vectors: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: dict[str, Any],
    *,
    fold: int,
) -> FoldResult:
    """Fit every learned object on training observations and evaluate test once."""
    started = time.perf_counter()
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    groups = records["group_id"].to_numpy()
    FitAuditTrail.assert_group_disjoint(groups[train], groups[test])
    audit = FitAuditTrail(fold, train, test)

    seed = int(config["random_seed"]) + int(fold)
    fitted_representation = fit_representation(
        np.asarray(vectors),
        train,
        test,
        config.get("representation", "l2_normalized_float32"),
        audit,
    )
    train_values = fitted_representation.train
    test_values = fitted_representation.test

    audit.record_fit("clustering", train)
    model = _fit_cluster(train_values, config, seed)
    train_labels = model.labels_

    audit.record_fit("target_selection", train)
    target = int(choose_target_cluster(train_labels, config["target_rule"]))
    y_train = (train_labels == target).astype(int)

    audit.record_fit("centroid", train)
    centroid = l2_normalize(train_values[y_train == 1].mean(axis=0, keepdims=True))[0]
    raw_train = train_values @ centroid
    raw_test = test_values @ centroid
    test_labels = model.predict(test_values)
    y_test = (test_labels == target).astype(int)

    calibration_name = str(config.get("calibration", "none"))
    calibrator_state: list[np.ndarray] = []
    probabilistic = False
    if calibration_name == "logistic" and len(np.unique(y_train)) == 2:
        audit.record_fit("calibration", train)
        calibrator = LogisticRegression(random_state=seed, solver="lbfgs")
        calibrator.fit(raw_train.reshape(-1, 1), y_train)
        train_score = calibrator.predict_proba(raw_train.reshape(-1, 1))[:, 1]
        test_score = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
        calibrator_state = [calibrator.coef_, calibrator.intercept_]
        probabilistic = True
        calibration_status = "logistic_train_only"
    else:
        train_score = raw_train
        test_score = raw_test
        calibration_status = "none" if calibration_name == "none" else "not_fitted_single_class"

    audit.record_fit("threshold", train)
    threshold = _training_threshold(y_train, train_score)
    state_sha256 = _state_hash(
        fitted_representation.specification,
        *fitted_representation.state_arrays,
        model.cluster_centers_,
        target,
        centroid,
        *calibrator_state,
        threshold,
    )
    positives = int(y_test.sum())
    n = int(len(test))
    metrics: dict[str, object] = {
        "design": "grouped_cross_fitted",
        "fold": int(fold),
        "seed": seed,
        "k": int(config["k"]),
        "n": n,
        "n_groups": int(len(np.unique(groups[test]))),
        "positives": positives,
        "negatives": n - positives,
        "prevalence": float(y_test.mean()),
        "train_n": int(len(train)),
        "train_n_groups": int(len(np.unique(groups[train]))),
        "train_positives": int(y_train.sum()),
        "train_negatives": int(len(train) - y_train.sum()),
        "train_prevalence": float(y_train.mean()),
        "target_cluster_train": target,
        "threshold": threshold,
        "calibration": calibration_status,
        "group_overlap": 0,
        "training_state_sha256": state_sha256,
        "representation": fitted_representation.specification["name"],
        "representation_explained_variance": float(fitted_representation.explained_variance_ratio.sum())
        if len(fitted_representation.explained_variance_ratio)
        else float("nan"),
        "runtime_seconds": time.perf_counter() - started,
        **_metrics(y_test, test_score, threshold, probabilistic),
    }
    composition = [
        {
            "fold": int(fold),
            "split": "train",
            "n": int(len(train)),
            "n_groups": int(len(np.unique(groups[train]))),
            "positives": int(y_train.sum()),
            "negatives": int(len(train) - y_train.sum()),
            "prevalence": float(y_train.mean()),
        },
        {
            "fold": int(fold),
            "split": "test",
            "n": n,
            "n_groups": int(len(np.unique(groups[test]))),
            "positives": positives,
            "negatives": n - positives,
            "prevalence": float(y_test.mean()),
        },
    ]
    record_ids = (
        records.iloc[test]["record_id"].astype(str).to_numpy()
        if "record_id" in records
        else records.index[test].astype(str).to_numpy()
    )
    predictions = pd.DataFrame(
        {
            "record_id": record_ids,
            "group_id": groups[test].astype(str),
            "fold": int(fold),
            "y_true": y_test,
            "score_raw": raw_test,
            "prob_calibrated": test_score,
            "cluster_label": test_labels,
            "distance_to_centroid": np.linalg.norm(
                test_values - model.cluster_centers_[test_labels], axis=1
            ),
        }
    )
    return FoldResult(
        metrics=metrics,
        composition=composition,
        audit_events=audit.events,
        predictions=predictions,
    )


def run_cross_fitting(
    records: pd.DataFrame,
    vectors: np.ndarray,
    config: dict[str, Any],
    tables: Path,
    reports: Path,
) -> pd.DataFrame:
    groups = records["group_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(config["outer_folds"]), len(unique_groups))
    if n_splits < 2:
        raise ValueError("Grouped cross-fitting requires at least two probable-duplicate groups.")
    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, object]] = []
    composition: list[dict[str, object]] = []
    audit_events: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold, (train, test) in enumerate(splitter.split(vectors, groups=groups)):
        result = run_fold(records, vectors, train, test, config, fold=fold)
        rows.append(result.metrics)
        composition.extend(result.composition)
        audit_events.extend(result.audit_events)
        prediction_frames.append(result.predictions)

    metrics = pd.DataFrame(rows)
    write_csv(metrics, tables / "cross_fitted_metrics.csv")
    write_csv(
        pd.concat(prediction_frames, ignore_index=True).sort_values("record_id", ignore_index=True),
        tables / "oof_predictions.csv",
    )
    write_csv(pd.DataFrame(composition), tables / "split_composition.csv")
    audit_frame = pd.DataFrame(audit_events)
    write_csv(audit_frame, tables / "fit_audit_events.csv")
    write_csv(audit_frame, tables / "leakage_audit.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "stage": stage,
                    "fit_scope": "train_only",
                    "test_use": "evaluation_only" if stage == "prediction" else "none",
                    "leakage_guard": "FitAuditTrail",
                }
                for stage in [
                    "representation",
                    "clustering",
                    "target_selection",
                    "centroid",
                    "calibration",
                    "threshold",
                    "prediction",
                ]
            ]
        ),
        tables / "data_usage_matrix.csv",
    )
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        reports / "leakage_and_cross_fitting_report.md",
        "# Leakage and cross-fitting\n\n"
        "All fitted stages use only the outer training split. The test split is used once for evaluation, "
        "and probable-duplicate groups never cross splits. Metrics quantify internal recovery of a reconstructed "
        "synthetic target only; they do not establish biometric, identity, social, legal or criminal validity.\n",
    )
    return metrics
