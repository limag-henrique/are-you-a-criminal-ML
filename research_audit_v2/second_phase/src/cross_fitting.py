"""Leakage-resistant grouped cross-fitting for a reconstructed synthetic target."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from research_audit_v2.src.clustering_stability import jaccard
from research_audit_v2.src.common import l2_normalize, write_csv
from research_audit_v2.src.target_heuristic import choose_target_cluster


def _scores(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    threshold = np.quantile(score, 1-y.mean()) if y.mean() not in {0, 1} else np.inf
    pred = score >= threshold
    return {"roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan, "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan, "balanced_accuracy": balanced_accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "brier": brier_score_loss(y, (score-score.min()) / max(score.max()-score.min(), 1e-12))}


def _fit(values: np.ndarray, cfg: dict, seed: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(n_clusters=cfg["k"], random_state=seed, batch_size=cfg["batch_size"], max_iter=cfg["max_iter"], n_init=cfg["n_init"], reassignment_ratio=.01).fit(values)


def run_cross_fitting(records: pd.DataFrame, vectors: np.ndarray, cfg: dict, tables: Path, reports: Path) -> pd.DataFrame:
    """Fit all learned components only on each external training fold.

    Test labels are model-assigned synthetic memberships and test scores use a
    training-only centroid. This is still internal synthetic-label recovery, but
    it removes direct reuse of test observations while defining the centroid.
    """
    values = l2_normalize(np.asarray(vectors, dtype=np.float32))
    groups = records["group_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(cfg["outer_folds"], len(np.unique(groups))))
    rows, composition = [], []
    for fold, (train, test) in enumerate(splitter.split(values, groups=groups)):
        started = time.perf_counter()
        if set(groups[train]).intersection(groups[test]):
            raise RuntimeError("Probable duplicate group crosses a train/test split.")
        model = _fit(values[train], cfg, cfg["random_seed"] + fold)
        train_labels = model.labels_
        target = choose_target_cluster(train_labels, cfg["target_rule"])
        centroid = l2_normalize(values[train][train_labels == target].mean(axis=0, keepdims=True))[0]
        test_labels = model.predict(values[test])
        y_test = (test_labels == target).astype(int)
        score = values[test] @ centroid
        rows.append({"design": "grouped_cross_fitted", "fold": fold, "seed": cfg["random_seed"] + fold, "k": cfg["k"], "train_records": len(train), "test_records": len(test), "target_prevalence_train": float((train_labels == target).mean()), "target_prevalence_test": float(y_test.mean()), "target_cluster_train": int(target), "centroid_from_test": False, "cluster_fit_uses_test": False, "group_overlap": 0, "runtime_seconds": time.perf_counter()-started, **_scores(y_test, score)})
        composition.append({"fold": fold, "split": "train", "records": len(train), "groups": len(np.unique(groups[train])), "target_prevalence": float((train_labels == target).mean())})
        composition.append({"fold": fold, "split": "test", "records": len(test), "groups": len(np.unique(groups[test])), "target_prevalence": float(y_test.mean())})
    result = pd.DataFrame(rows)
    write_csv(result, tables / "cross_fitted_metrics.csv")
    write_csv(pd.DataFrame(composition), tables / "split_composition.csv")
    usage = pd.DataFrame([
        {"stage": "embedding representation", "artifact": "l2 normalization", "train": "not fit; per-vector deterministic", "validation": "none", "test": "applied", "uses_global_information": False, "possible_leakage": False, "severity": "none", "correction_applied": "No fitted global transform."},
        {"stage": "clustering", "artifact": "MiniBatchKMeans", "train": "fit", "validation": "none", "test": "predict only", "uses_global_information": False, "possible_leakage": False, "severity": "none", "correction_applied": "Separate fit in each outer fold."},
        {"stage": "target selection", "artifact": "largest cluster", "train": "selected", "validation": "none", "test": "not used", "uses_global_information": False, "possible_leakage": False, "severity": "none", "correction_applied": "Target selected only from training labels."},
        {"stage": "centroid score", "artifact": "target centroid", "train": "fit", "validation": "none", "test": "scored", "uses_global_information": False, "possible_leakage": False, "severity": "none", "correction_applied": "Centroid uses training target only."},
        {"stage": "internal original diagnostic", "artifact": "all-record target centroid", "train": "all", "validation": "all", "test": "all", "uses_global_information": True, "possible_leakage": True, "severity": "high", "correction_applied": "Retained only as internal synthetic-label recovery comparator."},
    ])
    write_csv(usage, tables / "data_usage_matrix.csv")
    write_csv(usage, tables / "leakage_audit.csv")
    reports.mkdir(parents=True, exist_ok=True)
    reports.joinpath("leakage_and_cross_fitting_report.md").write_text("# Leakage and cross-fitting\n\nThe original all-record diagnostic is labelled **internal recovery of a synthetic label**. Cross-fitting prevents test observations from defining clusters, the target, or the centroid. The remaining label is still algorithm-assigned and therefore does not establish external, biometric, identity, social, legal or criminal validity.\n", encoding="utf-8")
    return result
