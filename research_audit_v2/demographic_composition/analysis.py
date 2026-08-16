"""Fixed clustering, target scoring, and stability metrics for A-D scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from research_audit_v2.src.common import l2_normalize
from research_audit_v2.src.target_heuristic import choose_target_cluster


@dataclass(frozen=True)
class RunResult:
    metrics: dict[str, object]
    partition: pd.DataFrame
    composition: pd.DataFrame


def _model(k: int, seed: int, config: Mapping[str, object]) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        batch_size=int(config["batch_size"]),
        max_iter=int(config["max_iter"]),
        n_init=int(config["n_init"]),
        reassignment_ratio=0.01,
    )


def _scenario_values(
    records: pd.DataFrame, vectors: np.ndarray, scenario: str
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = records[records["scenario"].eq(scenario)].sort_values("record_id").reset_index(drop=True)
    if frame.empty or frame["record_id"].duplicated().any():
        raise ValueError(f"Scenario {scenario} must contain unique records.")
    values = l2_normalize(np.asarray(vectors[frame["embedding_index"].to_numpy()], dtype=np.float32))
    return frame, values


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def fit_scenario_run(
    records: pd.DataFrame,
    vectors: np.ndarray,
    scenario: str,
    seed: int,
    k: int,
    config: Mapping[str, object],
) -> RunResult:
    frame, values = _scenario_values(records, vectors, scenario)
    if int(k) > len(values):
        raise ValueError("k cannot exceed the scenario sample size.")
    fitted = _model(k, seed, config).fit(values)
    labels = fitted.labels_.astype(int)
    target_cluster = int(choose_target_cluster(labels, str(config["target_rule"])))
    target = labels == target_cluster
    centroid = l2_normalize(values[target].mean(axis=0, keepdims=True))[0]
    score = values @ centroid
    group_column = str(config["group_column"])
    partition = frame.loc[:, ["record_id", "scenario", group_column]].copy()
    partition["cluster"] = labels
    partition["is_target"] = target
    partition["score"] = score
    composition = (
        partition.groupby(["cluster", group_column], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    composition["proportion_within_cluster"] = composition["count"] / composition.groupby("cluster")[
        "count"
    ].transform("sum")
    composition.insert(0, "scenario", scenario)
    composition.insert(1, "seed", int(seed))
    composition.insert(2, "k", int(k))
    return RunResult(
        metrics={
            "scenario": scenario,
            "seed": int(seed),
            "k": int(k),
            "target_cluster": target_cluster,
            "target_size": int(target.sum()),
            "target_prevalence": float(target.mean()),
            "inertia": float(fitted.inertia_),
            "iterations": int(fitted.n_iter_),
        },
        partition=partition,
        composition=composition,
    )


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    columns = ["record_id", "cluster", "is_target"]
    return left.loc[:, columns].merge(
        right.loc[:, columns], on="record_id", suffixes=("_a", "_b"), validate="one_to_one"
    )


def pairwise_seed_stability(partitions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for left_name, right_name in combinations(sorted(partitions), 2):
        aligned = _aligned(partitions[left_name], partitions[right_name])
        if aligned.empty:
            raise ValueError("Partitions have no shared records.")
        rows.append(
            {
                "run_a": left_name,
                "run_b": right_name,
                "intersection_n": int(len(aligned)),
                "ari": float(adjusted_rand_score(aligned["cluster_a"], aligned["cluster_b"])),
                "target_jaccard": _jaccard(
                    aligned["is_target_a"].to_numpy(bool), aligned["is_target_b"].to_numpy(bool)
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_scenarios_on_intersection(partitions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for left_name, right_name in combinations(sorted(partitions), 2):
        aligned = _aligned(partitions[left_name], partitions[right_name])
        if aligned.empty:
            raise ValueError(f"Scenarios {left_name} and {right_name} have no shared records.")
        rows.append(
            {
                "scenario_a": left_name,
                "scenario_b": right_name,
                "intersection_n": int(len(aligned)),
                "ari": float(adjusted_rand_score(aligned["cluster_a"], aligned["cluster_b"])),
                "target_jaccard": _jaccard(
                    aligned["is_target_a"].to_numpy(bool), aligned["is_target_b"].to_numpy(bool)
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_fitted_scores(
    records: pd.DataFrame,
    vectors: np.ndarray,
    scenario: str,
    seed: int,
    k: int,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Evaluate cosine recovery of train-derived cluster membership on held-out records."""
    _, values = _scenario_values(records, vectors, scenario)
    folds = min(int(config["outer_folds"]), len(values))
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two records.")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
    rows = []
    for fold, (train, test) in enumerate(splitter.split(values)):
        if int(k) > len(train):
            raise ValueError("k cannot exceed a cross-fitting training split.")
        fitted = _model(k, int(seed) + fold, config).fit(values[train])
        train_labels = fitted.labels_.astype(int)
        target_cluster = int(choose_target_cluster(train_labels, str(config["target_rule"])))
        train_target = train_labels == target_cluster
        centroid = l2_normalize(values[train][train_target].mean(axis=0, keepdims=True))[0]
        score = values[test] @ centroid
        target = fitted.predict(values[test]) == target_cluster
        prevalence = float(target.mean())
        two_classes = len(np.unique(target)) == 2
        rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "k": int(k),
                "fold": int(fold),
                "n": int(len(test)),
                "target_size": int(target.sum()),
                "prevalence": prevalence,
                "roc_auc": float(roc_auc_score(target, score)) if two_classes else np.nan,
                "pr_auc": float(average_precision_score(target, score)) if two_classes else np.nan,
                "pr_auc_baseline": prevalence,
            }
        )
    return pd.DataFrame(rows)
