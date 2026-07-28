"""Seed and k sensitivity for a fixed, documented clustering procedure."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

from .common import l2_normalize, write_csv
from .target_heuristic import choose_target_cluster, target_membership


def best_match_proportion(reference: np.ndarray, candidate: np.ndarray) -> float:
    left, left_index = np.unique(reference, return_inverse=True)
    right, right_index = np.unique(candidate, return_inverse=True)
    contingency = np.zeros((len(left), len(right)), dtype=int)
    np.add.at(contingency, (left_index, right_index), 1)
    rows, cols = linear_sum_assignment(-contingency)
    return float(contingency[rows, cols].sum() / len(reference))


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def fit_clusters(values: np.ndarray, k: int, seed: int, config: dict[str, Any], algorithm: str | None = None) -> tuple[object, np.ndarray]:
    chosen = algorithm or config["cluster_algorithm"]
    params = {"n_clusters": k, "random_state": seed, "n_init": config["n_init"], "max_iter": config["cluster_max_iter"]}
    model = MiniBatchKMeans(**params, batch_size=config["cluster_batch_size"], reassignment_ratio=0.01) if chosen == "minibatch" else KMeans(**params)
    labels = model.fit_predict(values)
    return model, labels


def _summary(frame: pd.DataFrame, grouping: str) -> pd.DataFrame:
    numerical = ["ari", "nmi", "hungarian_match", "target_jaccard", "target_size", "target_proportion", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz", "runtime_seconds"]
    return frame.groupby(grouping)[numerical].agg(["mean", "median", "std", "min", "max", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).reset_index()


def run_stability(records: pd.DataFrame, vectors: np.ndarray, config: dict[str, Any], tables: Path, figures: Path, reports: Path) -> pd.DataFrame:
    values = l2_normalize(np.asarray(vectors, dtype=np.float32))
    rng = np.random.default_rng(config["random_seed"])
    seeds = rng.integers(1, 2**31 - 1, size=config["seeds"], endpoint=False).tolist()
    rows: list[dict[str, object]] = []
    baseline: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in config["k_values"]:
        for ordinal, seed in enumerate(seeds):
            started = time.perf_counter()
            model, labels = fit_clusters(values, int(k), int(seed), config)
            target = choose_target_cluster(labels, config["target_rule"])
            membership = target_membership(labels, target)
            if k not in baseline:
                baseline[k] = (labels, membership)
            reference_labels, reference_target = baseline[k]
            sample_indices = np.arange(len(values)) if len(values) <= config["max_silhouette_sample"] else rng.choice(len(values), config["max_silhouette_sample"], replace=False)
            sample = values[sample_indices]
            sample_labels = labels[sample_indices]
            rows.append({"algorithm": config["cluster_algorithm"], "k": k, "seed": seed, "seed_ordinal": ordinal, "ari": adjusted_rand_score(reference_labels, labels), "nmi": normalized_mutual_info_score(reference_labels, labels), "hungarian_match": best_match_proportion(reference_labels, labels), "target_jaccard": jaccard(reference_target, membership), "heuristic_target_jaccard": jaccard(reference_target, membership), "target_cluster": target, "target_size": int(membership.sum()), "target_proportion": float(membership.mean()), "inertia": float(model.inertia_), "silhouette": float(silhouette_score(sample, sample_labels)) if len(np.unique(sample_labels)) > 1 else np.nan, "davies_bouldin": float(davies_bouldin_score(sample, sample_labels)), "calinski_harabasz": float(calinski_harabasz_score(sample, sample_labels)), "runtime_seconds": time.perf_counter() - started, "converged": True, "iterations": int(getattr(model, "n_iter_", 0))})
    result = pd.DataFrame(rows)
    result.to_parquet(tables / "clustering_all_runs.parquet", index=False)
    write_csv(_summary(result, "k"), tables / "clustering_summary_by_k.csv")
    write_csv(_summary(result, "seed_ordinal"), tables / "clustering_summary_by_seed.csv")
    figures.mkdir(parents=True, exist_ok=True)
    for metric, name in [("ari", "ari_by_k.svg"), ("nmi", "nmi_by_k.svg"), ("target_jaccard", "target_jaccard_distribution.svg"), ("target_size", "target_size_distribution.svg")]:
        fig, ax = plt.subplots(figsize=(6, 4)); result.boxplot(column=metric, by="k", ax=ax); fig.suptitle(""); ax.set_title(metric.replace("_", " ")); ax.set_xlabel("k"); fig.tight_layout(); fig.savefig(figures / name); plt.close(fig)
    reports.mkdir(parents=True, exist_ok=True)
    reports.joinpath("clustering_stability_report.md").write_text("# Clustering stability\n\nThe target rule is a declared largest-cluster reconstruction because the original rule was not found. Internal clustering indices measure geometry only and do not validate social constructs. Dispersion is conditional to this audited dataset.\n", encoding="utf-8")
    return result
