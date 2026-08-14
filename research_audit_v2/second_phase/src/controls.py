"""Negative and ground-truth controls that never use restricted inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from research_audit_v2.src.clustering_stability import jaccard
from research_audit_v2.src.common import write_csv
from research_audit_v2.src.target_heuristic import choose_target_cluster


def negative_controls(seed: int, tables: Path, reports: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random_values = rng.normal(size=(300, 32))
    random_labels = KMeans(3, n_init=5, random_state=seed).fit_predict(random_values)
    known_centres = np.array([[0]*16, [5]*16, [-5]*16])
    truth = np.repeat(np.arange(3), 100)
    structured = np.vstack([rng.normal(loc=centre, scale=.3, size=(100, 16)) for centre in known_centres])
    recovered = KMeans(3, n_init=5, random_state=seed).fit_predict(structured)
    duplicate = np.vstack([structured[:20], structured[:20], structured[20:]])
    duplicate_groups = len(duplicate) - len(np.unique(duplicate, axis=0))
    frame = pd.DataFrame([
        {"control": "isotropic_random_embeddings", "expected": "no external ground truth or claim", "observed_ari": np.nan, "pass": True},
        {"control": "known_separated_clusters", "expected": "high ARI", "observed_ari": adjusted_rand_score(truth, recovered), "pass": adjusted_rand_score(truth, recovered) > .95},
        {"control": "controlled_exact_duplicates", "expected": "duplicates present", "observed_ari": duplicate_groups, "pass": duplicate_groups == 20},
        {"control": "permuted_cluster_labels", "expected": "partition metric invariant", "observed_ari": adjusted_rand_score(recovered, (recovered + 1) % 3), "pass": adjusted_rand_score(recovered, (recovered + 1) % 3) == 1.0},
    ])
    write_csv(frame, tables / "negative_control_results.csv")
    reports.joinpath("negative_controls_report.md").write_text("# Negative controls\n\nThese controls use generated data only. They verify that the software recovers simple inserted structure and preserves label-permutation invariance; they cannot validate the restricted dataset's social meaning.\n", encoding="utf-8")
    return frame


def synthetic_geometry_control(seed: int, output_path: Path) -> pd.DataFrame:
    """Demonstrate circular geometry and target instability on generated data."""
    rng = np.random.default_rng(seed)
    target_parts = [
        rng.normal(loc=[-4.0, -0.6], scale=0.12, size=(60, 2)),
        rng.normal(loc=[-4.0, 0.6], scale=0.12, size=(60, 2)),
    ]
    other_parts = [
        rng.normal(loc=[0.0, 0.0], scale=0.18, size=(100, 2)),
        rng.normal(loc=[4.0, 0.0], scale=0.18, size=(80, 2)),
    ]
    values = np.vstack([*target_parts, *other_parts])
    base_labels = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(values)
    base_cluster = choose_target_cluster(base_labels, "largest_cluster")
    base_target = base_labels == base_cluster
    centroid = values[base_target].mean(axis=0)
    score = -np.linalg.norm(values - centroid, axis=1)
    auc = float(roc_auc_score(base_target.astype(int), score))

    perturbed_labels = KMeans(n_clusters=4, n_init=10, random_state=seed).fit_predict(values)
    perturbed_cluster = choose_target_cluster(perturbed_labels, "largest_cluster")
    perturbed_target = perturbed_labels == perturbed_cluster
    target_jaccard = jaccard(base_target, perturbed_target)

    result = pd.DataFrame(
        [
            {
                "demonstration": "clustering_generates_synthetic_target",
                "base_k": 3,
                "perturbed_k": np.nan,
                "roc_auc": np.nan,
                "target_jaccard": np.nan,
                "pass": bool(base_target.any() and (~base_target).any()),
                "interpretation": "methodological_demonstration_only",
            },
            {
                "demonstration": "same_geometry_score_has_high_separability",
                "base_k": 3,
                "perturbed_k": np.nan,
                "roc_auc": auc,
                "target_jaccard": np.nan,
                "pass": auc > 0.95,
                "interpretation": "methodological_demonstration_only",
            },
            {
                "demonstration": "target_changes_when_clustering_is_perturbed",
                "base_k": 3,
                "perturbed_k": 4,
                "roc_auc": np.nan,
                "target_jaccard": target_jaccard,
                "pass": target_jaccard < 0.8,
                "interpretation": "methodological_demonstration_only",
            },
        ]
    )
    write_csv(result, output_path)
    return result
