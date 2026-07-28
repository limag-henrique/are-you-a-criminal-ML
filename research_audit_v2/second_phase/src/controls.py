"""Negative and ground-truth controls that never use restricted inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from research_audit_v2.src.common import write_csv


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
