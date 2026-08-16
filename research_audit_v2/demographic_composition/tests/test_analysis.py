from __future__ import annotations

import numpy as np
import pandas as pd

from research_audit_v2.demographic_composition.analysis import (
    compare_scenarios_on_intersection,
    cross_fitted_scores,
    fit_scenario_run,
    pairwise_seed_stability,
)


def _records(vectors: np.ndarray, scenario: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": scenario,
            "record_id": [f"r{index:03d}" for index in range(len(vectors))],
            "source_race_label": ["G1" if index % 2 else "G2" for index in range(len(vectors))],
            "embedding_index": np.arange(len(vectors)),
        }
    )


def _config() -> dict[str, object]:
    return {
        "group_column": "source_race_label",
        "batch_size": 16,
        "max_iter": 50,
        "n_init": 3,
        "outer_folds": 3,
        "target_rule": "largest_cluster",
    }


def test_run_reports_largest_target_cosine_scores_and_complete_cluster_composition():
    rng = np.random.default_rng(7)
    vectors = np.vstack(
        [
            rng.normal([4, 0, 0, 0], 0.1, size=(30, 4)),
            rng.normal([0, 4, 0, 0], 0.1, size=(20, 4)),
            rng.normal([0, 0, 4, 0], 0.1, size=(10, 4)),
        ]
    ).astype(np.float32)

    result = fit_scenario_run(_records(vectors), vectors, "A", seed=31, k=3, config=_config())

    counts = result.partition.groupby("cluster").size()
    assert result.metrics["target_size"] == counts.max()
    assert result.metrics["target_prevalence"] == counts.max() / 60
    assert result.partition.loc[result.partition["is_target"], "score"].mean() > result.partition.loc[
        ~result.partition["is_target"], "score"
    ].mean()
    assert result.composition["count"].sum() == 60
    assert np.allclose(result.composition.groupby("cluster")["proportion_within_cluster"].sum(), 1.0)


def test_pairwise_stability_is_label_invariant_and_uses_target_membership():
    common = pd.DataFrame({"record_id": ["a", "b", "c", "d"]})
    first = common.assign(cluster=[0, 0, 1, 1], is_target=[True, True, False, False])
    relabeled = common.assign(cluster=[8, 8, 3, 3], is_target=[True, True, False, False])
    changed = common.assign(cluster=[0, 1, 0, 1], is_target=[True, False, True, False])

    metrics = pairwise_seed_stability({"s1": first, "s2": relabeled, "s3": changed})

    identical = metrics[(metrics["run_a"] == "s1") & (metrics["run_b"] == "s2")].iloc[0]
    assert identical["ari"] == 1.0
    assert identical["target_jaccard"] == 1.0
    assert len(metrics) == 3


def test_scenario_comparison_aligns_only_shared_record_ids():
    scenario_a = pd.DataFrame(
        {"record_id": ["a", "b", "c", "d"], "cluster": [0, 0, 1, 1], "is_target": [1, 1, 0, 0]}
    )
    scenario_c = pd.DataFrame(
        {"record_id": ["b", "c", "d", "e"], "cluster": [4, 5, 5, 4], "is_target": [1, 0, 0, 1]}
    )

    compared = compare_scenarios_on_intersection({"A": scenario_a, "C": scenario_c})

    assert compared.loc[0, "intersection_n"] == 3
    assert compared.loc[0, "ari"] == 1.0
    assert compared.loc[0, "target_jaccard"] == 1.0


def test_cross_fitted_auc_is_nan_when_target_has_only_one_class():
    vectors = np.eye(12, dtype=np.float32)
    metrics = cross_fitted_scores(_records(vectors), vectors, "A", seed=9, k=1, config=_config())

    assert len(metrics) == 3
    assert metrics["roc_auc"].isna().all()
    assert metrics["pr_auc"].isna().all()
    assert metrics["prevalence"].eq(1.0).all()
