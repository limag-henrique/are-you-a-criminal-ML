import numpy as np
import pandas as pd

from research_audit_v2.second_phase.src.group_audit import (
    safe_threshold_review_sample,
    summarize_probable_duplicate_groups,
)


def test_group_summary_reports_all_required_aggregate_statistics():
    groups = pd.Series(["grp_a", "grp_b", "grp_b", "grp_c", "grp_c", "grp_c"])

    summary, distribution = summarize_probable_duplicate_groups(
        groups, metric="cosine_similarity", threshold=0.999
    )

    assert summary == {
        "records": 6,
        "groups": 3,
        "singleton_groups": 1,
        "non_singleton_groups": 2,
        "grouped_records": 5,
        "grouped_record_proportion": 5 / 6,
        "mean_group_size": 2.0,
        "median_group_size": 2.0,
        "max_group_size": 3,
        "metric": "cosine_similarity",
        "threshold": 0.999,
        "interpretation": "probable_duplicate_only_not_confirmed_identity",
    }
    assert distribution.to_dict("records") == [
        {"group_size": 1, "groups": 1, "records": 1},
        {"group_size": 2, "groups": 1, "records": 2},
        {"group_size": 3, "groups": 1, "records": 3},
    ]


def test_threshold_review_sample_contains_only_pair_hash_and_aggregate_similarity():
    threshold = 0.995
    vectors = np.array(
        [
            [1.0, 0.0],
            [threshold, np.sqrt(1 - threshold**2)],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    records = pd.DataFrame(
        {
            "record_id": ["rec_a", "rec_b", "rec_c", "rec_d"],
            "name": ["private_a", "private_b", "private_c", "private_d"],
            "path": ["private"] * 4,
        }
    )

    sample = safe_threshold_review_sample(
        records,
        vectors,
        threshold=threshold,
        window=0.0001,
        max_pairs=10,
        salt="synthetic-test-salt",
    )

    assert sample.columns.tolist() == [
        "pair_id",
        "similarity",
        "distance_from_threshold",
        "metric",
        "threshold",
        "review_status",
    ]
    assert len(sample) == 1
    assert sample.iloc[0]["pair_id"].startswith("pair_")
    assert sample.iloc[0]["similarity"] == threshold
    assert "private" not in sample.to_csv(index=False)
    assert "rec_a" not in sample.to_csv(index=False)
