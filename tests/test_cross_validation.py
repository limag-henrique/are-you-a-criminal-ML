from pathlib import Path

import numpy as np
import pandas as pd

from face_profile_ml.cross_validation import run_grouped_cluster_cv


def test_grouped_cluster_cv_emits_complete_oof_contract(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    values = np.vstack(
        [rng.normal(-2, 0.2, size=(20, 3)), rng.normal(2, 0.2, size=(20, 3))]
    )
    samples = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(40)],
            "group_id": [f"group-{index // 2}" for index in range(40)],
        }
    )

    output, metrics = run_grouped_cluster_cv(
        samples, values, n_splits=4, k=2, seed=5, target_rule="largest"
    )

    assert len(output) == 40
    assert output["sample_id"].is_unique
    assert set(output.columns) == {
        "sample_id", "group_id", "fold", "y_true", "score_raw",
        "prob_calibrated", "cluster_label", "distance_to_centroid",
        "seed", "k", "target_rule", "threshold",
    }
    assert set(metrics) >= {"auc", "pr_auc", "brier", "balanced_accuracy"}
    for _, fold in output.groupby("fold"):
        test_groups = set(fold["group_id"])
        assert test_groups
