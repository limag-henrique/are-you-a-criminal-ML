from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import face_profile_ml.cross_validation as cross_validation
import face_profile_ml.experiment_runner as experiment_runner
from face_profile_ml.cross_validation import run_grouped_cluster_cv
from face_profile_ml.experiment_specs import FitSpec


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


def test_grouped_cluster_cv_does_not_materialize_sample_cluster_embedding_tensor(monkeypatch) -> None:
    rng = np.random.default_rng(21)
    values = rng.normal(size=(160, 32))
    samples = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(len(values))],
            "group_id": [f"group-{index}" for index in range(len(values))],
        }
    )
    original_norm = experiment_runner.np.linalg.norm

    def reject_three_dimensional_norm(array, *args, **kwargs):
        if np.asarray(array).ndim == 3:
            raise AssertionError("distance scoring materialized an O(n_samples * k * dimensions) tensor")
        return original_norm(array, *args, **kwargs)

    monkeypatch.setattr(experiment_runner.np.linalg, "norm", reject_three_dimensional_norm)

    output, metrics = run_grouped_cluster_cv(samples, values, n_splits=4, k=8, seed=5)

    assert len(output) == len(samples)
    assert metrics["auc"] == roc_auc_score(output["y_true"], output["prob_calibrated"])
    assert metrics["pr_auc"] == average_precision_score(
        output["y_true"], output["prob_calibrated"]
    )


def test_legacy_wrapper_declares_distinct_target_seed_and_legacy_fold_schedule(monkeypatch) -> None:
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
    original_runner = cross_validation.run_specifications
    captured: dict[str, object] = {}

    def observing_runner(*args, **kwargs):
        captured.update(kwargs)
        return original_runner(*args, **kwargs)

    monkeypatch.setattr(cross_validation, "run_specifications", observing_runner)
    run_grouped_cluster_cv(
        samples, values, n_splits=4, k=2, seed=5, target_rule="random"
    )

    declared_target_seed = captured["target_seed"]
    assert declared_target_seed != 5
    fold_seed_factory = captured["target_seed_for_fold"]
    assert callable(fold_seed_factory)
    assert [
        fold_seed_factory(FitSpec("legacy", "minibatch", 3, 2, 5, fold, None), declared_target_seed)
        for fold in range(4)
    ] == [5, 6, 7, 8]
