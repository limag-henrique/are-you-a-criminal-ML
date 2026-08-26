import numpy as np
import pandas as pd
import pytest

from research_audit_v2.second_phase.src.cross_fitting import (
    FitAuditTrail,
    LeakageError,
    run_cross_fitting,
    run_fold,
)


def _fixture():
    rng = np.random.default_rng(19)
    positive = rng.normal(loc=[3.0, 0.0, 0.0, 0.0], scale=0.15, size=(36, 4))
    negative = rng.normal(loc=[-3.0, 0.0, 0.0, 0.0], scale=0.15, size=(24, 4))
    vectors = np.vstack([positive, negative]).astype("float32")
    records = pd.DataFrame({"group_id": [f"grp_{index // 2:03d}" for index in range(len(vectors))]})
    config = {
        "outer_folds": 3,
        "k": 2,
        "batch_size": 16,
        "max_iter": 20,
        "n_init": 3,
        "random_seed": 23,
        "target_rule": "largest_cluster",
        "representation": "l2_normalized_float32",
        "calibration": "logistic",
    }
    return records, vectors, config


@pytest.mark.parametrize(
    "stage",
    ["transformation", "clustering", "target_selection", "centroid", "threshold", "calibration"],
)
def test_fit_audit_rejects_any_test_record_used_by_a_fitted_stage(stage):
    audit = FitAuditTrail(fold=0, train_indices=np.array([0, 1, 2]), test_indices=np.array([3, 4]))
    with pytest.raises(LeakageError, match=stage):
        audit.record_fit(stage, np.array([0, 3]))


def test_fit_audit_rejects_duplicate_group_overlap():
    with pytest.raises(LeakageError, match="group_id"):
        FitAuditTrail.assert_group_disjoint(
            train_groups=np.array(["grp_a", "grp_b"]),
            test_groups=np.array(["grp_b", "grp_c"]),
        )


def test_test_only_perturbation_cannot_change_any_training_fitted_state():
    records, vectors, config = _fixture()
    train = np.arange(0, 40)
    test = np.arange(40, 60)
    first = run_fold(records, vectors, train, test, config, fold=0)
    perturbed = vectors.copy()
    perturbed[test] = np.random.default_rng(99).normal(size=(len(test), vectors.shape[1])) * 100

    second = run_fold(records, perturbed, train, test, config, fold=0)

    assert first.metrics["training_state_sha256"] == second.metrics["training_state_sha256"]
    assert first.metrics["threshold"] == second.metrics["threshold"]
    assert {event["stage"] for event in first.audit_events} == {
        "clustering",
        "target_selection",
        "centroid",
        "threshold",
        "calibration",
    }
    assert all(event["test_overlap"] == 0 for event in first.audit_events)


def test_grouped_cross_fitting_exports_required_fold_metrics_and_composition(tmp_path):
    records, vectors, config = _fixture()
    tables = tmp_path / "tables"
    reports = tmp_path / "reports"
    tables.mkdir()
    reports.mkdir()

    metrics = run_cross_fitting(records, vectors, config, tables, reports)

    required = {
        "n",
        "n_groups",
        "positives",
        "negatives",
        "prevalence",
        "roc_auc",
        "pr_auc",
        "pr_auc_baseline",
        "threshold",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "brier",
        "calibration",
        "group_overlap",
    }
    assert required.issubset(metrics.columns)
    assert len(metrics) == config["outer_folds"]
    assert metrics["group_overlap"].eq(0).all()
    assert metrics["n"].eq(metrics["positives"] + metrics["negatives"]).all()
    assert np.allclose(metrics["pr_auc_baseline"], metrics["prevalence"])
    assert metrics["brier"].notna().all()
    assert (tables / "cross_fitted_metrics.csv").exists()
    predictions = pd.read_csv(tables / "oof_predictions.csv")
    assert len(predictions) == len(records)
    assert predictions["record_id"].is_unique
    assert {
        "record_id", "group_id", "fold", "y_true", "score_raw",
        "prob_calibrated", "cluster_label", "distance_to_centroid",
    }.issubset(predictions.columns)
    assert (tables / "split_composition.csv").exists()
    assert (tables / "fit_audit_events.csv").exists()


def test_brier_is_omitted_when_no_probabilistic_calibration_exists(tmp_path):
    records, vectors, config = _fixture()
    config["calibration"] = "none"
    tables = tmp_path / "tables"
    reports = tmp_path / "reports"
    tables.mkdir()
    reports.mkdir()

    metrics = run_cross_fitting(records, vectors, config, tables, reports)

    assert metrics["brier"].isna().all()
    assert metrics["calibration"].eq("none").all()
