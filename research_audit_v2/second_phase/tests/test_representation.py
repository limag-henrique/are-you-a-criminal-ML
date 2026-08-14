import json

import numpy as np

from research_audit_v2.second_phase.src.cross_fitting import FitAuditTrail, run_fold
from research_audit_v2.second_phase.src.privacy_scan import scan_public_tree
from research_audit_v2.second_phase.src.representation import (
    PCA64Spec,
    fit_full_representation,
    fit_representation,
    write_pca_specification,
)


def _pca_config(seed=31):
    return {
        "name": "pca_64",
        "n_components": 64,
        "svd_solver": "randomized",
        "whiten": False,
        "random_state": seed,
        "centering": True,
        "l2_before": True,
        "l2_after": True,
    }


def test_pca64_specification_has_every_methodological_parameter():
    spec = PCA64Spec.from_config(_pca_config())
    assert spec.to_dict() == {
        "name": "pca_64",
        "n_components": 64,
        "svd_solver": "randomized",
        "whiten": False,
        "random_state": 31,
        "centering": True,
        "l2_before": True,
        "l2_after": True,
    }


def test_pca64_fits_only_training_rows_and_records_explained_variance(tmp_path):
    rng = np.random.default_rng(3)
    values = rng.normal(size=(100, 80)).astype("float32")
    train = np.arange(80)
    test = np.arange(80, 100)
    audit = FitAuditTrail(0, train, test)

    fitted = fit_representation(values, train, test, _pca_config(), audit)

    assert fitted.train.shape == (80, 64)
    assert fitted.test.shape == (20, 64)
    assert [event["stage"] for event in audit.events] == ["transformation"]
    assert len(fitted.explained_variance_ratio) == 64
    assert np.isclose(fitted.cumulative_explained_variance[-1], fitted.explained_variance_ratio.sum())
    assert np.allclose(np.linalg.norm(fitted.train, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(fitted.test, axis=1), 1.0)

    output = tmp_path / "pca_specification.json"
    write_pca_specification(output, fitted)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["fit_scope"] == "training_only"
    assert payload["representation"] == "pca_64"
    assert "name" not in payload
    assert payload["explained_variance_ratio"] == fitted.explained_variance_ratio.tolist()
    assert payload["cumulative_explained_variance"] == fitted.cumulative_explained_variance.tolist()
    assert scan_public_tree(tmp_path) == []


def test_pca_cross_fitting_state_is_invariant_to_test_only_perturbation():
    rng = np.random.default_rng(8)
    positive = rng.normal(loc=1.0, scale=0.3, size=(60, 80))
    negative = rng.normal(loc=-1.0, scale=0.3, size=(40, 80))
    vectors = np.vstack([positive, negative]).astype("float32")
    import pandas as pd

    records = pd.DataFrame({"group_id": [f"grp_{i // 2:03d}" for i in range(100)]})
    train = np.arange(80)
    test = np.arange(80, 100)
    config = {
        "outer_folds": 3,
        "k": 2,
        "batch_size": 16,
        "max_iter": 20,
        "n_init": 2,
        "random_seed": 31,
        "target_rule": "largest_cluster",
        "representation": _pca_config(),
        "calibration": "logistic",
    }

    first = run_fold(records, vectors, train, test, config, fold=0)
    perturbed = vectors.copy()
    perturbed[test] = rng.normal(size=(20, 80)) * 100
    second = run_fold(records, perturbed, train, test, config, fold=0)

    assert first.metrics["training_state_sha256"] == second.metrics["training_state_sha256"]
    assert any(event["stage"] == "transformation" for event in first.audit_events)


def test_full_data_pca_is_explicitly_labeled_as_representation_sensitivity():
    values = np.random.default_rng(44).normal(size=(90, 70)).astype("float32")

    fitted = fit_full_representation(values, _pca_config())

    assert fitted.train.shape == (90, 64)
    assert fitted.test.shape == (0, 64)
    assert fitted.specification["fit_scope"] == "all_records_representation_sensitivity"
    assert fitted.specification["historical_attribution"] == "reconstructed_new_method"
    assert np.allclose(np.linalg.norm(fitted.train, axis=1), 1.0)
