import numpy as np

from face_profile_ml.bootstrap_ci import bootstrap_grouped_fold_metric, bootstrap_metric


def test_bootstrap_auc_returns_reproducible_ordered_interval() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1] * 10)
    scores = np.tile(np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]), 10)

    first = bootstrap_metric(labels, scores, metric="auc", n_bootstrap=100, seed=9)
    second = bootstrap_metric(labels, scores, metric="auc", n_bootstrap=100, seed=9)

    assert first == second
    assert first.point == 1.0
    assert first.lower <= first.point <= first.upper


def test_bootstrap_jaccard_resamples_paired_target_membership() -> None:
    left = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    right = np.array([1, 0, 0, 0, 1, 1], dtype=bool)

    result = bootstrap_metric(left, right, metric="jaccard", n_bootstrap=80, seed=2)

    assert result.point == 0.5
    assert 0.0 <= result.lower <= result.upper <= 1.0


def test_grouped_fold_bootstrap_resamples_groups_and_averages_fold_metrics() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])

    result = bootstrap_grouped_fold_metric(
        labels,
        scores,
        folds,
        groups,
        metric="auc",
        n_bootstrap=100,
        seed=4,
    )

    assert result.point == 1.0
    assert result.valid_resamples > 0
    assert result.lower <= result.point <= result.upper
