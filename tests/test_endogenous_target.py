import numpy as np

from face_profile_ml.endogenous_target import (
    EndogenousProposition,
    construct_endogenous_pipeline,
    measure_separability_vs_validity,
)


def test_construct_endogenous_pipeline_preserves_explicit_data_flow() -> None:
    x = np.arange(24, dtype=float).reshape(8, 3)
    z, y, scores = construct_endogenous_pipeline(
        x,
        f=lambda value: value[:, :2],
        h=lambda value: (value[:, 0] > 10).astype(int),
        g=lambda value: value[:, 0],
        seed=7,
    )

    assert z.shape == (8, 2)
    np.testing.assert_array_equal(y, (x[:, 0] > 10).astype(int))
    np.testing.assert_array_equal(scores, x[:, 0])


def test_measure_separability_distinguishes_internal_target_from_external_null() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    external = np.array([0, 1, 0, 1, 0, 1])

    result = measure_separability_vs_validity(y, external, scores)

    assert result.auc_internal == 1.0
    assert 0.0 <= result.auc_vs_external <= 1.0


def test_proposition_demo_is_reproducible_and_internally_separable() -> None:
    result = EndogenousProposition(d=16, n=160, k=4, seed=3).run()
    rerun = EndogenousProposition(d=16, n=160, k=4, seed=3).run()

    assert result == rerun
    assert result.auc_internal > 0.95
    assert abs(result.auc_vs_null - 0.5) < 0.2
    assert 0.0 <= result.jaccard_rerun <= 1.0
    assert result.calibration_brier < 0.15

