from __future__ import annotations

import numpy as np
import pandas as pd

from face_profile_ml.fairness import audit_group_metrics


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [1, 1, 0, 0] * 10 + [1, 0],
            "score": [0.95, 0.85, 0.20, 0.10] * 10 + [0.99, 0.05],
            "skin_tone_protocol": ["I-II"] * 20 + ["V-VI"] * 20 + ["III-IV"] * 2,
            "ethnicity_self_described": ["group_a"] * 20 + ["group_b"] * 20 + ["group_c"] * 2,
            "quality": ["high", "mid", "high", "low"] * 10 + ["high", "low"],
        }
    )


def test_audit_reports_documented_groups_and_intersections() -> None:
    summary, rows = audit_group_metrics(
        _scores(),
        group_columns=["skin_tone_protocol", "ethnicity_self_described", "quality"],
        min_group_n=10,
        bootstrap_rounds=10,
        seed=7,
    )

    assert summary["groups_reported"] > 0
    assert summary["global_eer_threshold"] > 0
    assert "skin_tone_protocol × ethnicity_self_described" in set(rows["audit_dimension"])
    group_a = rows.loc[(rows["audit_dimension"] == "skin_tone_protocol") & (rows["group"] == "I-II")].iloc[0]
    assert group_a["status"] == "ok"
    assert np.isclose(group_a["fmr"], 0.0)
    assert isinstance(group_a["fmr_ci95"], list)


def test_audit_suppresses_small_groups() -> None:
    _, rows = audit_group_metrics(_scores(), group_columns=["skin_tone_protocol"], min_group_n=10)
    small = rows.loc[rows["group"] == "III-IV"].iloc[0]
    assert small["status"] == "suppressed_small_group"
    assert pd.isna(small["auc"])


def test_audit_uses_a_frozen_calibration_threshold_when_provided() -> None:
    summary, rows = audit_group_metrics(
        _scores(), group_columns=["skin_tone_protocol"], min_group_n=10, threshold=0.80
    )
    assert summary["threshold_source"] == "provided_calibration_threshold"
    assert (rows["threshold"] == 0.80).all()


def test_audit_rejects_non_binary_labels() -> None:
    frame = _scores()
    frame.loc[0, "label"] = 2
    try:
        audit_group_metrics(frame, group_columns=["skin_tone_protocol"])
    except ValueError as error:
        assert "0 (impostor) and 1 (genuine)" in str(error)
    else:
        raise AssertionError("non-binary labels must be rejected")
