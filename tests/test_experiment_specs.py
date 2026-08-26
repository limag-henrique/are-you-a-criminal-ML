from dataclasses import FrozenInstanceError

import pytest

from face_profile_ml.experiment_specs import (
    AnalysisSpec,
    FitSpec,
    canonical_json,
    stable_id,
)


def test_fit_id_is_stable_and_excludes_target_rule() -> None:
    fit = FitSpec("arcface", "minibatch", 10, 64, 7, 2, 0.999)

    assert fit.fit_id == FitSpec(
        "arcface", "minibatch", 10, 64, 7, 2, 0.999
    ).fit_id

    largest = AnalysisSpec(fit.fit_id, "largest", 101, "oof-v1")
    compact = AnalysisSpec(fit.fit_id, "compact", 101, "oof-v1")

    assert largest.fit_id == compact.fit_id
    assert largest.spec_id != compact.spec_id


def test_canonical_json_and_stable_id_have_portable_encoding() -> None:
    value = {"b": 2, "a": "café"}

    assert canonical_json(value) == '{"a":"café","b":2}'
    assert stable_id("fit", value) == "fit-6d25402cde044dae"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 1}, "k must be at least 2"),
        ({"n_init": 0}, "n_init must be at least 1"),
        ({"fold": -1}, "fold must be non-negative"),
        ({"encoder": ""}, "encoder must not be empty"),
        ({"encoder": "   "}, "encoder must not be empty"),
        ({"backend": ""}, "backend must not be empty"),
    ],
)
def test_fit_spec_rejects_invalid_identity_factors(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "encoder": "arcface",
        "backend": "minibatch",
        "n_init": 10,
        "k": 64,
        "seed": 7,
        "fold": 2,
        "grouping_threshold": 0.999,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        FitSpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fit_id": ""}, "fit_id must not be empty"),
        ({"fit_id": "   "}, "fit_id must not be empty"),
        ({"target_rule": ""}, "target_rule must not be empty"),
        ({"target_rule": "   "}, "target_rule must not be empty"),
        ({"protocol_id": ""}, "protocol_id must not be empty"),
        ({"protocol_id": "   "}, "protocol_id must not be empty"),
    ],
)
def test_analysis_spec_rejects_empty_categorical_factors(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "fit_id": "fit-123",
        "target_rule": "largest",
        "target_seed": 101,
        "protocol_id": "oof-v1",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        AnalysisSpec(**values)  # type: ignore[arg-type]


def test_specifications_are_immutable() -> None:
    fit = FitSpec("arcface", "minibatch", 10, 64, 7, 2, 0.999)

    with pytest.raises(FrozenInstanceError):
        fit.k = 32  # type: ignore[misc]
