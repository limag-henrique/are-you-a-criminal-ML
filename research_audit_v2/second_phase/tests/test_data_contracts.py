import numpy as np
import pandas as pd
import pytest

from research_audit_v2.second_phase.src.data_contracts import (
    ContractError,
    validate_audit_inputs,
    validate_embeddings,
    validate_manifest,
)


def test_contract_rejects_nonfinite_and_zero_embeddings():
    with pytest.raises(ContractError): validate_embeddings(np.array([[0., 0.], [1., 1.]]))
    with pytest.raises(ContractError): validate_embeddings(np.array([[np.nan, 1.]]))


def test_contract_rejects_missing_and_duplicate_manifest_indices():
    with pytest.raises(ContractError): validate_manifest(pd.DataFrame(), 1)
    frame = pd.DataFrame({"embedding_index": [0, 0], "embedding_status": ["ok", "ok"], "quality": ["low", "low"]})
    with pytest.raises(ContractError): validate_manifest(frame, 2)


@pytest.mark.parametrize(
    "values",
    [
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([1.0, 2.0]),
        np.array([[1.0, np.inf]]),
    ],
)
def test_embedding_contract_rejects_empty_wrong_dimensional_or_nonfinite(values):
    with pytest.raises(ContractError):
        validate_embeddings(values)


def test_manifest_contract_rejects_fractional_missing_and_noncontiguous_indices():
    base = {"embedding_status": ["ok", "ok"], "quality": ["high", "low"]}
    for indices in ([0.5, 1.0], [0, np.nan], [0, 2]):
        frame = pd.DataFrame({"embedding_index": indices, **base})
        with pytest.raises(ContractError):
            validate_manifest(frame, 2)


def test_audit_input_contract_returns_rows_aligned_to_embedding_index():
    manifest = pd.DataFrame(
        {
            "embedding_index": [1, 0, np.nan],
            "embedding_status": ["ok", "ok", "failed"],
            "quality": ["low", "high", "unknown"],
        }
    )
    embeddings = np.array([[10.0, 1.0], [20.0, 2.0]])

    aligned = validate_audit_inputs(manifest, embeddings)

    assert aligned.manifest["embedding_index"].tolist() == [0, 1]
    assert aligned.embeddings.tolist() == [[10.0, 1.0], [20.0, 2.0]]


def test_audit_input_contract_rejects_manifest_matrix_count_mismatch():
    manifest = pd.DataFrame(
        {
            "embedding_index": [0],
            "embedding_status": ["ok"],
            "quality": ["high"],
        }
    )
    with pytest.raises(ContractError, match="row count"):
        validate_audit_inputs(manifest, np.array([[1.0, 2.0], [3.0, 4.0]]))
