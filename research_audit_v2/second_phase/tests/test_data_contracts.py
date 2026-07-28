import numpy as np
import pandas as pd
import pytest

from research_audit_v2.second_phase.src.data_contracts import ContractError, validate_embeddings, validate_manifest


def test_contract_rejects_nonfinite_and_zero_embeddings():
    with pytest.raises(ContractError): validate_embeddings(np.array([[0., 0.], [1., 1.]]))
    with pytest.raises(ContractError): validate_embeddings(np.array([[np.nan, 1.]]))


def test_contract_rejects_missing_and_duplicate_manifest_indices():
    with pytest.raises(ContractError): validate_manifest(pd.DataFrame(), 1)
    frame = pd.DataFrame({"embedding_index": [0, 0], "embedding_status": ["ok", "ok"], "quality": ["low", "low"]})
    with pytest.raises(ContractError): validate_manifest(frame, 2)
