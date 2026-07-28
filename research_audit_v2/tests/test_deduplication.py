import numpy as np
import pandas as pd

from research_audit_v2.src.deduplication import embedding_duplicate_groups


def test_extreme_embedding_groups_are_deterministic():
    records = pd.DataFrame({"record_id": ["a", "b", "c"]})
    values = np.array([[1., 0.], [1., 0.], [0., 1.]])
    groups_a = embedding_duplicate_groups(records, values, .999, "salt")
    groups_b = embedding_duplicate_groups(records, values, .999, "salt")
    assert groups_a.tolist() == groups_b.tolist()
    assert groups_a.iloc[0] == groups_a.iloc[1]
    assert groups_a.iloc[0] != groups_a.iloc[2]
