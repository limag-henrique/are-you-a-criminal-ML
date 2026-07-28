import numpy as np
import pytest

from research_audit_v2.src.target_heuristic import choose_target_cluster


def test_largest_cluster_tie_breaking_is_explicit():
    assert choose_target_cluster(np.array([5, 5, 2, 2])) == 2


def test_unknown_rule_fails_clearly():
    with pytest.raises(ValueError):
        choose_target_cluster(np.array([0, 1]), "unavailable")
