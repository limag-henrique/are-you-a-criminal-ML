import numpy as np

from research_audit_v2.src.clustering_stability import best_match_proportion, jaccard
from research_audit_v2.src.statistical_uncertainty import benjamini_hochberg


def test_hungarian_and_partition_metrics_are_label_invariant():
    first = np.array([0, 0, 1, 1, 2, 2])
    second = np.array([2, 2, 0, 0, 1, 1])
    assert best_match_proportion(first, second) == 1.0


def test_jaccard_range_and_known_value():
    assert jaccard(np.array([1, 0, 1], dtype=bool), np.array([1, 1, 0], dtype=bool)) == 1 / 3
    assert 0 <= jaccard(np.array([0], dtype=bool), np.array([1], dtype=bool)) <= 1


def test_benjamini_hochberg_is_bounded_and_monotone_in_rank_order():
    corrected = benjamini_hochberg(np.array([.01, .04, .03]))
    assert np.all((0 <= corrected) & (corrected <= 1))
    assert corrected[0] <= corrected[2] <= corrected[1]
