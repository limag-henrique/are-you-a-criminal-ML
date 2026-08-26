import numpy as np

from face_profile_ml.target_rules import select_target_cluster


def test_target_rules_select_clusters_from_declared_geometry() -> None:
    values = np.array(
        [
            [-0.1, 0.0], [0.0, 0.0], [0.1, 0.0],
            [9.0, 0.0], [11.0, 0.0],
            [2.9, 0.0], [3.0, 0.0], [3.1, 0.0], [3.0, 0.1],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    centroids = np.vstack([values[labels == i].mean(axis=0) for i in range(3)])

    assert select_target_cluster("largest", labels, centroids, values) == 2
    assert select_target_cluster("compact", labels, centroids, values) == 0
    assert select_target_cluster("separated", labels, centroids, values) == 1
    assert select_target_cluster("central", labels, centroids, values) == 2
    assert select_target_cluster("outlier", labels, centroids, values) == 1
    assert select_target_cluster("random", labels, centroids, values, seed=4) in {0, 1, 2}


def test_target_rule_ties_are_deterministic() -> None:
    values = np.array([[0.0], [0.0], [2.0], [2.0]])
    labels = np.array([4, 4, 9, 9])
    centroids = np.zeros((10, 1))
    centroids[4] = 0.0
    centroids[9] = 2.0

    assert select_target_cluster("largest", labels, centroids, values) == 4

