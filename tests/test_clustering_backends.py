import numpy as np

from face_profile_ml.clustering_backends import build_backend


def test_all_clustering_backends_share_fit_predict_contract() -> None:
    rng = np.random.default_rng(4)
    values = np.vstack(
        [rng.normal(-3, 0.1, size=(12, 2)), rng.normal(3, 0.1, size=(12, 2))]
    )

    for name in ["minibatch", "kmeans", "gmm", "agglomerative"]:
        labels = build_backend(name, n_clusters=2).fit_predict(values, seed=6)
        assert labels.shape == (24,)
        assert len(np.unique(labels)) == 2
