import numpy as np

from face_profile_ml.clustering_backends import build_backend


def test_kmeans_backends_honor_declared_n_init() -> None:
    mini = build_backend("minibatch", n_clusters=2, n_init=20)
    full = build_backend("kmeans", n_clusters=2, n_init=50)

    assert mini.n_init == 20
    assert full.n_init == 50


def test_all_clustering_backends_share_fit_predict_contract() -> None:
    rng = np.random.default_rng(4)
    values = np.vstack(
        [rng.normal(-3, 0.1, size=(12, 2)), rng.normal(3, 0.1, size=(12, 2))]
    )

    for name in ["minibatch", "kmeans", "gmm", "agglomerative"]:
        labels = build_backend(name, n_clusters=2).fit_predict(values, seed=6)
        assert labels.shape == (24,)
        assert len(np.unique(labels)) == 2


def test_fitted_backends_expose_geometry_and_predict_contract() -> None:
    rng = np.random.default_rng(9)
    values = np.vstack(
        [rng.normal(-3, 0.1, size=(12, 2)), rng.normal(3, 0.1, size=(12, 2))]
    )

    for name in ["minibatch", "kmeans", "gmm"]:
        fitted = build_backend(name, n_clusters=2, n_init=2).fit(values, seed=6)

        assert fitted.labels.shape == (24,)
        assert fitted.centers.shape == (2, 2)
        assert fitted.predict(values).shape == (24,)

    gmm = build_backend("gmm", n_clusters=2, n_init=2).fit(values, seed=6)
    assert np.isnan(gmm.inertia)
    assert np.isfinite(gmm.objective)


def test_agglomerative_remains_legacy_fit_predict_only() -> None:
    rng = np.random.default_rng(10)
    values = np.vstack(
        [rng.normal(-3, 0.1, size=(12, 2)), rng.normal(3, 0.1, size=(12, 2))]
    )

    labels = build_backend("agglomerative", n_clusters=2).fit_predict(values, seed=6)

    assert labels.shape == (24,)
    assert len(np.unique(labels)) == 2
