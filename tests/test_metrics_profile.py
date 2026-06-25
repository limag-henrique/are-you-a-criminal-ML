from __future__ import annotations

import numpy as np

from face_profile_ml.metrics import binary_metrics
from face_profile_ml.profile import FaceProfileModel
from face_profile_ml.utils import l2_normalize
from scripts.serve_similarity_app import (
    GallerySimilarityScorer,
    encode_preview_jpeg,
    person_white_filter_bgr,
    white_face_filter_bgr,
)


def test_l2_normalize_rows() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(values)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_profile_scores_positive_direction() -> None:
    rng = np.random.default_rng(42)
    center = l2_normalize(rng.normal(size=32).astype(np.float32))
    positives = l2_normalize(center + rng.normal(scale=0.05, size=(12, 32)).astype(np.float32))
    negatives = l2_normalize(rng.normal(size=(12, 32)).astype(np.float32))

    model = FaceProfileModel(top_k=3).fit(positives)
    pos_score = model.score(positives)["score_raw"].mean()
    neg_score = model.score(negatives)["score_raw"].mean()

    assert pos_score > neg_score


def test_binary_metrics_reports_auc() -> None:
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.9, 0.8, 0.2, 0.1])
    metrics = binary_metrics(labels, scores)
    assert metrics["status"] == "ok"
    assert metrics["auc"] == 1.0


def test_white_face_filter_keeps_face_area_and_whitens_background() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[40:80, 60:100] = (120, 130, 140)

    filtered = white_face_filter_bgr(image, (60.0, 40.0, 100.0, 80.0))

    assert filtered.ndim == 3
    assert filtered.shape[2] == 3
    assert np.all(filtered[0, 0] == 255)
    assert filtered[filtered.shape[0] // 2, filtered.shape[1] // 2].mean() < 250


def test_person_white_filter_returns_previewable_image() -> None:
    image = np.full((160, 180, 3), 255, dtype=np.uint8)
    image[35:125, 65:115] = (95, 120, 150)
    image[70:150, 45:135] = (80, 110, 140)

    filtered = person_white_filter_bgr(image, (65.0, 35.0, 115.0, 125.0))
    encoded = encode_preview_jpeg(filtered)

    assert filtered.shape == image.shape
    assert np.all(filtered[0, 0] == 255)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_gallery_similarity_uses_nearest_visual_neighbor(tmp_path) -> None:
    embeddings = l2_normalize(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.92, 0.08, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "path,subject_id,quality,split,embedding_index\n"
        "a.jpg,a,high,profile,0\n"
        "b.jpg,b,high,profile,1\n"
        "c.jpg,c,high,profile,2\n"
        "d.jpg,d,high,profile,3\n"
        "e.jpg,e,high,profile,4\n",
        encoding="utf-8",
    )

    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_s", -1, 320, calibration_sample=5)
    near = scorer.score_embedding(np.asarray([0.99, 0.01, 0.0, 0.0], dtype=np.float32))
    far = scorer.score_embedding(np.asarray([0.0, 0.7, 0.7, 0.0], dtype=np.float32))

    assert near["nearest"]["subject_id"] in {"a", "b"}
    assert near["best_cosine"] > far["best_cosine"]
