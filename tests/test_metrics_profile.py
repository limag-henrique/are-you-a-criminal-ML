from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import cv2

from face_profile_ml.extractor import select_available_providers
from face_profile_ml.metrics import binary_metrics
from face_profile_ml.profile import FaceProfileModel
from face_profile_ml.utils import l2_normalize
from scripts.serve_similarity_app import (
    GallerySimilarityScorer,
    SimilarityThresholds,
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


def test_select_available_providers_uses_cpu_when_cuda_is_missing(monkeypatch) -> None:
    ort = types.SimpleNamespace(
        get_available_providers=lambda: ["AzureExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    assert select_available_providers() == ["CPUExecutionProvider"]


def test_select_available_providers_keeps_cuda_when_available(monkeypatch) -> None:
    ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    assert select_available_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


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
    for name, value in {
        "a.jpg": 40,
        "b.jpg": 80,
        "c.jpg": 120,
        "d.jpg": 160,
        "e.jpg": 200,
    }.items():
        cv2.imwrite(str(tmp_path / name), np.full((24, 20, 3), value, dtype=np.uint8))
    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "path,subject_id,quality,split,embedding_index\n"
        f"{tmp_path / 'a.jpg'},a,high,profile,0\n"
        f"{tmp_path / 'b.jpg'},b,high,profile,1\n"
        f"{tmp_path / 'c.jpg'},c,high,profile,2\n"
        f"{tmp_path / 'd.jpg'},d,high,profile,3\n"
        f"{tmp_path / 'e.jpg'},e,high,profile,4\n",
        encoding="utf-8",
    )

    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_s", -1, 320, calibration_sample=5)
    near = scorer.score_embedding(np.asarray([0.99, 0.01, 0.0, 0.0], dtype=np.float32))
    far = scorer.score_embedding(np.asarray([0.0, 0.7, 0.7, 0.0], dtype=np.float32))

    assert near["nearest"]["subject_id"] in {"a", "b"}
    assert near["best_cosine"] > far["best_cosine"]
    assert len(near["top_matches"]) == 5
    assert 0.0 <= near["estimated_false_match_rate"] <= 1.0
    assert near["nearest"]["cosine"] == near["best_cosine"]
    assert near["nearest"]["image_url"].startswith("/api/reference/")
    assert scorer.reference_image_bytes(near["nearest"]["match_id"]) is not None


def _write_gallery(tmp_path: Path, embeddings: np.ndarray) -> tuple[Path, Path]:
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    rows = ["path,subject_id,quality,split,embedding_index"]
    for index in range(embeddings.shape[0]):
        image_path = tmp_path / f"ref_{index:03d}.jpg"
        cv2.imwrite(str(image_path), np.full((18, 18, 3), 32 + index % 180, dtype=np.uint8))
        rows.append(f"{image_path},subject_{index},high,profile,{index}")
    features_path = tmp_path / "features.csv"
    features_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return features_path, embeddings_path


def _axis(dim: int, index: int) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[index] = 1.0
    return value


def _cosine_vector(dim: int, main_axis: int, side_axis: int, cosine: float) -> np.ndarray:
    value = np.zeros(dim, dtype=np.float32)
    value[main_axis] = cosine
    value[side_axis] = float(np.sqrt(max(0.0, 1.0 - cosine**2)))
    return value


def test_gallery_similarity_scores_exact_match_close_to_100(tmp_path) -> None:
    dim = 32
    query = _axis(dim, 0)
    embeddings = np.vstack([query, _axis(dim, 1), _axis(dim, 2), _axis(dim, 3)])
    features_path, embeddings_path = _write_gallery(tmp_path, embeddings)

    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_s", -1, 320, calibration_sample=4)
    score = scorer.score_embedding(query)

    assert score["best_cosine"] > 0.999
    assert score["similarity_percent"] >= 99.0
    assert score["best_match_similarity_percent"] >= 99.0
    assert score["reference_image_match"] is True
    assert any("duplicata" in warning for warning in score["warnings"])


def test_gallery_similarity_spreads_high_and_unrelated_scores(tmp_path) -> None:
    dim = 48
    query = _axis(dim, 0)
    similar = _cosine_vector(dim, 0, 1, 0.76)
    unrelated_query = _axis(dim, 47)
    unrelated_refs = np.vstack([_axis(dim, index) for index in range(2, 18)])
    embeddings = np.vstack([similar, unrelated_refs])
    features_path, embeddings_path = _write_gallery(tmp_path, embeddings)

    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_s", -1, 320, calibration_sample=16)
    high = scorer.score_embedding(query)
    low = scorer.score_embedding(unrelated_query)

    assert high["similarity_percent"] >= 75.0
    assert low["similarity_percent"] <= 30.0
    assert high["similarity_percent"] - low["similarity_percent"] >= 50.0
    assert high["best_cosine"] > low["best_cosine"]


def test_gallery_density_and_topk_raise_overall_similarity(tmp_path) -> None:
    dim = 64
    sparse_query = _axis(dim, 0)
    dense_query = _axis(dim, 2)
    sparse_best = _cosine_vector(dim, 0, 1, 0.62)
    dense_best = _cosine_vector(dim, 2, 3, 0.62)
    dense_neighbors = [_cosine_vector(dim, 2, axis, 0.45) for axis in range(4, 12)]
    unrelated = [_axis(dim, axis) for axis in range(20, 34)]
    embeddings = np.vstack([sparse_best, dense_best, *dense_neighbors, *unrelated])
    features_path, embeddings_path = _write_gallery(tmp_path, embeddings)

    thresholds = SimilarityThresholds(low=0.25, medium=0.40, high=0.55, very_high=0.70, near_duplicate=0.98)
    scorer = GallerySimilarityScorer(
        features_path,
        embeddings_path,
        "buffalo_s",
        -1,
        320,
        calibration_sample=24,
        top_matches=5,
        aggregation_top_k=10,
        similarity_thresholds=thresholds,
    )
    sparse = scorer.score_embedding(sparse_query)
    dense = scorer.score_embedding(dense_query)

    assert np.isclose(sparse["best_cosine"], dense["best_cosine"], atol=1e-6)
    assert dense["raw_scores"]["weighted_top_k_cosine"] > sparse["raw_scores"]["weighted_top_k_cosine"]
    assert dense["score_components"]["gallery_density_percent"] > sparse["score_components"]["gallery_density_percent"]
    assert dense["similarity_percent"] > sparse["similarity_percent"]


def test_gallery_same_person_style_variants_remain_high(tmp_path) -> None:
    dim = 64
    query = _axis(dim, 0)
    variants = [
        _cosine_vector(dim, 0, 1, 0.92),
        _cosine_vector(dim, 0, 2, 0.87),
        _cosine_vector(dim, 0, 3, 0.83),
    ]
    unrelated = [_axis(dim, index) for index in range(8, 24)]
    features_path, embeddings_path = _write_gallery(tmp_path, np.vstack([*variants, *unrelated]))

    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_l", -1, 640, calibration_sample=16)
    score = scorer.score_embedding(query)

    assert score["best_cosine"] >= 0.90
    assert score["similarity_percent"] >= 85.0
    assert score["similarity_label"] in {"high", "very_high"}


def test_gallery_lookalike_cohort_increases_similarity_without_duplicate_match(tmp_path) -> None:
    dim = 96
    query = _axis(dim, 0)
    lookalikes = [_cosine_vector(dim, 0, side_axis, 0.48) for side_axis in range(1, 9)]
    isolated = _cosine_vector(dim, 10, 11, 0.48)
    unrelated = [_axis(dim, index) for index in range(20, 36)]
    features_path, embeddings_path = _write_gallery(tmp_path, np.vstack([*lookalikes, isolated, *unrelated]))

    thresholds = SimilarityThresholds(low=0.25, medium=0.40, high=0.55, very_high=0.70, near_duplicate=0.98)
    scorer = GallerySimilarityScorer(
        features_path,
        embeddings_path,
        "buffalo_l",
        -1,
        640,
        calibration_sample=24,
        aggregation_top_k=10,
        similarity_thresholds=thresholds,
    )
    cohort = scorer.score_embedding(query)
    single = scorer.score_embedding(_axis(dim, 10))

    assert cohort["best_cosine"] == single["best_cosine"]
    assert cohort["threshold_counts"]["medium"]["count"] > single["threshold_counts"]["medium"]["count"]
    assert cohort["score_components"]["gallery_density_percent"] > single["score_components"]["gallery_density_percent"]
    assert cohort["similarity_percent"] > single["similarity_percent"]
    assert cohort["reference_image_match"] is False


def test_duplicate_groups_do_not_inflate_gallery_density_counts(tmp_path) -> None:
    dim = 32
    query = _axis(dim, 0)
    duplicate = _cosine_vector(dim, 0, 1, 0.995)
    embeddings = np.vstack([query, duplicate, query, _axis(dim, 5), _axis(dim, 6)])
    features_path, embeddings_path = _write_gallery(tmp_path, embeddings)

    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_l", -1, 640, calibration_sample=5)
    score = scorer.score_embedding(query)

    assert scorer.gallery_count == 5
    assert scorer.duplicate_group_count == 3
    assert len(score["top_matches"]) == 5
    assert score["threshold_counts"]["near_duplicate"]["count"] == 1
    assert score["reference_image_match"] is True


class _StubEmbedder:
    def __init__(self, result) -> None:
        self.result = result

    def extract_bgr(self, image_bgr: np.ndarray):
        return self.result


def _jpeg_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_score_jpeg_handles_invalid_upload_low_quality_and_multiple_faces(tmp_path) -> None:
    dim = 16
    query = _axis(dim, 0)
    features_path, embeddings_path = _write_gallery(tmp_path, np.vstack([query, _axis(dim, 1), _axis(dim, 2)]))
    scorer = GallerySimilarityScorer(features_path, embeddings_path, "buffalo_l", -1, 640, calibration_sample=3)

    invalid = scorer.score_jpeg(b"not an image")
    assert invalid["ok"] is False

    scorer._embedder = _StubEmbedder(
        types.SimpleNamespace(
            embedding=query,
            bbox=(20.0, 20.0, 35.0, 35.0),
            det_score=0.20,
            face_count=3,
        )
    )
    image = np.full((200, 200, 3), 180, dtype=np.uint8)
    result = scorer.score_jpeg(_jpeg_bytes(image))

    assert result["ok"] is True
    assert result["face_count"] == 3
    assert any("Mais de um rosto" in warning for warning in result["warnings"])
    assert any("Deteccao facial fraca" in warning for warning in result["warnings"])
    assert any("Rosto pequeno" in warning for warning in result["warnings"])
