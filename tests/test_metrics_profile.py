from __future__ import annotations

import numpy as np

from face_profile_ml.metrics import binary_metrics
from face_profile_ml.profile import FaceProfileModel
from face_profile_ml.utils import l2_normalize


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

