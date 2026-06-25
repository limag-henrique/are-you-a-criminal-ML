from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .utils import ensure_dir, write_json


@dataclass
class ScoreCalibrator:
    threshold: float = 0.5
    model: LogisticRegression | None = None

    def fit(self, score_raw: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        scores = np.asarray(score_raw, dtype=np.float32).reshape(-1, 1)
        y = np.asarray(labels, dtype=np.int32)
        if scores.shape[0] != y.shape[0]:
            raise ValueError("score_raw and labels must have the same length.")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Calibration requires both positive and negative examples.")
        model = LogisticRegression(class_weight="balanced", random_state=42)
        model.fit(scores, y)
        self.model = model
        return self

    def predict_proba(self, score_raw: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ScoreCalibrator is not fitted yet.")
        scores = np.asarray(score_raw, dtype=np.float32).reshape(-1, 1)
        return self.model.predict_proba(scores)[:, 1]

    def save(self, model_dir: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("ScoreCalibrator is not fitted yet.")
        target = ensure_dir(model_dir)
        with (target / "calibrator.pkl").open("wb") as handle:
            pickle.dump(self, handle)
        write_json(
            target / "calibrator_metadata.json",
            {
                "method": "logistic_regression",
                "threshold": self.threshold,
                "coef": self.model.coef_.ravel().tolist(),
                "intercept": self.model.intercept_.ravel().tolist(),
            },
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "ScoreCalibrator":
        with (Path(model_dir) / "calibrator.pkl").open("rb") as handle:
            calibrator = pickle.load(handle)
        if not isinstance(calibrator, cls):
            raise TypeError("calibrator.pkl does not contain a ScoreCalibrator.")
        return calibrator

