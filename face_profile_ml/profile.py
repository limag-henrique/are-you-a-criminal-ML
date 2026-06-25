from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from .utils import ensure_dir, l2_normalize, write_json


@dataclass
class ScoreWeights:
    cos_mean: float = 0.35
    topk: float = 0.35
    mahalanobis: float = 0.20
    ocsvm: float = 0.10

    def normalized(self, use_ocsvm: bool) -> "ScoreWeights":
        data = asdict(self)
        if not use_ocsvm:
            data["ocsvm"] = 0.0
        total = sum(max(0.0, float(v)) for v in data.values())
        if total <= 0:
            raise ValueError("At least one score weight must be positive.")
        return ScoreWeights(**{k: max(0.0, float(v)) / total for k, v in data.items()})


@dataclass
class FaceProfileModel:
    top_k: int = 5
    mahalanobis_regularization: float = 0.05
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    use_ocsvm: bool = False
    ocsvm_nu: float = 0.05

    profile_embeddings: np.ndarray | None = None
    profile_weights: np.ndarray | None = None
    mean_embedding: np.ndarray | None = None
    inv_covariance: np.ndarray | None = None
    ocsvm_model: OneClassSVM | None = None

    def fit(self, embeddings: np.ndarray, sample_weights: np.ndarray | None = None) -> "FaceProfileModel":
        x = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Profile training requires at least one embedding.")

        weights = np.ones(x.shape[0], dtype=np.float32) if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)
        if weights.shape[0] != x.shape[0]:
            raise ValueError("sample_weights must have the same length as embeddings.")
        if np.any(weights <= 0):
            raise ValueError("sample_weights must be positive.")
        weights = weights / weights.sum()

        mean = np.average(x, axis=0, weights=weights)
        self.mean_embedding = l2_normalize(mean)
        self.profile_embeddings = x
        self.profile_weights = weights
        self.inv_covariance = self._regularized_inverse_covariance(x, weights)

        if self.use_ocsvm and x.shape[0] >= 5:
            self.ocsvm_model = OneClassSVM(kernel="rbf", gamma="scale", nu=self.ocsvm_nu)
            self.ocsvm_model.fit(x)
        return self

    def score(self, embeddings: np.ndarray) -> pd.DataFrame:
        self._require_fitted()
        query = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)

        mean = self.mean_embedding
        profile = self.profile_embeddings
        inv_cov = self.inv_covariance
        assert mean is not None and profile is not None and inv_cov is not None

        cosine_mean = query @ mean
        similarities = query @ profile.T
        k = min(max(1, self.top_k), profile.shape[0])
        topk_values = np.partition(similarities, kth=profile.shape[0] - k, axis=1)[:, -k:]
        topk_cosine = topk_values.mean(axis=1)

        deltas = query - mean
        maha_sq = np.einsum("ij,jk,ik->i", deltas, inv_cov, deltas)
        maha_distance = np.sqrt(np.maximum(maha_sq, 0.0))
        maha_similarity = np.exp(-0.5 * np.minimum(maha_sq, 80.0))

        ocsvm_score = np.zeros(query.shape[0], dtype=np.float32)
        if self.ocsvm_model is not None:
            ocsvm_score = np.tanh(self.ocsvm_model.decision_function(query).astype(np.float32))

        weights = self.score_weights.normalized(self.ocsvm_model is not None)
        raw = (
            weights.cos_mean * cosine_mean
            + weights.topk * topk_cosine
            + weights.mahalanobis * maha_similarity
            + weights.ocsvm * ocsvm_score
        )
        return pd.DataFrame(
            {
                "cos_mean": cosine_mean,
                "topk_cosine": topk_cosine,
                "mahalanobis_distance": maha_distance,
                "mahalanobis_similarity": maha_similarity,
                "ocsvm_score": ocsvm_score,
                "score_raw": raw,
            }
        )

    def save(self, out_dir: str | Path) -> None:
        self._require_fitted()
        target = ensure_dir(out_dir)
        with (target / "profile_model.pkl").open("wb") as handle:
            pickle.dump(self, handle)

        assert self.profile_embeddings is not None
        assert self.mean_embedding is not None
        assert self.inv_covariance is not None
        np.save(target / "profile_embeddings.npy", self.profile_embeddings)
        np.save(target / "profile_mean.npy", self.mean_embedding)
        np.save(target / "profile_inv_cov.npy", self.inv_covariance)
        write_json(
            target / "model_metadata.json",
            {
                "top_k": self.top_k,
                "mahalanobis_regularization": self.mahalanobis_regularization,
                "score_weights": asdict(self.score_weights),
                "effective_score_weights": asdict(self.score_weights.normalized(self.ocsvm_model is not None)),
                "use_ocsvm": self.ocsvm_model is not None,
                "num_profile_embeddings": int(self.profile_embeddings.shape[0]),
                "embedding_dim": int(self.profile_embeddings.shape[1]),
            },
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "FaceProfileModel":
        with (Path(model_dir) / "profile_model.pkl").open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("profile_model.pkl does not contain a FaceProfileModel.")
        return model

    def _regularized_inverse_covariance(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        dim = x.shape[1]
        if x.shape[0] < 2:
            return np.eye(dim, dtype=np.float32)
        assert self.mean_embedding is not None
        centered = x - self.mean_embedding
        covariance = (centered * weights[:, None]).T @ centered
        scale = float(np.trace(covariance) / max(dim, 1))
        ridge = self.mahalanobis_regularization * max(scale, 1e-6)
        covariance = covariance + np.eye(dim, dtype=np.float32) * ridge
        return np.linalg.pinv(covariance).astype(np.float32)

    def _require_fitted(self) -> None:
        if self.profile_embeddings is None or self.mean_embedding is None or self.inv_covariance is None:
            raise RuntimeError("FaceProfileModel is not fitted yet.")

