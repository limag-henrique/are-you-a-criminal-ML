"""Compare target recovery and compatibility across clustering backends."""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from face_profile_ml.clustering_backends import build_backend
from face_profile_ml.target_rules import largest_cluster


def _centers(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([values[labels == cluster].mean(axis=0) for cluster in range(labels.max() + 1)])


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seeds", default=",".join(str(value) for value in range(20)))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="artifacts/clustering_comparison")
    args = parser.parse_args()
    values = np.load(args.embeddings)
    if args.max_samples:
        values = values[: args.max_samples]
    runs, targets = [], {}
    for backend_name in ["minibatch", "kmeans", "gmm", "agglomerative"]:
        for seed in [int(value) for value in args.seeds.split(",")]:
            labels = build_backend(backend_name, n_clusters=args.k).fit_predict(values, seed)
            centers = _centers(values, labels)
            target_cluster = largest_cluster(labels, centers, values)
            target = labels == target_cluster
            distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
            score = np.min(np.delete(distances, target_cluster, axis=1), axis=1) - distances[:, target_cluster]
            targets[(backend_name, seed)] = target
            runs.append({"backend": backend_name, "seed": seed, "k": args.k, "auc": float(roc_auc_score(target, score)), "target_size": int(target.sum())})
    comparisons = []
    for (left_name, left_seed), (right_name, right_seed) in combinations(targets, 2):
        if left_seed == right_seed:
            comparisons.append({"backend_a": left_name, "backend_b": right_name, "seed": left_seed, "jaccard": _jaccard(targets[(left_name, left_seed)], targets[(right_name, right_seed)])})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(out / "backend_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "cross_backend_jaccard.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
