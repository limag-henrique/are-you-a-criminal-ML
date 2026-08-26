"""Create targets in one representation and score them in another."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from face_profile_ml.target_rules import largest_cluster


def _experiment(target_values: np.ndarray, scoring_values: np.ndarray, k: int, seed: int) -> tuple[float, int]:
    fitted = KMeans(k, n_init=10, random_state=seed).fit(target_values)
    target_cluster = largest_cluster(fitted.labels_, fitted.cluster_centers_, target_values)
    target = fitted.labels_ == target_cluster
    positive_center = scoring_values[target].mean(axis=0)
    negative_center = scoring_values[~target].mean(axis=0)
    score = np.linalg.norm(scoring_values - negative_center, axis=1) - np.linalg.norm(scoring_values - positive_center, axis=1)
    return float(roc_auc_score(target, score)), int(target.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seeds", default=",".join(str(value) for value in range(20)))
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="artifacts/embedding_decoupling")
    args = parser.parse_args()
    original = np.load(args.embeddings)
    if args.max_samples:
        original = original[: args.max_samples]
    components = min(args.pca_components, original.shape[1], len(original) - 1)
    proxy = PCA(components, random_state=0).fit_transform(original)
    rows = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        for target_name, target_values, score_name, score_values in [
            ("arcface", original, "pca_proxy", proxy),
            ("pca_proxy", proxy, "arcface", original),
            ("arcface", original, "arcface", original),
        ]:
            auc, size = _experiment(target_values, score_values, args.k, seed)
            rows.append({"target_embedding": target_name, "scoring_embedding": score_name, "seed": seed, "k": args.k, "auc": auc, "target_size": size})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "decoupling_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
