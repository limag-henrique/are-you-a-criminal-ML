"""Specification-curve analysis across declared pipeline choices."""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from face_profile_ml.cross_validation import run_grouped_cluster_cv
from face_profile_ml.grouping import cosine_similarity_groups


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    aligned = left.rename("left").to_frame().join(right.rename("right"), how="inner")
    union = np.logical_or(aligned.left, aligned.right).sum()
    return float(np.logical_and(aligned.left, aligned.right).sum() / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="artifacts/embedding_manifest.csv")
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument("--seeds", default=",".join(str(value) for value in range(50)))
    parser.add_argument("--k-values", default="32,64,128")
    parser.add_argument("--rules", default="largest,compact,separated,random,central,outlier")
    parser.add_argument("--thresholds", default="0.995,0.997,0.999,0.9995")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="artifacts/multiverse")
    args = parser.parse_args()
    table, embeddings = pd.read_csv(args.features), np.load(args.embeddings)
    table = table[table["embedding_index"].astype(int).ge(0)].copy()
    if args.max_samples:
        table = table.head(args.max_samples)
    original = embeddings[table["embedding_index"].astype(int).to_numpy()]
    representations = {"original": original, "pca64": PCA(min(64, original.shape[1], len(original) - 1), random_state=0).fit_transform(original)}
    sample_ids = table.get("sample_id", table.index.astype(str)).astype(str).to_numpy()
    rows, targets = [], {}
    pipeline_id = 0
    for threshold in [float(value) for value in args.thresholds.split(",")]:
        groups = cosine_similarity_groups(original, threshold)
        samples = pd.DataFrame({"sample_id": sample_ids, "group_id": groups.astype(str)})
        for representation_name, values in representations.items():
            for k in [int(value) for value in args.k_values.split(",")]:
                for seed in [int(value) for value in args.seeds.split(",")]:
                    for rule in args.rules.split(","):
                        pipeline_id += 1
                        oof, metrics = run_grouped_cluster_cv(samples, values, n_splits=args.folds, k=k, seed=seed, target_rule=rule)
                        identifier = f"pipeline_{pipeline_id:05d}"
                        targets[identifier] = oof.set_index("sample_id")["y_true"].astype(bool)
                        rows.append({"pipeline_id": identifier, "embedding": representation_name, "pca": representation_name == "pca64", "k": k, "seed": seed, "target_rule": rule, "groupid_threshold": threshold, "auc": metrics["auc"], "pr_auc": metrics["pr_auc"], "prevalence": float(oof["y_true"].mean())})
    curve = pd.DataFrame(rows)
    qualified = curve[curve["auc"].gt(0.85)]["pipeline_id"].tolist()
    pair_rows = [{"pipeline_a": left, "pipeline_b": right, "jaccard": _jaccard(targets[left], targets[right])} for left, right in combinations(qualified, 2)]
    pairs = pd.DataFrame(pair_rows)
    medians = pairs.groupby("pipeline_a")["jaccard"].median() if not pairs.empty else pd.Series(dtype=float)
    curve["median_pairwise_jaccard"] = curve["pipeline_id"].map(medians)
    curve = curve.sort_values("auc", ascending=False, ignore_index=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "specification_curve.csv", index=False)
    pairs.to_csv(out / "pairwise_jaccard.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(curve.index, curve["auc"]); axes[0].axhline(0.85, color="red", linestyle="--"); axes[0].set_ylabel("ROC-AUC")
    axes[1].scatter(curve.index, curve["median_pairwise_jaccard"], s=8); axes[1].axhline(0.1, color="red", linestyle="--"); axes[1].set(xlabel="Pipelines ordenadas", ylabel="Jaccard mediano")
    fig.tight_layout(); fig.savefig(out / "specification_curve.png", dpi=220); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
