"""Sensitivity of grouped OOF results to identity similarity thresholds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from face_profile_ml.cross_validation import run_grouped_cluster_cv
from face_profile_ml.grouping import cosine_similarity_groups


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="artifacts/embedding_manifest.csv")
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument("--thresholds", default="0.995,0.997,0.999,0.9995")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="artifacts/groupid_sensitivity")
    args = parser.parse_args()
    table, embeddings = pd.read_csv(args.features), np.load(args.embeddings)
    table = table[table["embedding_index"].astype(int).ge(0)].copy()
    if args.max_samples:
        table = table.head(args.max_samples)
    values = embeddings[table["embedding_index"].astype(int).to_numpy()]
    sample_ids = table.get("sample_id", table.index.astype(str)).astype(str).to_numpy()
    outputs, rows = {}, []
    for threshold in [float(value) for value in args.thresholds.split(",")]:
        groups = cosine_similarity_groups(values, threshold)
        samples = pd.DataFrame({"sample_id": sample_ids, "group_id": groups.astype(str)})
        oof, metrics = run_grouped_cluster_cv(samples, values, n_splits=args.folds, k=args.k, seed=args.seed)
        outputs[threshold] = oof
        rows.append({
            "threshold": threshold, "n_groups": int(len(np.unique(groups))),
            "duplicate_fraction": float(1 - len(np.unique(groups)) / len(groups)),
            "auc": metrics["auc"], "pr_auc": metrics["pr_auc"],
            "target_size": int(oof["y_true"].sum()),
        })
    reference_threshold = max(outputs)
    reference = outputs[reference_threshold].set_index("sample_id")
    for row in rows:
        current = outputs[row["threshold"]].set_index("sample_id").loc[reference.index]
        row["jaccard"] = _jaccard(current["y_true"].to_numpy(bool), reference["y_true"].to_numpy(bool))
        row["ari"] = float(adjusted_rand_score(current["cluster_label"], reference["cluster_label"]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sensitivity_curve.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
