"""Systematic ablation of six endogenous target definitions."""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from face_profile_ml.cross_validation import run_grouped_cluster_cv


RULES = ["largest", "compact", "separated", "random", "central", "outlier"]


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    aligned = left.rename("left").to_frame().join(right.rename("right"), how="inner")
    union = np.logical_or(aligned.left, aligned.right).sum()
    return float(np.logical_and(aligned.left, aligned.right).sum() / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="research_audit_v2/.sensitive/demographic_composition/final_selection.csv")
    parser.add_argument("--embedding-cache", default="research_audit_v2/.sensitive/demographic_composition/embedding_cache.npz")
    parser.add_argument("--seeds", default=",".join(str(value) for value in range(50)))
    parser.add_argument("--k-values", default="32,64,128")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="artifacts/ablation")
    args = parser.parse_args()
    selection = pd.read_csv(args.selection)
    selection = selection[selection["scenario"].eq("A")].drop_duplicates("record_id")
    if args.max_samples:
        selection = selection.head(args.max_samples)
    cache = np.load(args.embedding_cache)
    lookup = {str(record): index for index, record in enumerate(cache["record_ids"])}
    selection = selection[selection["record_id"].astype(str).isin(lookup)].copy()
    indices = selection["record_id"].astype(str).map(lookup).to_numpy()
    values = cache["vectors"][indices]
    samples = selection[["record_id"]].rename(columns={"record_id": "sample_id"})
    samples["group_id"] = samples["sample_id"]
    rows, pair_rows = [], []
    for seed in [int(value) for value in args.seeds.split(",")]:
        for k in [int(value) for value in args.k_values.split(",")]:
            outputs = {}
            for rule in RULES:
                outputs[rule], _ = run_grouped_cluster_cv(samples, values, n_splits=args.folds, k=k, seed=seed, target_rule=rule)
            pair_values: dict[tuple[str, str], float] = {}
            for left, right in combinations(RULES, 2):
                value = _jaccard(outputs[left].set_index("sample_id")["y_true"], outputs[right].set_index("sample_id")["y_true"])
                pair_values[(left, right)] = value
                pair_rows.append({"rule_a": left, "rule_b": right, "seed": seed, "k": k, "jaccard": value})
            largest = outputs["largest"].set_index("sample_id")["y_true"]
            for rule, output in outputs.items():
                pairwise = [value for pair, value in pair_values.items() if rule in pair]
                for fold, frame in output.groupby("fold"):
                    y, score = frame["y_true"], frame["prob_calibrated"]
                    rows.append({
                        "rule": rule, "seed": seed, "k": k, "fold": fold,
                        "auc": float(roc_auc_score(y, score)),
                        "pr_auc": float(average_precision_score(y, score)),
                        "jaccard_vs_largest": _jaccard(frame.set_index("sample_id")["y_true"], largest),
                        "jaccard_pairwise": float(np.median(pairwise)),
                        "prevalence": float(y.mean()),
                    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ablation_results.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    matrix = pairs.groupby(["rule_a", "rule_b"], as_index=False)["jaccard"].median()
    matrix.to_csv(out / "incompatibility_matrix.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
