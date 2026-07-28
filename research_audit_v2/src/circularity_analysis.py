"""Internal-label circularity diagnostics and permutation references."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve

from .common import l2_normalize, write_csv


def score_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prediction = score >= np.quantile(score, 1-y.mean())
    return {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score), "balanced_accuracy": balanced_accuracy_score(y, prediction), "precision": precision_score(y, prediction, zero_division=0), "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0), "brier": brier_score_loss(y, (score-score.min()) / max(score.max()-score.min(), 1e-12))}


def circularity(values: np.ndarray, membership: np.ndarray, config: dict, tables: Path, figures: Path, reports: Path) -> None:
    values = l2_normalize(values.astype(np.float64))
    y = membership.astype(int)
    centroid = l2_normalize(values[y == 1].mean(axis=0, keepdims=True))[0]
    cosine = values @ centroid
    rows = [{"method": "cosine_to_target_centroid", **score_metrics(y, cosine), "interpretation": "internal recovery of synthetic label"}, {"method": "majority_baseline", "roc_auc": .5, "pr_auc": float(y.mean()), "balanced_accuracy": .5, "precision": float(y.mean()), "recall": 1.0, "f1": 2*y.mean()/(1+y.mean()), "brier": float(y.mean()*(1-y.mean())), "interpretation": "majority reference"}]
    rng = np.random.default_rng(config["random_seed"])
    permutation_rows = []
    for index in range(config["permutations"]):
        permuted = rng.permutation(y)
        permutation_rows.append({"permutation": index, "roc_auc": roc_auc_score(permuted, cosine), "pr_auc": average_precision_score(permuted, cosine)})
    permutations = pd.DataFrame(permutation_rows)
    write_csv(pd.DataFrame(rows), tables / "circularity_ablation.csv")
    write_csv(permutations, tables / "permutation_results.csv")
    fig, ax = plt.subplots(figsize=(6,4)); ax.hist(permutations["roc_auc"], bins=20, color="0.5"); ax.axvline(rows[0]["roc_auc"], color="black", linestyle="--"); ax.set_xlabel("ROC-AUC after label permutation"); fig.tight_layout(); fig.savefig(figures / "permutation_null_distribution.svg"); plt.close(fig)
    fpr, tpr, _ = roc_curve(y, cosine); fig, ax = plt.subplots(figsize=(5,4)); ax.plot(fpr,tpr,color="black"); ax.plot([0,1],[0,1],"--",color="0.5"); ax.set(xlabel="false-positive rate",ylabel="true-positive rate"); fig.tight_layout(); fig.savefig(figures / "roc_internal_labels.svg"); plt.close(fig)
    precision, recall, _ = precision_recall_curve(y, cosine); fig, ax = plt.subplots(figsize=(5,4)); ax.plot(recall,precision,color="black"); ax.set(xlabel="recall",ylabel="precision"); fig.tight_layout(); fig.savefig(figures / "pr_internal_labels.svg"); plt.close(fig)
    reports.joinpath("circularity_report.md").write_text("# Circularity\n\nA high score is expected when a target label is derived from the same embedding geometry used to calculate distance to its centroid. These are internal synthetic-label recovery diagnostics, not facial-recognition performance or external validity.\n", encoding="utf-8")
