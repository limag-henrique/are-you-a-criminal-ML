from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    fmr_targets: tuple[float, ...] = (0.1, 0.01, 0.001),
) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float32)
    if len(np.unique(y)) < 2:
        return {"status": "skipped", "reason": "requires both positive and negative labels", "n": int(y.size)}

    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    operating = {}
    for target in fmr_targets:
        valid = np.where(fpr <= target)[0]
        op_idx = int(valid[np.argmax(tpr[valid])]) if valid.size else 0
        operating[f"fmr<={target:g}"] = {
            "threshold": float(thresholds[op_idx]),
            "fmr": float(fpr[op_idx]),
            "fnmr": float(fnr[op_idx]),
            "tpr": float(tpr[op_idx]),
        }

    return {
        "status": "ok",
        "n": int(y.size),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "auc": float(roc_auc_score(y, s)),
        "roc_auc_trapezoid": float(auc(fpr, tpr)),
        "eer": eer,
        "eer_threshold": float(thresholds[idx]),
        "operating_points": operating,
    }


def metrics_by_quality(frame: pd.DataFrame, score_column: str = "score") -> dict[str, object]:
    output: dict[str, object] = {"overall": binary_metrics(frame["label"].to_numpy(), frame[score_column].to_numpy())}
    by_quality: dict[str, object] = {}
    for quality, group in frame.groupby("quality"):
        by_quality[str(quality)] = binary_metrics(group["label"].to_numpy(), group[score_column].to_numpy())
    output["by_quality"] = by_quality
    return output

