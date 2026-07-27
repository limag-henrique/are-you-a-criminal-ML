"""Auditoria de desempenho por coortes declaradas e documentadas.

Este módulo não tenta inferir raça, etnia, cor da pele, nacionalidade ou
qualquer atributo sensível a partir de uma imagem.  As colunas de coorte devem
ser obtidas de fonte autorizada, com base legal e documentação de proveniência.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .metrics import binary_metrics


def _rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int32)
    predicted = np.asarray(scores, dtype=float) >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))
    return {
        "n": int(labels.size),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fmr": float(fp / negative.sum()) if negative.any() else float("nan"),
        "fnmr": float(fn / positive.sum()) if positive.any() else float("nan"),
        "tpr": float(tp / positive.sum()) if positive.any() else float("nan"),
        "tnr": float(tn / negative.sum()) if negative.any() else float("nan"),
    }


def _bootstrap_interval(
    labels: np.ndarray, scores: np.ndarray, threshold: float, metric: str, rounds: int, rng: np.random.Generator
) -> list[float] | None:
    if not rounds or labels.size < 2:
        return None
    samples: list[float] = []
    for _ in range(rounds):
        index = rng.integers(0, labels.size, size=labels.size)
        value = _rates(labels[index], scores[index], threshold)[metric]
        if np.isfinite(value):
            samples.append(float(value))
    if not samples:
        return None
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _group_columns(columns: list[str], include_intersections: bool) -> list[tuple[str, ...]]:
    result = [(column,) for column in columns]
    if include_intersections:
        result.extend(combinations(columns, 2))
    return result


def audit_group_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
    group_columns: list[str],
    min_group_n: int = 30,
    bootstrap_rounds: int = 0,
    seed: int = 42,
    include_intersections: bool = True,
    threshold: float | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return aggregate and per-cohort verification metrics.

    `label` must use 1 for a genuine/positive comparison and 0 for an
    impostor/negative comparison.  A single global EER threshold is applied to
    all groups so that observed FMR/FNMR gaps are comparable.
    """
    if min_group_n < 1:
        raise ValueError("min_group_n must be at least 1")
    required = {"label", score_column, *group_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing audit columns: {', '.join(sorted(missing))}")

    work = frame.loc[:, ["label", score_column, *group_columns]].copy()
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(int)
    if not work["label"].isin([0, 1]).all():
        raise ValueError("label must contain only 0 (impostor) and 1 (genuine)")
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", score_column])
    overall = binary_metrics(work["label"].to_numpy(), work[score_column].to_numpy())
    if overall.get("status") != "ok":
        raise ValueError("Fairness audit requires both genuine and impostor comparisons.")

    threshold_source = "provided_calibration_threshold"
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(work["label"].to_numpy(), work[score_column].to_numpy())
        threshold = float(thresholds[int(np.nanargmin(np.abs(fpr - (1.0 - tpr))))])
        threshold_source = "test_derived_eer_exploratory"
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    overall_rates = _rates(work["label"].to_numpy(), work[score_column].to_numpy(), threshold)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for columns in _group_columns(group_columns, include_intersections):
        audit_name = " × ".join(columns)
        usable = work.dropna(subset=list(columns)).copy()
        if usable.empty:
            continue
        values = usable.loc[:, list(columns)].astype(str).agg(" | ".join, axis=1)
        for value, group in usable.groupby(values, sort=True):
            labels = group["label"].to_numpy()
            scores = group[score_column].to_numpy()
            record: dict[str, object] = {
                "audit_dimension": audit_name,
                "group": str(value),
                "status": "ok" if len(group) >= min_group_n else "suppressed_small_group",
                "min_group_n": min_group_n,
                "threshold": threshold,
            }
            record.update(_rates(labels, scores, threshold))
            if len(group) >= min_group_n and len(np.unique(labels)) == 2:
                metric = binary_metrics(labels, scores)
                record["auc"] = metric.get("auc")
                record["eer"] = metric.get("eer")
                record["fmr_ci95"] = _bootstrap_interval(labels, scores, threshold, "fmr", bootstrap_rounds, rng)
                record["fnmr_ci95"] = _bootstrap_interval(labels, scores, threshold, "fnmr", bootstrap_rounds, rng)
            else:
                record["auc"] = np.nan
                record["eer"] = np.nan
                record["fmr_ci95"] = None
                record["fnmr_ci95"] = None
                if len(group) >= min_group_n:
                    record["status"] = "skipped_single_class"
            record["fmr_gap_vs_overall"] = record["fmr"] - overall_rates["fmr"]
            record["fnmr_gap_vs_overall"] = record["fnmr"] - overall_rates["fnmr"]
            records.append(record)

    rows = pd.DataFrame(records)
    summary: dict[str, object] = {
        "purpose": "Verification-performance audit by documented cohorts; not demographic inference.",
        "n": int(len(work)),
        "score_column": score_column,
        "group_columns": group_columns,
        "include_intersections": include_intersections,
        "min_group_n": min_group_n,
        "bootstrap_rounds": bootstrap_rounds,
        "global_eer_threshold": threshold,
        "threshold_source": threshold_source,
        "overall": overall,
        "overall_at_global_threshold": overall_rates,
        "groups_reported": int((rows["status"] == "ok").sum()) if not rows.empty else 0,
        "groups_suppressed": int((rows["status"] == "suppressed_small_group").sum()) if not rows.empty else 0,
    }
    return summary, rows
