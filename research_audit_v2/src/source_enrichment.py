"""Descriptive source composition with transparent limitations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .common import write_csv
from .statistical_uncertainty import benjamini_hochberg, bootstrap_proportion


def source_enrichment(records: pd.DataFrame, target: np.ndarray, config: dict, tables: Path, figures: Path, reports: Path) -> None:
    rows = []
    for source, part in records.assign(_target=target).groupby("source", dropna=False):
        total, selected = len(part), int(part["_target"].sum())
        overall = len(records)
        baseline = target.mean()
        proportion = total / overall
        target_proportion = selected / max(target.sum(), 1)
        ratio = target_proportion / proportion if proportion else np.nan
        _, p_value = fisher_exact([[selected, total-selected], [int(target.sum())-selected, overall-total-(int(target.sum())-selected)]]) if overall > total else (np.nan, 1.0)
        lo, hi = bootstrap_proportion(part["_target"].to_numpy(), config["bootstrap_iterations"], config["random_seed"], part["group_id"].to_numpy())
        rows.append({"source": source, "n_valid": total, "proportion_valid": proportion, "n_target": selected, "proportion_target": target_proportion, "enrichment_ratio": ratio, "log2_enrichment_ratio": np.log2(ratio) if ratio and ratio > 0 else np.nan, "absolute_difference": target_proportion-proportion, "expected_count": baseline*total, "conditional_ci_low": lo, "conditional_ci_high": hi, "p_value": p_value})
    result = pd.DataFrame(rows)
    result["p_value_bh"] = benjamini_hochberg(result["p_value"].to_numpy()) if len(result) else []
    write_csv(result, tables / "source_enrichment_detailed.csv")
    write_csv(result.assign(sensitivity="source field as observed"), tables / "source_enrichment_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(result["source"].astype(str), result["enrichment_ratio"].fillna(0), color="0.4"); ax.set_ylabel("enrichment ratio"); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(figures / "source_enrichment_forest.svg"); plt.close(fig)
    reports.joinpath("source_enrichment_report.md").write_text("# Source enrichment\n\nThe observed embedding manifest lacks a documented source column, so all records are reported as `unresolved`. This table is a contract and a limitation record, not evidence about institutional selection. Intervals are conditional resampling intervals.\n", encoding="utf-8")
