"""Controlled comparisons only where preserved inputs can be established."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import write_csv


def compare_artifacts(tables: Path, reports: Path) -> None:
    matrix = pd.DataFrame([{"comparison": "historical_vs_current", "embeddings_fixed": False, "records_fixed": False, "seed_fixed": False, "weights_fixed": False, "status": "not_executed", "reason": "No preserved historical clustering embeddings, configuration, and target rule were located."}])
    write_csv(matrix, tables / "artifact_factor_matrix.csv")
    write_csv(pd.DataFrame(columns=["comparison", "ari", "nmi", "notes"]), tables / "artifact_pairwise_metrics.csv")
    reports.joinpath("artifact_comparison_report.md").write_text("# Artifact comparison\n\nNo controlled historical comparison was possible. Any future comparison with multiple uncontrolled differences must be called a **divergence between preserved system states**, not attributed causally to one component.\n", encoding="utf-8")
