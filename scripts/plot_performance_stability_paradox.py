"""Plot high internal performance beside low target stability."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", default="research_audit_v2/outputs/final/tables/stability_summary.csv")
    parser.add_argument("--metrics", default="research_audit_v2/second_phase/outputs/tables/cross_fitted_metrics.csv")
    parser.add_argument("--output", default="artigo/figuras/paradox_performance_stability.png")
    args = parser.parse_args()
    stability, metrics = pd.read_csv(args.stability), pd.read_csv(args.metrics)
    auc_column = "roc_auc" if "roc_auc" in metrics else "auc"
    auc = metrics.groupby("fold", as_index=False)[auc_column].mean()
    if "instability_type" in stability.columns:
        stochastic = stability[stability["instability_type"].eq("stochastic")]
        if not stochastic.empty:
            stability = stochastic
    if "metric" in stability:
        stability = stability[stability["metric"].isin(["target_jaccard", "jaccard"])].copy()
        stability["jaccard"] = stability.get("median", stability.get("mean"))
    elif "target_jaccard" in stability:
        stability["jaccard"] = stability["target_jaccard"]
    if "k" not in stability:
        stability["k"] = 64
    scatter = stability.groupby("k", as_index=False)["jaccard"].median()
    scatter["auc"] = float(metrics[auc_column].mean())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(scatter["auc"], scatter["jaccard"], c=scatter["k"], cmap="viridis", s=80)
    label_offsets = {32: (-24, -12), 48: (4, 2), 64: (4, 13)}
    for row in scatter.itertuples():
        axes[0].annotate(
            f"k={int(row.k)}",
            (row.auc, row.jaccard),
            xytext=label_offsets.get(int(row.k), (4, 4)),
            textcoords="offset points",
        )
    axes[0].set(xlabel="ROC-AUC OOF", ylabel="Jaccard mediano do alvo", title="Desempenho × estabilidade")
    axes[1].plot(auc["fold"], auc[auc_column], marker="o", label="ROC-AUC por dobra")
    median_jaccard = float(stability["jaccard"].median())
    axes[1].axhspan(0, median_jaccard, color="#d95f02", alpha=0.18, label="faixa de Jaccard mediano")
    axes[1].set(xlabel="Dobra", ylabel="Métrica", ylim=(0, 1), title="AUC alta com estabilidade baixa")
    axes[1].legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
