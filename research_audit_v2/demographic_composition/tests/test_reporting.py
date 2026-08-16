from __future__ import annotations

from pathlib import Path

import pandas as pd

from research_audit_v2.demographic_composition.reporting import (
    classify_relevance,
    summarize_results,
    write_report,
)


def _write_tables(root: Path) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"scenario": scenario, "source_race_label": group, "count": count, "proportion": count / 100}
            for scenario in "ABCD"
            for group, count in (("Middle Eastern", {"A": 11, "B": 14, "C": 7, "D": 29}[scenario]), ("Other groups", {"A": 89, "B": 86, "C": 93, "D": 71}[scenario]))
        ]
    ).to_csv(tables / "scenario_composition.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": scenario, "seed": seed, "k": 64, "target_size": int(prevalence * 1000), "target_prevalence": prevalence}
            for scenario, prevalence in (("A", .10), ("B", .11), ("C", .14), ("D", .10))
            for seed in (1, 2)
        ]
    ).to_csv(tables / "run_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": scenario, "k": 64, "run_a": "1", "run_b": "2", "ari": ari, "target_jaccard": jaccard}
            for scenario, ari, jaccard in (("A", .96, .91), ("B", .95, .90), ("C", .84, .74), ("D", .94, .88))
        ]
    ).to_csv(tables / "seed_stability.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_a": "A", "scenario_b": scenario, "seed": 1, "k": 64, "intersection_n": 80, "ari": ari, "target_jaccard": jaccard}
            for scenario, ari, jaccard in (("B", .95, .90), ("C", .85, .75), ("D", .99, .95))
        ]
    ).to_csv(tables / "scenario_comparisons.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": scenario, "seed": 1, "k": 64, "fold": fold, "roc_auc": roc, "pr_auc": pr, "prevalence": prevalence}
            for scenario, roc, pr, prevalence in (("A", .80, .50, .10), ("B", .81, .51, .11), ("C", .84, .54, .14), ("D", .79, .48, .10))
            for fold in (0, 1)
        ]
    ).to_csv(tables / "cross_fitted_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": scenario, "seed": 1, "k": 64, "cluster": cluster, "source_race_label": group, "count": count, "proportion_within_cluster": count / 50, "is_target_cluster": cluster == 0}
            for scenario in "ABCD"
            for cluster in (0, 1)
            for group, count in (("Middle Eastern", 10), ("Other groups", 40))
        ]
    ).to_csv(tables / "cluster_composition.csv", index=False)


def _config() -> dict[str, object]:
    return {
        "sample_size": 100,
        "execution_mode": "smoke",
        "primary_k": 64,
        "perturbed_group": "Middle Eastern",
        "relevance_thresholds": {
            "ari": .90,
            "target_jaccard": .80,
            "target_prevalence_delta": .02,
            "auc_delta": .03,
        },
    }


def test_relevance_uses_predeclared_boundaries_and_absolute_changes(tmp_path):
    _write_tables(tmp_path)
    summaries = summarize_results(tmp_path / "tables", _config())

    result = classify_relevance(summaries, _config()["relevance_thresholds"])

    assert result["overall_relevant"] is True
    assert result["scenarios"]["B"]["relevant"] is False
    assert result["scenarios"]["C"]["relevant"] is True
    assert "ARI" in " ".join(result["scenarios"]["C"]["reasons"])
    assert "preval" in " ".join(result["scenarios"]["C"]["reasons"])


def test_report_contains_all_scenarios_figures_limits_and_objective_conclusion(tmp_path):
    _write_tables(tmp_path)
    report = tmp_path / "DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md"

    write_report(tmp_path, report, _config())

    content = report.read_text(encoding="utf-8")
    assert all(f"| {scenario} |" in content for scenario in "ABCD")
    assert "Conclusão objetiva" in content
    assert "Estabilidade por seed" in content
    assert "Variações contra A" in content
    assert "Sensibilidade a k" in content
    assert "Distribuição no cluster-alvo" in content
    assert "Reprodutibilidade" in content
    assert "- **Clustering:**" in content
    assert "- **Cluster-alvo:**" in content
    assert "- **Prevalência:**" in content
    assert "- **Estabilidade:**" in content
    assert "- **Métricas internas:**" in content
    assert "100 amostras" in content
    assert "smoke" in content.lower()
    assert "36.456 amostras" not in content
    assert "categorias fornecidas pelo FairFace" in content
    assert "recuperação interna" in content
    assert "associação" in content
    assert "C:\\" not in content
    assert ".jpg" not in content.lower()
    assert "embedding_index" not in content
    expected = {
        "scenario_composition.svg",
        "target_prevalence.svg",
        "stability.svg",
        "auc_metrics.svg",
        "k_sensitivity.svg",
        "target_demographic_distribution.svg",
    }
    assert expected == {path.name for path in (tmp_path / "figures").glob("*.svg")}
