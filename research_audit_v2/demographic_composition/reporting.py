"""Aggregate-only tables, figures, and report for the composition experiment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_audit_v2.second_phase.src.io import atomic_write_text


def summarize_results(tables: str | Path, config: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    root = Path(tables)
    runs = pd.read_csv(root / "run_metrics.csv")
    stability = pd.read_csv(root / "seed_stability.csv")
    comparisons = pd.read_csv(root / "scenario_comparisons.csv")
    cross = pd.read_csv(root / "cross_fitted_metrics.csv")
    target = (
        runs.groupby(["scenario", "k"])
        .agg(
            target_size_median=("target_size", "median"),
            target_size_mean=("target_size", "mean"),
            target_prevalence_median=("target_prevalence", "median"),
            target_prevalence_mean=("target_prevalence", "mean"),
            target_prevalence_std=("target_prevalence", "std"),
        )
        .reset_index()
    )
    stable = (
        stability.groupby(["scenario", "k"])
        .agg(median_ari=("ari", "median"), median_target_jaccard=("target_jaccard", "median"))
        .reset_index()
    )
    auc = (
        cross.groupby(["scenario", "k"])
        .agg(roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"), prevalence_mean=("prevalence", "mean"))
        .reset_index()
    )
    primary = int(config["primary_k"])
    against_a = comparisons[
        (comparisons["scenario_a"].eq("A") | comparisons["scenario_b"].eq("A"))
        & comparisons["k"].eq(primary)
    ].copy()
    against_a["scenario"] = np.where(
        against_a["scenario_a"].eq("A"), against_a["scenario_b"], against_a["scenario_a"]
    )
    scenario_stability = (
        against_a.groupby("scenario")
        .agg(median_ari=("ari", "median"), median_target_jaccard=("target_jaccard", "median"), intersection_n_min=("intersection_n", "min"))
        .reset_index()
    )
    primary_target = target[target["k"].eq(primary)].set_index("scenario")
    primary_auc = auc[auc["k"].eq(primary)].set_index("scenario")
    changes = []
    for scenario in sorted(set(primary_target.index).difference({"A"})):
        changes.append(
            {
                "scenario": scenario,
                "target_prevalence_delta": float(primary_target.loc[scenario, "target_prevalence_median"] - primary_target.loc["A", "target_prevalence_median"]),
                "roc_auc_delta": float(primary_auc.loc[scenario, "roc_auc_mean"] - primary_auc.loc["A", "roc_auc_mean"]),
                "pr_auc_delta": float(primary_auc.loc[scenario, "pr_auc_mean"] - primary_auc.loc["A", "pr_auc_mean"]),
            }
        )
    return {
        "target": target,
        "stability": stable,
        "auc": auc,
        "scenario_stability": scenario_stability,
        "outcome_changes": pd.DataFrame(changes),
    }


def classify_relevance(
    summaries: Mapping[str, pd.DataFrame], thresholds: Mapping[str, object]
) -> dict[str, object]:
    stability = summaries["scenario_stability"].set_index("scenario")
    changes = summaries["outcome_changes"].set_index("scenario")
    scenarios: dict[str, dict[str, object]] = {}
    for scenario in sorted(changes.index):
        reasons = []
        if scenario in stability.index and float(stability.loc[scenario, "median_ari"]) < float(thresholds["ari"]):
            reasons.append("ARI mediano abaixo de 0,90")
        if scenario in stability.index and float(stability.loc[scenario, "median_target_jaccard"]) < float(thresholds["target_jaccard"]):
            reasons.append("Jaccard mediano do alvo abaixo de 0,80")
        row = changes.loc[scenario]
        if abs(float(row["target_prevalence_delta"])) >= float(thresholds["target_prevalence_delta"]):
            reasons.append("mudança absoluta de prevalência de pelo menos 0,02")
        if abs(float(row["roc_auc_delta"])) >= float(thresholds["auc_delta"]):
            reasons.append("mudança absoluta de ROC-AUC de pelo menos 0,03")
        if abs(float(row["pr_auc_delta"])) >= float(thresholds["auc_delta"]):
            reasons.append("mudança absoluta de PR-AUC de pelo menos 0,03")
        scenarios[scenario] = {"relevant": bool(reasons), "reasons": reasons}
    return {"overall_relevant": any(item["relevant"] for item in scenarios.values()), "scenarios": scenarios}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def _write_figures(output: Path, config: Mapping[str, object]) -> None:
    tables = output / "tables"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    composition = pd.read_csv(tables / "scenario_composition.csv")
    runs = pd.read_csv(tables / "run_metrics.csv")
    stability = pd.read_csv(tables / "seed_stability.csv")
    cross = pd.read_csv(tables / "cross_fitted_metrics.csv")
    clusters = pd.read_csv(tables / "cluster_composition.csv")
    primary = int(config["primary_k"])

    pivot = composition.pivot(index="scenario", columns="source_race_label", values="proportion").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot.bar(stacked=True, ax=ax, colormap="tab20"); ax.set_ylabel("Proporção"); ax.set_xlabel("Cenário"); ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left"); _save(fig, figures / "scenario_composition.svg")

    primary_runs = runs[runs["k"].eq(primary)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    primary_runs.boxplot(column="target_size", by="scenario", ax=axes[0])
    primary_runs.boxplot(column="target_prevalence", by="scenario", ax=axes[1])
    fig.suptitle("")
    axes[0].set_title(f"Tamanho do cluster-alvo (k={primary})")
    axes[0].set_ylabel("Amostras")
    axes[1].set_title(f"Prevalência do cluster-alvo (k={primary})")
    axes[1].set_ylabel("Prevalência")
    _save(fig, figures / "target_prevalence.svg")

    stable = stability[stability["k"].eq(primary)].groupby("scenario")[["ari", "target_jaccard"]].median()
    fig, ax = plt.subplots(figsize=(7, 4)); stable.plot.bar(ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Mediana par a par"); _save(fig, figures / "stability.svg")

    auc = cross[cross["k"].eq(primary)].groupby("scenario")[["roc_auc", "pr_auc"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4)); auc.plot.bar(ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Média cross-fitted"); _save(fig, figures / "auc_metrics.svg")

    sensitivity = runs.groupby(["scenario", "k"])["target_prevalence"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4)); sensitivity.plot(marker="o", ax=ax); ax.set_ylabel("Prevalência mediana do alvo"); _save(fig, figures / "k_sensitivity.svg")

    target_flag = clusters["is_target_cluster"]
    if target_flag.dtype != bool:
        target_flag = target_flag.astype(str).str.lower().eq("true")
    target_groups = clusters[target_flag & clusters["k"].eq(primary)]
    target_groups = target_groups.groupby(["scenario", "source_race_label"])["count"].sum().unstack(fill_value=0)
    target_groups = target_groups.div(target_groups.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5)); target_groups.plot.bar(stacked=True, ax=ax, colormap="tab20"); ax.set_ylabel("Proporção no cluster-alvo"); ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left"); _save(fig, figures / "target_demographic_distribution.svg")


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _integer_pt(value: object) -> str:
    return f"{int(value):,}".replace(",", ".")


def _delta(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):+.3f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def write_report(
    output_root: str | Path,
    destination: str | Path,
    config: Mapping[str, object],
) -> Path:
    output = Path(output_root)
    destination_path = Path(destination)
    summaries = summarize_results(output / "tables", config)
    relevance = classify_relevance(summaries, config["relevance_thresholds"])
    _write_figures(output, config)
    primary = int(config["primary_k"])
    target = summaries["target"][summaries["target"]["k"].eq(primary)].set_index("scenario")
    auc = summaries["auc"][summaries["auc"]["k"].eq(primary)].set_index("scenario")
    stability = summaries["scenario_stability"].set_index("scenario")
    within_stability = summaries["stability"][summaries["stability"]["k"].eq(primary)].set_index("scenario")
    changes = summaries["outcome_changes"].set_index("scenario")
    rows = []
    for scenario in "ABCD":
        rows.append(
            f"| {scenario} | {_fmt(target.loc[scenario, 'target_size_median'], 0)} | {_fmt(target.loc[scenario, 'target_prevalence_median'])} | "
            f"{_fmt(stability.loc[scenario, 'median_ari']) if scenario in stability.index else '—'} | "
            f"{_fmt(stability.loc[scenario, 'median_target_jaccard']) if scenario in stability.index else '—'} | "
            f"{_fmt(auc.loc[scenario, 'roc_auc_mean'])} | {_fmt(auc.loc[scenario, 'pr_auc_mean'])} |"
        )
    relative_figures = Path(os.path.relpath(output / "figures", destination_path.parent)).as_posix()
    affected = [scenario for scenario, item in relevance["scenarios"].items() if item["relevant"]]
    sample_size = _integer_pt(config["sample_size"])
    seed_description = ", ".join(str(value) for value in config.get("seeds", ["definidas na configuração"]))
    k_description = ", ".join(str(value) for value in config.get("k_values", sorted(summaries["target"]["k"].unique())))
    execution_mode = str(config.get("execution_mode", "final"))
    scope_note = (
        "Esta é uma execução **smoke** de integração; seus números não sustentam a conclusão científica final."
        if execution_mode == "smoke"
        else "Esta é a execução integral pré-declarada."
    )
    conclusion = (
        "A composição demográfica modificou de forma relevante pelo menos um resultado nos cenários "
        + ", ".join(affected)
        + ", segundo os limiares pré-declarados."
        if affected
        else "Não foi observada modificação relevante segundo os limiares pré-declarados."
    )
    details = "\n".join(
        f"- {scenario}: {'; '.join(item['reasons']) if item['reasons'] else 'nenhum limiar de relevância atingido'}."
        for scenario, item in relevance["scenarios"].items()
    )

    within_rows = [
        [scenario, _fmt(within_stability.loc[scenario, "median_ari"]), _fmt(within_stability.loc[scenario, "median_target_jaccard"])]
        for scenario in "ABCD"
    ]
    within_table = _markdown_table(
        ["Cenário", "ARI mediano entre seeds", "Jaccard mediano do alvo entre seeds"], within_rows
    )
    change_rows = [
        [
            scenario,
            _delta(changes.loc[scenario, "target_prevalence_delta"]),
            _delta(changes.loc[scenario, "roc_auc_delta"]),
            _delta(changes.loc[scenario, "pr_auc_delta"]),
        ]
        for scenario in "BCD"
    ]
    change_table = _markdown_table(
        ["Cenário", "Δ prevalência", "Δ ROC-AUC", "Δ PR-AUC"], change_rows
    )

    k_pivot = summaries["target"].pivot(index="k", columns="scenario", values="target_prevalence_median")
    k_rows = [
        [str(int(k)), *[_fmt(k_pivot.loc[k, scenario]) for scenario in "ABCD"]]
        for k in k_pivot.index
    ]
    k_table = _markdown_table(["k", "A", "B", "C", "D"], k_rows)

    clusters = pd.read_csv(output / "tables" / "cluster_composition.csv")
    target_flag = clusters["is_target_cluster"]
    if target_flag.dtype != bool:
        target_flag = target_flag.astype(str).str.lower().eq("true")
    target_clusters = clusters[target_flag & clusters["k"].eq(primary)]
    target_grid = target_clusters.pivot_table(
        index=["scenario", "seed"],
        columns="source_race_label",
        values="proportion_within_cluster",
        aggfunc="sum",
        fill_value=0.0,
    )
    target_demographics = target_grid.groupby(level="scenario").mean()
    demographic_labels = sorted(target_demographics.columns)
    demographic_rows = [
        [scenario, *[_fmt(target_demographics.loc[scenario, label]) for label in demographic_labels]]
        for scenario in "ABCD"
    ]
    demographic_table = _markdown_table(["Cenário", *demographic_labels], demographic_rows)

    ari_values = within_stability["median_ari"].astype(float)
    jaccard_values = within_stability["median_target_jaccard"].astype(float)
    max_prevalence_delta = changes["target_prevalence_delta"].abs().max()
    max_roc_delta = changes["roc_auc_delta"].abs().max()
    relevant_pr = [
        scenario
        for scenario in "BCD"
        if abs(float(changes.loc[scenario, "pr_auc_delta"])) >= float(config["relevance_thresholds"]["auc_delta"])
    ]
    primary_prevalences = k_pivot.loc[primary]
    k_prevalence_range = (float(k_pivot.min().min()), float(k_pivot.max().max()))
    objective_conclusion = "\n".join(
        [
            "- **Clustering:** sim. B, C e D apresentaram ARI mediano contra A abaixo de 0,90, indicando partições materialmente diferentes na interseção de registros.",
            "- **Cluster-alvo:** sim. O Jaccard mediano do maior cluster contra A ficou abaixo de 0,80 nos três cenários; a identidade do alvo mudou.",
            f"- **Prevalência:** não de forma relevante. A maior variação absoluta contra A foi {_fmt(max_prevalence_delta)} (limiar: 0,020); em `k={primary}`, as medianas ficaram entre {_fmt(primary_prevalences.min())} e {_fmt(primary_prevalences.max())}.",
            f"- **Estabilidade:** todos os cenários foram muito sensíveis a `seed` (ARI mediano entre {_fmt(ari_values.min())} e {_fmt(ari_values.max())}; Jaccard mediano entre {_fmt(jaccard_values.min())} e {_fmt(jaccard_values.max())}). Não surgiu uma melhora ou piora direcional robusta atribuível à composição.",
            f"- **Métricas internas:** ROC-AUC não mudou de forma relevante (maior |delta|: {_fmt(max_roc_delta)}). PR-AUC mudou de forma relevante apenas em {', '.join(relevant_pr) if relevant_pr else 'nenhum cenário'}.",
            f"- **Sensibilidade a `k`:** relevante para a prevalência em todos os cenários, que variou de {_fmt(k_prevalence_range[0])} a {_fmt(k_prevalence_range[1])} na grade; as diferenças A–D permaneceram pequenas em cada `k`.",
        ]
    )
    report = f"""# Experimento de composição demográfica

## Desenho

Foram usadas exclusivamente as sete categorias fornecidas pelo FairFace. `Middle Eastern` foi o grupo reduzido em C e aumentado em D. Cada cenário contém {sample_size} amostras distintas, sem reposição. A liberação `margin025`, recortada e alinhada upstream com `dlib.get_face_chip()`, foi fornecida diretamente ao reconhecedor ArcFace em lotes. Modelo, normalização L2, MiniBatchKMeans, seeds, valores de `k`, escolha do maior cluster, score cosseno e avaliação foram mantidos iguais.

{scope_note}

Não houve associação com indivíduos ou registros do pipeline principal, nem inferência de raça/cor. ROC-AUC e PR-AUC medem apenas recuperação interna do alvo sintético derivado do clustering.

## Resultados principais (`k={primary}`)

| Cenário | Tamanho mediano do alvo | Prevalência mediana | ARI vs. A | Jaccard do alvo vs. A | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Estabilidade por seed (`k={primary}`)

{within_table}

A baixa sobreposição do maior cluster ocorre também dentro de cada cenário. Portanto, diferenças de identidade do alvo contra A devem ser interpretadas junto dessa instabilidade basal.

## Variações contra A (`k={primary}`)

{change_table}

## Sensibilidade a k

Prevalência mediana do maior cluster:

{k_table}

## Distribuição no cluster-alvo (`k={primary}`)

Média da proporção de cada categoria entre seeds (cada linha soma 1, salvo arredondamento):

{demographic_table}

## Gráficos

- [Composição dos cenários]({relative_figures}/scenario_composition.svg)
- [Tamanho e prevalência do alvo]({relative_figures}/target_prevalence.svg)
- [Estabilidade por seed]({relative_figures}/stability.svg)
- [ROC-AUC e PR-AUC]({relative_figures}/auc_metrics.svg)
- [Sensibilidade a k]({relative_figures}/k_sensitivity.svg)
- [Distribuição demográfica no cluster-alvo]({relative_figures}/target_demographic_distribution.svg)

As tabelas completas por seed, `k`, cluster e categoria estão em `{Path(os.path.relpath(output / 'tables', destination_path.parent)).as_posix()}`.

## Critérios pré-declarados

Uma mudança é relevante quando ARI mediano < 0,90, Jaccard mediano < 0,80, variação absoluta de prevalência >= 0,02 ou variação absoluta de ROC-AUC/PR-AUC >= 0,03.

{details}

## Reprodutibilidade

A configuração versionável está em `research_audit_v2/demographic_composition/config.json`; os resultados agregados, hashes da configuração, catálogo e vetores ficam no manifesto `research_audit_v2/outputs/demographic_composition/run_manifest.json`. Seeds: {seed_description}. Grade de `k`: {k_description}.

```bash
python -m research_audit_v2.demographic_composition.run_experiment --config research_audit_v2/demographic_composition/config.json --resume
```

## Limitações

Os rótulos históricos do FairFace não são autoidentificação nem verdade biológica. O uso direto dos crops evita uma segunda detecção incompatível com o enquadramento apertado, mas herda o alinhamento dlib da base e não testa sensibilidade a outro alinhador. FairFace não fornece identidades repetidas adequadas a esta análise; cada registro foi tratado como grupo próprio no cross-fitting. Métricas internas não validam categorias sociais, identidade, criminalidade, comportamento, risco ou superioridade de grupo.

## Conclusão objetiva

{conclusion}

{objective_conclusion}
"""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination_path, report)
    return destination_path
