"""Paired and replicated FairFace composition experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from face_profile_ml.cross_validation import run_grouped_cluster_cv
from face_profile_ml.variance_decomposition import decompose_variance
from research_audit_v2.demographic_composition.cohorts import FAIRFACE_GROUPS, scenario_quotas


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    aligned = left.rename("left").to_frame().join(right.rename("right"), how="inner")
    union = np.logical_or(aligned["left"], aligned["right"]).sum()
    return float(np.logical_and(aligned["left"], aligned["right"]).sum() / union) if union else 1.0


def paired_scenarios(catalog: pd.DataFrame, config: dict[str, object], replication: int, core_fraction: float) -> pd.DataFrame:
    """Sample a shared category-stratified core, then independent complements."""
    quotas = scenario_quotas(catalog, config)
    group_column = str(config["group_column"])
    scenarios = list(quotas)
    capacities = {group: min(quotas[name][group] for name in scenarios) for group in FAIRFACE_GROUPS}
    core_total = min(int(int(config["sample_size"]) * core_fraction), sum(capacities.values()))
    raw = {group: core_total * capacities[group] / sum(capacities.values()) for group in FAIRFACE_GROUPS}
    core_quotas = {group: min(capacities[group], int(raw[group])) for group in FAIRFACE_GROUPS}
    for group in sorted(FAIRFACE_GROUPS, key=lambda item: (-(raw[item] - int(raw[item])), item)):
        if sum(core_quotas.values()) >= core_total:
            break
        if core_quotas[group] < capacities[group]:
            core_quotas[group] += 1
    rng = np.random.default_rng(int(config["random_seed"]) + replication)
    selected: dict[str, list[pd.DataFrame]] = {name: [] for name in scenarios}
    for group in FAIRFACE_GROUPS:
        pool = catalog[catalog[group_column].eq(group)]
        core_index = rng.choice(pool.index, size=core_quotas[group], replace=False)
        core = pool.loc[core_index]
        remaining = pool.drop(core_index)
        for scenario_index, scenario in enumerate(scenarios):
            needed = quotas[scenario][group] - len(core)
            scenario_rng = np.random.default_rng(int(config["random_seed"]) + replication * 101 + scenario_index)
            complement_index = scenario_rng.choice(remaining.index, size=needed, replace=False)
            frame = pd.concat([core, remaining.loc[complement_index]], ignore_index=True)
            frame = frame.copy()
            frame["shared_core"] = frame["record_id"].isin(core["record_id"])
            selected[scenario].append(frame)
    frames = []
    for scenario, pieces in selected.items():
        frame = pd.concat(pieces, ignore_index=True)
        frame.insert(0, "scenario", scenario)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="research_audit_v2/.sensitive/demographic_composition/final_selection.csv")
    parser.add_argument("--embedding-cache", default="research_audit_v2/.sensitive/demographic_composition/embedding_cache.npz")
    parser.add_argument("--n-replications", type=int, default=50)
    parser.add_argument("--seeds-per-rep", type=int, default=5)
    parser.add_argument("--k-values", default="32,64,128")
    parser.add_argument("--scenarios", default="A,B,C,D")
    parser.add_argument("--sample-size", type=int, default=36456)
    parser.add_argument("--core-fraction", type=float, default=0.70)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out-dir", default="artifacts/fairface_replicated")
    args = parser.parse_args()
    source = pd.read_csv(args.catalog).drop_duplicates("record_id").drop(columns=["scenario"], errors="ignore")
    cache = np.load(args.embedding_cache)
    lookup = {str(record): index for index, record in enumerate(cache["record_ids"])}
    source = source[source["record_id"].astype(str).isin(lookup)].copy()
    source["embedding_index"] = source["record_id"].astype(str).map(lookup)
    config = {
        "sample_size": args.sample_size,
        "group_column": "source_race_label",
        "perturbed_group": "Middle Eastern",
        "random_seed": 20260815,
    }
    scenarios = args.scenarios.split(",")
    predictions, run_rows = [], []
    for replication in range(args.n_replications):
        sampled = paired_scenarios(source, config, replication, args.core_fraction)
        for scenario in scenarios:
            cohort = sampled[sampled["scenario"].eq(scenario)].reset_index(drop=True)
            values = cache["vectors"][cohort["embedding_index"].to_numpy()]
            for local_seed in range(args.seeds_per_rep):
                seed = 20260815 + replication * args.seeds_per_rep + local_seed
                for k in [int(value) for value in args.k_values.split(",")]:
                    oof, metrics = run_grouped_cluster_cv(
                        cohort[["record_id"]].rename(columns={"record_id": "sample_id"}).assign(
                            group_id=lambda frame: frame["sample_id"]
                        ),
                        values, n_splits=args.folds, k=k, seed=seed, target_rule="largest",
                    )
                    oof = oof.merge(
                        cohort[["record_id", "source_race_label"]],
                        left_on="sample_id", right_on="record_id", validate="one_to_one",
                    ).drop(columns="record_id")
                    oof = oof.rename(columns={"y_true": "is_target", "score_raw": "score", "prob_calibrated": "prob_calibrated", "source_race_label": "fairface_category"})
                    oof["scenario"], oof["replication"] = scenario, replication
                    predictions.append(oof)
                    run_rows.append({"scenario": scenario, "replication": replication, "seed": seed, "k": k, "auc": metrics["auc"], "target_size": int(oof["is_target"].sum())})
    prediction_frame = pd.concat(predictions, ignore_index=True)
    run_frame = pd.DataFrame(run_rows)
    target_sets = {
        key: group.set_index("sample_id")["is_target"].astype(bool)
        for key, group in prediction_frame.groupby(["scenario", "replication", "seed", "k"])
    }
    jaccards = []
    for row in run_frame.itertuples(index=False):
        reference = target_sets.get(("A", row.replication, row.seed, row.k))
        current = target_sets[(row.scenario, row.replication, row.seed, row.k)]
        jaccards.append(_jaccard(reference, current) if reference is not None else np.nan)
    run_frame["jaccard"] = jaccards
    decompositions = pd.concat(
        [decompose_variance(run_frame, outcome, ["scenario", "replication", "seed", "k"]).as_frame() for outcome in ["auc", "jaccard", "target_size"]],
        ignore_index=True,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(out / "oof_predictions.parquet", index=False)
    decompositions.to_csv(out / "variance_decomposition.csv", index=False)
    run_frame.to_csv(out / "run_metrics.csv", index=False)
    (out / "run_manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
