"""CLI orchestration for the isolated FairFace composition experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from research_audit_v2.demographic_composition.analysis import (
    RunResult,
    compare_scenarios_on_intersection,
    cross_fitted_scores,
    fit_scenario_run,
    pairwise_seed_stability,
)
from research_audit_v2.demographic_composition.cohorts import build_scenarios
from research_audit_v2.demographic_composition.embeddings import (
    FairFaceAlignedCropEmbedder,
    extract_union_embeddings,
)
from research_audit_v2.demographic_composition.reporting import write_report
from research_audit_v2.second_phase.src.io import atomic_target, atomic_write_csv, atomic_write_json
from research_audit_v2.second_phase.src.privacy_scan import write_privacy_report
from research_audit_v2.src.common import sha256_file


def _pipeline_digest(root: Path) -> str:
    digest = hashlib.sha256()
    patterns = ("face_profile_ml/*.py", "research_audit_v2/src/*.py", "research_audit_v2/second_phase/src/*.py")
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _analysis_key(
    vectors_sha256: str,
    record_ids: pd.Series,
    scenario: str,
    seed: int,
    k: int,
    config: Mapping[str, object],
) -> str:
    payload = {
        "vectors_sha256": vectors_sha256,
        "record_ids_sha256": hashlib.sha256("\n".join(record_ids.astype(str)).encode("utf-8")).hexdigest(),
        "scenario": scenario,
        "seed": seed,
        "k": k,
        "batch_size": config["batch_size"],
        "max_iter": config["max_iter"],
        "n_init": config["n_init"],
        "target_rule": config["target_rule"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _checkpoint_result(
    result: RunResult, root: Path, run_id: str, key: str
) -> None:
    with atomic_target(root / f"{run_id}.npz") as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                record_ids=result.partition["record_id"].to_numpy(str),
                cluster=result.partition["cluster"].to_numpy(int),
                is_target=result.partition["is_target"].to_numpy(np.uint8),
                score=result.partition["score"].to_numpy(np.float32),
            )
    atomic_write_json(root / f"{run_id}.json", {"checkpoint_key": key, "metrics": result.metrics})


def _load_result(
    frame: pd.DataFrame,
    root: Path,
    run_id: str,
    key: str,
    config: Mapping[str, object],
) -> RunResult | None:
    metadata_path = root / f"{run_id}.json"
    arrays_path = root / f"{run_id}.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("checkpoint_key") != key:
            return None
        with np.load(arrays_path, allow_pickle=False) as stored:
            ids = stored["record_ids"].astype(str)
            labels = stored["cluster"].astype(int)
            target = stored["is_target"].astype(bool)
            score = stored["score"].astype(float)
        ordered = frame.sort_values("record_id").reset_index(drop=True)
        if not np.array_equal(ids, ordered["record_id"].to_numpy(str)):
            return None
        group_column = str(config["group_column"])
        partition = ordered[["record_id", "scenario", group_column]].copy()
        partition["cluster"] = labels
        partition["is_target"] = target
        partition["score"] = score
        composition = partition.groupby(["cluster", group_column], observed=True).size().rename("count").reset_index()
        composition["proportion_within_cluster"] = composition["count"] / composition.groupby("cluster")["count"].transform("sum")
        metrics = dict(metadata["metrics"])
        composition.insert(0, "scenario", metrics["scenario"])
        composition.insert(1, "seed", int(metrics["seed"]))
        composition.insert(2, "k", int(metrics["k"]))
        return RunResult(metrics=metrics, partition=partition, composition=composition)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _public_config(config: Mapping[str, object]) -> dict[str, object]:
    return {
        key: config[key]
        for key in (
            "random_seed", "sample_size", "group_column", "perturbed_group", "model_name",
            "det_size", "preprocessing_mode", "embedding_batch_size", "seeds", "k_values", "primary_k", "batch_size", "max_iter", "n_init",
            "outer_folds", "target_rule", "relevance_thresholds", "execution_mode",
        )
    }


def run_experiment(
    config: Mapping[str, object],
    *,
    destination: str | Path = "DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md",
    smoke: bool = False,
    resume: bool = False,
    embedder_factory: Callable[..., object] = FairFaceAlignedCropEmbedder,
) -> Path:
    cfg = dict(config)
    cfg["execution_mode"] = "smoke" if smoke else "final"
    if smoke and int(cfg["sample_size"]) > 840:
        cfg["sample_size"] = 840
        cfg["seeds"] = [list(cfg["seeds"])[0]]
        cfg["k_values"] = [int(cfg["primary_k"])]
        cfg["output_root"] = str(Path(str(cfg["output_root"])) / "smoke")
    output = Path(str(cfg["output_root"]))
    private = Path(str(cfg["private_root"]))
    tables = output / "tables"
    checkpoints = private / "analysis_checkpoints"
    tables.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    repository = Path.cwd()
    pipeline_before = _pipeline_digest(repository)
    catalog_path = Path(str(cfg["catalog_path"]))
    catalog = pd.read_csv(catalog_path)
    selected, reserves = build_scenarios(catalog, cfg)
    final, vectors, failures = extract_union_embeddings(
        selected,
        reserves,
        str(cfg["image_root"]),
        private,
        cfg,
        embedder_factory=embedder_factory,
    )
    atomic_write_csv(private / "final_selection.csv", final)
    atomic_write_csv(private / "extraction_failures.csv", failures)
    composition = final.groupby(["scenario", str(cfg["group_column"])]).size().rename("count").reset_index()
    composition["proportion"] = composition["count"] / composition.groupby("scenario")["count"].transform("sum")
    atomic_write_csv(tables / "scenario_composition.csv", composition)
    failure_summary = failures.groupby(["source_race_label", "error_type"]).size().rename("count").reset_index()
    atomic_write_csv(tables / "extraction_failure_summary.csv", failure_summary)

    vectors_sha256 = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()
    runs: dict[tuple[str, int, int], RunResult] = {}
    run_rows: list[dict[str, object]] = []
    composition_rows: list[pd.DataFrame] = []
    for k in [int(value) for value in cfg["k_values"]]:
        for seed in [int(value) for value in cfg["seeds"]]:
            for scenario in "ABCD":
                scenario_frame = final[final["scenario"].eq(scenario)]
                run_id = f"{scenario}_seed{seed}_k{k}"
                key = _analysis_key(vectors_sha256, scenario_frame["record_id"], scenario, seed, k, cfg)
                result = _load_result(scenario_frame, checkpoints, run_id, key, cfg) if resume else None
                if result is None:
                    result = fit_scenario_run(final, vectors, scenario, seed, k, cfg)
                    _checkpoint_result(result, checkpoints, run_id, key)
                runs[(scenario, seed, k)] = result
                run_rows.append(result.metrics)
                cluster_composition = result.composition.copy()
                cluster_composition["is_target_cluster"] = cluster_composition["cluster"].eq(result.metrics["target_cluster"])
                composition_rows.append(cluster_composition)
    atomic_write_csv(tables / "run_metrics.csv", pd.DataFrame(run_rows))
    atomic_write_csv(tables / "cluster_composition.csv", pd.concat(composition_rows, ignore_index=True))

    stability_rows = []
    for scenario in "ABCD":
        for k in [int(value) for value in cfg["k_values"]]:
            partitions = {str(seed): runs[(scenario, int(seed), k)].partition for seed in cfg["seeds"]}
            pairwise = pairwise_seed_stability(partitions)
            if pairwise.empty:
                only = str(list(partitions)[0])
                pairwise = pd.DataFrame([{"run_a": only, "run_b": only, "intersection_n": len(partitions[only]), "ari": 1.0, "target_jaccard": 1.0}])
            pairwise.insert(0, "scenario", scenario)
            pairwise.insert(1, "k", k)
            stability_rows.append(pairwise)
    atomic_write_csv(tables / "seed_stability.csv", pd.concat(stability_rows, ignore_index=True))

    comparison_rows = []
    for k in [int(value) for value in cfg["k_values"]]:
        for seed in [int(value) for value in cfg["seeds"]]:
            compared = compare_scenarios_on_intersection({scenario: runs[(scenario, seed, k)].partition for scenario in "ABCD"})
            compared.insert(0, "seed", seed)
            compared.insert(1, "k", k)
            comparison_rows.append(compared)
    atomic_write_csv(tables / "scenario_comparisons.csv", pd.concat(comparison_rows, ignore_index=True))

    cross_rows = []
    primary_k = int(cfg["primary_k"])
    for scenario in "ABCD":
        for seed in [int(value) for value in cfg["seeds"]]:
            cross_rows.append(cross_fitted_scores(final, vectors, scenario, seed, primary_k, cfg))
    atomic_write_csv(tables / "cross_fitted_metrics.csv", pd.concat(cross_rows, ignore_index=True))

    destination_path = write_report(output, destination, cfg)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination_path, reports / destination_path.name)
    parameters = json.dumps(
        {key: cfg[key] for key in ("model_name", "preprocessing_mode", "embedding_batch_size", "batch_size", "max_iter", "n_init", "target_rule")},
        sort_keys=True,
    )
    manifest = {
        "status": "privacy_pending",
        "execution_mode": "smoke" if smoke else "final",
        "catalog_sha256": sha256_file(catalog_path),
        "configuration_sha256": hashlib.sha256(json.dumps(_public_config(cfg), sort_keys=True).encode("utf-8")).hexdigest(),
        "vectors_sha256": vectors_sha256,
        "parameters_by_scenario": {scenario: parameters for scenario in "ABCD"},
        "run_count": len(run_rows),
        "cross_fitted_fold_count": int(sum(len(frame) for frame in cross_rows)),
        "privacy_status": "pending",
        "pipeline_digest_before": pipeline_before,
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    findings = write_privacy_report(output, output / "privacy_scan.json")
    pipeline_after = _pipeline_digest(repository)
    if pipeline_before != pipeline_after:
        raise RuntimeError("Main pipeline files changed during the isolated experiment.")
    if findings:
        raise RuntimeError(f"Public-output privacy scan failed with {len(findings)} finding(s).")
    manifest.update({"status": "complete", "privacy_status": "passed", "pipeline_digest_after": pipeline_after})
    atomic_write_json(output / "run_manifest.json", manifest)
    return destination_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the isolated FairFace demographic-composition experiment.")
    parser.add_argument("--config", default="research_audit_v2/demographic_composition/config.json")
    parser.add_argument("--destination", default="DEMOGRAPHIC_COMPOSITION_EXPERIMENT.md")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_experiment(config, destination=args.destination, smoke=args.smoke, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
