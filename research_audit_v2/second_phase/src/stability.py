"""Separated stochastic, operational and representation stability analyses."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

from research_audit_v2.src.clustering_stability import jaccard
from research_audit_v2.src.common import l2_normalize, write_csv
from research_audit_v2.src.target_heuristic import choose_target_cluster

from .representation import fit_full_representation, write_pca_specification
from .io import atomic_target, atomic_write_json


METRICS = ["ari", "target_jaccard", "target_size", "target_prevalence"]


def _explicit_seeds(config: Mapping[str, Any]) -> list[int]:
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(value, int) for value in seeds):
        raise ValueError("Stability seeds must be a non-empty explicit integer list.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Stability seeds must be unique.")
    return seeds


def build_stability_design(config: Mapping[str, Any]) -> pd.DataFrame:
    """Predeclare one-factor designs so interpretations stay separated."""
    seeds = _explicit_seeds(config)
    k_values = [int(value) for value in config["k_values"]]
    batches = [int(value) for value in config["batch_sizes"]]
    orderings = [str(value) for value in config["orderings"]]
    representations = [str(value) for value in config["representations"]]
    primary_k = int(config["primary_k"])
    primary_batch = int(config["primary_batch_size"])
    primary_seed = seeds[0]
    rows: list[dict[str, object]] = []

    for k in k_values:
        for seed in seeds:
            rows.append(
                {
                    "instability_type": "stochastic",
                    "comparison_group": f"stochastic_k_{k}",
                    "k": k,
                    "seed": seed,
                    "batch_size": primary_batch,
                    "order": "original",
                    "representation": "original_l2",
                }
            )
    for order in orderings:
        rows.append(
            {
                "instability_type": "operational_order",
                "comparison_group": "operational_order",
                "k": primary_k,
                "seed": primary_seed,
                "batch_size": primary_batch,
                "order": order,
                "representation": "original_l2",
            }
        )
    for batch in batches:
        rows.append(
            {
                "instability_type": "operational_batch",
                "comparison_group": "operational_batch",
                "k": primary_k,
                "seed": primary_seed,
                "batch_size": batch,
                "order": "original",
                "representation": "original_l2",
            }
        )
    for representation in representations:
        rows.append(
            {
                "instability_type": "representation",
                "comparison_group": "representation",
                "k": primary_k,
                "seed": primary_seed,
                "batch_size": primary_batch,
                "order": "original",
                "representation": representation,
            }
        )
    design = pd.DataFrame(rows)
    design.insert(0, "run_id", [f"stability_{index:04d}" for index in range(len(design))])
    return design


def _summarize_metrics(
    frame: pd.DataFrame, group_columns: list[str], metrics: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, dropna=False, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        prefix = dict(zip(group_columns, key_values))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "q1": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q3": float(values.quantile(0.75)) if len(values) else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                    "p05": float(values.quantile(0.05)) if len(values) else np.nan,
                    "p95": float(values.quantile(0.95)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_stability(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize frames that already contain all four declared metrics."""
    return _summarize_metrics(frame, group_columns, METRICS)


def summarize_pairwise_stability(
    runs: pd.DataFrame, pairwise: pd.DataFrame
) -> pd.DataFrame:
    """Summarize partition metrics without selecting an arbitrary reference run."""
    group_columns = ["instability_type", "k"]
    run_summary = _summarize_metrics(
        runs,
        group_columns,
        ["target_size", "target_prevalence"],
    )
    unique_pairs = pairwise.loc[pairwise["run_a"] < pairwise["run_b"]].copy()
    comparison_metadata = runs[
        ["comparison_group", "instability_type", "k"]
    ].drop_duplicates()
    unique_pairs = unique_pairs.merge(
        comparison_metadata,
        on="comparison_group",
        how="left",
        validate="many_to_one",
    )
    pair_summary = _summarize_metrics(
        unique_pairs,
        group_columns,
        ["ari", "target_jaccard"],
    )
    return pd.concat([pair_summary, run_summary], ignore_index=True).sort_values(
        [*group_columns, "metric"], ignore_index=True
    )


def pairwise_partition_metrics(
    partitions: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    run_ids = list(partitions)
    for left in run_ids:
        left_labels, left_target = partitions[left]
        for right in run_ids:
            right_labels, right_target = partitions[right]
            rows.append(
                {
                    "run_a": left,
                    "run_b": right,
                    "ari": float(adjusted_rand_score(left_labels, right_labels)),
                    "target_jaccard": jaccard(left_target, right_target),
                }
            )
    return pd.DataFrame(rows)


def _order_indices(size: int, name: str, seed: int) -> np.ndarray:
    if name == "original":
        return np.arange(size)
    if name == "reversed":
        return np.arange(size - 1, -1, -1)
    if name == "seeded_shuffle":
        return np.random.default_rng(seed).permutation(size)
    if name == "hashed_index":
        keys = [hashlib.sha256(str(index).encode("ascii")).digest() for index in range(size)]
        return np.argsort(keys)
    raise ValueError(f"Unknown operational order: {name}")


def _representation(
    values: np.ndarray,
    name: str,
    config: Mapping[str, Any],
    table_root: Path,
) -> np.ndarray:
    if name == "original_l2":
        return l2_normalize(np.asarray(values, dtype=np.float32))
    if name == "pca_64":
        fitted = fit_full_representation(values, config["pca_64"])
        write_pca_specification(table_root / "pca_specification.json", fitted)
        return fitted.train
    raise ValueError(f"Unknown representation: {name}")


def _fit_partition(values: np.ndarray, row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, int]:
    order = _order_indices(len(values), str(row["order"]), int(row["seed"]))
    started = time.perf_counter()
    model = MiniBatchKMeans(
        n_clusters=int(row["k"]),
        random_state=int(row["seed"]),
        batch_size=int(row["batch_size"]),
        max_iter=int(config["cluster_max_iter"]),
        n_init=int(config["n_init"]),
        reassignment_ratio=0.01,
    ).fit(values[order])
    labels = np.empty_like(model.labels_)
    labels[order] = model.labels_
    target_cluster = int(choose_target_cluster(labels, config["target_rule"]))
    target = labels == target_cluster
    return labels, target, time.perf_counter() - started, int(model.n_iter_)


def _checkpoint_key(
    input_sha256: str, row: Mapping[str, Any], config: Mapping[str, Any]
) -> str:
    payload = {
        "input_sha256": input_sha256,
        "design": dict(row),
        "cluster_max_iter": int(config["cluster_max_iter"]),
        "n_init": int(config["n_init"]),
        "target_rule": str(config["target_rule"]),
        "pca_64": config.get("pca_64") if row["representation"] == "pca_64" else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(
    root: Path, run_id: str, key: str, expected_rows: int
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    metadata_path = root / f"{run_id}.json"
    arrays_path = root / f"{run_id}.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("checkpoint_key") != key:
            return None
        with np.load(arrays_path, allow_pickle=False) as stored:
            labels = stored["labels"]
            target = stored["target"].astype(bool)
        if len(labels) != expected_rows or len(target) != expected_rows:
            return None
        return labels, target, float(metadata["runtime_seconds"]), int(metadata["iterations"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_checkpoint(
    root: Path,
    run_id: str,
    key: str,
    labels: np.ndarray,
    target: np.ndarray,
    runtime_seconds: float,
    iterations: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / f"{run_id}.npz"
    with atomic_target(arrays_path) as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, labels=labels, target=target.astype(np.uint8))
            handle.flush()
            os.fsync(handle.fileno())
    atomic_write_json(
        root / f"{run_id}.json",
        {
            "checkpoint_key": key,
            "runtime_seconds": runtime_seconds,
            "iterations": iterations,
            "status": "complete",
        },
    )


def run_stability_analysis(
    vectors: np.ndarray,
    config: Mapping[str, Any],
    tables: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_root = Path(tables)
    table_root.mkdir(parents=True, exist_ok=True)
    design = build_stability_design(config)
    representation_cache: dict[str, np.ndarray] = {}
    partitions_by_group: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    run_rows: list[dict[str, object]] = []
    raw_partitions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    checkpoint_path = Path(checkpoint_root) if checkpoint_root is not None else None
    input_array = np.ascontiguousarray(vectors)
    input_sha256 = hashlib.sha256(input_array.tobytes()).hexdigest()

    for row in design.to_dict("records"):
        representation = str(row["representation"])
        if representation not in representation_cache:
            representation_cache[representation] = _representation(
                vectors, representation, config, table_root
            )
        run_id = str(row["run_id"])
        checkpoint_key = _checkpoint_key(input_sha256, row, config)
        cached = (
            _load_checkpoint(checkpoint_path, run_id, checkpoint_key, len(vectors))
            if resume and checkpoint_path is not None
            else None
        )
        if cached is None:
            labels, target, elapsed, iterations = _fit_partition(
                representation_cache[representation], row, config
            )
            if checkpoint_path is not None:
                _write_checkpoint(
                    checkpoint_path,
                    run_id,
                    checkpoint_key,
                    labels,
                    target,
                    elapsed,
                    iterations,
                )
        else:
            labels, target, elapsed, iterations = cached
        comparison_group = str(row["comparison_group"])
        raw_partitions[run_id] = (labels, target)
        partitions_by_group.setdefault(comparison_group, {})[run_id] = (labels, target)
        run_rows.append(
            {
                **row,
                "target_size": int(target.sum()),
                "target_prevalence": float(target.mean()),
                "runtime_seconds": elapsed,
                "iterations": iterations,
            }
        )

    pairwise_frames: list[pd.DataFrame] = []
    for comparison_group, partitions in partitions_by_group.items():
        pairwise = pairwise_partition_metrics(partitions)
        pairwise.insert(0, "comparison_group", comparison_group)
        pairwise_frames.append(pairwise)

    runs = pd.DataFrame(run_rows)
    pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    summary = summarize_pairwise_stability(runs, pairwise)
    write_csv(runs, table_root / "stability_runs.csv")
    write_csv(summary, table_root / "stability_summary.csv")
    write_csv(pairwise, table_root / "stability_pairwise.csv")
    return runs, summary, pairwise
