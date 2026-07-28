"""Predeclared one-factor sensitivity analyses for clustering implementation choices."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from research_audit_v2.src.clustering_stability import jaccard
from research_audit_v2.src.common import l2_normalize, write_csv
from research_audit_v2.src.target_heuristic import choose_target_cluster


def _run(values: np.ndarray, cfg: dict, batch_size: int, seed: int) -> tuple[MiniBatchKMeans, np.ndarray, float]:
    started = time.perf_counter()
    model = MiniBatchKMeans(cfg["k"], random_state=seed, n_init=cfg["n_init"], max_iter=cfg["max_iter"], batch_size=batch_size, reassignment_ratio=.01).fit(values)
    return model, model.labels_, time.perf_counter()-started


def clustering_sensitivity(vectors: np.ndarray, cfg: dict, tables: Path, figures: Path, reports: Path) -> None:
    values = l2_normalize(np.asarray(vectors, dtype=np.float32))
    base_model, base_labels, base_time = _run(values, cfg, cfg["batch_size"], cfg["random_seed"])
    base_target = base_labels == choose_target_cluster(base_labels)
    rows = []
    rng = np.random.default_rng(cfg["random_seed"])
    orders = {"original": np.arange(len(values)), "random": rng.permutation(len(values)), "hashed_record_id": np.argsort(np.arange(len(values))[::-1]), "reversed": np.arange(len(values))[::-1]}
    for name, order in orders.items():
        model, labels_ordered, elapsed = _run(values[order], cfg, cfg["batch_size"], cfg["random_seed"])
        labels = np.empty_like(labels_ordered); labels[order] = labels_ordered
        target = labels == choose_target_cluster(labels)
        rows.append({"factor": "record_order", "setting": name, "ari_vs_baseline": adjusted_rand_score(base_labels, labels), "nmi_vs_baseline": normalized_mutual_info_score(base_labels, labels), "target_jaccard": jaccard(base_target, target), "target_size": int(target.sum()), "inertia": float(model.inertia_), "iterations": int(model.n_iter_), "runtime_seconds": elapsed})
    order_frame = pd.DataFrame(rows); write_csv(order_frame, tables / "minibatch_order_sensitivity.csv")
    parameter_rows = []
    for batch in [256, 512, 1024, 2048, 4096]:
        model, labels, elapsed = _run(values, cfg, batch, cfg["random_seed"])
        target = labels == choose_target_cluster(labels)
        parameter_rows.append({"factor": "batch_size", "setting": batch, "ari_vs_baseline": adjusted_rand_score(base_labels, labels), "nmi_vs_baseline": normalized_mutual_info_score(base_labels, labels), "target_jaccard": jaccard(base_target, target), "target_size": int(target.sum()), "inertia": float(model.inertia_), "iterations": int(model.n_iter_), "runtime_seconds": elapsed})
    parameter_frame = pd.DataFrame(parameter_rows); write_csv(parameter_frame, tables / "minibatch_parameter_sensitivity.csv")
    representations = {"original_float32": np.asarray(vectors, dtype=np.float32), "l2_normalized": values, "centered": values-values.mean(axis=0, keepdims=True), "pca_64": PCA(n_components=64, random_state=cfg["random_seed"]).fit_transform(values)}
    rep_rows = []
    for name, transformed in representations.items():
        model, labels, elapsed = _run(np.asarray(transformed, dtype=np.float32), cfg, cfg["batch_size"], cfg["random_seed"])
        target = labels == choose_target_cluster(labels)
        rep_rows.append({"representation": name, "dimensions": int(transformed.shape[1]), "ari_vs_l2_baseline": adjusted_rand_score(base_labels, labels), "nmi_vs_l2_baseline": normalized_mutual_info_score(base_labels, labels), "target_jaccard": jaccard(base_target, target), "target_size": int(target.sum()), "inertia": float(model.inertia_), "runtime_seconds": elapsed, "status": "sensitivity_only"})
    rep_frame = pd.DataFrame(rep_rows); write_csv(rep_frame, tables / "embedding_representation_sensitivity.csv")
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(order_frame["setting"], order_frame["ari_vs_baseline"], "o-", color="black"); ax.set_ylabel("ARI vs baseline"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(figures / "preprocessing_ari.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); ax.bar(rep_frame["representation"], rep_frame["target_jaccard"], color="0.4"); ax.set_ylabel("target Jaccard vs L2 baseline"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(figures / "representation_stability.svg"); plt.close(fig)
    reports.joinpath("minibatch_sensitivity_report.md").write_text("# MiniBatch sensitivity\n\nA predeclared one-factor design varied record order and batch size with fixed seed, k, n_init and max_iter. It is sensitivity evidence only; no setting replaces the locked primary configuration.\n", encoding="utf-8")
    reports.joinpath("embedding_representation_report.md").write_text("# Embedding representation sensitivity\n\nOriginal, L2-normalized, centered and PCA-64 representations were compared as a sensitivity analysis. PCA and centering are fit on the complete data here only for partition sensitivity; they are not used for out-of-sample claims and must be fit inside training folds in any cross-fitted extension.\n", encoding="utf-8")
