#!/usr/bin/env python3
"""Build a non-random face-profile manifest from pretrained embeddings.

The script uses an unsupervised embedding cohort as the positive facial profile,
selects distant embeddings as negatives, and marks ambiguous samples as ignore.
It does not train or fine-tune any neural network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


QUALITY_WEIGHTS = {"high": 1.0, "mid": 0.8, "low": 0.6}
POSITIVE_SPLITS = {"profile", "calib_pos", "test_pos"}
NEGATIVE_SPLITS = {"calib_neg", "test_neg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create semantic manifest splits from extracted face embeddings.")
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--features", default="artifacts/embedding_manifest.csv")
    parser.add_argument("--embeddings", default="artifacts/embeddings.npy")
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--backup", default="manifest.random_backup.csv")
    parser.add_argument("--report", default="artifacts/semantic_manifest_report.json")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--positive-threshold", type=float, default=0.26)
    parser.add_argument("--negative-threshold", type=float, default=0.02)
    parser.add_argument("--min-negative-rows", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    features_path = Path(args.features)
    embeddings_path = Path(args.embeddings)

    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(features_path)
    embeddings = np.load(embeddings_path).astype("float32")

    valid = features[features["embedding_index"].astype(int) >= 0].copy()
    embedding_indices = valid["embedding_index"].astype(int).to_numpy()
    x = embeddings[embedding_indices]
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=args.random_state,
        batch_size=2048,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_.astype("float32")
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

    valid["cluster"] = labels
    target_cluster, cluster_report = choose_target_cluster(valid, x, labels, centers)
    target_center = centers[target_cluster]
    valid["target_sim"] = x @ target_center

    manifest = manifest[["path", "subject_id", "quality", "split", "weight"]].copy()
    manifest["loose_identity"] = manifest["subject_id"].map(loose_identity)
    manifest["target_sim"] = np.nan
    manifest.loc[valid.index, "target_sim"] = valid["target_sim"]

    group_stats = (
        manifest.groupby("loose_identity")
        .agg(
            n=("path", "size"),
            max_sim=("target_sim", "max"),
            mean_sim=("target_sim", "mean"),
            valid_n=("target_sim", "count"),
        )
        .reset_index()
    )
    positive_groups = set(group_stats[group_stats["max_sim"] >= args.positive_threshold]["loose_identity"])
    negative_pool = group_stats[
        (group_stats["max_sim"] <= args.negative_threshold)
        & (~group_stats["loose_identity"].isin(positive_groups))
    ].sort_values(["max_sim", "loose_identity"], ascending=[True, True])

    negative_target = max(args.min_negative_rows, int(manifest["loose_identity"].isin(positive_groups).sum()))
    negative_groups = select_negative_groups(negative_pool, negative_target)

    manifest["split"] = "ignore"
    pos_mask = manifest["loose_identity"].isin(positive_groups)
    neg_mask = manifest["loose_identity"].isin(negative_groups)
    manifest.loc[pos_mask, "split"] = manifest.loc[pos_mask, "loose_identity"].map(split_positive)
    manifest.loc[neg_mask, "split"] = manifest.loc[neg_mask, "loose_identity"].map(split_negative)
    manifest["weight"] = manifest["quality"].map(QUALITY_WEIGHTS).astype(float)

    output = manifest[["path", "subject_id", "quality", "split", "weight"]]
    out_path = Path(args.out)
    if out_path.resolve() == manifest_path.resolve() and not Path(args.backup).exists():
        shutil.copy2(manifest_path, args.backup)
    output.to_csv(out_path, index=False)

    sync_features(features_path, output)
    report = build_report(output, valid, target_cluster, cluster_report, args)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def choose_target_cluster(
    valid: pd.DataFrame,
    x: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[int, list[dict[str, float | int]]]:
    candidates: list[dict[str, float | int]] = []
    for cluster_id, group in valid.groupby("cluster"):
        mask = labels == cluster_id
        n = int(mask.sum())
        if n < 100:
            continue
        sims = np.sum(x[mask] * centers[int(cluster_id)], axis=1)
        mean_sim = float(sims.mean())
        item = {
            "cluster": int(cluster_id),
            "n": n,
            "mean_sim": mean_sim,
            "p10_sim": float(np.quantile(sims, 0.10)),
            "score": float(mean_sim * np.log1p(n)),
        }
        candidates.append(item)
    if not candidates:
        raise RuntimeError("No suitable embedding cluster found.")
    candidates.sort(key=lambda item: (float(item["score"]), float(item["mean_sim"])), reverse=True)
    return int(candidates[0]["cluster"]), candidates[:10]


def select_negative_groups(pool: pd.DataFrame, target_rows: int) -> set[str]:
    selected: set[str] = set()
    rows = 0
    for _, item in pool.iterrows():
        selected.add(str(item["loose_identity"]))
        rows += int(item["n"])
        if rows >= target_rows:
            break
    return selected


def sync_features(features_path: Path, manifest: pd.DataFrame) -> None:
    features = pd.read_csv(features_path)
    mapping = manifest.set_index("path")[["split", "weight"]]
    features = features.drop(columns=[c for c in ["split", "weight"] if c in features.columns])
    features = features.merge(mapping, left_on="path", right_index=True, how="left")
    features["split"] = features["split"].fillna("ignore")
    features["weight"] = features["weight"].fillna(1.0).astype(float)
    features.to_csv(features_path, index=False)


def build_report(
    manifest: pd.DataFrame,
    valid: pd.DataFrame,
    target_cluster: int,
    cluster_report: list[dict[str, float | int]],
    args: argparse.Namespace,
) -> dict[str, object]:
    split_counts = manifest["split"].value_counts().sort_index().to_dict()
    quality_x_split = pd.crosstab(manifest["quality"], manifest["split"]).to_dict()
    return {
        "summary": {
            "target_cluster": target_cluster,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "rows": int(len(manifest)),
            "positive_rows": int(manifest["split"].isin(POSITIVE_SPLITS).sum()),
            "negative_rows": int(manifest["split"].isin(NEGATIVE_SPLITS).sum()),
            "ignore_rows": int((manifest["split"] == "ignore").sum()),
            "valid_embeddings": int(len(valid)),
            "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        },
        "quality_x_split": quality_x_split,
        "top_clusters": cluster_report,
    }


def split_positive(group: str) -> str:
    value = stable_u01(f"pos:{group}")
    if value < 0.60:
        return "profile"
    if value < 0.80:
        return "calib_pos"
    return "test_pos"


def split_negative(group: str) -> str:
    return "calib_neg" if stable_u01(f"neg:{group}") < 0.50 else "test_neg"


def stable_u01(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def loose_identity(stem: str) -> str:
    value = str(stem).lower()
    value = re.sub(r"^src_[0-9a-f]{10}_\d+_", "", value)
    value = re.sub(r"_[0-9a-f]{10}$", "", value)
    value = re.sub(r"_[0-9a-f]{8,}$", "", value)
    value = re.sub(r"_\d{4}-\d+(_\d+)?(_[0-9a-f]+)?$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or str(stem).lower()


if __name__ == "__main__":
    raise SystemExit(main())
