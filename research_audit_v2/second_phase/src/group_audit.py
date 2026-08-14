"""Aggregate validation for probable-duplicate group IDs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research_audit_v2.src.common import l2_normalize


def summarize_probable_duplicate_groups(
    groups: pd.Series,
    *,
    metric: str,
    threshold: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    if groups.empty or groups.isna().any():
        raise ValueError("Probable-duplicate group IDs must be non-empty and complete.")
    sizes = groups.astype(str).value_counts(sort=False)
    size_distribution = sizes.value_counts().sort_index()
    distribution = pd.DataFrame(
        {
            "group_size": size_distribution.index.astype(int),
            "groups": size_distribution.to_numpy(dtype=int),
        }
    )
    distribution["records"] = distribution["group_size"] * distribution["groups"]
    grouped_records = int(sizes[sizes > 1].sum())
    summary: dict[str, object] = {
        "records": int(len(groups)),
        "groups": int(len(sizes)),
        "singleton_groups": int((sizes == 1).sum()),
        "non_singleton_groups": int((sizes > 1).sum()),
        "grouped_records": grouped_records,
        "grouped_record_proportion": float(grouped_records / len(groups)),
        "mean_group_size": float(sizes.mean()),
        "median_group_size": float(sizes.median()),
        "max_group_size": int(sizes.max()),
        "metric": metric,
        "threshold": float(threshold),
        "interpretation": "probable_duplicate_only_not_confirmed_identity",
    }
    return summary, distribution.reset_index(drop=True)


def safe_threshold_review_sample(
    records: pd.DataFrame,
    vectors: np.ndarray,
    *,
    threshold: float,
    window: float,
    max_pairs: int,
    salt: str,
    block_size: int = 512,
) -> pd.DataFrame:
    """Return unlinkable pair hashes near the grouping threshold."""
    if "record_id" not in records or len(records) != len(vectors):
        raise ValueError("Pseudonymous record IDs and embeddings must be aligned.")
    values = l2_normalize(np.asarray(vectors, dtype=np.float32))
    candidates: list[dict[str, object]] = []
    lower, upper = threshold - window, threshold + window
    record_ids = records["record_id"].astype(str).to_numpy()
    for start in range(0, len(values), block_size):
        similarities = values[start : start + block_size] @ values.T
        local_rows, columns = np.where((similarities >= lower) & (similarities <= upper))
        for local, right in zip(local_rows, columns):
            left = start + int(local)
            right = int(right)
            if left >= right:
                continue
            pair_key = "|".join(sorted((record_ids[left], record_ids[right])))
            pair_hash = hashlib.sha256(f"{salt}|{pair_key}".encode("utf-8")).hexdigest()[:20]
            similarity = round(float(similarities[local, right]), 8)
            candidates.append(
                {
                    "pair_id": f"pair_{pair_hash}",
                    "similarity": similarity,
                    "distance_from_threshold": round(abs(similarity - threshold), 8),
                    "metric": "cosine_similarity",
                    "threshold": float(threshold),
                    "review_status": "manual_review_not_performed",
                }
            )
    columns = [
        "pair_id",
        "similarity",
        "distance_from_threshold",
        "metric",
        "threshold",
        "review_status",
    ]
    return (
        pd.DataFrame(candidates, columns=columns)
        .sort_values(["distance_from_threshold", "pair_id"])
        .head(max_pairs)
        .reset_index(drop=True)
    )
