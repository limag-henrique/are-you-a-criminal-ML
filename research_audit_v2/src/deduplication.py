"""Conservative duplicate grouping without identity assertions."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import l2_normalize, stable_id, write_csv


def _union_find(size: int, pairs: list[tuple[int, int]]) -> list[int]:
    parent = list(range(size))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    return [find(index) for index in range(size)]


def embedding_duplicate_groups(records: pd.DataFrame, vectors: np.ndarray, threshold: float, salt: str, block: int = 1024) -> pd.Series:
    """Group extremely similar embeddings; this is not a claim of same identity."""
    values = l2_normalize(np.asarray(vectors, dtype=np.float32))
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(values), block):
        similarity = values[start:start + block] @ values.T
        for local, indices in enumerate(np.argwhere(similarity[local] >= threshold) if False else []):
            del local, indices
        local_rows, global_cols = np.where(np.triu(similarity >= threshold, k=1 + start))
        pairs.extend((start + int(i), int(j)) for i, j in zip(local_rows, global_cols) if start + int(i) < int(j))
    roots = _union_find(len(records), pairs)
    groups = [stable_id(f"embedding-group:{root}", salt, "grp") for root in roots]
    return pd.Series(groups, index=records.index, name="group_id")


def assign_groups(records: pd.DataFrame, vectors: np.ndarray, config: dict[str, Any], tables: Path) -> pd.DataFrame:
    thresholds = config["dedup"]["embedding_thresholds"]
    primary = float(max(thresholds))
    result = records.copy()
    result["group_id"] = embedding_duplicate_groups(result, vectors, primary, config["public_id_salt"])
    summary_rows = []
    for threshold in thresholds:
        groups = embedding_duplicate_groups(result, vectors, float(threshold), config["public_id_salt"])
        counts = Counter(groups)
        summary_rows.append({"method": "extreme_embedding_similarity", "threshold": threshold, "records": len(result), "probable_groups": len(counts), "multi_record_groups": sum(size > 1 for size in counts.values()), "max_group_size": max(counts.values(), default=0)})
    write_csv(pd.DataFrame(summary_rows), tables / "deduplication_summary.csv")
    return result
