"""Deterministic FairFace scenario construction using source-provided labels."""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Mapping

import pandas as pd

from research_audit_v2.src.common import stable_id


FAIRFACE_GROUPS = (
    "Black",
    "East Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
    "Southeast Asian",
    "White",
)


def largest_remainder_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate an integer total proportionally with deterministic remainders."""
    population = sum(int(value) for value in counts.values())
    if total <= 0 or population < total:
        raise ValueError("Sample total must be positive and no larger than the catalog.")
    raw = {key: Fraction(int(value) * total, population) for key, value in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    missing = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:missing]:
        quotas[key] += 1
    return quotas


def scenario_quotas(catalog: pd.DataFrame, config: Mapping[str, object]) -> dict[str, dict[str, int]]:
    group_column = str(config["group_column"])
    total = int(config["sample_size"])
    perturbed = str(config["perturbed_group"])
    counts = {str(key): int(value) for key, value in catalog[group_column].value_counts().items()}
    if set(counts) != set(FAIRFACE_GROUPS) or perturbed not in counts:
        raise ValueError("Catalog must contain exactly the seven FairFace race categories.")
    if total % 42:
        raise ValueError("Sample size must be divisible by 42 for exact B/C/D quotas.")
    balanced = total // 7
    under = total // 14
    over = 2 * total // 7
    quotas = {
        "A": largest_remainder_quotas(counts, total),
        "B": {group: balanced for group in FAIRFACE_GROUPS},
        "C": {group: under if group == perturbed else (total - under) // 6 for group in FAIRFACE_GROUPS},
        "D": {group: over if group == perturbed else (total - over) // 6 for group in FAIRFACE_GROUPS},
    }
    for scenario, values in quotas.items():
        if sum(values.values()) != total or any(values[group] > counts[group] for group in FAIRFACE_GROUPS):
            raise ValueError(f"Scenario {scenario} cannot be sampled without replacement.")
    return quotas


def _rank_key(seed: int, group: str, relative_path: str) -> str:
    return hashlib.sha256(f"{seed}|{group}|{relative_path}".encode("utf-8")).hexdigest()


def build_scenarios(
    catalog: pd.DataFrame, config: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scenario selections and deterministic same-group reserves."""
    required = {"relative_path", str(config["group_column"])}
    if missing := required.difference(catalog.columns):
        raise ValueError(f"FairFace catalog is missing columns: {sorted(missing)}")
    if catalog["relative_path"].duplicated().any():
        raise ValueError("FairFace relative paths must be unique.")
    group_column = str(config["group_column"])
    seed = int(config["random_seed"])
    prepared = catalog.loc[:, ["relative_path", group_column]].copy()
    prepared[group_column] = prepared[group_column].astype(str)
    prepared["record_id"] = prepared["relative_path"].map(
        lambda value: stable_id(value, "fairface-demographic-composition-v1", prefix="ff")
    )
    prepared["_rank_key"] = [
        _rank_key(seed, group, path)
        for group, path in zip(prepared[group_column], prepared["relative_path"])
    ]
    prepared = prepared.sort_values([group_column, "_rank_key", "relative_path"]).reset_index(drop=True)
    prepared["selection_rank"] = prepared.groupby(group_column).cumcount()
    quotas = scenario_quotas(prepared, config)
    selections = []
    for scenario in ("A", "B", "C", "D"):
        mask = pd.Series(False, index=prepared.index)
        for group, quota in quotas[scenario].items():
            mask |= prepared[group_column].eq(group) & prepared["selection_rank"].lt(quota)
        frame = prepared.loc[mask, ["record_id", group_column, "relative_path", "selection_rank"]].copy()
        frame.insert(0, "scenario", scenario)
        selections.append(frame)
    max_quotas = {group: max(values[group] for values in quotas.values()) for group in FAIRFACE_GROUPS}
    reserve_mask = pd.Series(False, index=prepared.index)
    for group, quota in max_quotas.items():
        reserve_mask |= prepared[group_column].eq(group) & prepared["selection_rank"].ge(quota)
    reserves = prepared.loc[
        reserve_mask, ["record_id", group_column, "relative_path", "selection_rank"]
    ].reset_index(drop=True)
    return pd.concat(selections, ignore_index=True), reserves
