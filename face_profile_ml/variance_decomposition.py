"""Fixed-effect variance decomposition for replicated experiment summaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarianceComponent:
    factor: str
    sum_squares: float
    fraction_explained: float


@dataclass(frozen=True)
class VarianceDecompositionResult:
    outcome: str
    factors: tuple[str, ...]
    components: tuple[VarianceComponent, ...]
    residual_sum_squares: float
    n_observations: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "outcome": self.outcome,
                    "factor": item.factor,
                    "sum_squares": item.sum_squares,
                    "fraction_explained": item.fraction_explained,
                    "n_observations": self.n_observations,
                }
                for item in self.components
            ]
        )


def _design(frame: pd.DataFrame, factors: list[str]) -> tuple[np.ndarray, dict[str, list[int]]]:
    columns = [np.ones(len(frame), dtype=float)]
    positions: dict[str, list[int]] = {}
    for factor in factors:
        dummies = pd.get_dummies(frame[factor].astype(str), prefix=factor, drop_first=True, dtype=float)
        indices = []
        for name in dummies.columns:
            indices.append(len(columns))
            columns.append(dummies[name].to_numpy(dtype=float))
        positions[factor] = indices
    return np.column_stack(columns), positions


def _sse(design: np.ndarray, outcome: np.ndarray) -> float:
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residual = outcome - design @ coefficients
    return float(residual @ residual)


def decompose_variance(
    df: pd.DataFrame,
    outcome_col: str,
    factors: list[str] | tuple[str, ...],
) -> VarianceDecompositionResult:
    """Estimate each fixed factor's marginal contribution via reduced models."""
    factor_list = list(factors)
    required = [outcome_col, *factor_list]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing variance decomposition columns: {missing}")
    frame = df[required].dropna().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Variance decomposition requires at least two complete rows")
    y = frame[outcome_col].to_numpy(dtype=float)
    design, positions = _design(frame, factor_list)
    full_sse = _sse(design, y)
    sums: dict[str, float] = {}
    for factor in factor_list:
        retained = [index for index in range(design.shape[1]) if index not in positions[factor]]
        sums[factor] = max(0.0, _sse(design[:, retained], y) - full_sse)
    denominator = sum(sums.values()) + full_sse
    if denominator <= np.finfo(float).eps:
        fractions = {factor: 0.0 for factor in factor_list}
        residual_fraction = 1.0
    else:
        fractions = {factor: value / denominator for factor, value in sums.items()}
        residual_fraction = full_sse / denominator
    components = [
        VarianceComponent(factor, sums[factor], fractions[factor]) for factor in factor_list
    ]
    components.append(VarianceComponent("residual", full_sse, residual_fraction))
    return VarianceDecompositionResult(
        outcome_col, tuple(factor_list), tuple(components), full_sse, len(frame)
    )

