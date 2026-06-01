from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .problem import Solution


def select_by_ideal_point(solutions: Sequence[Solution]) -> Solution:
    """Select the solution closest to the scaled ideal point (1, 1)."""

    if not solutions:
        raise ValueError("Cannot select from an empty Pareto set.")
    values = np.asarray([s.objectives for s in solutions], dtype=float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = np.where(np.abs(maxs - mins) <= 1e-12, 1.0, maxs - mins)
    scaled = (values - mins) / ranges
    distances = np.linalg.norm(scaled - np.ones(values.shape[1]), axis=1)
    return solutions[int(np.argmin(distances))]


def topsis_rank(
    metric_table: pd.DataFrame,
    weights: Sequence[float] = (0.33, 0.17, 0.17, 0.33),
    benefit_criteria: Sequence[bool] = (False, False, True, True),
) -> pd.DataFrame:
    """Rank algorithms by TOPSIS using columns [SM, MID, DM, SNS].

    The weights match the manuscript order: SM and SNS receive 0.33, while MID
    and DM receive 0.17. Lower SM/MID are better; higher DM/SNS are better.
    """

    required = ["SM", "MID", "DM", "SNS"]
    missing = [c for c in required if c not in metric_table.columns]
    if missing:
        raise ValueError(f"metric_table is missing columns {missing}.")
    data = metric_table.loc[:, required].to_numpy(dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sqrt((data**2).sum(axis=0))
    denom = np.where(denom <= 1e-12, 1.0, denom)
    norm = data / denom
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != data.shape[1]:
        raise ValueError("weights must have length 4 for [SM, MID, DM, SNS].")
    weighted = norm * weights
    benefit = np.asarray(benefit_criteria, dtype=bool)
    ideal = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    negative = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - negative, axis=1)
    scores = np.divide(d_neg, d_pos + d_neg, out=np.zeros_like(d_neg), where=(d_pos + d_neg) > 1e-12)

    result = metric_table.copy()
    result["CLO"] = scores
    result["rank"] = pd.Series((-scores).argsort().argsort() + 1, index=result.index).astype(int)
    return result.sort_values(["rank", "CLO"], ascending=[True, False])
