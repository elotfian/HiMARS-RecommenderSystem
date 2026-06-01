from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np
import pandas as pd

from .problem import Solution


def recommendation_metrics(
    items: Sequence[Hashable],
    user_id: Hashable,
    similarity_df: pd.DataFrame,
    test_ratings: pd.DataFrame | None,
    all_ratings: pd.DataFrame | None = None,
    rating_threshold: float = 3.0,
) -> dict[str, float]:
    """Compute accuracy, diversity and novelty for a final top-s list.

    Accuracy is precision@s against relevant test items. Diversity is average
    pairwise item similarity, so lower is better, matching the manuscript's
    evaluation metric. Novelty is average item popularity count; lower is better.
    """

    items = list(items)
    n = len(items)
    if n == 0:
        return {"accuracy": 0.0, "diversity": 0.0, "novelty": 0.0}

    if test_ratings is None or test_ratings.empty:
        relevant = set()
    else:
        user_test = test_ratings[
            (test_ratings["user_id"] == user_id) & (test_ratings["rating"] >= rating_threshold)
        ]
        relevant = set(user_test["item_id"].tolist())
    accuracy = len(set(items) & relevant) / n

    valid = [item for item in items if item in similarity_df.index]
    if len(valid) <= 1:
        diversity = 0.0
    else:
        sims = similarity_df.loc[valid, valid].to_numpy(dtype=float)
        mask = ~np.eye(len(valid), dtype=bool)
        diversity = float(np.nan_to_num(sims[mask], nan=0.0).mean())

    popularity_source = all_ratings if all_ratings is not None else test_ratings
    if popularity_source is None or popularity_source.empty:
        novelty = 0.0
    else:
        pop = popularity_source.groupby("item_id")["user_id"].nunique()
        novelty = float(np.mean([pop.get(item, 0) for item in items]))

    return {"accuracy": float(accuracy), "diversity": diversity, "novelty": novelty}


def pareto_front_metrics(solutions: Sequence[Solution]) -> dict[str, float]:
    """Compute SM, MID, DM and SNS for a Pareto set.

    The function is numerically guarded against one-point fronts and zero objective
    ranges, avoiding the warnings present in the notebooks.
    """

    if len(solutions) == 0:
        return {"SM": 0.0, "MID": 0.0, "DM": 0.0, "SNS": 0.0, "n_solutions": 0}
    values = np.asarray([s.objectives for s in solutions], dtype=float)
    values = np.unique(values, axis=0)
    n = values.shape[0]
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = np.where(np.abs(maxs - mins) <= 1e-12, 1.0, maxs - mins)
    normalized = (values - mins) / ranges

    # Mean consecutive distance after sorting by the first objective, matching the
    # final notebook implementation more closely than the textbook spacing variant.
    if n <= 1:
        sm = 0.0
    else:
        order = np.argsort(values[:, 0])
        sm = float(np.linalg.norm(np.diff(values[order], axis=0), axis=1).mean())

    ideal = np.ones(values.shape[1])
    mid = float(np.linalg.norm(normalized - ideal, axis=1).mean())
    dm = float(np.linalg.norm(maxs - mins))
    if n <= 1:
        sns = 0.0
    else:
        ci = np.linalg.norm(normalized, axis=1)
        sns = float(np.std(ci, ddof=1))
    return {"SM": sm, "MID": mid, "DM": dm, "SNS": sns, "n_solutions": int(n)}
