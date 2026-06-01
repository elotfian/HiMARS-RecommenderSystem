from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ObjectiveEvaluator:
    """Objective and recommendation-quality metrics for one target user."""

    user_id: Hashable
    train_matrix: pd.DataFrame
    similarity_df: pd.DataFrame
    test_ratings: pd.DataFrame | None = None
    all_ratings: pd.DataFrame | None = None
    rating_threshold: float = 3.0

    def _valid_items(self, items: Sequence[Hashable]) -> list[Hashable]:
        return [item for item in items if item in self.similarity_df.index]

    def objectives(self, items: Sequence[Hashable]) -> tuple[float, float]:
        """Return paper objectives ``(f1, f2)`` to be maximized.

        ``f1`` is the average similarity between recommended items and items rated
        by the target user in the training set. ``f2`` is average pairwise
        dissimilarity within the recommendation list.
        """

        valid_items = self._valid_items(items)
        if not valid_items:
            return 0.0, 0.0
        f1 = self.accuracy_objective(valid_items)
        f2 = self.diversity_objective(valid_items)
        return float(f1), float(f2)

    def accuracy_objective(self, items: Sequence[Hashable]) -> float:
        if self.user_id not in self.train_matrix.index:
            return 0.0
        rated_items = self.train_matrix.loc[self.user_id].dropna().index
        rated_items = [item for item in rated_items if item in self.similarity_df.columns]
        if len(rated_items) == 0:
            return 0.0
        sims = self.similarity_df.loc[list(items), rated_items].to_numpy(dtype=float)
        return float(np.nan_to_num(sims, nan=0.0).sum() / len(items))

    def diversity_objective(self, items: Sequence[Hashable]) -> float:
        n = len(items)
        if n <= 1:
            return 0.0
        sims = self.similarity_df.loc[list(items), list(items)].to_numpy(dtype=float)
        mask = ~np.eye(n, dtype=bool)
        dissimilarities = 1.0 - np.nan_to_num(sims[mask], nan=0.0)
        return float(dissimilarities.mean())

    def recommendation_quality(self, items: Sequence[Hashable]) -> dict[str, float]:
        from .metrics import recommendation_metrics

        return recommendation_metrics(
            items=items,
            user_id=self.user_id,
            similarity_df=self.similarity_df,
            test_ratings=self.test_ratings,
            all_ratings=self.all_ratings,
            rating_threshold=self.rating_threshold,
        )
