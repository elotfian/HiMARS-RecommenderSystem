from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data import make_rating_matrix


@dataclass(slots=True)
class ItemBasedCF:
    """Item-based collaborative filtering used for HiMARS candidate generation.

    Corrections relative to the notebooks:
    - supports adjusted cosine similarity, as stated in the manuscript;
    - uses ``sum(abs(similarity))`` in the weighted-sum denominator;
    - avoids global variables and exposes a reusable class API.
    """

    n_neighbors: int = 20
    similarity: str = "adjusted_cosine"
    positive_neighbors_only: bool = True
    fallback: str = "user_mean"  # "zero", "user_mean", or "global_mean"

    train_matrix: pd.DataFrame | None = None
    similarity_df: pd.DataFrame | None = None
    global_mean_: float = 0.0

    def fit(self, train_ratings_or_matrix: pd.DataFrame) -> "ItemBasedCF":
        if {"user_id", "item_id", "rating"}.issubset(train_ratings_or_matrix.columns):
            matrix = make_rating_matrix(train_ratings_or_matrix)
        else:
            matrix = train_ratings_or_matrix.copy()
        self.train_matrix = matrix
        self.global_mean_ = float(np.nanmean(matrix.to_numpy())) if matrix.size else 0.0
        if np.isnan(self.global_mean_):
            self.global_mean_ = 0.0
        self.similarity_df = self._compute_item_similarity(matrix)
        return self

    def _compute_item_similarity(self, matrix: pd.DataFrame) -> pd.DataFrame:
        method = self.similarity.lower()
        if method == "adjusted_cosine":
            user_means = matrix.mean(axis=1, skipna=True)
            centered = matrix.sub(user_means, axis=0).fillna(0.0)
            values = centered.T.to_numpy(dtype=float)
        elif method == "cosine":
            values = matrix.fillna(0.0).T.to_numpy(dtype=float)
        else:
            raise ValueError("similarity must be 'adjusted_cosine' or 'cosine'.")

        sim = cosine_similarity(values)
        sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(sim, 1.0)
        return pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)

    def _check_fitted(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_matrix is None or self.similarity_df is None:
            raise RuntimeError("ItemBasedCF.fit must be called before prediction.")
        return self.train_matrix, self.similarity_df

    def _fallback_score(self, user_id: Hashable) -> float:
        matrix, _ = self._check_fitted()
        if self.fallback == "zero" or user_id not in matrix.index:
            return 0.0
        if self.fallback == "global_mean":
            return self.global_mean_
        user_mean = matrix.loc[user_id].mean(skipna=True)
        return self.global_mean_ if pd.isna(user_mean) else float(user_mean)

    def predict_rating(self, user_id: Hashable, item_id: Hashable) -> float:
        matrix, sim = self._check_fitted()
        if user_id not in matrix.index or item_id not in matrix.columns:
            return self._fallback_score(user_id)

        user_row = matrix.loc[user_id]
        rated = user_row.dropna()
        if rated.empty:
            return self._fallback_score(user_id)

        similarities = sim.loc[item_id, rated.index].astype(float)
        if self.positive_neighbors_only:
            similarities = similarities[similarities > 0]
        if similarities.empty:
            return self._fallback_score(user_id)

        similarities = similarities.reindex(similarities.abs().sort_values(ascending=False).index)
        similarities = similarities.iloc[: self.n_neighbors]
        ratings = rated.loc[similarities.index].astype(float)
        denominator = float(similarities.abs().sum())
        if denominator <= 1e-12:
            return self._fallback_score(user_id)
        return float(np.dot(similarities.to_numpy(), ratings.to_numpy()) / denominator)

    def recommend_top_k(self, user_id: Hashable, k: int) -> pd.DataFrame:
        """Return top-k unrated candidate items for ``user_id``.

        The returned dataframe has columns ``item_id`` and ``predicted_rating``.
        """

        matrix, _ = self._check_fitted()
        if user_id not in matrix.index:
            raise KeyError(f"User {user_id!r} not present in the training matrix.")
        user_row = matrix.loc[user_id]
        unrated_items = user_row.index[user_row.isna()].tolist()
        scored = [(item, self.predict_rating(user_id, item)) for item in unrated_items]
        scored.sort(key=lambda z: z[1], reverse=True)
        return pd.DataFrame(scored[:k], columns=["item_id", "predicted_rating"])
