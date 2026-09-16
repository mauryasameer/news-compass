from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from src.core.interfaces import RecommenderProvider, popularity_rank


class UserCFProvider(RecommenderProvider):
    """User-based collaborative filtering via cosine user-user similarity."""

    def __init__(self) -> None:
        self._ratings: pd.DataFrame | None = None
        self._matrix: np.ndarray | None = None  # dense user x item
        self._user_similarity: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame) -> None:
        self._ratings = ratings
        n_users = int(ratings["user_id"].max()) + 1
        n_items = int(ratings["item_id"].max()) + 1
        matrix = np.zeros((n_users, n_items))
        for row in ratings.itertuples():
            matrix[row.user_id, row.item_id] = row.rating
        self._matrix = matrix
        self._user_similarity = 1 - pairwise_distances(matrix, metric="cosine")

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        if self._ratings is None or self._matrix is None or self._user_similarity is None:
            raise RuntimeError("UserCFProvider.recommend() called before fit()")

        seen_items = set(self._ratings.loc[self._ratings.user_id == user_id, "item_id"])

        if user_id >= self._matrix.shape[0] or not seen_items:
            return popularity_rank(self._ratings, exclude_items=seen_items if exclude_seen else None, k=k)

        similarity_row = self._user_similarity[user_id]
        scores = similarity_row @ self._matrix
        ranked = np.argsort(scores)[::-1]

        results: list[tuple[int, float]] = []
        for item_id in ranked:
            if exclude_seen and item_id in seen_items:
                continue
            results.append((int(item_id), float(scores[item_id])))
            if len(results) == k:
                break
        return results
