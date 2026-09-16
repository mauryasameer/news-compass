from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix

from src.core.interfaces import RecommenderProvider, popularity_rank


class ALSProvider(RecommenderProvider):
    """Alternating Least Squares matrix factorization via the `implicit` library."""

    def __init__(self, factors: int = 64, iterations: int = 15, alpha: float = 40.0) -> None:
        self._factors = factors
        self._iterations = iterations
        self._alpha = alpha
        self._ratings: pd.DataFrame | None = None
        self._sparse_user_item: csr_matrix | None = None
        self._model = None

    def fit(self, ratings: pd.DataFrame) -> None:
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ImportError(
                "ALSProvider requires the 'implicit' package. Install it with: "
                "pip install -r requirements-als.txt"
            ) from exc

        self._ratings = ratings
        n_users = int(ratings["user_id"].max()) + 1
        n_items = int(ratings["item_id"].max()) + 1
        sparse = csr_matrix(
            ([self._alpha] * len(ratings), (ratings["user_id"], ratings["item_id"])),
            shape=(n_users, n_items),
        )
        self._sparse_user_item = sparse

        model = AlternatingLeastSquares(factors=self._factors, iterations=self._iterations)
        model.fit(sparse)
        self._model = model

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        if self._ratings is None or self._sparse_user_item is None or self._model is None:
            raise RuntimeError("ALSProvider.recommend() called before fit()")

        seen_items = set(self._ratings.loc[self._ratings.user_id == user_id, "item_id"])

        if user_id >= self._sparse_user_item.shape[0] or not seen_items:
            return popularity_rank(self._ratings, exclude_items=seen_items if exclude_seen else None, k=k)

        # implicit's filter_already_liked_items=True does not shrink the result count —
        # it still returns N items total, padding filtered-out ones with a large-negative
        # sentinel score instead of dropping them. Over-fetch and post-filter in Python to
        # actually get k real results.
        n_items = self._sparse_user_item.shape[1]
        fetch_n = min(k + len(seen_items), n_items) if exclude_seen else k
        item_ids, scores = self._model.recommend(
            user_id, self._sparse_user_item[user_id], N=fetch_n, filter_already_liked_items=exclude_seen
        )
        results = [
            (int(item_id), float(score))
            for item_id, score in zip(item_ids, scores, strict=True)
            if not (exclude_seen and item_id in seen_items)
        ]
        return results[:k]
