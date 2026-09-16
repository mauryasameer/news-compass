from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class RecommenderProvider(ABC):
    """Abstract interface for all news-recommendation backends.

    Concrete implementations live in src/providers/. `ratings` is always a
    long-format DataFrame with columns user_id, item_id, rating. Swap
    providers by changing one constructor argument — no service code changes.
    """

    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> None:
        """Fit the model on the full long-format ratings DataFrame."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        """Return up to k (item_id, score) pairs, sorted descending by score."""
        ...


def popularity_rank(
    ratings: pd.DataFrame,
    exclude_items: set[int] | None,
    k: int,
) -> list[tuple[int, float]]:
    """Most-interacted-with items overall, used as the cold-start fallback.

    Score is the raw interaction count for that item.
    """
    counts = ratings["item_id"].value_counts()
    if exclude_items:
        counts = counts[~counts.index.isin(exclude_items)]
    top = counts.head(k)
    return [(int(item_id), float(count)) for item_id, count in top.items()]
