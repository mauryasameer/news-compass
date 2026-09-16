from __future__ import annotations

import pandas as pd

from src.core.interfaces import RecommenderProvider


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    return {item_id: (score - lo) / (hi - lo) for item_id, score in scores.items()}


class HybridProvider(RecommenderProvider):
    """Blends any number of member providers by min-max-normalized weighted score."""

    def __init__(self, providers: dict[str, RecommenderProvider], weights: dict[str, float]) -> None:
        if set(providers) != set(weights):
            raise ValueError("providers and weights must have the same keys")
        self._providers = providers
        self._weights = weights
        self._fitted = False

    @property
    def members(self) -> dict[str, RecommenderProvider]:
        """Read-only access to member providers, so callers can forward setup calls
        (like ContentBasedProvider.set_articles()) that HybridProvider itself has no
        way to know about."""
        return self._providers

    def fit(self, ratings: pd.DataFrame) -> None:
        for provider in self._providers.values():
            provider.fit(ratings)
        self._fitted = True

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        if not self._fitted:
            raise RuntimeError("HybridProvider.recommend() called before fit()")

        # Ask each member for a generously-sized candidate list so the blend has enough
        # overlap to rank meaningfully, not just k items each.
        candidate_k = max(k * 5, 20)
        blended: dict[int, float] = {}
        for name, provider in self._providers.items():
            raw_scores = dict(provider.recommend(user_id, candidate_k, exclude_seen))
            normalized = _min_max_normalize(raw_scores)
            weight = self._weights[name]
            for item_id, score in normalized.items():
                blended[item_id] = blended.get(item_id, 0.0) + weight * score

        ranked = sorted(blended.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]
