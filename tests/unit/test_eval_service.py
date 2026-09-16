import pandas as pd

from src.core.interfaces import RecommenderProvider
from src.services.eval_service import run_full_evaluation


class _PerfectProvider(RecommenderProvider):
    """Always recommends exactly what the user already rated, for a deterministic score."""

    def __init__(self, ratings: pd.DataFrame) -> None:
        self._ratings = ratings

    def fit(self, ratings: pd.DataFrame) -> None:
        pass

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        user_items = self._ratings[self._ratings.user_id == user_id]
        return [(int(row.item_id), float(row.rating)) for row in user_items.itertuples()][:k]


def test_perfect_provider_scores_high_precision():
    ratings = pd.DataFrame({"user_id": [0, 1], "item_id": [0, 1], "rating": [5.0, 5.0]})
    provider = _PerfectProvider(ratings)
    result = run_full_evaluation(ratings, provider, k=1)
    assert result.recommender_metrics.precision_at_k == 1.0
    assert result.rmse == 0.0
