import pandas as pd
import pytest

from src.core.interfaces import RecommenderProvider, popularity_rank


class _StubProvider(RecommenderProvider):
    def fit(self, ratings: pd.DataFrame) -> None:
        self._fitted = True

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        return [(1, 0.9), (2, 0.8)][:k]


def test_cannot_instantiate_abstract_provider():
    with pytest.raises(TypeError):
        RecommenderProvider()


def test_stub_provider_implements_interface():
    provider = _StubProvider()
    provider.fit(pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5.0]}))
    result = provider.recommend(user_id=1, k=2)
    assert result == [(1, 0.9), (2, 0.8)]


def test_popularity_rank_orders_by_interaction_count():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "item_id": [10, 20, 10, 10],
            "rating": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = popularity_rank(ratings, exclude_items=None, k=2)
    assert result[0][0] == 10  # item 10 appears 3 times, most popular
    assert result[0][1] == pytest.approx(3.0)


def test_popularity_rank_excludes_seen_items():
    ratings = pd.DataFrame(
        {"user_id": [1, 2, 3], "item_id": [10, 10, 20], "rating": [1.0, 1.0, 1.0]}
    )
    result = popularity_rank(ratings, exclude_items={10}, k=5)
    assert all(item_id != 10 for item_id, _ in result)


def test_popularity_rank_truncates_to_k():
    ratings = pd.DataFrame(
        {"user_id": [1, 2, 3], "item_id": [10, 20, 30], "rating": [1.0, 1.0, 1.0]}
    )
    result = popularity_rank(ratings, exclude_items=None, k=2)
    assert len(result) == 2
