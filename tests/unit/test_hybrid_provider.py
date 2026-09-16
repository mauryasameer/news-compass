import pandas as pd
import pytest

from src.core.interfaces import RecommenderProvider
from src.providers.hybrid_provider import HybridProvider


class _FixedProvider(RecommenderProvider):
    """Returns a hardcoded ranking, for deterministic blend testing."""

    def __init__(self, ranking: list[tuple[int, float]]) -> None:
        self._ranking = ranking

    def fit(self, ratings: pd.DataFrame) -> None:
        pass

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        return self._ranking[:k]


def test_blends_normalized_scores_by_weight():
    provider_a = _FixedProvider([(1, 10.0), (2, 0.0)])
    provider_b = _FixedProvider([(2, 10.0), (1, 0.0)])
    hybrid = HybridProvider(
        providers={"a": provider_a, "b": provider_b}, weights={"a": 0.5, "b": 0.5}
    )
    hybrid.fit(pd.DataFrame({"user_id": [], "item_id": [], "rating": []}))
    result = hybrid.recommend(user_id=0, k=2)
    scores = dict(result)
    # both items get a 0.5/0.5 blend of a max-normalized (1.0) and min-normalized (0.0)
    # score from each provider -> tied at 0.5 each
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.5)


def test_single_member_weight_one_matches_that_members_ranking():
    provider_a = _FixedProvider([(1, 10.0), (2, 5.0)])
    provider_b = _FixedProvider([(3, 100.0)])
    hybrid = HybridProvider(
        providers={"a": provider_a, "b": provider_b}, weights={"a": 1.0, "b": 0.0}
    )
    hybrid.fit(pd.DataFrame({"user_id": [], "item_id": [], "rating": []}))
    result = hybrid.recommend(user_id=0, k=2)
    top_item = result[0][0]
    assert top_item == 1


def test_recommend_before_fit_raises():
    provider_a = _FixedProvider([(1, 10.0)])
    hybrid = HybridProvider(providers={"a": provider_a}, weights={"a": 1.0})
    with pytest.raises(RuntimeError):
        hybrid.recommend(user_id=0, k=1)


def test_members_exposes_the_providers_dict():
    provider_a = _FixedProvider([(1, 10.0)])
    provider_b = _FixedProvider([(2, 10.0)])
    hybrid = HybridProvider(providers={"a": provider_a, "b": provider_b}, weights={"a": 0.5, "b": 0.5})
    assert hybrid.members == {"a": provider_a, "b": provider_b}
