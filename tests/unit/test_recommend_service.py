import pandas as pd
import pytest

from src.core.interfaces import RecommenderProvider
from src.providers.hybrid_provider import HybridProvider
from src.services.data_service import ProcessedData
from src.services.recommend_service import build_provider, recommend_for_user, set_articles_if_needed


def test_build_provider_user_cf():
    provider = build_provider("user_cf", hybrid_spec=None)
    assert provider.__class__.__name__ == "UserCFProvider"


def test_build_provider_item_cf():
    provider = build_provider("item_cf", hybrid_spec=None)
    assert provider.__class__.__name__ == "ItemCFProvider"


def test_build_provider_hybrid_parses_spec():
    provider = build_provider("hybrid", hybrid_spec="item_cf:0.5,user_cf:0.5")
    assert provider.__class__.__name__ == "HybridProvider"


def test_build_provider_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_provider("not_a_real_strategy", hybrid_spec=None)


def test_recommend_for_user_merges_titles():
    # user 0 has only rated item 0; user 1 has rated item 1, so item 1 is a valid,
    # unseen-by-user-0 candidate that ItemCFProvider can actually recommend.
    data = ProcessedData(
        ratings=pd.DataFrame({"user_id": [0, 1], "item_id": [0, 1], "rating": [5.0, 5.0]}),
        articles=pd.DataFrame({"item_id": [1], "title": ["Test Article"], "text_description": ["x"], "item_url": ["http://x"]}),
        user_id_map={100: 0, 400: 1},
        item_id_map={200: 0, 300: 1},
    )
    provider = build_provider("item_cf", hybrid_spec=None)
    provider.fit(data.ratings)
    result = recommend_for_user(data, provider, user_id=0, k=1)
    assert result[0]["title"] == "Test Article"
    assert "score" in result[0]


class _NeedsArticlesProvider(RecommenderProvider):
    """Stands in for ContentBasedProvider: only usable after set_articles() is called."""

    def __init__(self) -> None:
        self._articles = None

    def set_articles(self, articles) -> None:
        self._articles = articles

    def fit(self, ratings: pd.DataFrame) -> None:
        pass

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        if self._articles is None:
            raise RuntimeError("set_articles() was never called")
        return [(0, 1.0)][:k]


def test_set_articles_if_needed_reaches_content_style_provider_directly():
    provider = _NeedsArticlesProvider()
    set_articles_if_needed(provider, articles="stub-articles")
    assert provider.recommend(user_id=0, k=1) == [(0, 1.0)]  # doesn't raise


def test_set_articles_if_needed_reaches_content_style_provider_nested_in_hybrid():
    content_like = _NeedsArticlesProvider()
    other = build_provider("item_cf", hybrid_spec=None)
    hybrid = HybridProvider(providers={"content": content_like, "item_cf": other}, weights={"content": 0.5, "item_cf": 0.5})

    set_articles_if_needed(hybrid, articles="stub-articles")

    # before the fix, this member never got set_articles() called and would raise here
    assert content_like.recommend(user_id=0, k=1) == [(0, 1.0)]
