import pandas as pd
import pytest

pytest.importorskip("implicit")

from src.providers.als_provider import ALSProvider  # noqa: E402


@pytest.fixture
def ratings():
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [0, 1, 0, 1, 2],
            "rating": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )


def test_recommend_returns_k_items(ratings):
    provider = ALSProvider(factors=4, iterations=2)
    provider.fit(ratings)
    result = provider.recommend(user_id=0, k=1, exclude_seen=False)
    assert len(result) == 1


def test_recommend_before_fit_raises():
    provider = ALSProvider()
    with pytest.raises(RuntimeError):
        provider.recommend(user_id=0, k=1)


def test_excludes_seen_items_by_default(ratings):
    provider = ALSProvider(factors=4, iterations=2)
    provider.fit(ratings)
    result = provider.recommend(user_id=0, k=3)
    recommended_items = {item_id for item_id, _ in result}
    assert 0 not in recommended_items
    assert 1 not in recommended_items
