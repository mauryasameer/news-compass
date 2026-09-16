import pandas as pd
import pytest

from src.providers.item_cf_provider import ItemCFProvider


@pytest.fixture
def ratings():
    # 3 users, 3 items. User 0 and 1 both like item 0 and 1 strongly (similar items);
    # item 2 is disjoint (liked only by user 2).
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [0, 1, 0, 1, 2],
            "rating": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )


def test_recommend_returns_k_items(ratings):
    provider = ItemCFProvider()
    provider.fit(ratings)
    result = provider.recommend(user_id=0, k=1, exclude_seen=False)
    assert len(result) == 1


def test_recommend_before_fit_raises():
    provider = ItemCFProvider()
    with pytest.raises(RuntimeError):
        provider.recommend(user_id=0, k=1)


def test_excludes_seen_items_by_default(ratings):
    provider = ItemCFProvider()
    provider.fit(ratings)
    result = provider.recommend(user_id=0, k=3)
    recommended_items = {item_id for item_id, _ in result}
    assert 0 not in recommended_items
    assert 1 not in recommended_items


def test_cold_start_user_falls_back_to_popularity(ratings):
    provider = ItemCFProvider()
    provider.fit(ratings)
    result = provider.recommend(user_id=99, k=1, exclude_seen=False)
    assert len(result) == 1  # popularity fallback, not an empty list or a crash
