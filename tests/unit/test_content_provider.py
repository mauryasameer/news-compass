import pandas as pd
import pytest

from src.providers.content_provider import ContentBasedProvider


@pytest.fixture
def articles():
    return pd.DataFrame(
        {
            "item_id": [0, 1, 2],
            "title": ["Ethereum blockchain guide", "Bitcoin blockchain primer", "Cooking pasta at home"],
            "text_description": [
                "ethereum blockchain smart contract crypto",
                "bitcoin blockchain crypto currency",
                "pasta recipe cooking italian food",
            ],
        }
    )


@pytest.fixture
def ratings():
    return pd.DataFrame(
        {"user_id": [0, 0], "item_id": [0, 2], "rating": [10.0, 1.0]}
    )


def test_recommend_similar_ranks_semantically_close_articles_first(articles):
    provider = ContentBasedProvider()
    provider.set_articles(articles)
    provider.fit(pd.DataFrame({"user_id": [], "item_id": [], "rating": []}))
    result = provider.recommend_similar("Ethereum blockchain guide", k=2)
    top_item_id = result[0][0]
    assert top_item_id == 1  # bitcoin article is the closest match to the ethereum seed


def test_recommend_similar_before_articles_set_raises():
    provider = ContentBasedProvider()
    with pytest.raises(RuntimeError):
        provider.recommend_similar("anything", k=1)


def test_recommend_seeds_from_users_highest_rated_item(articles, ratings):
    provider = ContentBasedProvider()
    provider.set_articles(articles)
    provider.fit(ratings)
    # user 0's highest-rated item is 0 (ethereum) -> should surface item 1 (bitcoin)
    result = provider.recommend(user_id=0, k=1)
    assert result[0][0] == 1


def test_recommend_cold_start_falls_back_to_popularity(articles):
    provider = ContentBasedProvider()
    provider.set_articles(articles)
    provider.fit(pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5.0]}))
    result = provider.recommend(user_id=99, k=1, exclude_seen=False)
    assert len(result) == 1
