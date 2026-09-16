import pandas as pd
import pytest

from src.services.data_service import load_interactions


@pytest.fixture
def consumer_csv(tmp_path):
    path = tmp_path / "consumer.csv"
    pd.DataFrame(
        {
            "event_timestamp": [1, 2, 3, 4, 5],
            "interaction_type": [
                "content_watched",
                "content_watched",
                "content_liked",
                "content_watched",
                "content_saved",
            ],
            "item_id": [100, 100, 100, 200, 200],
            "consumer_id": [1, 2, 1, 1, 2],
            "consumer_session_id": [1, 1, 1, 1, 1],
            "consumer_device_info": ["", "", "", "", ""],
            "consumer_location": ["", "", "", "", ""],
            "country": ["", "", "", "", ""],
        }
    ).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def content_csv(tmp_path):
    path = tmp_path / "content.csv"
    pd.DataFrame(
        {
            "event_timestamp": [1, 2],
            "interaction_type": ["content_present", "content_present"],
            "item_id": [100, 200],
            "producer_id": [1, 1],
            "producer_session_id": [1, 1],
            "producer_device_info": ["", ""],
            "producer_location": ["", ""],
            "producer_country": ["", ""],
            "item_type": ["HTML", "HTML"],
            "item_url": ["http://a", "http://b"],
            "title": ["Article A", "Article B"],
            "text_description": ["about a", "about b"],
            "language": ["en", "en"],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_ratings_have_expected_columns(consumer_csv, content_csv):
    data = load_interactions(consumer_csv, content_csv)
    assert set(data.ratings.columns) == {"user_id", "item_id", "rating"}


def test_rarer_interaction_type_scores_higher(consumer_csv, content_csv):
    data = load_interactions(consumer_csv, content_csv)
    # content_liked/content_saved (1/5 = 20% share each) are rarer than content_watched
    # (3/5 = 60% share), so their rating should be higher. Use user 2's item-100
    # interaction as the "pure watched" baseline: user 1's item-100 interaction is
    # deliberately duplicated (watched + liked) to exercise the max-collapse dedup path
    # elsewhere, so after that collapse it already carries the higher "liked" rating and
    # can't be used as a clean watched-only signal.
    watched_rating = data.ratings.loc[
        (data.ratings.user_id == data.user_id_map[2]) & (data.ratings.item_id == data.item_id_map[100]),
        "rating",
    ]
    liked_rows = data.ratings[data.ratings.rating > watched_rating.max()]
    assert len(liked_rows) > 0


def test_deduplicates_repeated_interactions(consumer_csv, content_csv):
    data = load_interactions(consumer_csv, content_csv)
    # user 1 interacted with item 100 twice (watched + liked) -> one row per (user, item)
    pair_counts = data.ratings.groupby(["user_id", "item_id"]).size()
    assert (pair_counts == 1).all()


def test_articles_have_titles(consumer_csv, content_csv):
    data = load_interactions(consumer_csv, content_csv)
    assert set(data.articles["title"]) == {"Article A", "Article B"}


def test_id_maps_are_dense_zero_indexed(consumer_csv, content_csv):
    data = load_interactions(consumer_csv, content_csv)
    assert set(data.user_id_map.values()) == {0, 1}
    assert set(data.item_id_map.values()) == {0, 1}
