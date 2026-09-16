from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ProcessedData:
    ratings: pd.DataFrame  # columns: user_id, item_id, rating (dense integer ids)
    articles: pd.DataFrame  # columns: item_id (dense), title, text_description, item_url
    user_id_map: dict[int, int]  # original consumer_id -> dense user_id
    item_id_map: dict[int, int]  # original item_id -> dense item_id


def _impute_ratings(interactions: pd.DataFrame) -> pd.DataFrame:
    """Rarer interaction types score higher (100 / pct-share-of-that-type)."""
    pct_share = interactions["interaction_type"].value_counts() * 100 / len(interactions)
    rating_by_type = 100 / pct_share
    interactions = interactions.copy()
    interactions["rating"] = interactions["interaction_type"].map(rating_by_type)
    return interactions


def load_interactions(consumer_csv: str, content_csv: str) -> ProcessedData:
    interactions = pd.read_csv(consumer_csv, low_memory=False)
    interactions = interactions[["consumer_id", "item_id", "interaction_type"]]
    interactions = _impute_ratings(interactions)

    # Collapse repeated interactions on the same (user, item) to their max rating.
    grouped = (
        interactions.groupby(["consumer_id", "item_id"], as_index=False)["rating"].max()
    )

    user_ids = sorted(grouped["consumer_id"].unique())
    item_ids = sorted(grouped["item_id"].unique())
    user_id_map = {original: dense for dense, original in enumerate(user_ids)}
    item_id_map = {original: dense for dense, original in enumerate(item_ids)}

    ratings = pd.DataFrame(
        {
            "user_id": grouped["consumer_id"].map(user_id_map),
            "item_id": grouped["item_id"].map(item_id_map),
            "rating": grouped["rating"],
        }
    )

    content = pd.read_csv(content_csv, low_memory=False)
    content = content[content["item_id"].isin(item_id_map)]
    content = content.drop_duplicates(subset="item_id")
    articles = pd.DataFrame(
        {
            "item_id": content["item_id"].map(item_id_map),
            "title": content["title"],
            "text_description": content["text_description"],
            "item_url": content["item_url"],
        }
    ).dropna(subset=["item_id"])
    articles["item_id"] = articles["item_id"].astype(int)

    return ProcessedData(
        ratings=ratings, articles=articles, user_id_map=user_id_map, item_id_map=item_id_map
    )
