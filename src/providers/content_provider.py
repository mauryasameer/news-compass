from __future__ import annotations

import pandas as pd
from gensim import corpora, similarities
from gensim.models import TfidfModel

from src.core.interfaces import RecommenderProvider, popularity_rank


class ContentBasedProvider(RecommenderProvider):
    """Content similarity over article text_description via gensim TF-IDF."""

    def __init__(self) -> None:
        self._articles: pd.DataFrame | None = None
        self._ratings: pd.DataFrame | None = None
        self._dictionary: corpora.Dictionary | None = None
        self._tfidf: TfidfModel | None = None
        self._index: similarities.MatrixSimilarity | None = None
        self._title_to_item_id: dict[str, int] = {}

    def set_articles(self, articles: pd.DataFrame) -> None:
        self._articles = articles.reset_index(drop=True)
        tokenized = [text.lower().split() for text in self._articles["text_description"]]
        self._dictionary = corpora.Dictionary(tokenized)
        corpus = [self._dictionary.doc2bow(tokens) for tokens in tokenized]
        self._tfidf = TfidfModel(corpus)
        self._index = similarities.MatrixSimilarity(self._tfidf[corpus])
        self._title_to_item_id = dict(zip(self._articles["title"], self._articles["item_id"], strict=True))

    def fit(self, ratings: pd.DataFrame) -> None:
        self._ratings = ratings

    def recommend_similar(self, title: str, k: int) -> list[tuple[int, float]]:
        if self._articles is None or self._dictionary is None or self._tfidf is None or self._index is None:
            raise RuntimeError("ContentBasedProvider.recommend_similar() called before set_articles()")

        seed_row = self._articles[self._articles["title"] == title]
        if seed_row.empty:
            return []
        seed_text = seed_row.iloc[0]["text_description"].lower().split()
        seed_vec = self._tfidf[self._dictionary.doc2bow(seed_text)]
        similarities_scores = self._index[seed_vec]

        ranked = sorted(enumerate(similarities_scores), key=lambda x: x[1], reverse=True)
        seed_item_id = int(seed_row.iloc[0]["item_id"])
        results: list[tuple[int, float]] = []
        for position, score in ranked:
            item_id = int(self._articles.iloc[position]["item_id"])
            if item_id == seed_item_id:
                continue
            results.append((item_id, float(score)))
            if len(results) == k:
                break
        return results

    def recommend(self, user_id: int, k: int, exclude_seen: bool = True) -> list[tuple[int, float]]:
        if self._ratings is None or self._articles is None:
            raise RuntimeError("ContentBasedProvider.recommend() called before fit()/set_articles()")

        user_ratings = self._ratings[self._ratings.user_id == user_id]
        if user_ratings.empty:
            # no interaction history at all -> nothing to exclude either way
            return popularity_rank(self._ratings, exclude_items=None, k=k)

        seed_item_id = int(user_ratings.sort_values("rating", ascending=False).iloc[0]["item_id"])
        seed_row = self._articles[self._articles["item_id"] == seed_item_id]
        if seed_row.empty:
            seen_items = set(user_ratings["item_id"])
            return popularity_rank(self._ratings, exclude_items=seen_items if exclude_seen else None, k=k)

        seed_title = seed_row.iloc[0]["title"]
        results = self.recommend_similar(seed_title, k=k + len(user_ratings))

        seen_items = set(user_ratings["item_id"]) if exclude_seen else set()
        filtered = [(item_id, score) for item_id, score in results if item_id not in seen_items]
        return filtered[:k]
