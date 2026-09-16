from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "data"
KAGGLE_CREDENTIALS = Path.home() / ".kaggle" / "kaggle.json"
DATASET_SLUG = "gspmoreira/articles-sharing-reading-from-cit-deskdrop"

# The Kaggle dataset ships shared_articles.csv / users_interactions.csv, but
# data_service.load_interactions() (and src/app.py's CLI defaults) expect
# platform_content.csv / consumer_transanctions.csv with these renamed columns.
CONTENT_COLUMN_MAP = {
    "timestamp": "event_timestamp",
    "eventType": "interaction_type",
    "contentId": "item_id",
    "authorPersonId": "producer_id",
    "authorSessionId": "producer_session_id",
    "authorUserAgent": "producer_device_info",
    "authorRegion": "producer_location",
    "authorCountry": "producer_country",
    "contentType": "item_type",
    "url": "item_url",
    "title": "title",
    "text": "text_description",
    "lang": "language",
}

INTERACTIONS_COLUMN_MAP = {
    "timestamp": "event_timestamp",
    "eventType": "interaction_type",
    "contentId": "item_id",
    "personId": "consumer_id",
    "sessionId": "consumer_session_id",
    "userAgent": "consumer_device_info",
    "userRegion": "consumer_location",
    "userCountry": "country",
}

# users_interactions.csv's eventType values, remapped to match this project's
# interaction-type vocabulary (verified against the already-local consumer_transanctions.csv).
INTERACTION_TYPE_MAP = {
    "VIEW": "content_watched",
    "LIKE": "content_liked",
    "BOOKMARK": "content_saved",
    "COMMENT CREATED": "content_commented_on",
    "FOLLOW": "content_followed",
}


def _transform_downloaded_files(data_dir: Path) -> None:
    shared_articles_path = data_dir / "shared_articles.csv"
    users_interactions_path = data_dir / "users_interactions.csv"

    content = pd.read_csv(shared_articles_path, low_memory=False)
    content = content.rename(columns=CONTENT_COLUMN_MAP)
    content.to_csv(data_dir / "platform_content.csv", index=False)

    interactions = pd.read_csv(users_interactions_path, low_memory=False)
    interactions = interactions.rename(columns=INTERACTIONS_COLUMN_MAP)
    interactions["interaction_type"] = interactions["interaction_type"].replace(INTERACTION_TYPE_MAP)
    interactions.to_csv(data_dir / "consumer_transanctions.csv", index=False)

    shared_articles_path.unlink()
    users_interactions_path.unlink()


def fetch_deskdrop_data() -> Path:
    if not KAGGLE_CREDENTIALS.exists():
        print(
            f"error: Kaggle API credentials not found at {KAGGLE_CREDENTIALS}.\n"
            "Set up your Kaggle API token: https://www.kaggle.com/docs/api#authentication",
            file=sys.stderr,
        )
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    _transform_downloaded_files(DATA_DIR)
    return DATA_DIR


if __name__ == "__main__":
    fetch_deskdrop_data()
