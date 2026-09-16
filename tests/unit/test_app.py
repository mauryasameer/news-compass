import pandas as pd

from src.app import main


def test_top_k_zero_rejected(capsys):
    exit_code = main(["--strategy", "item_cf", "--top-k", "0"])
    assert exit_code == 1
    assert "--top-k must be positive, got 0" in capsys.readouterr().err


def test_top_k_negative_rejected(capsys):
    exit_code = main(["--strategy", "item_cf", "--top-k", "-5"])
    assert exit_code == 1
    assert "--top-k must be positive, got -5" in capsys.readouterr().err


def _make_synthetic_csvs(tmp_path):
    consumer_path = tmp_path / "consumer.csv"
    pd.DataFrame(
        {
            "event_timestamp": range(20),
            "interaction_type": (["content_watched"] * 15 + ["content_liked"] * 5),
            "item_id": [i % 5 for i in range(20)],
            "consumer_id": [i % 4 for i in range(20)],
            "consumer_session_id": [1] * 20,
            "consumer_device_info": [""] * 20,
            "consumer_location": [""] * 20,
            "country": [""] * 20,
        }
    ).to_csv(consumer_path, index=False)

    content_path = tmp_path / "content.csv"
    pd.DataFrame(
        {
            "event_timestamp": range(5),
            "interaction_type": ["content_present"] * 5,
            "item_id": range(5),
            "producer_id": [1] * 5,
            "producer_session_id": [1] * 5,
            "producer_device_info": [""] * 5,
            "producer_location": [""] * 5,
            "producer_country": [""] * 5,
            "item_type": ["HTML"] * 5,
            "item_url": [f"http://x/{i}" for i in range(5)],
            "title": [f"Article {i}" for i in range(5)],
            "text_description": [f"description of article {i} topic" for i in range(5)],
            "language": ["en"] * 5,
        }
    ).to_csv(content_path, index=False)

    return str(consumer_path), str(content_path)


def test_malformed_hybrid_spec_rejected_cleanly(tmp_path, capsys):
    consumer_csv, content_csv = _make_synthetic_csvs(tmp_path)
    exit_code = main(
        [
            "--consumer-csv", consumer_csv,
            "--content-csv", content_csv,
            "--strategy", "hybrid",
            "--hybrid", "item_cf:abc",
        ]
    )
    assert exit_code == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_missing_hybrid_spec_rejected_cleanly(tmp_path, capsys):
    consumer_csv, content_csv = _make_synthetic_csvs(tmp_path)
    exit_code = main(
        [
            "--consumer-csv", consumer_csv,
            "--content-csv", content_csv,
            "--strategy", "hybrid",
        ]
    )
    assert exit_code == 1
    assert "--hybrid spec is required" in capsys.readouterr().err
