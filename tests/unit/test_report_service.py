from src.services.report_service import build_report


def test_report_includes_narrative_and_run_info():
    recommendations = [{"item_id": 1, "title": "Test Article", "item_url": "http://x", "score": 0.9}]
    narratives = {1: "This is relevant because of X."}
    report = build_report(
        "Test Report", recommendations, narratives, strategy_name="item_cf", user_id=0
    )
    html = report.to_html()
    assert "Test Article" in html
    assert "This is relevant because of X." in html
    assert "Run Info" in html
    assert "item_cf" in html


def test_report_escapes_html_in_narrative():
    recommendations = [{"item_id": 1, "title": "A", "item_url": "http://x", "score": 0.9}]
    narratives = {1: "<script>alert(1)</script>"}
    report = build_report("Test Report", recommendations, narratives, strategy_name="item_cf", user_id=0)
    html = report.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
