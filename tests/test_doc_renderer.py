"""
Unit tests for doc_renderer.py
"""

from src.agent.doc_renderer import render_doc_section
from src.agent.summarizer import Cluster, ValidatedQuote

def test_render_doc_section_heading():
    metadata = {
        "product_name": "Groww",
        "iso_week": "2026-W23",
        "review_count": 847,
        "review_window_weeks": 12,
        "review_window": {"start": "2026-03-30", "end": "2026-06-01"}
    }
    
    clusters = [
        Cluster(
            cluster_id=1,
            theme_name="App performance & bugs",
            summary="Lag, crashes during trading hours; login/session timeouts.",
            review_count=312,
            avg_rating=2.1,
            quotes=[ValidatedQuote(text="The app freezes exactly when the market opens, very frustrating.", review_id="1", rating=1)],
            action_ideas=["Stabilize peak-time performance"]
        )
    ]
    
    payload = render_doc_section(clusters, metadata)
    
    assert "requests" in payload
    assert "_raw_markdown" in payload
    
    raw_md = payload["_raw_markdown"]
    assert "Groww — 2026-W23" in raw_md
    assert "2026-03-30 → 2026-06-01" in raw_md
    assert "847" in raw_md
    assert "App performance & bugs" in raw_md
    assert "The app freezes exactly when the market opens, very frustrating." in raw_md
    assert "Stabilize peak-time performance" in raw_md

def test_render_doc_section_empty_clusters():
    metadata = {
        "product_name": "Groww",
        "iso_week": "2026-W23",
    }
    
    payload = render_doc_section([], metadata)
    
    raw_md = payload["_raw_markdown"]
    assert "Groww — 2026-W23" in raw_md
    assert "Top Themes" in raw_md
