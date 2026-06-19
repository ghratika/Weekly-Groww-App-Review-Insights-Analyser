"""
Unit tests for email_renderer.py
"""

from src.agent.email_renderer import render_email
from src.agent.summarizer import Cluster, ValidatedQuote

def test_render_email():
    metadata = {
        "iso_week": "2026-W23",
        "doc_heading_id": "heading=h.abc123"
    }
    
    config = {
        "product": {"name": "Groww"},
        "delivery": {
            "email_subject_template": "{product_name} Review Pulse — {iso_week}",
            "recipients": ["test@example.com"],
            "google_doc_id": "12345abcdef"
        }
    }
    
    clusters = [
        Cluster(
            cluster_id=1,
            theme_name="App performance & bugs",
            summary="",
            review_count=312,
            avg_rating=2.1,
        ),
        Cluster(
            cluster_id=2,
            theme_name="Customer support friction",
            summary="",
            review_count=198,
            avg_rating=1.8,
        )
    ]
    
    payload = render_email(clusters, metadata, config)
    
    assert payload.subject == "Groww Review Pulse — 2026-W23"
    assert payload.to == ["test@example.com"]
    
    # Check HTML
    assert "<b>510</b> Groww reviews" in payload.html_body
    assert "<b>App performance & bugs</b> (312 mentions)" in payload.html_body
    assert "<b>Customer support friction</b> (198 mentions)" in payload.html_body
    assert "https://docs.google.com/document/d/12345abcdef/edit#heading=h.abc123" in payload.html_body
    
    # Check Plain text
    assert "from 510 Groww reviews" in payload.text_body
    assert "• App performance & bugs (312 mentions)" in payload.text_body
    assert "https://docs.google.com/document/d/12345abcdef/edit#heading=h.abc123" in payload.text_body
