"""
Unit tests for the review ingestion module.

Tests mock the underlying tool call to avoid real network requests.
Covers:
  - Successful review fetch and Layer 2 PII scrubbing
  - 0-review response triggers RuntimeError abort
  - Tool errors (ValueError, ConnectionError) propagate correctly
  - Config parameters are passed correctly to the tool call
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agent.ingestion import fetch_reviews_via_mcp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    app_id: str = "com.groww.v1",
    weeks: int = 12,
    lang: str = "en",
    country: str = "in",
) -> dict:
    """Build a minimal config dict for ingestion tests."""
    return {
        "product": {
            "name": "Groww",
            "play_store_app_id": app_id,
            "review_window_weeks": weeks,
        },
        "mcp_servers": {
            "playstore_reviews": {
                "lang": lang,
                "country": country,
                "server_module": "src.mcp_servers.playstore_reviews.server",
            }
        },
        "delivery": {
            "google_doc_id": "doc123",
            "recipients": ["pm@example.com"],
            "email_mode": "draft",
            "email_subject_template": "Pulse {iso_week}",
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens_per_run": 50000,
            "cost_limit_usd": 5.0,
        },
        "clustering": {
            "embedding_model": "all-MiniLM-L6-v2",
            "umap_n_neighbors": 15,
            "umap_n_components": 5,
            "hdbscan_min_cluster_size": 10,
            "max_themes": 7,
        },
    }


def _make_review(review_id: str = "r1", text: str = "The app works really well for trading") -> dict:
    return {
        "review_id": review_id,
        "author": "User_abc123",
        "rating": 4,
        "text": text,
        "date": "2026-05-01",
        "app_version": "4.8.1",
        "thumbs_up": 5,
        "language": "en",
    }


# Patch target: the tool function imported into ingestion.py
_PATCH_TARGET = "src.agent.ingestion._tool_fetch_reviews"


# ---------------------------------------------------------------------------
# Tests: fetch_reviews_via_mcp
# ---------------------------------------------------------------------------

class TestFetchReviewsViaMcp:

    @patch(_PATCH_TARGET)
    def test_successful_fetch_returns_scrubbed_reviews(self, mock_fetch):
        """Happy path: reviews are fetched and returned after Layer 2 PII scrub."""
        reviews = [_make_review("r1"), _make_review("r2")]
        mock_fetch.return_value = reviews

        result = fetch_reviews_via_mcp(_make_config())

        assert len(result) == 2
        assert result[0]["review_id"] == "r1"
        assert result[1]["review_id"] == "r2"

    @patch(_PATCH_TARGET)
    def test_zero_reviews_raises_runtime_error(self, mock_fetch):
        """0 reviews → abort the run with RuntimeError (architecture §10)."""
        mock_fetch.return_value = []

        with pytest.raises(RuntimeError, match="No reviews returned"):
            fetch_reviews_via_mcp(_make_config())

    @patch(_PATCH_TARGET)
    def test_pii_in_text_is_scrubbed(self, mock_fetch):
        """Layer 2 PII scrubbing is applied: emails and phones are redacted."""
        reviews = [
            _make_review("r1", "Contact me at user@example.com for refund"),
            _make_review("r2", "Call 9876543210 for support"),
        ]
        mock_fetch.return_value = reviews

        result = fetch_reviews_via_mcp(_make_config())

        assert "user@example.com" not in result[0]["text"]
        assert "9876543210" not in result[1]["text"]
        assert "[REDACTED]" in result[0]["text"]
        assert "[REDACTED]" in result[1]["text"]

    @patch(_PATCH_TARGET)
    def test_non_pii_text_preserved(self, mock_fetch):
        """Reviews without PII have their text preserved exactly."""
        clean_text = "The SIP calculator is accurate and easy to use."
        mock_fetch.return_value = [_make_review("r1", clean_text)]

        result = fetch_reviews_via_mcp(_make_config())

        assert result[0]["text"] == clean_text

    @patch(_PATCH_TARGET)
    def test_connection_error_propagates(self, mock_fetch):
        """ConnectionError from the tool propagates to the caller."""
        mock_fetch.side_effect = ConnectionError("Network unavailable after retries")

        with pytest.raises(ConnectionError, match="Network unavailable"):
            fetch_reviews_via_mcp(_make_config())

    @patch(_PATCH_TARGET)
    def test_value_error_propagates(self, mock_fetch):
        """ValueError from the tool (e.g., app not found) propagates to the caller."""
        mock_fetch.side_effect = ValueError("App ID 'com.fake.app' not found on Google Play")

        with pytest.raises(ValueError, match="not found on Google Play"):
            fetch_reviews_via_mcp(_make_config())

    @patch(_PATCH_TARGET)
    def test_config_params_passed_to_tool(self, mock_fetch):
        """app_id, weeks, lang, and country are forwarded from config."""
        mock_fetch.return_value = [_make_review()]
        config = _make_config(app_id="com.zerodha.kite", weeks=8, lang="en", country="in")

        fetch_reviews_via_mcp(config)

        mock_fetch.assert_called_once_with(
            app_id="com.zerodha.kite",
            weeks=8,
            lang="en",
            country="in",
        )

    @patch(_PATCH_TARGET)
    def test_unexpected_response_type_raises(self, mock_fetch):
        """If tool returns a non-list, a ValueError is raised."""
        mock_fetch.return_value = {"error": "unexpected"}

        with pytest.raises(ValueError, match="Unexpected response type"):
            fetch_reviews_via_mcp(_make_config())

    @patch(_PATCH_TARGET)
    def test_author_fields_preserved(self, mock_fetch):
        """Layer 2 PII scrub must not touch the author field (already anonymized by L1)."""
        reviews = [_make_review("r1", "Good app, no issues at all")]
        mock_fetch.return_value = reviews

        result = fetch_reviews_via_mcp(_make_config())

        # Author was already anonymized by Layer 1; Layer 2 should not alter it
        assert result[0]["author"] == "User_abc123"

    @patch(_PATCH_TARGET)
    def test_default_lang_and_country_from_config(self, mock_fetch):
        """If mcp_servers config is absent, defaults (en, in) are used."""
        mock_fetch.return_value = [_make_review()]
        config = _make_config()
        # Remove mcp_servers section to test defaults
        del config["mcp_servers"]

        fetch_reviews_via_mcp(config)

        mock_fetch.assert_called_once_with(
            app_id="com.groww.v1",
            weeks=12,
            lang="en",
            country="in",
        )
