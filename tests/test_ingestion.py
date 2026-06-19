"""
Unit tests for the review ingestion module (Phase 3).

Tests use mocked MCP subprocess calls to avoid spawning real servers.
Covers:
  - Successful review fetch and Layer 2 PII scrubbing
  - 0-review response triggers RuntimeError abort
  - MCP server subprocess failure raises ConnectionError
  - MCP error response raises ValueError
  - Config parameters are passed correctly to the tool call
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agent.ingestion import fetch_reviews_via_mcp, _parse_mcp_response


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


# ---------------------------------------------------------------------------
# Test: _parse_mcp_response
# ---------------------------------------------------------------------------

class TestParseMcpResponse:
    def test_valid_result(self):
        raw = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"data": 42}})
        result = _parse_mcp_response(raw)
        assert result == {"data": 42}

    def test_json_rpc_error_raises(self):
        raw = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32601, "message": "Method not found"}
        })
        with pytest.raises(ValueError, match="Method not found"):
            _parse_mcp_response(raw)

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="Malformed JSON-RPC"):
            _parse_mcp_response("not json at all")


# ---------------------------------------------------------------------------
# Test: fetch_reviews_via_mcp
# ---------------------------------------------------------------------------

class TestFetchReviewsViaMcp:

    def _make_mcp_stdout(self, reviews: list[dict]) -> str:
        """Build the stdout that the MCP server subprocess would emit."""
        # Response to initialize (id=1)
        init_response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "playstore-reviews", "version": "1.0"},
        }})
        # Response to tools/call (id=2)
        tool_response = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {
            "content": [{"type": "text", "text": json.dumps(reviews)}]
        }})
        return init_response + "\n" + tool_response + "\n"

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_successful_fetch_returns_scrubbed_reviews(self, mock_mcp):
        """Happy path: reviews are fetched and returned after Layer 2 PII scrub."""
        reviews = [_make_review("r1"), _make_review("r2")]
        mock_mcp.return_value = reviews

        result = fetch_reviews_via_mcp(_make_config())

        assert len(result) == 2
        assert result[0]["review_id"] == "r1"
        assert result[1]["review_id"] == "r2"

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_zero_reviews_raises_runtime_error(self, mock_mcp):
        """0 reviews → abort the run with RuntimeError (architecture §10)."""
        mock_mcp.return_value = []

        with pytest.raises(RuntimeError, match="No reviews returned"):
            fetch_reviews_via_mcp(_make_config())

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_pii_in_text_is_scrubbed(self, mock_mcp):
        """Layer 2 PII scrubbing is applied: emails and phones are redacted."""
        reviews = [
            _make_review("r1", "Contact me at user@example.com for refund"),
            _make_review("r2", "Call 9876543210 for support"),
        ]
        mock_mcp.return_value = reviews

        result = fetch_reviews_via_mcp(_make_config())

        assert "user@example.com" not in result[0]["text"]
        assert "9876543210" not in result[1]["text"]
        assert "[REDACTED]" in result[0]["text"]
        assert "[REDACTED]" in result[1]["text"]

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_non_pii_text_preserved(self, mock_mcp):
        """Reviews without PII have their text preserved exactly."""
        clean_text = "The SIP calculator is accurate and easy to use."
        mock_mcp.return_value = [_make_review("r1", clean_text)]

        result = fetch_reviews_via_mcp(_make_config())

        assert result[0]["text"] == clean_text

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_connection_error_propagates(self, mock_mcp):
        """ConnectionError from the MCP subprocess propagates to the caller."""
        mock_mcp.side_effect = ConnectionError("Failed to start subprocess")

        with pytest.raises(ConnectionError, match="Failed to start subprocess"):
            fetch_reviews_via_mcp(_make_config())

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_mcp_value_error_propagates(self, mock_mcp):
        """ValueError from MCP error response propagates to the caller."""
        mock_mcp.side_effect = ValueError("MCP server returned error: App not found")

        with pytest.raises(ValueError, match="App not found"):
            fetch_reviews_via_mcp(_make_config())

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_config_params_passed_to_mcp(self, mock_mcp):
        """app_id, weeks, lang, and country are forwarded from config."""
        mock_mcp.return_value = [_make_review()]
        config = _make_config(app_id="com.zerodha.kite", weeks=8, lang="en", country="in")

        fetch_reviews_via_mcp(config)

        call_kwargs = mock_mcp.call_args
        assert call_kwargs.kwargs["tool_name"] == "fetch_reviews"
        args = call_kwargs.kwargs["arguments"]
        assert args["app_id"] == "com.zerodha.kite"
        assert args["weeks"] == 8
        assert args["lang"] == "en"
        assert args["country"] == "in"

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_unexpected_response_type_raises(self, mock_mcp):
        """If MCP returns a non-list, a ValueError is raised."""
        mock_mcp.return_value = {"error": "unexpected"}

        with pytest.raises(ValueError, match="Unexpected response type"):
            fetch_reviews_via_mcp(_make_config())

    @patch("src.agent.ingestion._call_mcp_tool_via_stdio")
    def test_author_fields_preserved(self, mock_mcp):
        """Layer 2 PII scrub must not touch the author field (already anonymized by L1)."""
        reviews = [_make_review("r1", "Good app, no issues at all")]
        mock_mcp.return_value = reviews

        result = fetch_reviews_via_mcp(_make_config())

        # Author was already anonymized by Layer 1; Layer 2 should not alter it
        assert result[0]["author"] == "User_abc123"
