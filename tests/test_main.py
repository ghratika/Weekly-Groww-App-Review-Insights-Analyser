"""
Unit tests for src/agent/main.py — Phase 7 orchestrator.

Tests cover:
 - CLI --help renders correctly
 - Config load failure → exit(1)
 - Idempotency: status=success → exit(0) immediately
 - 0 reviews returned → abort with exit(1)
 - 0 clusters returned → abort with exit(1)
 - LLM summarisation failure → abort with exit(1)
 - Doc delivery failure → status=partial, exit(1)
 - Email delivery failure → status=partial, exit(1)
 - --dry-run happy path → status=success, no deliver_doc/deliver_email calls
 - Partial resume: doc already in run_log → skip doc, deliver email
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from src.agent.main import cli
from src.agent.summarizer import Cluster, ValidatedQuote


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "product": {
        "name": "Groww",
        "play_store_app_id": "com.nextbillion.groww",
        "review_window_weeks": 12,
    },
    "mcp_servers": {
        "playstore_reviews": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "src.mcp_servers.playstore_reviews.server"],
        },
        "google_docs": {"transport": "sse", "url": "http://test", "api_key": "secret"},
        "gmail": {"transport": "sse", "url": "http://test", "api_key": "secret"},
    },
    "delivery": {
        "google_doc_id": "doc123",
        "recipients": ["test@example.com"],
        "email_mode": "draft",
        "email_subject_template": "Groww Review Pulse — {iso_week}",
    },
    "llm": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "requests_per_minute": 30,
        "requests_per_day": 1000,
        "tokens_per_minute": 12000,
        "tokens_per_day": 100000,
    },
    "clustering": {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "umap_n_neighbors": 15,
        "umap_n_components": 5,
        "hdbscan_min_cluster_size": 5,
        "max_themes": 8,
    },
}

SAMPLE_REVIEWS = [
    MagicMock(
        review_id=f"r{i}",
        text=f"Review text number {i} — app is sometimes slow.",
        rating=3,
        date="2026-05-01",
        author=f"User_{i:04x}",
        app_version="4.8.0",
        thumbs_up=0,
        language="en",
    )
    for i in range(10)
]

SAMPLE_CLUSTER = Cluster(
    cluster_id=0,
    theme_name="App performance",
    summary="Users report slowness.",
    review_count=10,
    avg_rating=3.0,
    quotes=[ValidatedQuote(text="app is sometimes slow.", review_id="r0", rating=3)],
    action_ideas=["Fix perf"],
)


def _make_runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Helper: patch the full happy-path stack (analysis only, delivery mocked out)
# ---------------------------------------------------------------------------

def _patch_analysis(monkeypatch, tmp_path, *, dry_run=False):
    """
    Patch all heavy pipeline steps so the CLI can run without real MCP/ML.
    Returns a dict of mock objects keyed by step name.
    """
    mocks = {}

    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG) as m:
        mocks["load_config"] = m
    with patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS) as m:
        mocks["fetch_reviews"] = m
    with patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))) as m:
        mocks["embed_reviews"] = m
    with patch("src.agent.main.cluster_reviews", return_value=[MagicMock(cluster_id=0, review_indices=[0], avg_rating=3.0, review_count=10)]) as m:
        mocks["cluster_reviews"] = m
    with patch("src.agent.main.summarize_clusters", return_value=[SAMPLE_CLUSTER]) as m:
        mocks["summarize_clusters"] = m
    with patch("src.agent.main.validate_quotes", return_value=[SAMPLE_CLUSTER]) as m:
        mocks["validate_quotes"] = m
    with patch("src.agent.main.render_doc_section", return_value={"requests": [], "_raw_markdown": "## Groww — 2026-W23\n"}) as m:
        mocks["render_doc_section"] = m
    with patch("src.agent.main.render_email", return_value=MagicMock(subject="Groww Review Pulse — 2026-W23", html_body="<p>test</p>", text_body="test", to=["test@example.com"])) as m:
        mocks["render_email"] = m
    with patch("src.agent.main.deliver_doc", return_value="heading=Groww — 2026-W23") as m:
        mocks["deliver_doc"] = m
    with patch("src.agent.main.deliver_email", return_value="msg_id_001") as m:
        mocks["deliver_email"] = m

    return mocks


# ---------------------------------------------------------------------------
# 1. --help
# ---------------------------------------------------------------------------

def test_cli_help():
    runner = _make_runner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--week" in result.output
    assert "--product" in result.output


# ---------------------------------------------------------------------------
# 2. Invalid ISO week format
# ---------------------------------------------------------------------------

def test_invalid_iso_week():
    runner = _make_runner()
    result = runner.invoke(cli, ["--week", "2026-23"])
    assert result.exit_code != 0
    assert "Invalid ISO week format" in result.output


# ---------------------------------------------------------------------------
# 3. Config load failure → exit(1)
# ---------------------------------------------------------------------------

def test_config_load_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    runner = _make_runner()
    with patch("src.agent.main.load_config", side_effect=FileNotFoundError("config not found")):
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 4. Idempotency: status=success → exit(0)
# ---------------------------------------------------------------------------

def test_already_succeeded_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    success_log = {"product": "Groww", "iso_week": "2026-W23", "status": "success", "delivery": {}}
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=success_log):
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 5. 0 reviews returned → abort, exit(1), run_log status=failed
# ---------------------------------------------------------------------------

def test_zero_reviews_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=None), \
         patch("src.agent.main.fetch_reviews_via_mcp", side_effect=RuntimeError("No reviews returned")), \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 1
    written_log = mock_write.call_args[0][0]
    assert written_log["status"] == "failed"
    assert any("No reviews returned" in e for e in written_log["errors"])


# ---------------------------------------------------------------------------
# 6. 0 clusters produced → abort, exit(1), run_log status=failed
# ---------------------------------------------------------------------------

def test_zero_clusters_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=None), \
         patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS), \
         patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))), \
         patch("src.agent.main.cluster_reviews", side_effect=RuntimeError("No clusters found")), \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 1
    written_log = mock_write.call_args[0][0]
    assert written_log["status"] == "failed"
    assert any("No clusters found" in e for e in written_log["errors"])


# ---------------------------------------------------------------------------
# 7. LLM failure → abort, exit(1), run_log status=failed
# ---------------------------------------------------------------------------

def test_llm_failure_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    raw_cluster = MagicMock(cluster_id=0, review_indices=[0], avg_rating=3.0, review_count=10)
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=None), \
         patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS), \
         patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))), \
         patch("src.agent.main.cluster_reviews", return_value=[raw_cluster]), \
         patch("src.agent.main.summarize_clusters", side_effect=RuntimeError("Groq daily token limit reached")), \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 1
    written_log = mock_write.call_args[0][0]
    assert written_log["status"] == "failed"
    assert any("token limit" in e for e in written_log["errors"])


# ---------------------------------------------------------------------------
# 8. Doc delivery failure → status=partial, exit(1)
# ---------------------------------------------------------------------------

def test_doc_delivery_failure_sets_partial(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    raw_cluster = MagicMock(cluster_id=0, review_indices=[0], avg_rating=3.0, review_count=10)
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=None), \
         patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS), \
         patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))), \
         patch("src.agent.main.cluster_reviews", return_value=[raw_cluster]), \
         patch("src.agent.main.summarize_clusters", return_value=([SAMPLE_CLUSTER], 1500)), \
         patch("src.agent.main.validate_quotes", return_value=[SAMPLE_CLUSTER]), \
         patch("src.agent.main.render_doc_section", return_value={"requests": [], "_raw_markdown": ""}), \
         patch("src.agent.main.deliver_doc", side_effect=Exception("MCP server unreachable")), \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 1
    written_log = mock_write.call_args[0][0]
    assert written_log["status"] == "partial"
    assert any("Doc delivery failed" in e for e in written_log["errors"])


# ---------------------------------------------------------------------------
# 9. Email delivery failure → status=partial, exit(1), doc already saved
# ---------------------------------------------------------------------------

def test_email_delivery_failure_sets_partial(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    raw_cluster = MagicMock(cluster_id=0, review_indices=[0], avg_rating=3.0, review_count=10)
    email_mock = MagicMock(subject="S", html_body="<p/>", text_body="t", to=["x@x.com"])
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=None), \
         patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS), \
         patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))), \
         patch("src.agent.main.cluster_reviews", return_value=[raw_cluster]), \
         patch("src.agent.main.summarize_clusters", return_value=([SAMPLE_CLUSTER], 1500)), \
         patch("src.agent.main.validate_quotes", return_value=[SAMPLE_CLUSTER]), \
         patch("src.agent.main.render_doc_section", return_value={"requests": [], "_raw_markdown": ""}), \
         patch("src.agent.main.deliver_doc", return_value="heading=Groww — 2026-W23"), \
         patch("src.agent.main.render_email", return_value=email_mock), \
         patch("src.agent.main.deliver_email", side_effect=Exception("Gmail MCP down")), \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23"])
    assert result.exit_code == 1
    written_log = mock_write.call_args[0][0]
    assert written_log["status"] == "partial"
    assert any("Email delivery failed" in e for e in written_log["errors"])


# ---------------------------------------------------------------------------
# 10. --dry-run happy path → exit(0), status=success, no real MCP calls
# ---------------------------------------------------------------------------

def test_dry_run_skips_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    raw_cluster = MagicMock(cluster_id=0, review_indices=[0], avg_rating=3.0, review_count=10)
    email_mock = MagicMock(subject="S", html_body="<p/>", text_body="t", to=["x@x.com"])
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=None), \
         patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS), \
         patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))), \
         patch("src.agent.main.cluster_reviews", return_value=[raw_cluster]), \
         patch("src.agent.main.summarize_clusters", return_value=([SAMPLE_CLUSTER], 1500)), \
         patch("src.agent.main.validate_quotes", return_value=[SAMPLE_CLUSTER]), \
         patch("src.agent.main.render_doc_section", return_value={"requests": [], "_raw_markdown": ""}), \
         patch("src.agent.main.deliver_doc") as mock_doc, \
         patch("src.agent.main.render_email", return_value=email_mock), \
         patch("src.agent.main.deliver_email") as mock_email, \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23", "--dry-run"])

    assert result.exit_code == 0
    mock_doc.assert_not_called()
    mock_email.assert_not_called()
    # Final log should have status=success
    final_log = mock_write.call_args[0][0]
    assert final_log["status"] == "success"


# ---------------------------------------------------------------------------
# 11. Partial resume: doc already in run_log → skip deliver_doc, call deliver_email
# ---------------------------------------------------------------------------

def test_partial_resume_skips_doc_delivers_email(tmp_path, monkeypatch):
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    partial_log = {
        "product": "Groww",
        "iso_week": "2026-W23",
        "status": "partial",
        "delivery": {"doc_heading_id": "heading=Groww — 2026-W23"},
        "errors": [],
        "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile", "tokens_used": 0},
        "reviews_fetched": 0,
        "clusters_found": 0,
        "review_window": {},
    }
    raw_cluster = MagicMock(cluster_id=0, review_indices=[0], avg_rating=3.0, review_count=10)
    email_mock = MagicMock(subject="S", html_body="<p/>", text_body="t", to=["x@x.com"])
    runner = _make_runner()
    with patch("src.agent.main.load_config", return_value=MINIMAL_CONFIG), \
         patch("src.agent.main.check_run_log", return_value=partial_log), \
         patch("src.agent.main.fetch_reviews_via_mcp", return_value=SAMPLE_REVIEWS), \
         patch("src.agent.main.embed_reviews", return_value=np.zeros((10, 32))), \
         patch("src.agent.main.cluster_reviews", return_value=[raw_cluster]), \
         patch("src.agent.main.summarize_clusters", return_value=([SAMPLE_CLUSTER], 1500)), \
         patch("src.agent.main.validate_quotes", return_value=[SAMPLE_CLUSTER]), \
         patch("src.agent.main.render_doc_section", return_value={"requests": [], "_raw_markdown": ""}), \
         patch("src.agent.main.deliver_doc") as mock_doc, \
         patch("src.agent.main.render_email", return_value=email_mock), \
         patch("src.agent.main.deliver_email", return_value="msg_resume_001") as mock_email, \
         patch("src.agent.main.write_run_log") as mock_write:
        result = runner.invoke(cli, ["--week", "2026-W23"])

    assert result.exit_code == 0
    # Doc must NOT be re-delivered (already in run_log)
    mock_doc.assert_not_called()
    # Email MUST be delivered (not in run_log)
    mock_email.assert_called_once()
    final_log = mock_write.call_args[0][0]
    assert final_log["status"] == "success"
    assert final_log["delivery"]["gmail_message_id"] == "msg_resume_001"
