"""
Unit tests for delivery.py
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from src.agent.delivery import deliver_doc, deliver_email, _http1_sse_client_factory
from src.agent.email_renderer import EmailPayload

@pytest.fixture
def mock_mcp_session():
    with patch("src.agent.delivery.sse_client") as mock_sse:
        mock_ctx = AsyncMock()
        mock_sse.return_value = mock_ctx
        mock_ctx.__aenter__.return_value = (MagicMock(), MagicMock())
        
        with patch("src.agent.delivery.ClientSession") as mock_session_cls:
            mock_session_ctx = AsyncMock()
            mock_session_cls.return_value = mock_session_ctx
            
            mock_session = AsyncMock()
            mock_session_ctx.__aenter__.return_value = mock_session
            yield mock_session


# ---------------------------------------------------------------------------
# Factory tests — verify HTTP/1.1 enforcement and header forwarding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http1_factory_is_context_manager_and_forces_http11():
    """Factory must work as async ctx-manager and produce an HTTP/1.1 client."""
    async with _http1_sse_client_factory() as client:
        assert isinstance(client, httpx.AsyncClient)
        # http2 must be disabled on the underlying connection pool
        assert not client._transport._pool._http2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_http1_factory_forwards_auth_headers():
    """Authorization header passed to factory must appear on the client."""
    headers = {"Authorization": "Bearer secret-key-123"}
    async with _http1_sse_client_factory(headers=headers) as client:
        # httpx merges headers to lowercase; check case-insensitively
        assert client.headers.get("authorization") == "Bearer secret-key-123"


@pytest.mark.asyncio
async def test_http1_factory_forwards_timeout():
    """Timeout passed to factory must be applied to the client."""
    timeout = httpx.Timeout(42.0, read=120.0)
    async with _http1_sse_client_factory(timeout=timeout) as client:
        assert client.timeout.connect == 42.0
        assert client.timeout.read == 120.0


# ---------------------------------------------------------------------------

def test_deliver_doc_idempotency_skip(mock_mcp_session):
    # Setup mock to return doc content that already has the heading
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="## Groww — 2026-W23\nSome content.")]
    mock_mcp_session.call_tool.return_value = mock_result
    
    config = {
        "mcp_servers": {"google_docs": {"url": "http://test"}},
        "delivery": {"google_doc_id": "doc123"}
    }
    
    metadata = {"product_name": "Groww", "iso_week": "2026-W23"}
    doc_payload = {"_raw_markdown": "new content"}
    
    heading_id = deliver_doc(doc_payload, metadata, config)
    
    assert heading_id == "heading=Groww — 2026-W23"
    # Should only call get_document, not update_document
    mock_mcp_session.call_tool.assert_called_once_with("get_document", arguments={"document_id": "doc123"})


def test_deliver_doc_append(mock_mcp_session):
    # Setup mock to return doc content without the heading
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="Some other content.")]
    mock_mcp_session.call_tool.return_value = mock_result
    
    config = {
        "mcp_servers": {"google_docs": {"url": "http://test"}},
        "delivery": {"google_doc_id": "doc123"}
    }
    
    metadata = {"product_name": "Groww", "iso_week": "2026-W23"}
    doc_payload = {"_raw_markdown": "new content"}
    
    heading_id = deliver_doc(doc_payload, metadata, config)
    
    assert heading_id == "heading=Groww — 2026-W23"
    assert mock_mcp_session.call_tool.call_count == 2
    mock_mcp_session.call_tool.assert_any_call("update_document", arguments={
        "document_id": "doc123",
        "content": "new content\n\n",
        "mode": "append"
    })

def test_deliver_email_skip_idempotent():
    config = {}
    run_log = {"delivery": {"gmail_message_id": "msg123"}}
    email_payload = EmailPayload(subject="Subj", html_body="html", text_body="txt", to=["test@test.com"])
    
    # Should return early without connecting
    msg_id = deliver_email(email_payload, config, run_log)
    assert msg_id == "msg123"

def test_deliver_email_draft(mock_mcp_session):
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="draft_id_456")]
    mock_mcp_session.call_tool.return_value = mock_result
    
    config = {
        "mcp_servers": {"gmail": {"url": "http://test"}},
        "delivery": {"email_mode": "draft"}
    }
    email_payload = EmailPayload(subject="Subj", html_body="html", text_body="txt", to=["test@test.com"])
    
    msg_id = deliver_email(email_payload, config, None)
    
    assert msg_id == "draft_id_456"
    mock_mcp_session.call_tool.assert_called_once_with("create_draft", arguments={
        "to": "test@test.com",
        "subject": "Subj",
        "body": "html"
    })

def test_deliver_email_send(mock_mcp_session):
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="msg_id_789")]
    mock_mcp_session.call_tool.return_value = mock_result
    
    config = {
        "mcp_servers": {"gmail": {"url": "http://test"}},
        "delivery": {"email_mode": "sent"}
    }
    email_payload = EmailPayload(subject="Subj", html_body="html", text_body="txt", to=["test@test.com"])
    
    msg_id = deliver_email(email_payload, config, None)
    
    assert msg_id == "msg_id_789"
    mock_mcp_session.call_tool.assert_called_once_with("send_email", arguments={
        "to": "test@test.com",
        "subject": "Subj",
        "body": "html"
    })
