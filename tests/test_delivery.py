"""
Unit tests for delivery.py
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.agent.delivery import deliver_doc, deliver_email
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
