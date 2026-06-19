"""
Delivery layer (Phase 6).

Appends report sections to Google Docs and sends/drafts emails via the
remote Railway SSE MCP server. Passes the MCP_API_SECRET_KEY as an
Authorization: Bearer header for authentication.

Note: Railway's proxy returns HTTP 421 with HTTP/2 SSE connections.
We force HTTP/1.1 via a custom httpx transport to work around this.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

from src.agent.email_renderer import EmailPayload

logger = logging.getLogger("pulse.delivery")


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Unwrap ExceptionGroup to expose the root cause for cleaner logging."""
    if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        return _unwrap_exception(exc.exceptions[0])
    return exc


def _http1_client_factory(**kwargs) -> httpx.AsyncClient:
    """Factory that creates a fresh HTTP/1.1 client for each SSE connection.
    Avoids Railway's 421 Misdirected Request caused by HTTP/2 multiplexing.
    Ignores kwargs passed by sse_client (headers/auth/timeout) since we set them separately.
    """
    return httpx.AsyncClient(http2=False)


def _get_headers(config_section: dict) -> dict[str, str]:
    """
    Build authentication headers for the SSE MCP server.

    Prefers the api_key from config (which may be substituted from env vars),
    falling back to the MCP_API_SECRET_KEY environment variable directly.
    """
    api_key = config_section.get("api_key") or os.environ.get("MCP_API_SECRET_KEY", "")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


# ---------------------------------------------------------------------------
# Google Docs delivery
# ---------------------------------------------------------------------------

async def _deliver_doc_async(doc_payload: dict, metadata: dict, config: dict) -> str:
    delivery_cfg = config.get("delivery", {})
    doc_id = delivery_cfg.get("google_doc_id")
    if not doc_id:
        raise ValueError("google_doc_id not found in config")

    docs_cfg = config.get("mcp_servers", {}).get("google_docs", {})
    url = docs_cfg.get("url")
    if not url:
        raise ValueError("Google Docs MCP server URL not found in config")

    headers = _get_headers(docs_cfg)
    heading = f"{metadata.get('product_name')} — {metadata.get('iso_week')}"
    raw_markdown = doc_payload.get("_raw_markdown", "")

    logger.debug("Connecting to Google Docs MCP server at %s", url)
    try:
        async with sse_client(url, headers=headers, httpx_client_factory=_http1_client_factory) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()

                # Check for idempotency: does the doc already contain this heading?
                logger.info("Checking if section '%s' already exists in doc...", heading)
                res = await session.call_tool("get_document", arguments={"document_id": doc_id})
                doc_text = res.content[0].text if res.content else ""

                if heading in doc_text:
                    logger.info("Heading '%s' already exists in Doc. Skipping append.", heading)
                    return f"heading={heading}"

                # Append the section
                logger.info("Appending section to Google Doc...")
                await session.call_tool("update_document", arguments={
                    "document_id": doc_id,
                    "content": raw_markdown + "\n\n",
                    "mode": "append"
                })
                logger.info("Successfully appended to Google Doc.")
                return f"heading={heading}"
    except Exception as exc:
        root = _unwrap_exception(exc)
        logger.error("Failed to deliver doc: %s: %s", type(root).__name__, root)
        raise


def deliver_doc(doc_payload: dict, metadata: dict, config: dict) -> str:
    """Synchronous wrapper for doc delivery."""
    return asyncio.run(_deliver_doc_async(doc_payload, metadata, config))


# ---------------------------------------------------------------------------
# Gmail delivery
# ---------------------------------------------------------------------------

async def _deliver_email_async(
    email_payload: EmailPayload,
    config: dict,
    run_log: dict[str, Any] | None,
) -> str | None:
    # Check idempotency first — no network call needed
    if run_log and run_log.get("delivery", {}).get("gmail_message_id"):
        msg_id = run_log["delivery"]["gmail_message_id"]
        logger.info("Email already delivered (id: %s). Skipping.", msg_id)
        return msg_id

    gmail_cfg = config.get("mcp_servers", {}).get("gmail", {})
    url = gmail_cfg.get("url")
    if not url:
        raise ValueError("Gmail MCP server URL not found in config")

    headers = _get_headers(gmail_cfg)
    mode = config.get("delivery", {}).get("email_mode", "draft")

    logger.debug("Connecting to Gmail MCP server at %s", url)
    try:
        async with sse_client(url, headers=headers, httpx_client_factory=_http1_client_factory) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()

                args = {
                    "to": ", ".join(email_payload.to),
                    "subject": email_payload.subject,
                    "body": email_payload.html_body,
                }

                if mode == "sent":
                    logger.info("Sending email to %s...", args["to"])
                    res = await session.call_tool("send_email", arguments=args)
                else:
                    logger.info("Creating email draft for %s...", args["to"])
                    res = await session.call_tool("create_draft", arguments=args)

                output = res.content[0].text if res.content else "unknown_id"
                logger.info("Email %s successful: %s", mode, output)
                return output
    except Exception as exc:
        root = _unwrap_exception(exc)
        logger.error("Failed to deliver email: %s: %s", type(root).__name__, root)
        raise


def deliver_email(
    email_payload: EmailPayload,
    config: dict,
    run_log: dict[str, Any] | None,
) -> str | None:
    """Synchronous wrapper for email delivery."""
    return asyncio.run(_deliver_email_async(email_payload, config, run_log))
