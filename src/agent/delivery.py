"""
Delivery layer (Phase 6).

Appends report sections to Google Docs and sends/drafts emails via the
remote Railway SSE MCP server. Passes the MCP_API_SECRET_KEY as an
Authorization: Bearer header for authentication.

Railway proxy quirks
--------------------
Railway's reverse proxy returns HTTP 421 (Misdirected Request) and also
rejects connections with an "Invalid Host header" error when clients use
HTTP/2. Two compounding issues existed in the original factory:

  1. ``_http1_client_factory`` was NOT a context manager — it returned a
     bare ``httpx.AsyncClient``, not the ``async with``-able object that
     ``sse_client`` expects from its ``httpx_client_factory`` callback.
     This caused the SSE connection to bypass the factory entirely in
     some MCP SDK versions and always use the default HTTP/2 client.

  2. The factory silently discarded the ``headers``, ``auth``, and
     ``timeout`` kwargs that ``sse_client`` passes in, so the
     ``Authorization: Bearer`` token was never sent. Railway then
     rejected the unauthenticated request before host-validation even ran.

Fix: ``_http1_sse_client_factory`` is now a proper ``@asynccontextmanager``
that forces ``http2=False`` and forwards all kwargs correctly.
"""

from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def _http1_sse_client_factory(
    headers: dict | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
):
    """
    Async context-manager factory for MCP's ``httpx_client_factory`` hook.

    Fixes two Railway-specific problems:
      - Forces HTTP/1.1 (``http2=False``) to avoid Railway's 421
        Misdirected Request on the ``/sse`` endpoint.
      - Correctly forwards ``headers``, ``timeout``, and ``auth`` so
        the ``Authorization: Bearer`` token is actually sent.

    The ``@asynccontextmanager`` decorator is required because
    ``sse_client`` uses the factory as ``async with factory(...) as client``.
    """
    client_kwargs: dict[str, Any] = {
        "http2": False,
        "follow_redirects": True,
    }
    if headers:
        client_kwargs["headers"] = headers
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if auth is not None:
        client_kwargs["auth"] = auth

    async with httpx.AsyncClient(**client_kwargs) as client:
        yield client


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
        async with sse_client(url, headers=headers, httpx_client_factory=_http1_sse_client_factory) as streams:
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
                logger.info(
                    "Appending section to Google Doc (doc_id=%s, text_len=%d chars)...",
                    doc_id, len(raw_markdown),
                )
                if not raw_markdown:
                    raise ValueError(
                        "_raw_markdown is empty in doc_payload — nothing to write. "
                        "Check that render_doc_section populated _raw_markdown and "
                        "that the run log JSON preserved it correctly."
                    )
                update_res = await session.call_tool("update_document", arguments={
                    "document_id": doc_id,
                    "text": raw_markdown + "\n\n",   # tool param is "text", not "content"
                    "mode": "append"
                })
                # Log the raw MCP response so Railway logs show exactly what the
                # Docs API replied (characters_written, replies[], status).
                update_text = update_res.content[0].text if update_res.content else "{}"
                logger.info("update_document response: %s", update_text)

                # Fail loudly if the tool reported writing 0 characters.
                import json as _json
                try:
                    update_data = _json.loads(update_text)
                    chars = update_data.get("characters_written", -1)
                    if chars == 0:
                        raise RuntimeError(
                            f"update_document reported characters_written=0. "
                            f"Full response: {update_text}"
                        )
                    logger.info("Confirmed %d characters written to Google Doc.", chars)
                except (_json.JSONDecodeError, AttributeError):
                    logger.warning("Could not parse update_document response as JSON: %s", update_text)

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
        async with sse_client(url, headers=headers, httpx_client_factory=_http1_sse_client_factory) as streams:
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
