"""
Review ingestion via the Play Store Reviews MCP server.

This module is the agent-side MCP client for Phase 3. It:
  1. Spawns the Play Store Reviews MCP server as a subprocess (stdio transport).
  2. Calls the ``fetch_reviews`` MCP tool to retrieve normalized, Layer-1-scrubbed
     reviews from Google Play.
  3. Validates the response (aborts if 0 reviews are returned).
  4. Applies Layer 2 PII scrubbing (regex + NER) to all review texts.

Architecture references:
  - §3   — Ingestion layer
  - §8.1 — Two-layer PII strategy
  - §10  — Error handling: 0-review abort
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.agent.pii_scrubber import scrub_reviews

logger = logging.getLogger("pulse.ingestion")


# ---------------------------------------------------------------------------
# MCP client helpers
# ---------------------------------------------------------------------------

def _build_mcp_request(method: str, params: dict, request_id: int = 1) -> str:
    """Serialize a JSON-RPC 2.0 request as a newline-terminated string."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    return json.dumps(payload) + "\n"


def _parse_mcp_response(raw: str) -> Any:
    """
    Parse a single JSON-RPC 2.0 response line.

    Raises:
        ValueError: If the response contains a JSON-RPC error or is malformed.
    """
    try:
        response = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON-RPC response: {raw!r}") from exc

    if "error" in response:
        err = response["error"]
        raise ValueError(
            f"MCP server returned error [{err.get('code')}]: {err.get('message')}"
        )

    return response.get("result")


def _call_mcp_tool_via_stdio(
    server_module: str,
    tool_name: str,
    arguments: dict,
    python_executable: str | None = None,
) -> Any:
    """
    Invoke an MCP tool by spawning the server as a subprocess (stdio transport).

    Sends:
      1. ``initialize`` handshake
      2. ``tools/call`` request

    Args:
        server_module: Python module path for the server
                       (e.g., ``src.mcp_servers.playstore_reviews.server``).
        tool_name:     Name of the MCP tool to call.
        arguments:     Tool arguments dict.
        python_executable: Path to the Python interpreter. Defaults to sys.executable.

    Returns:
        The parsed ``result`` from the tool call response.

    Raises:
        ConnectionError: If the subprocess fails to start or communication errors occur.
        ValueError:      If the MCP server returns an error.
    """
    python = python_executable or sys.executable

    try:
        process = subprocess.Popen(
            [python, "-m", server_module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConnectionError(
            f"Failed to start MCP server subprocess '{server_module}': {exc}"
        ) from exc

    try:
        # --- Step 1: initialize handshake ---
        init_request = _build_mcp_request(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pulse-agent", "version": "1.0"},
            },
            request_id=1,
        )
        process.stdin.write(init_request)

        # --- Step 2: initialized notification ---
        initialized_notif = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }) + "\n"
        process.stdin.write(initialized_notif)

        # --- Step 3: tool call ---
        tool_request = _build_mcp_request(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=2,
        )
        process.stdin.write(tool_request)
        process.stdin.flush()
        process.stdin.close()

        # --- Read all stdout ---
        stdout_data, stderr_data = process.communicate(timeout=120)

    except subprocess.TimeoutExpired:
        process.kill()
        raise ConnectionError(
            f"MCP server '{server_module}' timed out after 120 seconds."
        )
    except OSError as exc:
        raise ConnectionError(
            f"I/O error communicating with MCP server '{server_module}': {exc}"
        ) from exc

    if process.returncode != 0 and not stdout_data.strip():
        raise ConnectionError(
            f"MCP server '{server_module}' exited with code {process.returncode}. "
            f"stderr: {stderr_data[:500]}"
        )

    # Parse JSON-RPC responses — we need the response with id=2 (tool call)
    tool_response_raw = None
    for line in stdout_data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if parsed.get("id") == 2:
                tool_response_raw = line
                break
        except json.JSONDecodeError:
            continue

    if tool_response_raw is None:
        raise ConnectionError(
            f"No tool-call response received from MCP server '{server_module}'. "
            f"stdout: {stdout_data[:500]}"
        )

    result = _parse_mcp_response(tool_response_raw)

    # MCP tool results are wrapped in TextContent: [{"type": "text", "text": "<json>"}]
    if isinstance(result, dict) and "content" in result:
        content = result["content"]
        if isinstance(content, list) and content:
            text_content = content[0].get("text", "")
            return json.loads(text_content)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_reviews_via_mcp(config: dict) -> list[dict]:
    """
    Fetch reviews from the Play Store Reviews MCP server and apply Layer 2 PII scrubbing.

    Workflow:
      1. Extract connection parameters from ``config``.
      2. Spawn the Play Store Reviews MCP server subprocess (stdio).
      3. Call the ``fetch_reviews`` tool.
      4. Validate: abort if 0 reviews returned.
      5. Apply Layer 2 PII scrubbing (``pii_scrubber.scrub_reviews``).
      6. Return the cleaned list of ``Review`` dicts.

    Args:
        config: Loaded configuration dict (from ``src.agent.config.load_config``).

    Returns:
        List of review dicts with PII-scrubbed ``text`` fields.

    Raises:
        RuntimeError: If 0 reviews are returned (abort signal to the orchestrator).
        ConnectionError: If the MCP server subprocess fails.
        ValueError: If the MCP server returns an error response.
    """
    product_cfg = config["product"]
    app_id: str = product_cfg["play_store_app_id"]
    weeks: int = product_cfg.get("review_window_weeks", 12)

    mcp_cfg = config.get("mcp_servers", {}).get("playstore_reviews", {})
    lang: str = mcp_cfg.get("lang", "en")
    country: str = mcp_cfg.get("country", "in")
    server_module: str = mcp_cfg.get(
        "server_module", "src.mcp_servers.playstore_reviews.server"
    )

    logger.info(
        "Fetching reviews via MCP: app_id=%s, weeks=%d, lang=%s, country=%s",
        app_id, weeks, lang, country,
    )

    # --- Connect to MCP server and call fetch_reviews ---
    reviews = _call_mcp_tool_via_stdio(
        server_module=server_module,
        tool_name="fetch_reviews",
        arguments={
            "app_id": app_id,
            "weeks": weeks,
            "lang": lang,
            "country": country,
        },
    )

    if not isinstance(reviews, list):
        raise ValueError(
            f"Unexpected response type from fetch_reviews MCP tool: {type(reviews)}"
        )

    logger.info("Received %d reviews from MCP server.", len(reviews))

    # --- Abort on 0 reviews (architecture §10) ---
    if len(reviews) == 0:
        raise RuntimeError(
            f"No reviews returned for app '{app_id}' in the last {weeks} weeks. "
            "Aborting run — will not append an empty section to the report."
        )

    # --- Layer 2 PII scrubbing ---
    logger.info("Applying Layer 2 PII scrubbing to %d reviews...", len(reviews))
    scrubbed = scrub_reviews(reviews)

    logger.info(
        "Ingestion complete: %d reviews ready for analysis.", len(scrubbed)
    )
    return scrubbed
