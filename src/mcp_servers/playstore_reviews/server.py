"""
Play Store Reviews MCP Server.

A lightweight, synchronous JSON-RPC 2.0 dispatcher that exposes
Google Play Store review-fetching capabilities over stdio transport.
This replaces the anyio/stdio_server-based implementation to avoid
the "I/O operation on closed file" crash caused by the MCP SDK's
anyio streams receiving EOF before the tool response is written.

Usage:
    python -m src.mcp_servers.playstore_reviews.server

Protocol:
    Reads newline-delimited JSON-RPC 2.0 messages from stdin.
    Writes newline-delimited JSON-RPC 2.0 responses to stdout.
    All logging goes to stderr (stdout is reserved for the protocol).

Tools exposed:
    - fetch_reviews:    Fetch PII-scrubbed reviews for a Google Play app.
    - get_app_metadata: Fetch app metadata (name, category, rating, version).
"""

import json
import logging
import sys

from src.mcp_servers.playstore_reviews.tools import fetch_reviews, get_app_metadata

# ---------------------------------------------------------------------------
# Logging — stderr only; stdout is the JSON-RPC channel
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("pulse.mcp_server")

# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _ok(request_id, result) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _send(obj: dict) -> None:
    """Write a JSON-RPC object to stdout as a newline-terminated string."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

def _handle_initialize(request_id, params: dict) -> None:
    """Respond to the MCP initialize handshake."""
    _send(_ok(request_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "playstore-reviews", "version": "1.0"},
    }))


def _handle_tools_list(request_id) -> None:
    """Return the list of tools this server exposes."""
    tools = [
        {
            "name": "fetch_reviews",
            "description": (
                "Fetch public Google Play Store reviews for an app within a "
                "rolling date window. Reviews are returned with PII-scrubbed "
                "author names. Returns a JSON array of Review objects."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Google Play package name (e.g., 'com.groww.v1').",
                    },
                    "weeks": {
                        "type": "integer",
                        "description": "Number of weeks back from today. Default: 12.",
                        "default": 12,
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language code (BCP 47, e.g., 'en'). Default: 'en'.",
                        "default": "en",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (e.g., 'in'). Default: 'in'.",
                        "default": "in",
                    },
                },
                "required": ["app_id"],
            },
        },
        {
            "name": "get_app_metadata",
            "description": (
                "Fetch metadata for a Google Play app: name, category, "
                "current rating, and version."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Google Play package name (e.g., 'com.groww.v1').",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language code. Default: 'en'.",
                        "default": "en",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code. Default: 'in'.",
                        "default": "in",
                    },
                },
                "required": ["app_id"],
            },
        },
    ]
    _send(_ok(request_id, {"tools": tools}))


def _handle_tool_call(request_id, params: dict) -> None:
    """Dispatch a tools/call request to the appropriate tool function."""
    name = params.get("name", "")
    arguments = params.get("arguments", {})

    logger.info("Tool call: %s(%s)", name, arguments)

    try:
        if name == "fetch_reviews":
            result = fetch_reviews(
                app_id=arguments["app_id"],
                weeks=arguments.get("weeks", 12),
                lang=arguments.get("lang", "en"),
                country=arguments.get("country", "in"),
            )
        elif name == "get_app_metadata":
            result = get_app_metadata(
                app_id=arguments["app_id"],
                lang=arguments.get("lang", "en"),
                country=arguments.get("country", "in"),
            )
        else:
            _send(_error(request_id, -32601, f"Unknown tool: {name}"))
            return

        # Wrap result as MCP TextContent (JSON string inside text field)
        _send(_ok(request_id, {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
        }))

    except KeyError as exc:
        logger.error("Missing required argument for tool %s: %s", name, exc)
        _send(_error(request_id, -32602, f"Missing required argument: {exc}"))
    except (ValueError, ConnectionError) as exc:
        logger.error("Tool %s failed: %s", name, exc)
        _send(_error(request_id, -32000, str(exc)))
    except Exception as exc:
        logger.exception("Unexpected error in tool %s", name)
        _send(_error(request_id, -32000, f"Internal server error: {exc}"))


# ---------------------------------------------------------------------------
# Main dispatch loop
# ---------------------------------------------------------------------------

def _dispatch(line: str) -> None:
    """Parse one JSON-RPC line and dispatch to the appropriate handler."""
    try:
        msg = json.loads(line.strip())
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON-RPC message: %s", exc)
        _send(_error(None, -32700, f"Parse error: {exc}"))
        return

    request_id = msg.get("id")
    method = msg.get("method", "")

    # Notifications (no id) are silently acknowledged
    if request_id is None:
        logger.debug("Notification received: %s", method)
        return

    if method == "initialize":
        _handle_initialize(request_id, msg.get("params", {}))
    elif method == "tools/list":
        _handle_tools_list(request_id)
    elif method == "tools/call":
        _handle_tool_call(request_id, msg.get("params", {}))
    else:
        logger.warning("Unknown method: %s", method)
        _send(_error(request_id, -32601, f"Method not found: {method}"))


def main() -> None:
    """
    Read newline-delimited JSON-RPC messages from stdin and dispatch them.

    Runs until stdin is closed (EOF).  All responses go to stdout.
    Logging goes to stderr to keep the JSON-RPC channel clean.
    """
    logger.info("Play Store Reviews MCP Server started (synchronous stdio transport).")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        _dispatch(line)

    logger.info("Stdin closed — server exiting cleanly.")


if __name__ == "__main__":
    main()
