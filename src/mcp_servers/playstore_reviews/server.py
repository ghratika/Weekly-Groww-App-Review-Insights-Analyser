"""
Play Store Reviews MCP Server.

An MCP server that exposes Google Play Store review-fetching capabilities
as MCP tools via stdio transport. This is the only custom MCP server built
in this project.

Usage:
    python -m src.mcp_servers.playstore_reviews.server

Tools exposed:
    - fetch_reviews:    Fetch PII-scrubbed reviews for a Google Play app.
    - get_app_metadata: Fetch app metadata (name, category, rating, version).
"""

import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.mcp_servers.playstore_reviews.tools import fetch_reviews, get_app_metadata

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,  # MCP uses stdout for protocol; logs go to stderr
)
logger = logging.getLogger("pulse.mcp_server")

# ---------------------------------------------------------------------------
# MCP Server Setup
# ---------------------------------------------------------------------------

server = Server("playstore-reviews")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools this MCP server exposes."""
    return [
        Tool(
            name="fetch_reviews",
            description=(
                "Fetch public Google Play Store reviews for an app within a "
                "rolling date window. Reviews are returned with PII-scrubbed "
                "author names. Returns a JSON array of Review objects."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Google Play package name (e.g., 'com.groww.v1').",
                    },
                    "weeks": {
                        "type": "integer",
                        "description": "Number of weeks back from today (review window). Default: 12.",
                        "default": 12,
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language code for reviews (BCP 47, e.g., 'en'). Default: 'en'.",
                        "default": "en",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code for the Play Store (e.g., 'in'). Default: 'in'.",
                        "default": "in",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_app_metadata",
            description=(
                "Fetch metadata for a Google Play app: name, category, "
                "current rating, and version."
            ),
            inputSchema={
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
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle incoming MCP tool calls.

    Routes to the appropriate tool function and returns the result
    as JSON-encoded TextContent.
    """
    logger.info("Tool call received: %s(%s)", name, arguments)

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
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )
            ]

        return [
            TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False),
            )
        ]

    except ValueError as exc:
        logger.error("Tool %s raised ValueError: %s", name, exc)
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": str(exc)}),
            )
        ]
    except ConnectionError as exc:
        logger.error("Tool %s raised ConnectionError: %s", name, exc)
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Network error: {exc}"}),
            )
        ]
    except Exception as exc:
        logger.exception("Unexpected error in tool %s", name)
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Internal server error: {exc}"}),
            )
        ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    """Run the MCP server on stdio transport."""
    logger.info("Starting Play Store Reviews MCP Server (stdio transport)...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
