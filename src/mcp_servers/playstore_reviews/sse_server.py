"""
Play Store Reviews MCP Server — HTTP/SSE transport for Railway deployment.

This module wraps the same tool implementations as server.py (stdio) but
exposes them over HTTP using FastMCP's SSE transport. Railway binds the
process to $PORT; this server listens on that port and serves the MCP SSE
protocol at the ``/sse`` path.

Why a separate file?
  - ``server.py`` uses a raw synchronous stdin/stdout loop — it has no HTTP
    server and cannot bind to a TCP port. Railway's gateway requires a
    process that listens on ``$PORT`` for HTTP traffic, so using server.py
    on Railway results in "Invalid Host header" / 421 errors because
    Railway's ingress proxy handles the TCP connection but nothing in the
    process accepts HTTP requests.
  - This file is the Railway entrypoint. The local agent pipeline still
    uses ``server.py`` via ``transport: stdio`` in config.yaml.

Deployment:
  Railway start command (in railway.toml):
    python -m src.mcp_servers.playstore_reviews.sse_server

Local smoke-test:
    PORT=8080 python -m src.mcp_servers.playstore_reviews.sse_server
    curl -v http://localhost:8080/sse       # should start SSE stream
    curl -v http://localhost:8080/health    # should return 200 OK

Authentication:
  All requests must carry ``Authorization: Bearer <MCP_API_SECRET_KEY>``.
  Requests without a valid key receive 401 Unauthorized.
  The secret key is read from the ``MCP_API_SECRET_KEY`` environment variable.

Allowed hosts:
  FastMCP / Starlette's default ``TrustedHostMiddleware`` is NOT used here.
  Railway terminates TLS at its edge and forwards requests with the original
  ``Host`` header. We accept any host so Railway's proxy hostname
  (``mcp-server-ghratika-production.up.railway.app``) is not rejected.
"""

from __future__ import annotations

import logging
import os
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("pulse.sse_server")


# ---------------------------------------------------------------------------
# Auth middleware — validate Bearer token for every request
# ---------------------------------------------------------------------------

def _get_expected_key() -> str:
    """Read the expected API key from the environment."""
    key = os.environ.get("MCP_API_SECRET_KEY", "")
    if not key:
        logger.warning(
            "MCP_API_SECRET_KEY is not set. "
            "All requests will be rejected with 401."
        )
    return key


class BearerAuthMiddleware:
    """
    ASGI middleware that validates ``Authorization: Bearer <key>`` headers.

    Returns 401 if the header is missing or the token does not match
    ``MCP_API_SECRET_KEY``.  The health-check endpoint is exempt.
    """

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Health check is always exempt
        path = scope.get("path", "")
        if path in ("/health", "/"):
            await self._app(scope, receive, send)
            return

        expected_key = _get_expected_key()
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")

        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:]

        if not expected_key or token != expected_key:
            logger.warning(
                "Rejected unauthenticated request to %s (token present: %s)",
                path, bool(token),
            )
            response = Response(
                content='{"error": "Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


# ---------------------------------------------------------------------------
# Build FastMCP server
# ---------------------------------------------------------------------------

def _build_mcp_server():
    """
    Construct the FastMCP server and register both tools.

    Tools are thin wrappers around the same implementation used by the
    stdio server (tools.py), so behaviour is identical between transports.
    """
    from mcp.server.fastmcp import FastMCP
    from src.mcp_servers.playstore_reviews.tools import fetch_reviews, get_app_metadata

    port = int(os.environ.get("PORT", "8080"))

    mcp = FastMCP(
        name="playstore-reviews",
        host="0.0.0.0",   # bind to all interfaces so Railway can route traffic
        port=port,
        sse_path="/sse",
        message_path="/messages",
    )

    @mcp.tool()
    def fetch_reviews_tool(
        app_id: str,
        weeks: int = 12,
        lang: str = "en",
        country: str = "in",
    ) -> list[dict]:
        """
        Fetch public Google Play Store reviews for an app within a rolling
        date window. Reviews are returned with PII-scrubbed author names.
        Returns a JSON array of Review objects.
        """
        return fetch_reviews(app_id=app_id, weeks=weeks, lang=lang, country=country)

    @mcp.tool()
    def get_app_metadata_tool(
        app_id: str,
        lang: str = "en",
        country: str = "in",
    ) -> dict:
        """
        Fetch metadata for a Google Play app: name, category, current rating,
        and version.
        """
        return get_app_metadata(app_id=app_id, lang=lang, country=country)

    return mcp


# ---------------------------------------------------------------------------
# Health check route
# ---------------------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
    """Simple health-check endpoint for Railway's HTTP health probe."""
    return JSONResponse({"status": "ok", "service": "playstore-reviews-mcp"})


# ---------------------------------------------------------------------------
# Assemble the full ASGI application
# ---------------------------------------------------------------------------

def build_app() -> Starlette:
    """
    Build and return the full ASGI application.

    Layout:
      GET  /health   — Railway health probe (unauthenticated)
      GET  /sse      — MCP SSE stream (Bearer auth required)
      POST /messages — MCP message endpoint (Bearer auth required)
    """
    mcp = _build_mcp_server()

    # Get the FastMCP SSE Starlette sub-app
    mcp_starlette = mcp.sse_app()

    # Wrap both the health route and the MCP app together
    app = Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
        ],
    )

    # Mount the MCP SSE app at root so /sse and /messages are reachable
    # We use the Starlette Router's mount mechanism
    from starlette.routing import Mount
    app.router.routes.append(Mount("/", app=mcp_starlette))

    # Wrap with Bearer auth middleware (health is exempt inside the middleware)
    return BearerAuthMiddleware(app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start uvicorn serving the MCP SSE app on $PORT."""
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    logger.info(
        "Starting Play Store Reviews MCP SSE server on 0.0.0.0:%d", port
    )
    logger.info("  SSE endpoint  : http://0.0.0.0:%d/sse", port)
    logger.info("  Health check  : http://0.0.0.0:%d/health", port)

    uvicorn.run(
        "src.mcp_servers.playstore_reviews.sse_server:build_app",
        host="0.0.0.0",
        port=port,
        factory=True,
        log_level="info",
        # http2=False is the uvicorn default when h2 is not installed
        # Explicitly disable it to prevent Railway 421 issues
        http="h11",
    )


if __name__ == "__main__":
    main()
