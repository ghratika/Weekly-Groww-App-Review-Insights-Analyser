"""
Review ingestion module.

Fetches Google Play Store reviews by directly calling the Play Store
Reviews tool functions and applies two-layer PII scrubbing.

Architecture:
  - Layer 1 PII (author anonymization) is applied inside ``fetch_reviews``
    (in ``src.mcp_servers.playstore_reviews.tools``).
  - Layer 2 PII scrubbing (regex + NER on review text) is applied here,
    after the reviews are returned.

Note on MCP architecture:
  The Play Store Reviews MCP server (``src.mcp_servers.playstore_reviews``)
  exposes ``fetch_reviews`` as a tool for external MCP clients. Within this
  Python package the agent calls the tool functions directly to avoid
  subprocess and pipe-lifecycle complexity (no spawned subprocesses, no
  JSON-RPC serialisation overhead, no "I/O operation on closed file" races).
  The ``server.py`` module remains available for external MCP clients.

Architecture references:
  - §3   — Ingestion layer
  - §8.1 — Two-layer PII strategy
  - §10  — Error handling: 0-review abort
"""

import logging

from src.mcp_servers.playstore_reviews.tools import fetch_reviews as _tool_fetch_reviews
from src.agent.pii_scrubber import scrub_reviews

logger = logging.getLogger("pulse.ingestion")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_reviews_via_mcp(config: dict) -> list[dict]:
    """
    Fetch reviews from the Play Store Reviews tool and apply Layer 2 PII scrubbing.

    Calls ``fetch_reviews`` (the MCP tool function) directly — no subprocess
    or JSON-RPC communication needed since the tool is in the same Python package.

    Workflow:
      1. Extract connection parameters from ``config``.
      2. Call ``fetch_reviews`` (Layer 1 PII — author anonymization — is applied
         inside this function).
      3. Validate: abort if 0 reviews returned.
      4. Apply Layer 2 PII scrubbing (``pii_scrubber.scrub_reviews``).
      5. Return the cleaned list of ``Review`` dicts.

    Args:
        config: Loaded configuration dict (from ``src.agent.config.load_config``).

    Returns:
        List of review dicts with PII-scrubbed ``text`` fields.

    Raises:
        RuntimeError:    If 0 reviews are returned (abort signal to the orchestrator).
        ValueError:      If the app_id is not found on Google Play.
        ConnectionError: If the network is unavailable after retries.
    """
    product_cfg = config["product"]
    app_id: str = product_cfg["play_store_app_id"]
    weeks: int = product_cfg.get("review_window_weeks", 12)

    mcp_cfg = config.get("mcp_servers", {}).get("playstore_reviews", {})
    lang: str = mcp_cfg.get("lang", "en")
    country: str = mcp_cfg.get("country", "in")

    logger.info(
        "Fetching reviews via MCP: app_id=%s, weeks=%d, lang=%s, country=%s",
        app_id, weeks, lang, country,
    )

    # --- Call the MCP tool function directly (Layer 1 PII included) ---
    reviews = _tool_fetch_reviews(
        app_id=app_id,
        weeks=weeks,
        lang=lang,
        country=country,
    )

    if not isinstance(reviews, list):
        raise ValueError(
            f"Unexpected response type from fetch_reviews: {type(reviews)}"
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
