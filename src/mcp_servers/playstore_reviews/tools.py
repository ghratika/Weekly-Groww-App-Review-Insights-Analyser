"""
MCP tool definitions for the Play Store Reviews MCP server.

Exposes two tools:
- `fetch_reviews`: Fetch and return PII-scrubbed reviews for a Google Play app.
- `get_app_metadata`: Fetch app metadata (name, category, rating, version).

These tool functions wire the scraper and PII modules together.
"""

import logging
from typing import Any

from src.mcp_servers.playstore_reviews.scraper import scrape_reviews, scrape_app_info
from src.mcp_servers.playstore_reviews.pii import anonymize_reviews

logger = logging.getLogger("pulse.mcp_tools")


def fetch_reviews(
    app_id: str,
    weeks: int = 12,
    lang: str = "en",
    country: str = "in",
) -> list[dict]:
    """
    Fetch reviews for a Google Play app, with PII-scrubbed author names.

    This is the primary tool exposed by the Play Store Reviews MCP server.
    It:
      1. Calls the scraper to fetch raw reviews within the date window.
      2. Passes reviews through Layer 1 PII scrubbing (author anonymization).
      3. Returns the cleaned `Review[]` list.

    Args:
        app_id:   Google Play package name (e.g., "com.groww.v1").
        weeks:    Number of weeks back from today (review window).
        lang:     Language code (BCP 47, e.g., "en").
        country:  Country code (e.g., "in" for India).

    Returns:
        List of review dicts conforming to the Review schema:
        {
            "review_id": "gp_abc123",
            "author": "User_a3f2b1c8",  # PII-scrubbed
            "rating": 3,
            "text": "The app crashes during market hours...",
            "date": "2026-05-28",
            "app_version": "4.8.1",
            "thumbs_up": 12,
            "language": "en"
        }

    Raises:
        ValueError: If app_id is not found on Google Play.
        ConnectionError: If the network is unavailable after retries.
    """
    logger.info(
        "fetch_reviews called: app_id=%s, weeks=%d, lang=%s, country=%s",
        app_id, weeks, lang, country,
    )

    # Step 1: Scrape raw reviews
    raw_reviews = scrape_reviews(
        app_id=app_id,
        weeks=weeks,
        lang=lang,
        country=country,
    )

    logger.info("Fetched %d raw reviews from Google Play.", len(raw_reviews))

    # Step 2: Apply Layer 1 PII scrubbing (author anonymization)
    scrubbed_reviews = anonymize_reviews(raw_reviews)

    logger.info(
        "Returning %d PII-scrubbed reviews for %s.", len(scrubbed_reviews), app_id
    )
    return scrubbed_reviews


def get_app_metadata(
    app_id: str,
    lang: str = "en",
    country: str = "in",
) -> dict:
    """
    Fetch metadata for a Google Play app.

    Args:
        app_id:   Google Play package name (e.g., "com.groww.v1").
        lang:     Language code.
        country:  Country code.

    Returns:
        Dict with keys:
        {
            "app_name": "Groww",
            "category": "Finance",
            "current_rating": 4.3,
            "version": "4.8.1"
        }

    Raises:
        ValueError: If app_id is not found on Google Play.
        ConnectionError: If the network is unavailable after retries.
    """
    logger.info("get_app_metadata called: app_id=%s", app_id)

    metadata = scrape_app_info(
        app_id=app_id,
        lang=lang,
        country=country,
    )

    logger.info("App metadata: %s", metadata)
    return metadata
