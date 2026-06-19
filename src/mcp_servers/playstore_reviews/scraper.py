"""
Google Play Store review scraper.

Uses the `google-play-scraper` library to fetch app reviews and metadata.
Handles date-window filtering, pagination, rate limiting, and edge cases
(S-01 through S-12 in edge-cases.md).
"""

import hashlib
import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

from google_play_scraper import Sort, reviews, app as app_info
from google_play_scraper.exceptions import NotFoundError

logger = logging.getLogger("pulse.scraper")

# Maximum reviews to request per pagination batch
_BATCH_SIZE = 200

# Maximum retry attempts on rate-limit / transient errors
_MAX_RETRIES = 3

# Base delay for exponential backoff (seconds)
_BACKOFF_BASE = 2.0

# Minimum number of words for a review to be kept
_MIN_WORD_COUNT = 8

# Regex pattern to detect emoji characters (Unicode emoji ranges)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Misc Symbols
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U0000200D"             # Zero Width Joiner
    "\U00002B50"             # Star
    "\U000023F0-\U000023FA"  # Clock symbols
    "]",
    flags=re.UNICODE,
)


def _generate_review_id(review: dict) -> str:
    """
    Generate a stable review ID from a review dict.

    The google-play-scraper library provides a `reviewId` field, but we prefix
    it with `gp_` to namespace it and truncate to a reasonable length.
    """
    raw_id = review.get("reviewId", "")
    if raw_id:
        return f"gp_{raw_id[:32]}"

    # Fallback: hash author + date + text for a deterministic ID
    fingerprint = f"{review.get('userName', '')}{review.get('at', '')}{review.get('content', '')}"
    short_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"gp_{short_hash}"


def _parse_review_date(review: dict) -> datetime | None:
    """
    Parse the review date from the scraper output.

    The `at` field is a datetime object from google-play-scraper.
    Returns None if the date cannot be parsed (edge case S-07).
    """
    raw_date = review.get("at")
    if raw_date is None:
        return None

    if isinstance(raw_date, datetime):
        # Ensure timezone-aware (assume UTC if naive)
        if raw_date.tzinfo is None:
            return raw_date.replace(tzinfo=timezone.utc)
        return raw_date

    # If it's a string, try parsing common formats
    if isinstance(raw_date, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                parsed = datetime.strptime(raw_date, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        logger.warning("Unparseable review date: %s", raw_date)
        return None

    return None


def _has_emoji(text: str) -> bool:
    """Check if the text contains any emoji characters."""
    return bool(_EMOJI_PATTERN.search(text))


def _is_non_english(text: str) -> bool:
    """
    Check if the text contains significant non-English (non-Latin) content.

    Uses Unicode script detection: if more than 15% of the alphabetic
    characters are outside the Basic Latin range (a-z, A-Z), the text
    is considered non-English and will be excluded.

    The 15% threshold (down from 30%) catches mixed Hindi-English reviews
    that are common in Indian Play Store data, where a few English words
    are sprinkled into a primarily Devanagari sentence.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False  # No alphabetic chars → can't determine; keep it

    non_latin_count = sum(
        1 for c in alpha_chars
        if not ('a' <= c <= 'z' or 'A' <= c <= 'Z')
    )
    ratio = non_latin_count / len(alpha_chars)
    return ratio > 0.15


def _normalize_review(review: dict) -> dict | None:
    """
    Normalize a raw google-play-scraper review dict into our standard schema.

    Applies the following filters:
    - Malformed reviews (missing fields) → skipped (edge case S-05)
    - Empty text → skipped (edge case S-08)
    - Reviews with fewer than 8 words → skipped (too short for meaningful analysis)
    - Reviews containing emojis → skipped (noise reduction)
    - Reviews in non-English languages → skipped (pipeline is English-only)

    Returns None if the review should be excluded.
    """
    review_date = _parse_review_date(review)
    if review_date is None:
        logger.debug("Skipping review with unparseable date: %s", review.get("reviewId"))
        return None

    text = review.get("content")
    if text is None or (isinstance(text, str) and text.strip() == ""):
        logger.debug("Skipping review with empty text: %s", review.get("reviewId"))
        return None  # Edge case S-08

    text = str(text).strip()

    # Filter: minimum word count (< 8 words are too short for clustering)
    word_count = len(text.split())
    if word_count < _MIN_WORD_COUNT:
        logger.debug(
            "Skipping review with too few words (%d < %d): %s",
            word_count, _MIN_WORD_COUNT, review.get("reviewId"),
        )
        return None

    # Filter: reviews containing emojis (noise for text analysis)
    if _has_emoji(text):
        logger.debug("Skipping review with emojis: %s", review.get("reviewId"))
        return None

    # Filter: non-English text (pipeline is English-only)
    if _is_non_english(text):
        logger.debug("Skipping non-English review: %s", review.get("reviewId"))
        return None

    rating = review.get("score")
    if rating is None:
        logger.debug("Skipping review with missing rating: %s", review.get("reviewId"))
        return None

    return {
        "review_id": _generate_review_id(review),
        "author": review.get("userName", "Unknown"),
        "rating": int(rating),
        "text": text,
        "date": review_date.strftime("%Y-%m-%d"),
        "app_version": review.get("reviewCreatedVersion") or "unknown",
        "thumbs_up": review.get("thumbsUpCount", 0) or 0,
        "language": "en",  # Language is set by the caller
    }


def scrape_reviews(
    app_id: str,
    weeks: int = 12,
    lang: str = "en",
    country: str = "in",
) -> list[dict]:
    """
    Fetch reviews for a Google Play app within a rolling date window.

    Args:
        app_id:   Google Play package name (e.g., "com.groww.v1").
        weeks:    Number of weeks back from today (the review window).
        lang:     Language code for reviews (BCP 47, e.g., "en").
        country:  Country code for the Play Store (e.g., "in" for India).

    Returns:
        List of normalized review dicts, ordered newest-first.

    Raises:
        ValueError: If app_id is not found on Google Play (edge case S-01).
        ConnectionError: If the network is unavailable after retries (S-12).
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    logger.info(
        "Scraping reviews for %s (last %d weeks, cutoff: %s, lang: %s, country: %s)",
        app_id, weeks, cutoff_date.strftime("%Y-%m-%d"), lang, country,
    )

    all_reviews: list[dict] = []
    continuation_token = None
    seen_ids: set[str] = set()
    batch_num = 0

    while True:
        batch_num += 1
        logger.debug("Fetching batch %d (token: %s)", batch_num, bool(continuation_token))

        # Retry loop for rate limiting (edge case S-04)
        result = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=_BATCH_SIZE,
                    continuation_token=continuation_token,
                )
                break  # Success
            except NotFoundError:
                raise ValueError(
                    f"App ID '{app_id}' not found on Google Play. "
                    f"Verify the package name is correct."
                )  # Edge case S-01
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Scraper error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to fetch reviews after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc  # Edge case S-12

        if result is None or len(result) == 0:
            logger.debug("No more reviews returned; stopping pagination.")
            break

        # Process each review in the batch
        reached_cutoff = False
        for raw_review in result:
            normalized = _normalize_review(raw_review)
            if normalized is None:
                continue  # Malformed review; already logged

            # Set language from the call parameter
            normalized["language"] = lang

            # Date-window filtering (edge case S-06)
            review_date = datetime.strptime(normalized["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            if review_date < cutoff_date:
                reached_cutoff = True
                continue  # Outside window

            # Deduplication (edge case S-10)
            if normalized["review_id"] in seen_ids:
                logger.debug("Skipping duplicate review: %s", normalized["review_id"])
                continue
            seen_ids.add(normalized["review_id"])

            all_reviews.append(normalized)

        # If we've gone past the cutoff date, stop fetching
        if reached_cutoff:
            logger.debug("Reached cutoff date; stopping pagination.")
            break

        # If no continuation token, we've exhausted all reviews
        if continuation_token is None:
            logger.debug("No continuation token; all reviews fetched.")
            break

        # Brief pause between batches to be respectful
        time.sleep(0.5)

    logger.info("Scraped %d reviews within the %d-week window.", len(all_reviews), weeks)
    return all_reviews


def scrape_app_info(app_id: str, lang: str = "en", country: str = "in") -> dict:
    """
    Fetch metadata for a Google Play app.

    Args:
        app_id:   Google Play package name (e.g., "com.groww.v1").
        lang:     Language code.
        country:  Country code.

    Returns:
        Dict with keys: app_name, category, current_rating, version.

    Raises:
        ValueError: If app_id is not found on Google Play.
    """
    logger.info("Fetching app metadata for %s", app_id)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = app_info(app_id, lang=lang, country=country)
            break
        except NotFoundError:
            raise ValueError(
                f"App ID '{app_id}' not found on Google Play. "
                f"Verify the package name is correct."
            )
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE ** attempt
                logger.warning(
                    "App info fetch error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            else:
                raise ConnectionError(
                    f"Failed to fetch app info after {_MAX_RETRIES} attempts: {exc}"
                ) from exc

    return {
        "app_name": result.get("title", "Unknown"),
        "category": result.get("genre", "Unknown"),
        "current_rating": result.get("score", 0.0),
        "version": result.get("version", "unknown"),
    }
