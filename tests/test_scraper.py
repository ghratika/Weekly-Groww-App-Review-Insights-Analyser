"""
Unit tests for the Play Store Reviews scraper module.

Tests use mocked `google-play-scraper` responses to avoid network calls.
Covers edge cases S-01 through S-12 plus normalization filters.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.mcp_servers.playstore_reviews.scraper import (
    scrape_reviews,
    scrape_app_info,
    _normalize_review,
    _generate_review_id,
    _parse_review_date,
    _has_emoji,
    _is_non_english,
)


# ---------------------------------------------------------------------------
# Fixtures: sample review data
# ---------------------------------------------------------------------------

# A valid English review with >= 8 words and no emojis
_VALID_TEXT = "The app works really well and I love using it every day"


def _make_raw_review(
    review_id: str = "abc123",
    user_name: str = "Test User",
    score: int = 4,
    content: str = _VALID_TEXT,
    at: datetime | None = None,
    version: str = "4.8.1",
    thumbs_up: int = 5,
) -> dict:
    """Create a mock google-play-scraper review dict."""
    if at is None:
        at = datetime.now(timezone.utc) - timedelta(days=3)
    return {
        "reviewId": review_id,
        "userName": user_name,
        "score": score,
        "content": content,
        "at": at,
        "reviewCreatedVersion": version,
        "thumbsUpCount": thumbs_up,
    }


# ---------------------------------------------------------------------------
# Test: _generate_review_id
# ---------------------------------------------------------------------------

class TestGenerateReviewId:
    def test_uses_review_id_field(self):
        review = {"reviewId": "abc123xyz"}
        result = _generate_review_id(review)
        assert result == "gp_abc123xyz"

    def test_truncates_long_ids(self):
        review = {"reviewId": "a" * 100}
        result = _generate_review_id(review)
        assert result == f"gp_{'a' * 32}"
        assert len(result) == 35  # "gp_" + 32 chars

    def test_fallback_hash_when_no_review_id(self):
        review = {"userName": "User", "at": "2026-01-01", "content": "Test"}
        result = _generate_review_id(review)
        assert result.startswith("gp_")
        assert len(result) == 15  # "gp_" + 12 hex chars

    def test_deterministic_fallback(self):
        review = {"userName": "User", "at": "2026-01-01", "content": "Test"}
        assert _generate_review_id(review) == _generate_review_id(review)


# ---------------------------------------------------------------------------
# Test: _parse_review_date
# ---------------------------------------------------------------------------

class TestParseReviewDate:
    def test_datetime_object(self):
        dt = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_review_date({"at": dt})
        assert result == dt

    def test_naive_datetime_gets_utc(self):
        dt = datetime(2026, 5, 28, 12, 0, 0)
        result = _parse_review_date({"at": dt})
        assert result.tzinfo == timezone.utc

    def test_string_date(self):
        result = _parse_review_date({"at": "2026-05-28"})
        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 28

    def test_none_returns_none(self):
        assert _parse_review_date({"at": None}) is None
        assert _parse_review_date({}) is None

    def test_unparseable_string_returns_none(self):
        """Edge case S-07: unexpected date format."""
        assert _parse_review_date({"at": "not-a-date"}) is None


# ---------------------------------------------------------------------------
# Test: _has_emoji
# ---------------------------------------------------------------------------

class TestHasEmoji:
    def test_no_emoji(self):
        assert _has_emoji("This is a plain text review without any special chars") is False

    def test_with_smiley(self):
        assert _has_emoji("Great app! 😊") is True

    def test_with_thumbs_up(self):
        assert _has_emoji("Love it 👍") is True

    def test_with_star(self):
        assert _has_emoji("Five stars ⭐") is True

    def test_with_heart(self):
        assert _has_emoji("I ❤ this app so much it is wonderful") is True

    def test_plain_punctuation_is_not_emoji(self):
        assert _has_emoji("Great app!!! Very good??? Yes.") is False


# ---------------------------------------------------------------------------
# Test: _is_non_english
# ---------------------------------------------------------------------------

class TestIsNonEnglish:
    def test_english_text(self):
        assert _is_non_english("This is a great app and I use it daily") is False

    def test_hindi_text(self):
        assert _is_non_english("यह एक बहुत अच्छा ऐप है") is True

    def test_tamil_text(self):
        assert _is_non_english("இது ஒரு நல்ல பயன்பாடு") is True

    def test_mixed_mostly_english(self):
        """Mixed text that is mostly English should pass."""
        assert _is_non_english("The app is great for trading stocks online") is False

    def test_mixed_mostly_non_english(self):
        """Mixed text that is mostly non-English should fail."""
        assert _is_non_english("ऐप अच्छा है app") is True

    def test_numbers_only(self):
        """No alphabetic chars → treated as English (kept)."""
        assert _is_non_english("12345 67890") is False

    def test_empty_string(self):
        assert _is_non_english("") is False


# ---------------------------------------------------------------------------
# Test: _normalize_review
# ---------------------------------------------------------------------------

class TestNormalizeReview:
    def test_valid_review(self):
        raw = _make_raw_review()
        result = _normalize_review(raw)
        assert result is not None
        assert result["review_id"].startswith("gp_")
        assert result["author"] == "Test User"
        assert result["rating"] == 4
        assert result["text"] == _VALID_TEXT
        assert result["app_version"] == "4.8.1"
        assert result["thumbs_up"] == 5

    def test_empty_text_returns_none(self):
        """Edge case S-08: empty review text."""
        raw = _make_raw_review(content="")
        assert _normalize_review(raw) is None

    def test_none_text_returns_none(self):
        """Edge case S-08: null review text."""
        raw = _make_raw_review()
        raw["content"] = None
        assert _normalize_review(raw) is None

    def test_missing_rating_returns_none(self):
        """Edge case S-05: missing fields."""
        raw = _make_raw_review()
        raw["score"] = None
        assert _normalize_review(raw) is None

    def test_unparseable_date_returns_none(self):
        """Edge case S-07."""
        raw = _make_raw_review()
        raw["at"] = "invalid"
        assert _normalize_review(raw) is None

    def test_short_review_filtered(self):
        """Reviews with fewer than 8 words are filtered out."""
        raw = _make_raw_review(content="Too short review")
        assert _normalize_review(raw) is None

    def test_exactly_8_words_passes(self):
        """Reviews with exactly 8 words should pass."""
        raw = _make_raw_review(content="This app works well for trading stocks daily")
        result = _normalize_review(raw)
        assert result is not None

    def test_7_words_filtered(self):
        """Reviews with 7 words should be filtered."""
        raw = _make_raw_review(content="This app works well for trading stocks")
        assert _normalize_review(raw) is None

    def test_emoji_review_filtered(self):
        """Reviews containing emojis are filtered out."""
        raw = _make_raw_review(
            content="This is a great app and I love it so much 😊"
        )
        assert _normalize_review(raw) is None

    def test_non_english_review_filtered(self):
        """Reviews in non-English languages are filtered out."""
        raw = _make_raw_review(
            content="यह एक बहुत अच्छा ऐप है मुझे बहुत पसंद है"
        )
        assert _normalize_review(raw) is None

    def test_clean_english_review_passes(self):
        """A clean, 8+ word English review with no emojis passes all filters."""
        raw = _make_raw_review(
            content="The customer support team resolved my issue very quickly and efficiently"
        )
        result = _normalize_review(raw)
        assert result is not None
        assert result["text"] == "The customer support team resolved my issue very quickly and efficiently"


# ---------------------------------------------------------------------------
# Test: scrape_reviews (mocked)
# ---------------------------------------------------------------------------

class TestScrapeReviews:
    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_basic_scrape(self, mock_reviews):
        """Scraping returns normalized, filtered reviews."""
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        mock_reviews.return_value = (
            [_make_raw_review(review_id="r1", at=recent_date)],
            None,  # No continuation token
        )

        result = scrape_reviews("com.groww.v1", weeks=12)
        assert len(result) == 1
        assert result[0]["review_id"] == "gp_r1"
        mock_reviews.assert_called_once()

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_date_window_filtering(self, mock_reviews):
        """Edge case S-06: reviews outside the window are excluded."""
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        old = datetime.now(timezone.utc) - timedelta(weeks=20)

        mock_reviews.return_value = (
            [
                _make_raw_review(review_id="recent", at=recent),
                _make_raw_review(review_id="old", at=old),
            ],
            None,
        )

        result = scrape_reviews("com.groww.v1", weeks=12)
        assert len(result) == 1
        assert result[0]["review_id"] == "gp_recent"

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_empty_results(self, mock_reviews):
        """Edge case S-02: app has zero reviews."""
        mock_reviews.return_value = ([], None)
        result = scrape_reviews("com.groww.v1")
        assert result == []

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_deduplication(self, mock_reviews):
        """Edge case S-10: duplicate review_ids are deduplicated."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mock_reviews.return_value = (
            [
                _make_raw_review(review_id="dup", at=recent, content=_VALID_TEXT),
                _make_raw_review(
                    review_id="dup", at=recent,
                    content="Another review that has at least eight words in it definitely"
                ),
            ],
            None,
        )

        result = scrape_reviews("com.groww.v1")
        assert len(result) == 1
        assert result[0]["text"] == _VALID_TEXT  # Keeps first occurrence

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_app_not_found(self, mock_reviews):
        """Edge case S-01: app ID does not exist."""
        from google_play_scraper.exceptions import NotFoundError
        mock_reviews.side_effect = NotFoundError("Not found")

        with pytest.raises(ValueError, match="not found on Google Play"):
            scrape_reviews("com.nonexistent.app")

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    @patch("src.mcp_servers.playstore_reviews.scraper.time.sleep")
    def test_retry_on_transient_error(self, mock_sleep, mock_reviews):
        """Edge case S-04: retries on transient errors."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mock_reviews.side_effect = [
            Exception("Rate limited"),
            ([_make_raw_review(review_id="r1", at=recent)], None),
        ]

        result = scrape_reviews("com.groww.v1")
        assert len(result) == 1
        assert mock_reviews.call_count == 2
        mock_sleep.assert_called()  # Backoff delay was applied

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    @patch("src.mcp_servers.playstore_reviews.scraper.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep, mock_reviews):
        """Edge case S-12: all retries exhausted."""
        mock_reviews.side_effect = Exception("Network error")

        with pytest.raises(ConnectionError, match="Failed to fetch reviews"):
            scrape_reviews("com.groww.v1")

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_pagination(self, mock_reviews):
        """Reviews are fetched across multiple pages."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        token = "next_page_token"

        mock_reviews.side_effect = [
            ([_make_raw_review(review_id="r1", at=recent)], token),
            ([_make_raw_review(review_id="r2", at=recent)], None),
        ]

        with patch("src.mcp_servers.playstore_reviews.scraper.time.sleep"):
            result = scrape_reviews("com.groww.v1")

        assert len(result) == 2
        assert mock_reviews.call_count == 2

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_skips_malformed_reviews(self, mock_reviews):
        """Edge case S-05: reviews with null fields are skipped."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        good_review = _make_raw_review(review_id="good", at=recent)
        bad_review = _make_raw_review(review_id="bad", at=recent)
        bad_review["content"] = None  # Missing text

        mock_reviews.return_value = ([good_review, bad_review], None)
        result = scrape_reviews("com.groww.v1")
        assert len(result) == 1
        assert result[0]["review_id"] == "gp_good"

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_filters_short_reviews(self, mock_reviews):
        """Reviews with < 8 words are filtered during scraping."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mock_reviews.return_value = (
            [
                _make_raw_review(review_id="long", at=recent, content=_VALID_TEXT),
                _make_raw_review(review_id="short", at=recent, content="Too short"),
            ],
            None,
        )

        result = scrape_reviews("com.groww.v1")
        assert len(result) == 1
        assert result[0]["review_id"] == "gp_long"

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_filters_emoji_reviews(self, mock_reviews):
        """Reviews with emojis are filtered during scraping."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mock_reviews.return_value = (
            [
                _make_raw_review(review_id="clean", at=recent, content=_VALID_TEXT),
                _make_raw_review(
                    review_id="emoji", at=recent,
                    content="This app is really great and amazing for investing 🚀🔥"
                ),
            ],
            None,
        )

        result = scrape_reviews("com.groww.v1")
        assert len(result) == 1
        assert result[0]["review_id"] == "gp_clean"

    @patch("src.mcp_servers.playstore_reviews.scraper.reviews")
    def test_filters_non_english_reviews(self, mock_reviews):
        """Non-English reviews are filtered during scraping."""
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mock_reviews.return_value = (
            [
                _make_raw_review(review_id="en", at=recent, content=_VALID_TEXT),
                _make_raw_review(
                    review_id="hi", at=recent,
                    content="यह ऐप बहुत अच्छा है मुझे यह बहुत पसंद है"
                ),
            ],
            None,
        )

        result = scrape_reviews("com.groww.v1")
        assert len(result) == 1
        assert result[0]["review_id"] == "gp_en"


# ---------------------------------------------------------------------------
# Test: scrape_app_info (mocked)
# ---------------------------------------------------------------------------

class TestScrapeAppInfo:
    @patch("src.mcp_servers.playstore_reviews.scraper.app_info")
    def test_basic_info(self, mock_app_info):
        mock_app_info.return_value = {
            "title": "Groww",
            "genre": "Finance",
            "score": 4.3,
            "version": "4.8.1",
        }

        result = scrape_app_info("com.groww.v1")
        assert result["app_name"] == "Groww"
        assert result["category"] == "Finance"
        assert result["current_rating"] == 4.3
        assert result["version"] == "4.8.1"

    @patch("src.mcp_servers.playstore_reviews.scraper.app_info")
    def test_app_not_found(self, mock_app_info):
        from google_play_scraper.exceptions import NotFoundError
        mock_app_info.side_effect = NotFoundError("Not found")

        with pytest.raises(ValueError, match="not found on Google Play"):
            scrape_app_info("com.nonexistent.app")

    @patch("src.mcp_servers.playstore_reviews.scraper.app_info")
    def test_missing_fields_use_defaults(self, mock_app_info):
        """App info with missing fields returns safe defaults."""
        mock_app_info.return_value = {}
        result = scrape_app_info("com.groww.v1")
        assert result["app_name"] == "Unknown"
        assert result["category"] == "Unknown"
        assert result["current_rating"] == 0.0
        assert result["version"] == "unknown"
