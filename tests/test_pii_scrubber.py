"""
Unit tests for the agent-side PII scrubber (Layer 2).

Tests cover:
  - Email address redaction
  - Indian and international phone number redaction
  - Non-PII text is preserved exactly
  - scrub_reviews applies scrub_text to every review's text field
  - Graceful handling of None / empty text
"""

import pytest

from src.agent.pii_scrubber import scrub_text, scrub_reviews

_REDACTED = "[REDACTED]"


# ---------------------------------------------------------------------------
# Test: scrub_text — email redaction
# ---------------------------------------------------------------------------

class TestEmailRedaction:
    def test_plain_email(self):
        text = "Contact me at john.doe@example.com for help"
        result = scrub_text(text)
        assert "john.doe@example.com" not in result
        assert _REDACTED in result

    def test_email_with_plus(self):
        text = "Send to user+tag@gmail.com please"
        result = scrub_text(text)
        assert "user+tag@gmail.com" not in result
        assert _REDACTED in result

    def test_email_with_subdomain(self):
        text = "Reply to support@mail.company.co.in"
        result = scrub_text(text)
        assert "support@mail.company.co.in" not in result
        assert _REDACTED in result

    def test_multiple_emails(self):
        text = "Email a@b.com or c@d.org for info"
        result = scrub_text(text)
        assert "a@b.com" not in result
        assert "c@d.org" not in result

    def test_no_email_preserved(self):
        text = "The app is great and works well"
        assert scrub_text(text) == text


# ---------------------------------------------------------------------------
# Test: scrub_text — phone number redaction
# ---------------------------------------------------------------------------

class TestPhoneRedaction:
    def test_indian_mobile_10_digits(self):
        text = "Call me on 9876543210 anytime"
        result = scrub_text(text)
        assert "9876543210" not in result
        assert _REDACTED in result

    def test_indian_mobile_with_country_code(self):
        text = "Reach me at +919876543210"
        result = scrub_text(text)
        assert "9876543210" not in result
        assert _REDACTED in result

    def test_indian_mobile_with_spaces(self):
        text = "My number is +91 98765 43210"
        result = scrub_text(text)
        assert "98765" not in result

    def test_no_phone_preserved(self):
        text = "The version is 4.8.1 and works fine on Android 14"
        assert scrub_text(text) == text


# ---------------------------------------------------------------------------
# Test: scrub_text — non-PII preservation
# ---------------------------------------------------------------------------

class TestNonPIIPreservation:
    def test_plain_review_unchanged(self):
        text = "The app crashes during market hours. Please fix this issue."
        assert scrub_text(text) == text

    def test_review_with_version_unchanged(self):
        text = "Great update in version 4.8.1, the UI is much smoother now."
        assert scrub_text(text) == text

    def test_review_with_rating_unchanged(self):
        text = "I give it 5 stars out of 5 for ease of use."
        assert scrub_text(text) == text

    def test_empty_string(self):
        assert scrub_text("") == ""

    def test_none_returns_none(self):
        result = scrub_text(None)
        assert result is None

    def test_whitespace_only(self):
        assert scrub_text("   ") == "   "


# ---------------------------------------------------------------------------
# Test: scrub_text — combined PII in single review
# ---------------------------------------------------------------------------

class TestCombinedPII:
    def test_email_and_phone_both_redacted(self):
        text = "Contact john@example.com or call 9876543210 for refund"
        result = scrub_text(text)
        assert "john@example.com" not in result
        assert "9876543210" not in result
        assert result.count(_REDACTED) >= 2

    def test_pii_at_start_of_text(self):
        text = "9876543210 is my number, please call back"
        result = scrub_text(text)
        assert "9876543210" not in result

    def test_pii_at_end_of_text(self):
        text = "For help email me at support@groww.in"
        result = scrub_text(text)
        assert "support@groww.in" not in result


# ---------------------------------------------------------------------------
# Test: scrub_reviews — list-level scrubbing
# ---------------------------------------------------------------------------

class TestScrubReviews:
    def _make_review(self, review_id: str, text: str) -> dict:
        return {
            "review_id": review_id,
            "author": "User_abc",
            "rating": 4,
            "text": text,
            "date": "2026-05-01",
            "app_version": "4.8.1",
            "thumbs_up": 0,
            "language": "en",
        }

    def test_scrubs_all_reviews(self):
        reviews = [
            self._make_review("r1", "Email me at foo@bar.com for more info"),
            self._make_review("r2", "Call 9876543210 to complain"),
            self._make_review("r3", "Great app overall, no issues"),
        ]
        result = scrub_reviews(reviews)
        assert "foo@bar.com" not in result[0]["text"]
        assert "9876543210" not in result[1]["text"]
        assert result[2]["text"] == "Great app overall, no issues"

    def test_returns_same_list(self):
        """scrub_reviews modifies in-place and returns the same list object."""
        reviews = [self._make_review("r1", "Clean text no PII")]
        returned = scrub_reviews(reviews)
        assert returned is reviews

    def test_other_fields_untouched(self):
        """Only the text field is modified; other fields remain unchanged."""
        reviews = [self._make_review("r1", "Email foo@bar.com for help")]
        scrub_reviews(reviews)
        assert reviews[0]["review_id"] == "r1"
        assert reviews[0]["author"] == "User_abc"
        assert reviews[0]["rating"] == 4
        assert reviews[0]["date"] == "2026-05-01"

    def test_empty_list(self):
        assert scrub_reviews([]) == []

    def test_review_with_no_pii_unchanged(self):
        original_text = "The SIP feature stopped working after the last update."
        reviews = [self._make_review("r1", original_text)]
        scrub_reviews(reviews)
        assert reviews[0]["text"] == original_text
