"""
Unit tests for the server-side PII scrubbing module (Layer 1).

Tests cover edge cases P1-01 through P1-04.
"""

import pytest

from src.mcp_servers.playstore_reviews.pii import anonymize_author, anonymize_reviews


# ---------------------------------------------------------------------------
# Test: anonymize_author
# ---------------------------------------------------------------------------

class TestAnonymizeAuthor:
    def test_basic_anonymization(self):
        """Normal author name is replaced with User_<hash>."""
        result = anonymize_author("John Doe")
        assert result.startswith("User_")
        assert len(result) == 13  # "User_" (5) + 8 hex chars
        assert result != "John Doe"

    def test_deterministic(self):
        """Same input always produces the same pseudonym."""
        assert anonymize_author("Jane Smith") == anonymize_author("Jane Smith")

    def test_different_names_different_hashes(self):
        """Different names produce different pseudonyms (with high probability)."""
        result_a = anonymize_author("Alice")
        result_b = anonymize_author("Bob")
        assert result_a != result_b

    def test_none_returns_anon(self):
        """Edge case P1-04: None author name."""
        assert anonymize_author(None) == "User_anon"

    def test_empty_string_returns_anon(self):
        """Edge case P1-04: empty author name."""
        assert anonymize_author("") == "User_anon"

    def test_whitespace_only_returns_anon(self):
        """Edge case P1-04: whitespace-only author name."""
        assert anonymize_author("   ") == "User_anon"

    def test_already_anonymous(self):
        """Edge case P1-01: 'A Google user' is still hashed."""
        result = anonymize_author("A Google user")
        assert result.startswith("User_")
        assert result != "A Google user"
        assert result != "User_anon"

    def test_non_ascii_characters(self):
        """Edge case P1-03: non-ASCII characters work correctly."""
        result = anonymize_author("राहुल शर्मा")
        assert result.startswith("User_")
        assert len(result) == 13

    def test_emoji_in_name(self):
        """Edge case P1-03: emoji characters work correctly."""
        result = anonymize_author("User 🚀✨")
        assert result.startswith("User_")
        assert len(result) == 13

    def test_unicode_deterministic(self):
        """Non-ASCII names produce deterministic results."""
        assert anonymize_author("用户名") == anonymize_author("用户名")

    def test_long_name(self):
        """Very long names still produce 8-char hashes."""
        result = anonymize_author("A" * 10000)
        assert result.startswith("User_")
        assert len(result) == 13


# ---------------------------------------------------------------------------
# Test: anonymize_reviews
# ---------------------------------------------------------------------------

class TestAnonymizeReviews:
    def test_batch_anonymization(self):
        """All reviews in a list have their authors anonymized."""
        reviews = [
            {"author": "Alice", "text": "Great app!"},
            {"author": "Bob", "text": "Needs work."},
            {"author": "Charlie", "text": "Love it!"},
        ]
        result = anonymize_reviews(reviews)
        assert len(result) == 3
        for review in result:
            assert review["author"].startswith("User_")
            assert review["author"] != "Alice"
            assert review["author"] != "Bob"
            assert review["author"] != "Charlie"

    def test_preserves_other_fields(self):
        """Non-author fields are not modified."""
        reviews = [
            {"author": "Alice", "text": "Great app!", "rating": 5},
        ]
        result = anonymize_reviews(reviews)
        assert result[0]["text"] == "Great app!"
        assert result[0]["rating"] == 5

    def test_empty_list(self):
        """Empty review list returns empty list."""
        assert anonymize_reviews([]) == []

    def test_original_names_never_in_output(self):
        """Edge case: original names must never appear in output."""
        names = ["John Doe", "Jane Smith", "राहुल", "User 🚀"]
        reviews = [{"author": name, "text": "Review"} for name in names]
        result = anonymize_reviews(reviews)
        output_authors = [r["author"] for r in result]
        for name in names:
            assert name not in output_authors

    def test_same_author_same_pseudonym(self):
        """Reviews by the same author get the same pseudonym."""
        reviews = [
            {"author": "Duplicate User", "text": "Review 1"},
            {"author": "Duplicate User", "text": "Review 2"},
        ]
        result = anonymize_reviews(reviews)
        assert result[0]["author"] == result[1]["author"]

    def test_none_author_in_batch(self):
        """None author in a batch doesn't crash."""
        reviews = [
            {"author": None, "text": "No name"},
            {"author": "Real Name", "text": "Has name"},
        ]
        result = anonymize_reviews(reviews)
        assert result[0]["author"] == "User_anon"
        assert result[1]["author"].startswith("User_")
        assert result[1]["author"] != "User_anon"
