"""
Unit tests for src/agent/quote_validator.py (Phase 4).

Tests cover:
  - Verbatim quotes that appear in the source review are kept
  - Paraphrased / fabricated quotes that do NOT appear verbatim are discarded
  - Quotes referencing an unknown review_id are discarded
  - Case-sensitive matching (uppercase substring ≠ lowercase review text)
  - Clusters with 0 surviving quotes are still retained (summary + action_ideas valid)
  - Logging of discarded quotes (observability)
"""

from __future__ import annotations

import logging

import pytest

from src.agent.clustering import RawCluster
from src.agent.quote_validator import validate_quotes
from src.agent.summarizer import Cluster, ValidatedQuote


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_review(review_id: str, text: str, rating: int = 3) -> dict:
    return {"review_id": review_id, "text": text, "rating": rating}


def _make_quote(text: str, review_id: str, rating: int = 3) -> ValidatedQuote:
    return ValidatedQuote(text=text, review_id=review_id, rating=rating)


def _make_cluster(
    cluster_id: int,
    quotes: list[ValidatedQuote],
    theme_name: str = "Test Theme",
) -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        theme_name=theme_name,
        summary="A summary.",
        review_count=len(quotes) or 1,
        avg_rating=3.0,
        quotes=quotes,
        action_ideas=["Improve X"],
    )


# ---------------------------------------------------------------------------
# Core validation tests
# ---------------------------------------------------------------------------

class TestValidateQuotes:
    def test_verbatim_quote_passes(self):
        """A quote that is a verbatim substring of the source review must be kept."""
        reviews = [_make_review("r1", "The app crashes during market hours unexpectedly.")]
        quote = _make_quote("crashes during market hours", "r1")
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)

        assert len(result[0].quotes) == 1
        assert result[0].quotes[0].text == "crashes during market hours"

    def test_full_review_text_as_quote_passes(self):
        """The entire review text as a quote is a valid verbatim match."""
        text = "Great app but needs better UI."
        reviews = [_make_review("r2", text)]
        quote = _make_quote(text, "r2")
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)
        assert len(result[0].quotes) == 1

    def test_fabricated_quote_is_discarded(self):
        """A quote that is NOT a substring of the referenced review must be removed."""
        reviews = [_make_review("r1", "The app is slow on older devices.")]
        quote = _make_quote("crashes during market hours", "r1")  # not in review
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)
        assert len(result[0].quotes) == 0

    def test_paraphrased_quote_is_discarded(self):
        """Paraphrased text that differs even slightly from the original is discarded."""
        reviews = [_make_review("r1", "The loading is too slow for me.")]
        # Paraphrase — similar but not verbatim
        quote = _make_quote("loading is very slow", "r1")
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)
        assert len(result[0].quotes) == 0

    def test_unknown_review_id_is_discarded(self):
        """A quote referencing a review_id not in the reviews list must be dropped."""
        reviews = [_make_review("r1", "Some review text here.")]
        quote = _make_quote("Some review text", "r99")  # r99 does not exist
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)
        assert len(result[0].quotes) == 0

    def test_case_sensitive_matching(self):
        """Matching is case-sensitive: uppercase quote ≠ lowercase review text."""
        reviews = [_make_review("r1", "the app crashes frequently")]
        # Uppercase first letter — not an exact match
        quote = _make_quote("The app crashes frequently", "r1")
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)
        assert len(result[0].quotes) == 0

    def test_exact_case_match_passes(self):
        """Case-sensitive verbatim match must pass."""
        reviews = [_make_review("r1", "The app crashes frequently")]
        quote = _make_quote("The app crashes frequently", "r1")
        cluster = _make_cluster(0, [quote])

        result = validate_quotes([cluster], reviews)
        assert len(result[0].quotes) == 1

    def test_cluster_with_zero_quotes_is_retained(self):
        """
        A cluster where all quotes are discarded must still be returned.
        Its theme_name, summary, and action_ideas remain intact.
        """
        reviews = [_make_review("r1", "Decent app overall.")]
        quote = _make_quote("fabricated text", "r1")
        cluster = _make_cluster(0, [quote], theme_name="UI Issues")

        result = validate_quotes([cluster], reviews)

        assert len(result) == 1, "Cluster with 0 validated quotes must still be returned."
        assert result[0].quotes == []
        assert result[0].theme_name == "UI Issues"
        assert result[0].action_ideas == ["Improve X"]

    def test_mixed_quotes_partial_pass(self):
        """Only verbatim quotes pass; fabricated ones are dropped within the same cluster."""
        review_text = "The KYC process is broken and login fails."
        reviews = [_make_review("r1", review_text)]
        quotes = [
            _make_quote("KYC process is broken", "r1"),        # ✓ verbatim
            _make_quote("KYC is totally broken", "r1"),         # ✗ paraphrased
            _make_quote("login fails", "r1"),                    # ✓ verbatim
        ]
        cluster = _make_cluster(0, quotes)

        result = validate_quotes([cluster], reviews)

        surviving = [q.text for q in result[0].quotes]
        assert "KYC process is broken" in surviving
        assert "login fails" in surviving
        assert "KYC is totally broken" not in surviving

    def test_multiple_clusters_independently_validated(self):
        """Validation operates per-cluster independently."""
        reviews = [
            _make_review("r1", "app crashes frequently"),
            _make_review("r2", "slow loading screen on startup"),
        ]
        clusters = [
            _make_cluster(0, [_make_quote("app crashes frequently", "r1")]),
            _make_cluster(1, [_make_quote("fabricated text here", "r2")]),
        ]

        result = validate_quotes(clusters, reviews)

        assert len(result[0].quotes) == 1
        assert len(result[1].quotes) == 0

    def test_empty_clusters_list(self):
        """validate_quotes with an empty cluster list returns an empty list."""
        reviews = [_make_review("r1", "some review")]
        result = validate_quotes([], reviews)
        assert result == []

    def test_empty_reviews_list(self):
        """If reviews list is empty, all quotes reference unknown IDs and are discarded."""
        quote = _make_quote("some text", "r1")
        cluster = _make_cluster(0, [quote])
        result = validate_quotes([cluster], [])
        assert len(result[0].quotes) == 0

    def test_original_clusters_mutated_in_place(self):
        """validate_quotes modifies cluster.quotes in-place and returns the same list."""
        reviews = [_make_review("r1", "real review text")]
        quote = _make_quote("fabricated", "r1")
        cluster = _make_cluster(0, [quote])
        original_list = [cluster]

        result = validate_quotes(original_list, reviews)

        assert result is original_list  # same list object returned
        assert cluster.quotes == []     # mutated in-place

    def test_logging_of_discarded_quotes(self, caplog):
        """Discarded quotes should generate at least one debug log entry."""
        reviews = [_make_review("r1", "valid review text here")]
        quote = _make_quote("not in review", "r1")
        cluster = _make_cluster(0, [quote])

        with caplog.at_level(logging.DEBUG, logger="pulse.quote_validator"):
            validate_quotes([cluster], reviews)

        # At least one debug message should mention the discarded quote
        assert any("Discarding" in record.message for record in caplog.records)

    def test_count_logged_matches_discarded(self, caplog):
        """The logged summary line should reflect the correct discard count."""
        reviews = [
            _make_review("r1", "good quote here"),
            _make_review("r2", "another real review"),
        ]
        clusters = [
            _make_cluster(0, [
                _make_quote("good quote here", "r1"),  # ✓ pass
                _make_quote("fabricated one", "r1"),   # ✗ discard
            ]),
        ]

        with caplog.at_level(logging.INFO, logger="pulse.quote_validator"):
            result = validate_quotes(clusters, reviews)

        # 1 of 2 quotes should survive
        assert len(result[0].quotes) == 1
        # Check summary log
        summary_logs = [r.message for r in caplog.records if "validation complete" in r.message.lower()]
        assert len(summary_logs) == 1
        assert "1/2" in summary_logs[0]
