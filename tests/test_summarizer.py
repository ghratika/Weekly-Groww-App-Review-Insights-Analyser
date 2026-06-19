"""
Unit tests for src/agent/summarizer.py (Phase 4).

Tests cover:
  - LLM response is correctly parsed into Cluster objects
  - Token/daily-request limit enforcement (RuntimeError)
  - Retry logic on transient LLM failure (exponential backoff)
  - Graceful fallback for malformed LLM JSON fields
  - _build_user_message respects _MAX_REVIEWS_PER_CALL and _MAX_CHARS_PER_REVIEW caps
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.agent.clustering import RawCluster
from src.agent.summarizer import (
    Cluster,
    ValidatedQuote,
    _RateLimitTracker,
    _build_user_message,
    _call_llm_with_retry,
    _parse_llm_response,
    summarize_clusters,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_raw_cluster(
    cluster_id: int = 0,
    indices: list[int] | None = None,
    avg_rating: float = 3.0,
) -> RawCluster:
    if indices is None:
        indices = [0, 1, 2]
    return RawCluster(
        cluster_id=cluster_id,
        review_indices=indices,
        avg_rating=avg_rating,
        review_count=len(indices),
    )


def _make_reviews(n: int) -> list[dict]:
    return [
        {
            "review_id": f"r{i}",
            "text": f"This is review number {i} with some meaningful content about the app.",
            "rating": (i % 5) + 1,
        }
        for i in range(n)
    ]


def _valid_llm_json(review_id: str = "r0") -> dict:
    return {
        "theme_name": "App Crashes During Login",
        "summary": "Many users report crashes when trying to log in.",
        "quotes": [
            {"text": "This is review number 0 with some", "review_id": review_id},
        ],
        "action_ideas": ["Fix the login crash", "Add crash reporting"],
    }


def _base_config() -> dict:
    return {
        "llm": {
            "requests_per_day": 1000,
            "tokens_per_day": 100_000,
        }
    }


# ---------------------------------------------------------------------------
# _build_user_message tests
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_returns_valid_json(self):
        cluster = _make_raw_cluster(indices=[0, 1, 2])
        reviews = _make_reviews(3)
        msg = _build_user_message(cluster, reviews)
        parsed = json.loads(msg)
        assert "reviews" in parsed
        assert isinstance(parsed["reviews"], list)

    def test_respects_max_reviews_per_call(self):
        """Should not include more than _MAX_REVIEWS_PER_CALL reviews."""
        from src.agent.summarizer import _MAX_REVIEWS_PER_CALL

        n = _MAX_REVIEWS_PER_CALL + 10
        cluster = _make_raw_cluster(indices=list(range(n)))
        reviews = _make_reviews(n)
        msg = _build_user_message(cluster, reviews)
        parsed = json.loads(msg)
        assert len(parsed["reviews"]) <= _MAX_REVIEWS_PER_CALL

    def test_respects_max_chars_per_review(self):
        """Review text must be truncated to _MAX_CHARS_PER_REVIEW characters."""
        from src.agent.summarizer import _MAX_CHARS_PER_REVIEW

        long_text = "x" * (_MAX_CHARS_PER_REVIEW * 5)
        reviews = [{"review_id": "r0", "text": long_text, "rating": 4}]
        cluster = _make_raw_cluster(indices=[0])
        msg = _build_user_message(cluster, reviews)
        parsed = json.loads(msg)
        assert len(parsed["reviews"][0]["text"]) <= _MAX_CHARS_PER_REVIEW


# ---------------------------------------------------------------------------
# _RateLimitTracker tests
# ---------------------------------------------------------------------------

class TestRateLimitTracker:
    def test_accumulates_tokens_and_requests(self):
        tracker = _RateLimitTracker(_base_config())
        tracker.check_and_record(500)
        tracker.check_and_record(300)
        assert tracker.tokens_used == 800
        assert tracker.requests_used == 2

    def test_raises_on_token_limit_exceeded(self):
        config = {"llm": {"requests_per_day": 1000, "tokens_per_day": 100}}
        tracker = _RateLimitTracker(config)
        with pytest.raises(RuntimeError, match="token limit"):
            tracker.check_and_record(200)  # 200 > 100

    def test_raises_on_request_limit_exceeded(self):
        config = {"llm": {"requests_per_day": 2, "tokens_per_day": 100_000}}
        tracker = _RateLimitTracker(config)
        tracker.check_and_record(10)  # request 1
        tracker.check_and_record(10)  # request 2
        with pytest.raises(RuntimeError, match="request limit"):
            tracker.check_and_record(10)  # request 3 > limit


# ---------------------------------------------------------------------------
# _call_llm_with_retry tests
# ---------------------------------------------------------------------------

class TestCallLLMWithRetry:
    def _mock_response(self, content_dict: dict, total_tokens: int = 200):
        """Build a litellm-style mock response object."""
        response = MagicMock()
        response.choices[0].message.content = json.dumps(content_dict)
        response.usage.total_tokens = total_tokens
        return response

    def test_succeeds_on_first_attempt(self):
        tracker = _RateLimitTracker(_base_config())
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = self._mock_response(_valid_llm_json())

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):  # suppress actual sleeps
                result = _call_llm_with_retry("sys", "user", tracker)

        assert result["theme_name"] == "App Crashes During Login"
        mock_litellm.completion.assert_called_once()

    def test_retries_on_transient_exception(self):
        """Should retry up to _MAX_RETRIES times on generic exceptions."""
        from src.agent.summarizer import _MAX_RETRIES

        tracker = _RateLimitTracker(_base_config())
        mock_litellm = MagicMock()

        # Fail twice, succeed on the third attempt
        mock_litellm.completion.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            self._mock_response(_valid_llm_json()),
        ]

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                result = _call_llm_with_retry("sys", "user", tracker)

        assert mock_litellm.completion.call_count == 3
        assert "theme_name" in result

    def test_raises_after_all_retries_exhausted(self):
        """Should raise RuntimeError once all retries are used up."""
        from src.agent.summarizer import _MAX_RETRIES

        tracker = _RateLimitTracker(_base_config())
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = Exception("permanent failure")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="failed after"):
                    _call_llm_with_retry("sys", "user", tracker)

        assert mock_litellm.completion.call_count == _MAX_RETRIES

    def test_does_not_retry_on_rate_limit_runtime_error(self):
        """RuntimeError (e.g. daily limit) must not be retried."""
        tracker = _RateLimitTracker(_base_config())
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = RuntimeError("daily limit")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="daily limit"):
                    _call_llm_with_retry("sys", "user", tracker)

        mock_litellm.completion.assert_called_once()

    def test_retries_on_invalid_json(self):
        """JSON parse errors should be retried."""
        from src.agent.summarizer import _MAX_RETRIES

        tracker = _RateLimitTracker(_base_config())
        mock_litellm = MagicMock()

        bad_response = MagicMock()
        bad_response.choices[0].message.content = "not json {{{"
        bad_response.usage.total_tokens = 50

        good_response = self._mock_response(_valid_llm_json())

        # Fail with bad JSON twice, then succeed
        mock_litellm.completion.side_effect = [bad_response, bad_response, good_response]

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                result = _call_llm_with_retry("sys", "user", tracker)

        assert "theme_name" in result


# ---------------------------------------------------------------------------
# _parse_llm_response tests
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    def test_parses_valid_response(self):
        raw_cluster = _make_raw_cluster(indices=[0, 1])
        reviews = _make_reviews(2)
        raw = _valid_llm_json(review_id="r0")

        cluster = _parse_llm_response(raw, raw_cluster, reviews)

        assert isinstance(cluster, Cluster)
        assert cluster.theme_name == raw["theme_name"]
        assert cluster.summary == raw["summary"]
        assert len(cluster.action_ideas) == 2
        assert cluster.cluster_id == raw_cluster.cluster_id
        assert cluster.review_count == raw_cluster.review_count

    def test_missing_fields_use_defaults(self):
        """Gracefully handles missing theme_name / summary / action_ideas."""
        raw_cluster = _make_raw_cluster(indices=[0])
        reviews = _make_reviews(1)
        raw = {}  # Empty LLM response

        cluster = _parse_llm_response(raw, raw_cluster, reviews)

        assert cluster.theme_name.startswith("Theme")
        assert cluster.summary == ""
        assert cluster.action_ideas == []
        assert cluster.quotes == []

    def test_quote_enriched_with_rating(self):
        """Quotes should carry the rating from the source review."""
        raw_cluster = _make_raw_cluster(indices=[0, 1])
        reviews = _make_reviews(2)
        # Review r0 has rating = 1 (index 0, (0 % 5) + 1 = 1)
        raw = {
            "theme_name": "T",
            "summary": "S",
            "quotes": [{"text": "some text", "review_id": "r0"}],
            "action_ideas": [],
        }

        cluster = _parse_llm_response(raw, raw_cluster, reviews)

        assert len(cluster.quotes) == 1
        assert cluster.quotes[0].rating == 1

    def test_malformed_quotes_skipped(self):
        """Non-dict quote entries should be silently skipped."""
        raw_cluster = _make_raw_cluster(indices=[0])
        reviews = _make_reviews(1)
        raw = {
            "theme_name": "T",
            "summary": "S",
            "quotes": ["not a dict", 42, None],
            "action_ideas": [],
        }

        cluster = _parse_llm_response(raw, raw_cluster, reviews)
        assert cluster.quotes == []


# ---------------------------------------------------------------------------
# summarize_clusters integration tests
# ---------------------------------------------------------------------------

class TestSummarizeClusters:
    def _mock_litellm_response(self, content_dict: dict, tokens: int = 300):
        response = MagicMock()
        response.choices[0].message.content = json.dumps(content_dict)
        response.usage.total_tokens = tokens
        return response

    def test_returns_one_cluster_per_raw_cluster(self):
        raw_clusters = [_make_raw_cluster(cluster_id=i, indices=[i]) for i in range(3)]
        reviews = _make_reviews(3)
        config = _base_config()

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = self._mock_litellm_response(
            _valid_llm_json("r0")
        )

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                clusters, tokens_used = summarize_clusters(raw_clusters, reviews, config)

        assert len(clusters) == 3
        assert all(isinstance(c, Cluster) for c in clusters)
        assert tokens_used == 900  # 3 clusters × 300 tokens each

    def test_aborts_when_daily_token_limit_exceeded(self):
        """summarize_clusters must propagate RuntimeError from tracker."""
        raw_clusters = [_make_raw_cluster(cluster_id=0, indices=[0])]
        reviews = _make_reviews(1)
        # Set a very low token limit
        config = {"llm": {"requests_per_day": 1000, "tokens_per_day": 1}}

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = self._mock_litellm_response(
            _valid_llm_json(), tokens=500  # 500 > 1
        )

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="token limit"):
                    summarize_clusters(raw_clusters, reviews, config)

    def test_retry_on_failure_still_produces_cluster(self):
        """Retry mechanism allows recovery — final cluster list still populated."""
        raw_clusters = [_make_raw_cluster(indices=[0])]
        reviews = _make_reviews(1)
        config = _base_config()

        mock_litellm = MagicMock()
        bad_side_effect = [
            Exception("first failure"),
            self._mock_litellm_response(_valid_llm_json("r0")),
        ]
        mock_litellm.completion.side_effect = bad_side_effect

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("time.sleep"):
                clusters, tokens_used = summarize_clusters(raw_clusters, reviews, config)

        assert len(clusters) == 1
        assert clusters[0].theme_name == "App Crashes During Login"
        assert tokens_used > 0
