"""
LLM-based cluster summarization via Groq (Phase 4).

For each RawCluster produced by the clustering engine, this module:
  1. Builds a structured prompt containing the cluster's review texts.
  2. Calls ``groq/llama-3.3-70b-versatile`` via ``litellm``.
  3. Parses the JSON response into a ``Cluster`` object with theme_name,
     summary, quotes, and action_ideas.
  4. Enforces Groq free-tier rate limits with per-call delays and daily counters.
  5. Retries up to 3× with exponential backoff on transient failures.

Data models produced:
  ValidatedQuote: { text, review_id, rating }
  Cluster:        { cluster_id, theme_name, summary, review_count,
                    avg_rating, quotes, action_ideas }

Architecture references:
  - §4   — Analysis engine
  - §8.2 — Prompt injection safety (reviews as data, not in system prompt)
  - §10  — Error handling: retry, rate-limit abort
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("pulse.summarizer")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ValidatedQuote:
    """A quote extracted from a review (verbatim, to be validated later)."""
    text: str
    review_id: str
    rating: int


@dataclass
class Cluster:
    """
    A fully summarized theme cluster (output of LLM summarization).

    Quotes at this stage are pre-validated; final validation happens in
    quote_validator.py.
    """
    cluster_id: int
    theme_name: str
    summary: str
    review_count: int
    avg_rating: float
    quotes: list[ValidatedQuote] = field(default_factory=list)
    action_ideas: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Groq / LiteLLM constants
# ---------------------------------------------------------------------------

_MODEL = "groq/llama-3.3-70b-versatile"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds

# Safety margin: leave 10% headroom below hard limits
_SAFE_RPM = 27          # 30 * 0.9
_SAFE_TPD = 90_000      # 100_000 * 0.9
_SAFE_RPD = 900         # 1_000 * 0.9

# Minimum delay between calls to respect RPM limit (seconds per request)
_MIN_CALL_INTERVAL = 60.0 / _SAFE_RPM  # ~2.2 seconds

# Maximum reviews to include per cluster call (to stay within 12K TPM)
_MAX_REVIEWS_PER_CALL = 30
_MAX_CHARS_PER_REVIEW = 400


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior product analyst summarizing Google Play Store reviews "
    "for a fintech app. Your job is to identify a clear product theme from a "
    "batch of user reviews, write a concise summary, extract 2-3 representative "
    "verbatim quotes, and propose 2-3 actionable product improvements.\n\n"
    "IMPORTANT: Quotes MUST be verbatim substrings copied exactly from the "
    "provided review texts. Do NOT paraphrase or fabricate quotes.\n\n"
    "Respond ONLY with valid JSON matching this schema:\n"
    "{\n"
    "  \"theme_name\": \"<short descriptive label, ≤6 words>\",\n"
    "  \"summary\": \"<1-2 sentence summary of the theme>\",\n"
    "  \"quotes\": [\n"
    "    {\"text\": \"<exact verbatim text from a review>\", \"review_id\": \"<review_id>\"}\n"
    "  ],\n"
    "  \"action_ideas\": [\"<actionable improvement>\", ...]\n"
    "}"
)


def _build_user_message(cluster: "RawCluster", reviews: list[dict]) -> str:  # type: ignore[name-defined]
    """
    Build the user-turn message containing the cluster's review texts as data.

    Reviews are passed as structured JSON (never interpolated into the system
    prompt) to prevent prompt injection — per architecture §8.2.
    """
    from src.agent.clustering import RawCluster  # local import to avoid circular

    review_data = []
    for idx in cluster.review_indices[:_MAX_REVIEWS_PER_CALL]:
        r = reviews[idx]
        text = r.get("text", "")[:_MAX_CHARS_PER_REVIEW]
        review_data.append({
            "review_id": r.get("review_id", ""),
            "rating": r.get("rating", 0),
            "text": text,
        })

    payload = {
        "task": "Summarize the following app reviews into a single product theme.",
        "reviews": review_data,
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Rate-limit tracker (per-run, in-memory)
# ---------------------------------------------------------------------------

class _RateLimitTracker:
    """
    Tracks running totals of requests and tokens within a pipeline run.
    Raises RuntimeError when daily Groq limits would be exceeded.
    """

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        self.max_rpd: int = llm_cfg.get("requests_per_day", 1000)
        self.max_tpd: int = llm_cfg.get("tokens_per_day", 100_000)
        self.requests_used: int = 0
        self.tokens_used: int = 0
        self._last_call_time: float = 0.0

    def check_and_record(self, tokens_used: int) -> None:
        """Record a completed call. Raises RuntimeError if limits are exceeded."""
        self.requests_used += 1
        self.tokens_used += tokens_used

        if self.requests_used > self.max_rpd:
            raise RuntimeError(
                f"Groq daily request limit reached: {self.requests_used}/{self.max_rpd}. "
                "Aborting summarization to protect quota."
            )
        if self.tokens_used > self.max_tpd:
            raise RuntimeError(
                f"Groq daily token limit reached: {self.tokens_used}/{self.max_tpd}. "
                "Aborting summarization to protect quota."
            )

    def enforce_rpm_delay(self) -> None:
        """Sleep if needed to stay within the requests-per-minute limit."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            sleep_for = _MIN_CALL_INTERVAL - elapsed
            logger.debug("Rate-limit delay: sleeping %.2fs before next call.", sleep_for)
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def _call_llm_with_retry(
    system_prompt: str,
    user_message: str,
    tracker: _RateLimitTracker,
) -> dict:
    """
    Call the Groq LLM via litellm with up to 3 retries and exponential backoff.

    Args:
        system_prompt: The system-turn text.
        user_message:  The user-turn text (contains review data as JSON).
        tracker:       Rate-limit tracker (updated on success).

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        RuntimeError: If all retries are exhausted or daily limits exceeded.
    """
    try:
        import litellm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "litellm is required for LLM summarization. "
            "Install with: pip install litellm"
        ) from exc

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        tracker.enforce_rpm_delay()
        try:
            logger.debug("LLM call attempt %d/%d...", attempt, _MAX_RETRIES)
            response = litellm.completion(
                model=_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"},
                api_key=os.environ.get("GROQ_API_KEY"),
            )

            content = response.choices[0].message.content or ""
            usage = response.usage
            tokens = (usage.total_tokens if usage else 0)

            tracker.check_and_record(tokens)
            logger.debug(
                "LLM call succeeded (attempt %d). Tokens used: %d (total: %d).",
                attempt, tokens, tracker.tokens_used,
            )

            return json.loads(content)

        except RuntimeError:
            # Rate-limit abort — do not retry
            raise
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM returned invalid JSON (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, exc,
            )
            last_exc = exc
        except Exception as exc:
            delay = _BACKOFF_BASE ** attempt
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt, _MAX_RETRIES, delay, exc,
            )
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(delay)

    raise RuntimeError(
        f"LLM summarization failed after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(
    raw: dict,
    cluster: "RawCluster",  # type: ignore[name-defined]
    reviews: list[dict],
) -> Cluster:
    """
    Parse and validate the raw LLM JSON response into a Cluster object.

    Gracefully handles missing or malformed fields by falling back to safe defaults.
    """
    from src.agent.clustering import RawCluster  # local import

    theme_name: str = str(raw.get("theme_name", f"Theme {cluster.cluster_id}")).strip()
    summary: str = str(raw.get("summary", "")).strip()
    action_ideas: list[str] = [
        str(a) for a in raw.get("action_ideas", []) if isinstance(a, str)
    ]

    # Build a lookup from review_id → rating for quote enrichment
    rating_lookup: dict[str, int] = {
        reviews[i]["review_id"]: reviews[i].get("rating", 0)
        for i in cluster.review_indices
    }

    raw_quotes = raw.get("quotes", [])
    quotes: list[ValidatedQuote] = []
    for q in raw_quotes:
        if not isinstance(q, dict):
            continue
        text = str(q.get("text", "")).strip()
        review_id = str(q.get("review_id", "")).strip()
        if text and review_id:
            quotes.append(ValidatedQuote(
                text=text,
                review_id=review_id,
                rating=rating_lookup.get(review_id, 0),
            ))

    return Cluster(
        cluster_id=cluster.cluster_id,
        theme_name=theme_name,
        summary=summary,
        review_count=cluster.review_count,
        avg_rating=cluster.avg_rating,
        quotes=quotes,
        action_ideas=action_ideas,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_clusters(
    raw_clusters: list,  # list[RawCluster]
    reviews: list[dict],
    config: dict,
) -> tuple[list[Cluster], int]:
    """
    Summarize each RawCluster into a named, quoted Cluster using Groq LLM.

    For each cluster:
      1. Enforce RPM rate-limit delay.
      2. Build prompt with review texts as structured JSON data.
      3. Call ``groq/llama-3.3-70b-versatile`` via litellm (retry up to 3×).
      4. Parse JSON response → Cluster object.
      5. Track token and request usage; abort if daily limits exceeded.

    Args:
        raw_clusters: Output of ``cluster_reviews()``.
        reviews:      Full list of review dicts.
        config:       Full config dict (reads ``llm`` section for limits).

    Returns:
        Tuple of (list[Cluster], total_tokens_used).
        - Cluster objects are ready for quote validation.
        - total_tokens_used is the sum of all Groq API tokens consumed.

    Raises:
        RuntimeError: If daily Groq rate limits are exceeded or all retries fail.
    """
    tracker = _RateLimitTracker(config)
    clusters: list[Cluster] = []

    logger.info(
        "Summarizing %d clusters via %s...", len(raw_clusters), _MODEL
    )

    for i, raw_cluster in enumerate(raw_clusters):
        logger.info(
            "Summarizing cluster %d/%d (id=%d, %d reviews, avg_rating=%.1f)...",
            i + 1, len(raw_clusters),
            raw_cluster.cluster_id,
            raw_cluster.review_count,
            raw_cluster.avg_rating,
        )

        user_msg = _build_user_message(raw_cluster, reviews)

        try:
            raw_response = _call_llm_with_retry(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_msg,
                tracker=tracker,
            )
        except RuntimeError as exc:
            # Daily limit abort or total retry exhaustion — propagate up
            logger.error("Summarization aborted at cluster %d: %s", i + 1, exc)
            raise

        cluster = _parse_llm_response(raw_response, raw_cluster, reviews)
        clusters.append(cluster)
        logger.info(
            "Cluster %d summarized: theme='%s', quotes=%d, action_ideas=%d",
            i + 1, cluster.theme_name, len(cluster.quotes), len(cluster.action_ideas),
        )

    logger.info(
        "Summarization complete: %d clusters. "
        "Total Groq usage: %d requests, %d tokens.",
        len(clusters), tracker.requests_used, tracker.tokens_used,
    )
    return clusters, tracker.tokens_used

