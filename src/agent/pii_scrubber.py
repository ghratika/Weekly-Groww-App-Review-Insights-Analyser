"""
Agent-side PII scrubbing (Layer 2).

Applies text-level PII redaction to review content before any LLM processing.
This is the second PII layer; Layer 1 (author name anonymization) is handled
server-side in src/mcp_servers/playstore_reviews/pii.py.

Scrubbing strategy (per architecture §8.1):
  1. Regex patterns for high-recall, low-latency detection of emails and
     Indian/international phone numbers.
  2. Presidio Analyzer NER for PERSON, PHONE_NUMBER, EMAIL_ADDRESS entities
     that regex may miss (e.g., names embedded in free text).

All detected PII is replaced with the ``[REDACTED]`` placeholder.
"""

import logging
import re
from typing import Any

logger = logging.getLogger("pulse.pii_l2")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Email addresses: anything that looks like user@domain.tld
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Phone numbers — covers:
#   Indian mobile: +91-XXXXX-XXXXX, 91XXXXXXXXXX, 0XXXXXXXXXX, XXXXXXXXXX (10 digits)
#   Generic international: +1 (555) 123-4567, +44 20 7946 0958, etc.
_PHONE_RE = re.compile(
    r"""
    (?:
        # Indian formats
        (?:\+?91[\s\-]?)?                  # optional +91 country code
        [6-9]\d{9}                         # 10-digit mobile starting 6-9
    |
        # Generic international with country code
        \+\d{1,3}[\s\-.]?                  # +CC
        (?:\(?\d{1,4}\)?[\s\-.]?){2,5}    # area + number groups
        \d{2,4}                            # final digits
    )
    """,
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Presidio setup (lazy-loaded to avoid import cost when not needed)
# ---------------------------------------------------------------------------

_presidio_analyzer = None
_presidio_available = False


def _get_presidio_analyzer():
    """
    Lazy-load the Presidio AnalyzerEngine.

    Returns None if Presidio is not installed (graceful degradation —
    regex-only mode is still applied).
    """
    global _presidio_analyzer, _presidio_available

    if _presidio_analyzer is not None:
        return _presidio_analyzer

    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        _presidio_analyzer = AnalyzerEngine()
        _presidio_available = True
        logger.info("Presidio AnalyzerEngine loaded successfully.")
    except ImportError:
        logger.warning(
            "presidio-analyzer not installed. "
            "Layer 2 PII scrubbing will use regex only (no NER). "
            "Install with: pip install presidio-analyzer"
        )
        _presidio_available = False

    return _presidio_analyzer


# ---------------------------------------------------------------------------
# Core scrubbing logic
# ---------------------------------------------------------------------------

_PRESIDIO_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]
_REDACTED = "[REDACTED]"


def scrub_text(text: str) -> str:
    """
    Scrub PII from a single review text string.

    Applies two passes:
      Pass 1 — Regex: fast, high-recall detection of emails and phone numbers.
      Pass 2 — Presidio NER: entity-level detection of PERSON, PHONE_NUMBER,
               EMAIL_ADDRESS (skipped if Presidio is not installed).

    Args:
        text: Raw review text.

    Returns:
        Text with all detected PII replaced by ``[REDACTED]``.
    """
    if not text or not isinstance(text, str):
        return text

    # --- Pass 1: Regex substitution ---
    scrubbed = _EMAIL_RE.sub(_REDACTED, text)
    scrubbed = _PHONE_RE.sub(_REDACTED, scrubbed)

    # --- Pass 2: Presidio NER ---
    analyzer = _get_presidio_analyzer()
    if analyzer is not None:
        try:
            results = analyzer.analyze(
                text=scrubbed,
                entities=_PRESIDIO_ENTITIES,
                language="en",
            )
            # Sort by start position descending so replacements don't shift offsets
            results_sorted = sorted(results, key=lambda r: r.start, reverse=True)
            scrubbed_chars = list(scrubbed)
            for result in results_sorted:
                scrubbed_chars[result.start : result.end] = list(_REDACTED)
            scrubbed = "".join(scrubbed_chars)
        except Exception as exc:
            # Presidio failures must not crash the pipeline — log and continue
            logger.warning("Presidio analysis failed, skipping NER pass: %s", exc)

    return scrubbed


def scrub_reviews(reviews: list[dict]) -> list[dict]:
    """
    Apply Layer 2 PII scrubbing to a list of review dicts.

    Scrubs the ``text`` field of each review in-place and returns the same list.

    Args:
        reviews: List of review dicts, each containing a ``text`` field.

    Returns:
        The same list with all ``text`` fields PII-scrubbed.
    """
    redacted_count = 0
    for review in reviews:
        original_text = review.get("text", "")
        scrubbed_text = scrub_text(original_text)
        if scrubbed_text != original_text:
            redacted_count += 1
        review["text"] = scrubbed_text

    logger.info(
        "Layer 2 PII scrub complete: %d/%d reviews had PII redacted.",
        redacted_count,
        len(reviews),
    )
    return reviews
