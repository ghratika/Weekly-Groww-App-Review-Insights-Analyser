"""
Verbatim quote validator (Phase 4).

Ensures that every quote extracted by the LLM summarizer is a genuine
verbatim substring of the review it claims to originate from.

This guards against LLM hallucinations: if the LLM paraphrases or fabricates
a quote, it won't survive verbatim matching and will be silently dropped
(with a debug log for observability).

Architecture references:
  - §4 — Analysis engine
  - §8.2 — Quote traceability guarantee
"""

from __future__ import annotations

import logging

logger = logging.getLogger("pulse.quote_validator")


def validate_quotes(
    clusters: list,   # list[Cluster]
    reviews: list[dict],
) -> list:            # list[Cluster]
    """
    Filter each cluster's quotes to only those that appear verbatim in their
    referenced source review.

    Algorithm:
      For each quote in each cluster:
        1. Look up the source review by ``review_id``.
        2. Check if ``quote.text`` is a substring of ``review.text``
           (case-sensitive, exact match).
        3. Keep the quote if it matches; discard and log if not.

    Args:
        clusters: List of ``Cluster`` objects from ``summarize_clusters()``.
        reviews:  Full list of review dicts (used as the source-of-truth).

    Returns:
        The same cluster list with non-verbatim quotes removed.
        Clusters that end up with 0 quotes are retained (the summary and
        action ideas are still valuable).
    """
    # Build lookup: review_id → review text for O(1) access
    review_text_by_id: dict[str, str] = {
        r["review_id"]: r.get("text", "")
        for r in reviews
        if "review_id" in r
    }

    total_quotes_before = sum(len(c.quotes) for c in clusters)
    total_discarded = 0

    for cluster in clusters:
        validated = []
        for quote in cluster.quotes:
            source_text = review_text_by_id.get(quote.review_id)

            if source_text is None:
                logger.debug(
                    "Discarding quote from unknown review_id '%s' "
                    "(cluster %d, theme='%s'): \"%s\"",
                    quote.review_id, cluster.cluster_id, cluster.theme_name,
                    quote.text[:80],
                )
                total_discarded += 1
                continue

            if quote.text in source_text:
                validated.append(quote)
            else:
                logger.debug(
                    "Discarding fabricated/paraphrased quote "
                    "(cluster %d, theme='%s', review=%s): \"%s\"",
                    cluster.cluster_id, cluster.theme_name,
                    quote.review_id, quote.text[:80],
                )
                total_discarded += 1

        cluster.quotes = validated

    total_quotes_after = sum(len(c.quotes) for c in clusters)
    logger.info(
        "Quote validation complete: %d/%d quotes passed "
        "(%d discarded as fabrications or unmatched).",
        total_quotes_after,
        total_quotes_before,
        total_discarded,
    )
    return clusters
