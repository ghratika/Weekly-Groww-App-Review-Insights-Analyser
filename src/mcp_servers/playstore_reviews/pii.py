"""
Server-side PII scrubbing (Layer 1).

Anonymizes author display names by replacing them with deterministic
`User_<hash>` pseudonyms before returning reviews to the agent.

Covers edge cases P1-01 through P1-04 in edge-cases.md.
"""

import hashlib
import logging

logger = logging.getLogger("pulse.pii_l1")

# Length of the hex hash suffix in the pseudonym (8 hex chars = 4.3B values)
_HASH_LENGTH = 8


def anonymize_author(name: str | None) -> str:
    """
    Replace an author display name with a deterministic pseudonym.

    The pseudonym is `User_<hash>` where `<hash>` is the first 8 hex characters
    of the SHA-256 digest of the original name (UTF-8 encoded).

    This is deterministic: the same input name always produces the same
    pseudonym, allowing consistency across runs.

    Args:
        name: The original author display name. May be None or empty.

    Returns:
        A pseudonym string like `User_a3f2b1c8`.

    Examples:
        >>> anonymize_author("John Doe")
        'User_a8cfcd74'
        >>> anonymize_author("John Doe")  # Same input → same output
        'User_a8cfcd74'
        >>> anonymize_author(None)
        'User_anon'
        >>> anonymize_author("")
        'User_anon'
    """
    # Edge case P1-04: null or empty author name
    if name is None or (isinstance(name, str) and name.strip() == ""):
        return "User_anon"

    # Edge case P1-03: non-ASCII / emoji characters are handled by UTF-8 encoding
    name_bytes = name.encode("utf-8")
    digest = hashlib.sha256(name_bytes).hexdigest()

    # Edge case P1-02: collision risk is ~1 in 4.3 billion with 8 hex chars
    pseudonym = f"User_{digest[:_HASH_LENGTH]}"

    return pseudonym


def anonymize_reviews(reviews: list[dict]) -> list[dict]:
    """
    Apply author anonymization to a list of review dicts.

    Modifies the `author` field in-place and returns the same list.

    Args:
        reviews: List of review dicts, each with an `author` field.

    Returns:
        The same list with `author` fields replaced by pseudonyms.
    """
    for review in reviews:
        original = review.get("author")
        review["author"] = anonymize_author(original)

    logger.info("Anonymized authors for %d reviews.", len(reviews))
    return reviews
