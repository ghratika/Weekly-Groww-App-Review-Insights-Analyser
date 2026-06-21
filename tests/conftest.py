"""
tests/conftest.py — shared pytest configuration and fixtures.

Key safety fixture
------------------
``_block_live_llm_calls`` (session-scoped, autouse):

  Removes GROQ_API_KEY from the environment for the entire test session.
  This is a defence-in-depth measure: even if the key accidentally leaks
  into the CI environment (e.g., future workflow changes), unit tests will
  never be able to make live LLM calls and hit provider rate limits.

  All LLM interactions in unit tests are expected to be mocked via
  ``patch.dict("sys.modules", {"litellm": <MagicMock>})``.
  If a test genuinely needs a live key (integration tests), it should be
  tagged with a custom marker (e.g. ``@pytest.mark.integration``) and run
  in a separate CI job that explicitly sets the key.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _block_live_llm_calls():
    """
    Ensure no real LLM provider key is available during the unit-test session.

    Removes GROQ_API_KEY (and any other provider keys) from ``os.environ``
    at the start of the session and restores them when the session ends.
    This prevents accidental live calls even if the key is injected into
    the CI environment.
    """
    _KEYS_TO_CLEAR = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    saved = {}
    for key in _KEYS_TO_CLEAR:
        if key in os.environ:
            saved[key] = os.environ.pop(key)

    yield  # ← run the entire test session

    # Restore after the session (good practice for local dev runs)
    os.environ.update(saved)
