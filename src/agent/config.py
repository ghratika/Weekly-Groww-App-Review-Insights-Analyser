"""
Configuration loader for the Weekly Product Review Pulse.

Reads config.yaml, performs environment variable substitution,
and validates all required fields.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Default config path relative to project root
DEFAULT_CONFIG_PATH = "config/config.yaml"

# Required top-level config keys
REQUIRED_SECTIONS = ["product", "mcp_servers", "delivery", "llm", "clustering"]

# Required keys within each section
REQUIRED_KEYS = {
    "product": ["name", "play_store_app_id", "review_window_weeks"],
    "mcp_servers": ["playstore_reviews"],
    "delivery": ["google_doc_id", "recipients", "email_mode", "email_subject_template"],
    "llm": [
        "provider",
        "model",
        "requests_per_minute",
        "requests_per_day",
        "tokens_per_minute",
        "tokens_per_day",
    ],
    "clustering": [
        "embedding_model",
        "umap_n_neighbors",
        "umap_n_components",
        "hdbscan_min_cluster_size",
        "max_themes",
    ],
}

# Regex pattern for ${ENV_VAR} substitution
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute ${ENV_VAR} references in config values
    with actual environment variable values.
    """
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                # Return the original placeholder if env var is not set
                # (will be caught during validation if the field is required)
                return match.group(0)
            return env_value

        return _ENV_VAR_PATTERN.sub(_replace, value)

    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]

    return value


def _validate_config(config: dict) -> list[str]:
    """
    Validate that all required sections and keys are present in the config.
    Returns a list of error messages (empty if valid).
    """
    errors = []

    for section in REQUIRED_SECTIONS:
        if section not in config:
            errors.append(f"Missing required config section: '{section}'")
            continue

        if section in REQUIRED_KEYS:
            for key in REQUIRED_KEYS[section]:
                if key not in config[section]:
                    errors.append(
                        f"Missing required key '{key}' in config section '{section}'"
                    )

    return errors


def load_config(config_path: str | None = None) -> dict:
    """
    Load and validate the configuration from a YAML file.

    Args:
        config_path: Path to the config.yaml file. If None, uses the default
                     path relative to the project root.

    Returns:
        Validated configuration dictionary with env vars substituted.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required config keys are missing.
    """
    # Load .env file for environment variable access
    load_dotenv()

    # Resolve config path
    if config_path is None:
        # Find project root (look for pyproject.toml or config/ directory)
        project_root = Path(__file__).resolve().parent.parent.parent
        resolved_path = project_root / DEFAULT_CONFIG_PATH
    else:
        resolved_path = Path(config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {resolved_path}\n"
            f"Copy config/config.example.yaml to config/config.yaml and fill in your values."
        )

    # Load YAML
    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {resolved_path}")

    # Substitute environment variables
    config = _substitute_env_vars(config)

    # Validate required keys
    errors = _validate_config(config)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)

    return config


def get_iso_week() -> str:
    """
    Get the current ISO week string (e.g., '2026-W23').
    """
    from datetime import date

    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"
