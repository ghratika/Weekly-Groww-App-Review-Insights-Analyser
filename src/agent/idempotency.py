"""
Idempotency and run logging (Phase 6).

Reads and writes the run_log.json for a given product and iso_week.
This ensures we don't re-run expensive operations or send duplicate emails
if a previous run succeeded or partially succeeded.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("pulse.idempotency")

def get_run_log_path(product: str, iso_week: str) -> Path:
    """Get the path to the run_log.json file for a given run."""
    # Assuming the root of the project is where we run from, or we can use a fixed path.
    # We will put it in runs/<product>/<iso_week>/run_log.json
    base_dir = Path(os.getcwd()) / "runs" / product / iso_week
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "run_log.json"

def check_run_log(product: str, iso_week: str) -> dict[str, Any] | None:
    """
    Check if a run log exists.
    
    Returns:
        The run log dict if found, otherwise None.
    """
    path = get_run_log_path(product, iso_week)
    if not path.exists():
        logger.debug("No run log found at %s", path)
        return None
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            log = json.load(f)
            logger.debug("Loaded existing run log from %s (status: %s)", path, log.get("status"))
            return log
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read run log at %s: %s", path, exc)
        return None

def write_run_log(log: dict[str, Any]) -> None:
    """
    Write or update the run log.
    
    Args:
        log: The run log dictionary. Must contain 'product' and 'iso_week'.
    """
    product = log.get("product")
    iso_week = log.get("iso_week")
    
    if not product or not iso_week:
        logger.error("Cannot write run log: missing 'product' or 'iso_week'")
        return
        
    path = get_run_log_path(product, iso_week)
    
    try:
        # Make sure the directory exists (in case it was deleted)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
            logger.debug("Wrote run log to %s", path)
    except OSError as exc:
        logger.error("Failed to write run log to %s: %s", path, exc)
