"""
Unit tests for idempotency.py
"""

import json
from pathlib import Path
from src.agent.idempotency import check_run_log, write_run_log, get_run_log_path

def test_idempotency_flow(tmp_path, monkeypatch):
    # Mock os.getcwd to use tmp_path
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    
    product = "groww"
    iso_week = "2026-W23"
    
    # 1. Fresh run
    log = check_run_log(product, iso_week)
    assert log is None
    
    # 2. Write partial run log
    partial_log = {
        "product": product,
        "iso_week": iso_week,
        "status": "partial",
        "delivery": {
            "doc_heading_id": "heading=h.abc"
        }
    }
    write_run_log(partial_log)
    
    # Verify file exists
    log_path = get_run_log_path(product, iso_week)
    assert log_path.exists()
    
    # 3. Check partial log
    read_log = check_run_log(product, iso_week)
    assert read_log is not None
    assert read_log["status"] == "partial"
    assert read_log["delivery"]["doc_heading_id"] == "heading=h.abc"
