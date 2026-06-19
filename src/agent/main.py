"""
CLI entrypoint and orchestrator for the Weekly Product Review Pulse.

Phase 7 — Fully hardened end-to-end pipeline:
  config → idempotency check → ingest → PII scrub → embed → cluster →
  summarize → validate quotes → render doc → deliver doc → render email →
  deliver email → write run log

Usage:
    python -m src.agent.main --product groww --week 2026-W23
    python -m src.agent.main --product groww --dry-run
    python -m src.agent.main --help
"""

import logging
import re
import sys
from datetime import date, datetime, timezone

import click

from src.agent.clustering import embed_reviews, cluster_reviews
from src.agent.config import load_config, get_iso_week
from src.agent.ingestion import fetch_reviews_via_mcp
from src.agent.quote_validator import validate_quotes
from src.agent.summarizer import summarize_clusters
from src.agent.doc_renderer import render_doc_section
from src.agent.email_renderer import render_email
from src.agent.delivery import deliver_doc, deliver_email
from src.agent.idempotency import check_run_log, write_run_log


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure structured logging with ISO timestamps to stderr."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )
    return logging.getLogger("pulse")


# ---------------------------------------------------------------------------
# ISO week validation
# ---------------------------------------------------------------------------

_ISO_WEEK_PATTERN = re.compile(r"^\d{4}-W(?:0[1-9]|[1-4]\d|5[0-3])$")


def _validate_iso_week(ctx: click.Context, param: click.Parameter, value: str | None) -> str:
    """Validate and default the ISO week parameter."""
    if value is None:
        return get_iso_week()

    if not _ISO_WEEK_PATTERN.match(value):
        raise click.BadParameter(
            f"Invalid ISO week format: '{value}'. Expected format: YYYY-Wnn (e.g., 2026-W23)"
        )
    return value


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@click.command(
    name="pulse",
    help="Weekly Product Review Pulse — automated insight reports from Google Play reviews.",
)
@click.option(
    "--product",
    default="groww",
    show_default=True,
    help="Product name (must match config.yaml product.name, case-insensitive).",
)
@click.option(
    "--week",
    default=None,
    callback=_validate_iso_week,
    help="ISO week to analyze (e.g., 2026-W23). Defaults to the current week.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to config.yaml. Defaults to config/config.yaml.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run analysis but skip MCP delivery (no Doc append, no email).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def cli(product: str, week: str, config_path: str | None, dry_run: bool, verbose: bool) -> None:
    """
    Main CLI entrypoint. Orchestrates the full 14-step pipeline.
    Partial failures are logged and the run log is written before exiting.
    Re-running after a partial failure resumes from the last incomplete step.
    """
    logger = _setup_logging(verbose)

    logger.info("=" * 60)
    logger.info("Weekly Product Review Pulse")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load configuration
    # ------------------------------------------------------------------
    logger.info("[1/14] Loading configuration...")
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    product_name = config["product"]["name"]
    app_id = config["product"]["play_store_app_id"]
    review_window = config["product"]["review_window_weeks"]

    logger.info("  Product       : %s (%s)", product_name, app_id)
    logger.info("  ISO Week      : %s", week)
    logger.info("  Review Window : %d weeks", review_window)
    logger.info("  LLM           : %s / %s", config["llm"]["provider"], config["llm"]["model"])
    logger.info("  Email Mode    : %s", config["delivery"]["email_mode"])
    logger.info("  Dry Run       : %s", dry_run)
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # Step 2: Idempotency check
    # ------------------------------------------------------------------
    logger.info("[2/14] Checking run log for idempotency...")
    run_log = check_run_log(product_name, week)
    if run_log and run_log.get("status") == "success":
        logger.info(
            "Run for %s %s already completed successfully. Nothing to do.",
            product_name, week,
        )
        sys.exit(0)
    elif run_log:
        logger.info("Found partial run log (status=%s). Will skip already-completed steps.", run_log.get("status"))
    else:
        logger.info("No existing run log found. Starting fresh.")
        run_log = {
            "product": product_name,
            "iso_week": week,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "review_window": {},
            "reviews_fetched": 0,
            "clusters_found": 0,
            "delivery": {},
            "llm": {
                "provider": config["llm"]["provider"],
                "model": config["llm"]["model"],
                "tokens_used": 0,
            },
            "status": "partial",
            "errors": [],
        }

    # Check if we can skip steps 3-9 by loading cached payloads from the run log
    _cached_doc_payload = run_log.get("_cached_doc_payload")
    _cached_email_metadata = run_log.get("_cached_email_metadata")
    _can_skip_analysis = bool(_cached_doc_payload and _cached_email_metadata)

    # Initialise these so the finalise step can always reference them safely,
    # even on the skip-analysis (partial-resume) path.
    reviews: list = []
    clusters: list = []

    # ------------------------------------------------------------------
    # Steps 3–9: Analysis pipeline (skipped if resuming with cached payloads)
    # ------------------------------------------------------------------
    if _can_skip_analysis:
        logger.info("[3-9/14] Resuming partial run — skipping analysis (using cached payloads).")
        doc_payload = _cached_doc_payload
        metadata = _cached_email_metadata
    else:
        # ------------------------------------------------------------------
        # Step 3 + 4: Fetch reviews via Play Store Reviews MCP & Layer 2 PII
        # ------------------------------------------------------------------
        logger.info("[3/14] Fetching reviews via Play Store Reviews MCP server...")
        try:
            reviews = fetch_reviews_via_mcp(config)
        except RuntimeError as exc:
            # 0 reviews returned — abort cleanly; don't write an empty report
            logger.error("ABORT: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)
        except (ConnectionError, ValueError) as exc:
            logger.error("FAILED — could not fetch reviews: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)

        logger.info("[3+4/14] %d reviews fetched and Layer-2 PII-scrubbed.", len(reviews))
        run_log["reviews_fetched"] = len(reviews)

        # Compute review window dates
        from datetime import timedelta
        today = date.today()
        window_start = today - timedelta(weeks=review_window)
        run_log["review_window"] = {
            "start": window_start.isoformat(),
            "end": today.isoformat(),
        }

        # ------------------------------------------------------------------
        # Step 5: Generate embeddings
        # ------------------------------------------------------------------
        logger.info("[5/14] Generating review embeddings...")
        embedding_model = config["clustering"].get("embedding_model", "BAAI/bge-small-en-v1.5")
        try:
            embeddings = embed_reviews(reviews, model_name=embedding_model)
        except ImportError as exc:
            logger.error("FAILED — missing embedding dependency: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)
        except ValueError as exc:
            logger.error("FAILED — embedding error: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)
        logger.info("[5/14] Embeddings shape: %s", embeddings.shape)

        # ------------------------------------------------------------------
        # Step 6: Cluster reviews
        # ------------------------------------------------------------------
        logger.info("[6/14] Clustering reviews with UMAP + HDBSCAN...")
        try:
            raw_clusters = cluster_reviews(embeddings, reviews, config)
        except RuntimeError as exc:
            # 0 clusters produced — abort
            logger.error("ABORT: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)
        except ImportError as exc:
            logger.error("FAILED — missing clustering dependency: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)
        logger.info("[6/14] %d clusters found.", len(raw_clusters))
        run_log["clusters_found"] = len(raw_clusters)

        # ------------------------------------------------------------------
        # Step 7: LLM summarization
        # ------------------------------------------------------------------
        logger.info("[7/14] Summarizing clusters via Groq LLM...")
        try:
            clusters, tokens_used = summarize_clusters(raw_clusters, reviews, config)
        except RuntimeError as exc:
            # Rate-limit exceeded or all retries exhausted
            logger.error("ABORT: %s", exc)
            run_log["errors"].append(str(exc))
            run_log["status"] = "failed"
            write_run_log(run_log)
            sys.exit(1)
        # Wire actual token usage into the run log
        run_log["llm"]["tokens_used"] = tokens_used
        logger.info("[7/14] %d clusters summarized. Tokens used: %d.", len(clusters), tokens_used)

        # ------------------------------------------------------------------
        # Step 8: Quote validation
        # ------------------------------------------------------------------
        logger.info("[8/14] Validating quotes against source reviews...")
        clusters = validate_quotes(clusters, reviews)
        total_quotes = sum(len(c.quotes) for c in clusters)
        logger.info("[8/14] %d verified quotes across %d clusters.", total_quotes, len(clusters))

        # ------------------------------------------------------------------
        # Step 9: Render Google Doc section
        # ------------------------------------------------------------------
        logger.info("[9/14] Rendering Google Doc section payload...")
        metadata = {
            "product_name": product_name,
            "iso_week": week,
            "review_count": len(reviews),
            "review_window_weeks": review_window,
            "review_window": run_log["review_window"],
        }
        doc_payload = render_doc_section(clusters, metadata)
        logger.info("[9/14] Doc payload generated.")

        # Cache the rendered payloads in the run log so a retry can skip steps 3-9
        run_log["_cached_doc_payload"] = doc_payload
        run_log["_cached_email_metadata"] = metadata
        write_run_log(run_log)

    # ------------------------------------------------------------------
    # Step 10: Deliver Doc via MCP
    # ------------------------------------------------------------------
    doc_heading_id = run_log.get("delivery", {}).get("doc_heading_id")

    if doc_heading_id:
        logger.info("[10/14] Doc already delivered in a previous run (heading=%s). Skipping.", doc_heading_id)
    elif dry_run:
        logger.info("[10/14] Deliver Doc — SKIPPED (dry-run mode).")
        doc_heading_id = f"heading={product_name} — {week}"
    else:
        logger.info("[10/14] Delivering Google Doc section via MCP...")
        try:
            doc_heading_id = deliver_doc(doc_payload, metadata, config)
            run_log["delivery"]["doc_heading_id"] = doc_heading_id
            # Save progress immediately after doc delivery
            write_run_log(run_log)
            logger.info("[10/14] Doc section delivered (heading=%s).", doc_heading_id)
        except Exception as exc:
            logger.error("[10/14] FAILED — Doc delivery error: %s", exc)
            logger.warning("Setting status=partial. Email step will be skipped.")
            run_log["errors"].append(f"Doc delivery failed: {exc}")
            run_log["status"] = "partial"
            write_run_log(run_log)
            sys.exit(1)

    # Update metadata with heading ID for the email deep link
    metadata["doc_heading_id"] = doc_heading_id

    # ------------------------------------------------------------------
    # Step 11: Render email (after doc delivery to embed real heading ID)
    # ------------------------------------------------------------------
    logger.info("[11/14] Rendering email payload...")
    email_payload = render_email(clusters, metadata, config)
    logger.info("[11/14] Email HTML and plain-text generated.")

    # ------------------------------------------------------------------
    # Step 12: Deliver email via MCP
    # ------------------------------------------------------------------
    existing_gmail_id = run_log.get("delivery", {}).get("gmail_message_id")

    if existing_gmail_id:
        logger.info("[12/14] Email already delivered in a previous run (id=%s). Skipping.", existing_gmail_id)
    elif dry_run:
        logger.info("[12/14] Deliver email — SKIPPED (dry-run mode).")
        run_log["status"] = "success"
    else:
        logger.info("[12/14] Delivering email via MCP...")
        try:
            gmail_msg_id = deliver_email(email_payload, config, run_log)
            if gmail_msg_id:
                run_log["delivery"]["gmail_message_id"] = gmail_msg_id
            run_log["status"] = "success"
            logger.info("[12/14] Email delivered/drafted (id=%s).", gmail_msg_id)
        except Exception as exc:
            logger.error("[12/14] FAILED — Email delivery error: %s", exc)
            logger.warning("Doc was already delivered. Setting status=partial.")
            run_log["errors"].append(f"Email delivery failed: {exc}")
            run_log["status"] = "partial"
            write_run_log(run_log)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 13: Finalize and write run log
    # ------------------------------------------------------------------
    logger.info("[13/14] Finalising run log...")
    # Only overwrite these counts when we actually ran analysis (not on a cached resume)
    if not _can_skip_analysis:
        run_log["reviews_fetched"] = len(reviews)
        run_log["clusters_found"] = len(clusters)
    run_log["delivery"]["doc_id"] = config.get("delivery", {}).get("google_doc_id")
    run_log["delivery"]["gmail_mode"] = config.get("delivery", {}).get("email_mode")
    write_run_log(run_log)
    logger.info("[13/14] Run log written.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    if run_log["status"] == "success":
        logger.info("[14/14] Pipeline complete. Status: SUCCESS ✓")
    else:
        logger.info("[14/14] Pipeline complete. Status: %s", run_log["status"].upper())
    logger.info("=" * 60)


# Allow running as: python -m src.agent.main
if __name__ == "__main__":
    cli()
