# Weekly Product Review Pulse — Edge Cases & Corner Cases

> Derived from [architecture.md](file:///c:/Users/Lavanya%20gupta/OneDrive/Documents/Ghratika/Playstore/docs/architecture.md) · [problemStatement.md](file:///c:/Users/Lavanya%20gupta/OneDrive/Documents/Ghratika/Playstore/docs/problemStatement.md) · [implementation_plan.md](file:///c:/Users/Lavanya%20gupta/OneDrive/Documents/Ghratika/Playstore/docs/implementation_plan.md)
>
> This document catalogs every identified corner case, boundary condition, and failure mode across the full pipeline — grouped by component — along with expected system behavior and recommended mitigations.

---

## Table of Contents

1. [Configuration & Environment](#1-configuration--environment)
2. [Play Store Reviews MCP Server (Scraper)](#2-play-store-reviews-mcp-server-scraper)
3. [Review Ingestion (MCP Client)](#3-review-ingestion-mcp-client)
4. [PII Scrubbing — Layer 1 (MCP Server)](#4-pii-scrubbing--layer-1-mcp-server)
5. [PII Scrubbing — Layer 2 (Agent-Side)](#5-pii-scrubbing--layer-2-agent-side)
6. [Embedding Generation](#6-embedding-generation)
7. [Clustering (UMAP + HDBSCAN)](#7-clustering-umap--hdbscan)
8. [LLM Summarization](#8-llm-summarization)
9. [Quote Validation](#9-quote-validation)
10. [Report Rendering (Google Doc)](#10-report-rendering-google-doc)
11. [Report Rendering (Email)](#11-report-rendering-email)
12. [Google Docs MCP Delivery](#12-google-docs-mcp-delivery)
13. [Gmail MCP Delivery](#13-gmail-mcp-delivery)
14. [Idempotency & Run Logging](#14-idempotency--run-logging)
15. [Cost & Token Limits](#15-cost--token-limits)
16. [ISO Week & Date Handling](#16-iso-week--date-handling)
17. [Security Edge Cases](#17-security-edge-cases)
18. [System-Level Edge Cases](#18-system-level-edge-cases)

---

## 1. Configuration & Environment

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| C-01 | `config.yaml` is missing | Raise `FileNotFoundError` with instructions to copy `config.example.yaml` | `load_config()` checks file existence before parsing |
| C-02 | `config.yaml` is empty / all-whitespace | Raise `ValueError: Configuration file is empty` | Explicit `None` check after `yaml.safe_load()` |
| C-03 | `config.yaml` has invalid YAML syntax | Raise `yaml.YAMLError` with line/column info | Let PyYAML's exception propagate with a wrapped user-friendly message |
| C-04 | `${ENV_VAR}` reference in config but env var not set in `.env` | Placeholder remains as-is (e.g., `${GOOGLE_CREDENTIALS_PATH}`) | Log a warning; fail loudly only if the field is actually used at runtime |
| C-05 | `.env` file is missing entirely | `python-dotenv` silently no-ops; env vars from shell still apply | Document that `.env` is optional if vars are set in the shell environment |
| C-06 | `review_window_weeks` set to `0` or negative | Validation should reject; default to minimum of `1` week | Add range validation: `1 ≤ review_window_weeks ≤ 52` |
| C-07 | `max_tokens_per_run` or `cost_limit_usd` set to `0` | Would immediately abort every LLM call | Validate `> 0`; raise `ValueError` on zero/negative values |
| C-08 | `email_mode` set to an unrecognized value (not `"draft"` or `"sent"`) | Default to `"draft"` (safe fallback) with a warning log | Enum validation: only `{"draft", "sent"}` accepted |
| C-09 | `recipients` list is empty | Skip email delivery, log a warning | Warn at config-load time; don't abort the entire run |
| C-10 | `google_doc_id` is a placeholder value (e.g., still set to example ID) | Doc delivery fails at MCP call time | Warn if value matches known placeholder patterns |

---

## 2. Play Store Reviews MCP Server (Scraper)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| S-01 | App ID does not exist on Google Play | `google-play-scraper` raises a `404`-equivalent error | Catch and convert to a clear `AppNotFoundError`; abort the run |
| S-02 | App has zero reviews | Return empty `Review[]`; agent aborts (per architecture §10) | Return `[]`; agent checks and logs error |
| S-03 | App has fewer reviews than `hdbscan_min_cluster_size` | Clustering may produce 0 clusters | Log warning in agent; HDBSCAN edge case handled in Phase 4 |
| S-04 | Google Play rate-limits or blocks scraping | `google-play-scraper` raises HTTP 429 or connection error | Retry with exponential backoff (up to 3×); then abort |
| S-05 | Scraper returns reviews with `null`/missing fields | Partial `Review` objects break downstream steps | Validate each review object on ingestion; skip malformed entries, log count |
| S-06 | All returned reviews are outside the `weeks` window | Date filtering produces empty list → same as zero-reviews | Treat as S-02; abort with descriptive message |
| S-07 | Review `date` field is in an unexpected format | Date parsing fails | Normalize dates at scrape time; skip reviews with unparseable dates |
| S-08 | Review `text` is `null` or empty string | Embedding step fails on empty text | Filter out reviews with blank `text` before embedding; log count |
| S-09 | Review `text` is extremely long (e.g., 50,000 chars) | May exceed embedding model token limits | Truncate at a safe maximum (e.g., 512 tokens) before embedding |
| S-10 | Scraper returns duplicate `review_id`s | Downstream deduplication might fail | Deduplicate by `review_id` at ingestion; keep the first occurrence |
| S-11 | Google Play UI changes break the scraper | `google-play-scraper` returns garbage or raises | Pin library version; add integration test with real app on CI |
| S-12 | Network is unavailable | Connection error / timeout | Catch `requests.exceptions.ConnectionError`; retry then abort |

---

## 3. Review Ingestion (MCP Client)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| I-01 | Play Store Reviews MCP server fails to start | `subprocess` for stdio transport fails | Catch process startup errors; log the stderr output and abort |
| I-02 | MCP server starts but tool call times out | No response received within timeout window | Set a reasonable timeout (e.g., 60s); retry once; then abort |
| I-03 | MCP server returns an error response (not a tool result) | Agent receives an MCP error object | Parse MCP error message; surface it clearly in the run log |
| I-04 | MCP tool response is not valid JSON | Deserialization fails | Wrap JSON parse in try/except; abort with parse error |
| I-05 | MCP server returns 0 reviews (empty list) | Abort run; do not append empty section to Doc (per §10) | Explicit `len(reviews) == 0` check; write `status: "failed"` to run log |
| I-06 | Agent cannot find the MCP server binary/command | `FileNotFoundError` for the subprocess command | Validate MCP server command exists before calling; provide setup instructions |

---

## 4. PII Scrubbing — Layer 1 (MCP Server)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| P1-01 | Author name is already anonymous (e.g., "A Google user") | Hash is still applied; result is a unique `User_<hash>` | No special handling needed; hashing is unconditional |
| P1-02 | Two different real names produce the same short hash (collision) | Different usernames map to the same pseudonym | Use sufficient hash length (e.g., 8 hex chars = 4.3 billion values); log collisions |
| P1-03 | Author name contains non-ASCII / emoji characters | SHA-256 encodes UTF-8 bytes, so this is safe | Encode name as UTF-8 before hashing |
| P1-04 | Author name is `null` / empty string | Hash of empty string is deterministic; no crash | Handle `None` → fallback to `"User_anon"` |

---

## 5. PII Scrubbing — Layer 2 (Agent-Side)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| P2-01 | Review text contains no PII | Text returned unchanged | No issue; regex/NER finds nothing to redact |
| P2-02 | Review text is in a non-English language (Hindi, Tamil, etc.) | Presidio NER models are primarily English-trained | Log a warning for non-`en` language reviews; apply regex-only scrub |
| P2-03 | PII spans a sentence boundary (e.g., name split across lines) | Regex/NER may miss it | Normalize whitespace before scrubbing |
| P2-04 | Regex false positive redacts non-PII (e.g., "support@groww" in app name) | Over-redaction reduces report quality | Tune regex patterns; add allowlist for known app-domain terms |
| P2-05 | Presidio `analyzer` model fails to load | Import error or model download failure | Catch import errors; fall back to regex-only mode with warning |
| P2-06 | Review text contains only PII (e.g., "Call me at 9876543210") | After scrub, text is `"Call me at [REDACTED]"` | Such reviews may produce low-quality embeddings; acceptable trade-off |
| P2-07 | Very high volume of reviews makes NER slow | Scrubbing becomes a pipeline bottleneck | Batch Presidio calls; consider parallelism if >5,000 reviews |

---

## 6. Embedding Generation

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| E-01 | `sentence-transformers` model not downloaded yet | First run triggers a model download | Ensure network access; add a pre-flight model check with clear error |
| E-02 | Review text is empty after PII scrubbing | Embedding model returns a zero vector or fails | Filter out empty-text reviews before embedding; log count |
| E-03 | Review text is a single character or punctuation | Produces a low-quality embedding | No special handling; HDBSCAN may classify it as noise |
| E-04 | OpenAI Embeddings API is selected but `OPENAI_API_KEY` is missing | API call fails with `AuthenticationError` | Validate API key presence before calling; abort with clear message |
| E-05 | OpenAI Embeddings API rate limit hit | `RateLimitError` exception | Retry with exponential backoff; log the delay |
| E-06 | Embedding model returns `NaN` or `Inf` values | UMAP/HDBSCAN produce undefined results | Validate embedding array; replace or drop invalid rows |
| E-07 | Very large review corpus (>10,000 reviews) | Embedding generation is slow / memory-intensive | Batch in chunks of 500; log progress |

---

## 7. Clustering (UMAP + HDBSCAN)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| CL-01 | HDBSCAN produces **0 clusters** (all noise) | Abort run; log warning (per architecture §10) | Explicit check; try relaxing `hdbscan_min_cluster_size` once before aborting |
| CL-02 | HDBSCAN produces **1 cluster** (all reviews in one theme) | Valid output; produce a single-theme report | No special handling; single cluster is fine |
| CL-03 | Number of reviews < `hdbscan_min_cluster_size` | HDBSCAN treats all points as noise → 0 clusters | Detect early; log: "Too few reviews to cluster" |
| CL-04 | All reviews have identical or near-identical embeddings | UMAP collapses them; HDBSCAN sees one dense blob | Produces 1 cluster; acceptable |
| CL-05 | UMAP `n_components` ≥ number of reviews | UMAP raises a `ValueError` | Clamp `n_components` to `min(config_value, n_reviews - 1)` |
| CL-06 | UMAP `n_neighbors` ≥ number of reviews | UMAP raises a `ValueError` | Clamp `n_neighbors` to `min(config_value, n_reviews - 1)` |
| CL-07 | Produced clusters > `max_themes` | Truncate to top `max_themes` by review count | Sort clusters by size descending; take first `max_themes` |
| CL-08 | Cluster IDs are non-contiguous (HDBSCAN assigns -1 for noise) | Noise points (label = -1) are ignored | Explicitly filter `cluster_id != -1` |
| CL-09 | Memory error during UMAP on very large corpus | `MemoryError` | Log clearly; suggest reducing `review_window_weeks` or increasing system memory |

---

## 8. LLM Summarization

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| L-01 | LLM API call fails (network error, 5xx) | Retry up to 3× with exponential backoff; then abort (per §10) | `tenacity` or manual retry loop with backoff |
| L-02 | LLM returns malformed JSON (not parseable) | JSON parse fails | Retry the call once; if still malformed, skip the cluster and log |
| L-03 | LLM response is missing a required field (e.g., no `theme_name`) | Downstream steps receive incomplete `Cluster` object | Validate LLM response schema; fill missing fields with safe defaults |
| L-04 | LLM generates 0 quotes for a cluster | `quotes: []` in the cluster | Acceptable; report section renders without quotes section |
| L-05 | LLM generates more quotes than needed | Excess quotes pass validation | Cap at a maximum (e.g., 3 per cluster) to keep reports concise |
| L-06 | LLM generates 0 action ideas | `action_ideas: []` | Acceptable; action ideas section is omitted from the report |
| L-07 | LLM call exceeds `max_tokens_per_run` mid-run | Abort remaining LLM calls; produce partial report with completed clusters | Track running token count; check before each cluster call |
| L-08 | LLM estimated cost exceeds `cost_limit_usd` | Same as L-07: abort remaining calls | Track cumulative cost; check before each cluster call |
| L-09 | LLM `context window` is too small for a large cluster | Cluster's reviews exceed the model's context limit | Truncate cluster reviews to fit; log how many were truncated |
| L-10 | LLM prompt injection in review text | Malicious review tries to hijack agent behavior | Reviews are passed as structured **data** in user message, never interpolated into system prompt |
| L-11 | LLM provider API key is invalid or expired | `AuthenticationError` on first call | Validate API key at startup; fail fast with clear error |
| L-12 | LLM model name in config is unavailable / deprecated | API returns model-not-found error | Catch model errors specifically; suggest updating config |
| L-13 | All LLM retries exhausted across all clusters | All clusters fail summarization | Abort run; write `status: "failed"` with detailed error log |

---

## 9. Quote Validation

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| Q-01 | LLM returns a **fabricated quote** (not in any review) | Quote is discarded | Substring search across all reviews; discard if not found |
| Q-02 | LLM returns a **paraphrased quote** (semantically similar but not verbatim) | Not found as substring → discarded | Substring match is strict (verbatim only); paraphrases are discarded |
| Q-03 | Quote appears in a **different cluster's** review (cross-cluster) | Still a real quote → allowed | `review_id` is stored for traceability; cross-cluster quotes are valid |
| Q-04 | Quote is an empty string | Trivially matches everywhere | Filter: discard quotes shorter than a minimum length (e.g., 10 chars) |
| Q-05 | All quotes in a cluster are discarded (all fabricated) | Cluster is rendered with 0 quotes | Acceptable; log how many quotes were discarded per cluster |
| Q-06 | Quote contains special regex characters (if regex matching is used) | Regex may fail or produce wrong matches | Use `str.find()` or `in` operator (plain substring), not regex |
| Q-07 | Quote matching is case-sensitive, but LLM changed capitalization | Quote fails validation despite being real | Try case-insensitive match as a fallback before discarding |
| Q-08 | `review_id` in the quote does not exist in the review list | Traceability is broken | Validate `review_id` exists; if not, search all reviews for the quote text |

---

## 10. Report Rendering (Google Doc)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| R-01 | `clusters` list is empty (all clusters aborted) | Do not append an empty section; abort the run | Guard: if `len(clusters) == 0` before rendering, abort |
| R-02 | A theme name contains Google Docs-unsupported characters | `batchUpdate` may fail or render incorrectly | Sanitize theme names: strip control characters and BIDI overrides |
| R-03 | Section heading is extremely long | Heading anchor may be truncated by Docs API | Cap `theme_name` at 100 characters |
| R-04 | Doc section for the same week already exists (re-run) | Skip appending (idempotency — per architecture §5) | `documents.get` heading scan before `batchUpdate` |
| R-05 | The weekly section heading format changes (e.g., "Groww — 2026-W23" vs "Groww - 2026-W23") | Idempotency check fails; duplicate section appended | Standardize heading format in a single constant; never vary it |
| R-06 | Quotes contain Markdown or special formatting characters | May render unexpectedly in Google Docs | Strip or escape Markdown in quote text |
| R-07 | `batchUpdate` payload is too large (Docs API limit) | API returns a 400 error | Split large sections into multiple `batchUpdate` calls |

---

## 11. Report Rendering (Email)

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| EM-01 | `google_doc_id` is missing or placeholder | Deep link in email points to an invalid URL | Validate `google_doc_id` at config-load time |
| EM-02 | `doc_heading_id` was not saved to run log (Doc step failed) | Email deep link is broken (missing anchor) | If `doc_heading_id` is `null`, omit the anchor from the deep link |
| EM-03 | Email `recipients` list contains an invalid email address | Gmail MCP rejects the send | Validate email format with regex before constructing payload |
| EM-04 | `email_subject_template` references an undefined variable | `str.format()` raises `KeyError` | Use `.format_map()` with a safe default dict; log missing variables |
| EM-05 | HTML email is too large for Gmail (limit: 25 MB) | Gmail rejects the send | Keep email to a short teaser only; enforce max ~3 theme bullets |
| EM-06 | Plain-text fallback is missing | Email clients that don't render HTML show nothing | Always generate both `html_body` and `text_body` |

---

## 12. Google Docs MCP Delivery

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| GD-01 | Google Docs MCP server fails to start | stdio subprocess error | Catch process startup failure; log stderr; set `status: "partial"` |
| GD-02 | `documents.get` returns an empty heading list | Doc is new/empty → no existing heading → proceed to append | Handle empty headings gracefully; treat as "heading not found" |
| GD-03 | `documents.get` times out | Cannot determine if heading exists | Retry once; if still timing out, skip Doc step and log `status: "partial"` |
| GD-04 | `documents.batchUpdate` fails (e.g., permission denied) | Section not appended | Log error; set `status: "partial"`; email step is skipped |
| GD-05 | Google OAuth token is expired | MCP server returns `401 Unauthorized` | MCP server should handle token refresh; if not, surface auth error clearly |
| GD-06 | Google Doc is in read-only mode (e.g., shared as viewer) | `batchUpdate` returns permission error | Catch and surface clearly; check Doc sharing settings |
| GD-07 | Doc heading ID returned by MCP is `null` | Cannot build deep link for email | Use Doc URL without anchor as fallback |
| GD-08 | The shared Google Doc is deleted | `documents.get` returns `404` | Abort run with clear message; notify in run log |

---

## 13. Gmail MCP Delivery

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| GM-01 | Gmail MCP server fails to start | Same as GD-01; log error; `status: "partial"` | Catch subprocess failure |
| GM-02 | `email_mode` is `"draft"` but `drafts.create` fails | Draft not created; log error; `status: "partial"` | Catch MCP error; the Doc section is already saved |
| GM-03 | `email_mode` is `"sent"` but `messages.send` fails | Email not delivered; log error; `status: "partial"` | Same as above; retry once before failing |
| GM-04 | Gmail API quota exceeded | MCP server returns `429 Resource Exhausted` | Retry after delay; if quota persists, fall back to `draft` mode |
| GM-05 | `messages.send` is called twice for the same week | Duplicate email sent to stakeholders | Idempotency: check `gmail_message_id` in run log before calling |
| GM-06 | Gmail rejects the email (spam detection, policy violation) | `messages.send` returns an error | Log the rejection reason; do not retry automatically |
| GM-07 | `gmail_message_id` returned by MCP is `null` | Cannot confirm delivery in run log | Store `null` and set `status: "partial"`; do not retry indefinitely |

---

## 14. Idempotency & Run Logging

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| ID-01 | Run log file is corrupted (invalid JSON) | `json.loads()` raises `JSONDecodeError` | Catch error; treat as a fresh run with a warning log |
| ID-02 | Run log directory is not writable | `open(..., 'w')` raises `PermissionError` | Catch and log; the pipeline still completes but audit trail is lost |
| ID-03 | Two pipeline instances run simultaneously for the same `(product, iso_week)` | Race condition: both read "no log exists" and both proceed | Implement file-locking (e.g., `fcntl` / `msvcrt` lock) on run log write |
| ID-04 | Run log shows `status: "partial"` — Doc was appended but email was not | Resume: skip Doc step, only attempt email | `delivery.gmail_message_id is None` and `delivery.doc_heading_id is not None` → resume from email |
| ID-05 | Run log shows `status: "partial"` — Doc was not appended | Resume: redo both Doc and email steps | `delivery.doc_heading_id is None` → resume from Doc delivery |
| ID-06 | `iso_week` in the run log does not match the requested week | Should never happen if path is `runs/{product}/{iso_week}/run_log.json` | Validate on read: if log's `iso_week` ≠ requested `iso_week`, treat as corrupt |
| ID-07 | Run log exists but `status` field is missing | Cannot determine run state | Treat missing `status` as `"failed"` (safe fallback) |
| ID-08 | The `runs/` directory is git-ignored but deployed to a new machine | No run history exists | Acceptable; first run on each machine starts fresh |

---

## 15. Cost & Token Limits

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| TC-01 | Single cluster's reviews exceed model context window | Truncate reviews to fit; log truncation | Implement `cluster_reviews_to_fit(max_tokens)` helper |
| TC-02 | `max_tokens_per_run` is exceeded after the first cluster | Abort remaining clusters; produce partial report | Check cumulative tokens before each cluster call |
| TC-03 | `cost_limit_usd` is exceeded mid-run | Same abort behavior as TC-02 | Estimate cost per call using `litellm.cost_calculator` |
| TC-04 | LLM provider changes pricing (cost estimate is wrong) | `cost_limit_usd` guard under/over-triggers | Use `litellm`'s built-in cost tracking; update pricing config when provider changes |
| TC-05 | Token count cannot be estimated before the call | Skip pre-call check; check post-call | Use post-call token usage from API response to update running total |

---

## 16. ISO Week & Date Handling

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| DT-01 | ISO week `2026-W53` — only some years have 53 weeks | Week number may be invalid for the given year | Validate with Python's `date.fromisocalendar()`; raise `BadParameter` if invalid |
| DT-02 | `--week` is in the future | Review window returns 0 reviews (no future reviews exist) | Detect zero reviews; abort with message "no reviews exist for a future week" |
| DT-03 | `--week` is very old (e.g., `2020-W01`) | Google Play may not have reviews that old; scraper returns empty list | Same as S-02; abort with "0 reviews" message |
| DT-04 | Run executed at year boundary (e.g., Jan 1) | ISO week may belong to the previous year (e.g., 2026-W53 is Dec 2025) | Use Python's `date.isocalendar()` correctly; do NOT use `strftime('%Y-W%W')` |
| DT-05 | Timezone mismatch between scraper and run date | Reviews from "today" may or may not be included depending on timezone | Normalize all dates to UTC; document the timezone assumption |
| DT-06 | `review_window_weeks` spans across a year boundary | Date arithmetic must handle Dec→Jan correctly | Use `timedelta` arithmetic, not week arithmetic; Python handles this correctly |

---

## 17. Security Edge Cases

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| SEC-01 | A review contains a prompt injection attempt (e.g., "Ignore previous instructions and...") | LLM follows injected instructions | Reviews are always passed as **data** in the user message inside a structured JSON field, never interpolated into the system prompt |
| SEC-02 | A review contains malicious HTML/JS (XSS attempt) | HTML injected into email body is executed by email client | Escape all user-generated content in `html_body` with `html.escape()` |
| SEC-03 | A review contains a BIDI override character (RTL text attack) | Doc section renders deceptively | Strip Unicode BIDI override characters (`U+202A`–`U+202E`, `U+2066`–`U+2069`) from review text |
| SEC-04 | `OPENAI_API_KEY` is accidentally committed to git | API key exposure | `.env` is in `.gitignore`; never store keys in `config.yaml` |
| SEC-05 | `GOOGLE_CREDENTIALS_PATH` points to a world-readable file | OAuth credentials exposed | Document that credentials file should be `chmod 600`; validate file permissions at startup |
| SEC-06 | Review text contains a file path or system command | Could mislead LLM into producing dangerous action ideas | Action ideas are always presented as suggestions to humans, not executed programmatically |

---

## 18. System-Level Edge Cases

| # | Edge Case | Expected Behavior | Mitigation |
|---|-----------|-------------------|------------|
| SY-01 | Pipeline is interrupted mid-run (Ctrl+C, power loss) | Run log is in `"partial"` or absent state | Next run detects partial state and resumes correctly |
| SY-02 | Disk is full when writing run log | `PermissionError` or `OSError` | Catch write errors; log to stderr; the pipeline still completes |
| SY-03 | System clock is wrong (far in the past/future) | Auto-detected ISO week is incorrect | Log the detected week; always allow `--week` override |
| SY-04 | Python version < 3.11 | `match` statements, `str | None` type hints may fail | `pyproject.toml` enforces `requires-python = ">=3.11"`; add version check in `main.py` |
| SY-05 | Virtual environment is not activated | Imports fail with `ModuleNotFoundError` | Document venv activation; entry point script can check |
| SY-06 | Multiple products run in parallel (future) | `runs/groww/` and `runs/indmoney/` are independent | Each product has its own run log path; no conflict |
| SY-07 | `runs/` directory grows unboundedly over time | Disk usage increases weekly | Document a periodic cleanup policy (e.g., retain last 52 weeks) |
| SY-08 | MCP server binary not found on PATH | `FileNotFoundError` when spawning subprocess | Check command existence before spawning; provide install instructions |
| SY-09 | `npx` not installed (required for Google Docs / Gmail MCP) | MCP server fails to start | Add `npx` / Node.js to prerequisites; check at startup |
| SY-10 | Running on Windows vs. Linux/macOS (path separators, line endings) | File paths may break on Windows | Use `pathlib.Path` throughout; never hardcode `/` separators |

---

## Summary by Severity

| Severity | Count | Examples |
|----------|-------|---------|
| 🔴 **Critical** (data loss / silent wrong output) | 6 | L-01 (fabricated quotes reach report), SEC-01 (prompt injection), ID-03 (race condition), R-05 (duplicate Doc sections), GM-05 (duplicate email sends), CL-01 (0 clusters, no abort) |
| 🟠 **High** (run fails silently or produces bad output) | 18 | S-01, S-04, I-01, L-07, L-08, GD-04, GM-03, ID-01, TC-02, DT-04, SEC-02, SEC-03, SY-01, SY-04, CL-05, CL-06, Q-01, Q-07 |
| 🟡 **Medium** (run degrades gracefully) | 30 | Most scraper, ingestion, PII, rendering edge cases |
| 🟢 **Low** (cosmetic or informational) | Remaining | Empty quotes, over-long themes, disk growth, etc. |

> [!IMPORTANT]
> **Critical cases to address in Phase 1 / Phase 6 before production:**
> - `ID-03` (race condition on run log) — add file locking
> - `R-05` (heading format consistency) — define heading format as a single constant
> - `GM-05` / `GD-04` (duplicate delivery) — idempotency guards are mandatory
> - `SEC-01` (prompt injection) — enforce structured data separation in LLM prompts
