# Weekly Product Review Pulse

> Automated weekly insights from Google Play Store reviews, delivered via MCP to Google Docs & Gmail.

## Overview

This system generates a **weekly insight report** from [Groww](https://play.google.com/store/apps/details?id=com.groww.v1) Google Play Store reviews. It uses embedding-based clustering and LLM summarization to identify top themes, extract representative quotes, and propose action ideas — then delivers the report via **MCP (Model Context Protocol)** to a shared Google Doc and stakeholder emails.

### Architecture

The agent is an **MCP host/client** — every external interaction flows through an MCP server's tool interface:

| MCP Server | Transport | Purpose |
|---|---|---|
| **Play Store Reviews** (this repo) | stdio (local) | Scrape & return Groww reviews from Google Play |
| **Google Docs** (Railway cloud) | SSE (remote) | Append weekly report sections to a shared Doc |
| **Gmail** (Railway cloud) | SSE (remote) | Send/draft stakeholder notification emails |

See [docs/architecture.md](docs/architecture.md) for the full technical design.

---

## Setup Guide

### Prerequisites

- Python 3.11+
- A **Groq** API key (free — [console.groq.com](https://console.groq.com))
- A **Google Doc ID** for the shared pulse document

### 1. Clone & install

```bash
git clone <repo-url>
cd Playstore

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install all dependencies
pip install -e ".[dev]"
```

### 2. Configure environment variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
copy .env.example .env
```

Fill in your values:

```env
# Groq free-tier LLM
GROQ_API_KEY=gsk_...your_key_here...

# MCP server authentication (used by the Railway cloud MCP server)
MCP_API_SECRET_KEY=your_secret

# Google Doc ID — the shared document where weekly reports are appended
GOOGLE_DOC_ID=1m9zzBIJ...your_doc_id...
```

### 3. Configure the application

```bash
copy config\config.example.yaml config\config.yaml
```

Edit `config/config.yaml` to update the product, recipients, and any other settings. Key fields:

| Field | Description |
|---|---|
| `product.play_store_app_id` | Google Play package name (e.g. `com.groww.v1`) |
| `product.review_window_weeks` | Rolling window of reviews to analyse (8–12 weeks) |
| `delivery.recipients` | List of email addresses to notify |
| `delivery.email_mode` | `"draft"` (staging) or `"sent"` (production) |
| `clustering.embedding_model` | Sentence-transformer model (default: `BAAI/bge-small-en-v1.5`) |

---

## Usage

```bash
# Run for the current week (auto-detects ISO week)
python -m src.agent.main --product groww

# Run for a specific ISO week
python -m src.agent.main --product groww --week 2026-W23

# Dry run — full analysis but skip all MCP delivery (safe for testing)
python -m src.agent.main --product groww --dry-run

# Enable verbose/debug logging
python -m src.agent.main --product groww --verbose

# Show all options
python -m src.agent.main --help
```

### CLI Reference

| Option | Default | Description |
|---|---|---|
| `--product` | `groww` | Product name (matches `config.yaml product.name`) |
| `--week` | Current week | ISO week to analyze (e.g., `2026-W23`) |
| `--config` | `config/config.yaml` | Path to config file |
| `--dry-run` | `false` | Skip MCP delivery (no Doc append, no email) |
| `--verbose` / `-v` | `false` | Enable debug logging |

---

## Pipeline Overview

The agent runs a 13-step pipeline:

| Step | Description |
|---|---|
| 1 | Load `config.yaml` and validate all keys |
| 2 | Idempotency check — skip if this week already succeeded |
| 3+4 | Fetch reviews via Play Store MCP (stdio) + Layer 2 PII scrub |
| 5 | Generate sentence-transformer embeddings |
| 6 | UMAP dimensionality reduction + HDBSCAN clustering |
| 7 | LLM summarization via Groq (`llama-3.3-70b-versatile`) |
| 8 | Quote validation — discard any LLM-fabricated quotes |
| 9 | Render Google Docs `batchUpdate` payload |
| 10 | Deliver Doc section via Google Docs MCP (SSE) |
| 11 | Render HTML + plain-text email |
| 12 | Deliver email via Gmail MCP (SSE) |
| 13 | Write `runs/<product>/<week>/run_log.json` |

### Idempotency & Partial-Failure Recovery

Every run writes a `run_log.json`. If the pipeline is re-run:

- `status: "success"` → the run is **skipped entirely** (no duplicates)
- `status: "partial"` → already-delivered steps are **skipped**, resuming from where it left off
- Doc delivery is saved immediately after success, before email is attempted

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Configuration error: Missing required key 'google_doc_id'` | `GOOGLE_DOC_ID` not in `.env` | Add `GOOGLE_DOC_ID=...` to your `.env` file |
| `ABORT: No reviews returned for app...` | App ID wrong or no reviews in window | Check `play_store_app_id` in `config.yaml` |
| `LLM call failed: AuthenticationError` | Invalid `GROQ_API_KEY` | Check your Groq key at console.groq.com |
| `ABORT: Groq daily token limit reached` | Groq free-tier quota hit | Wait for quota reset (midnight UTC) or reduce `max_themes` |
| `Failed to deliver doc: ...` | MCP server unreachable | Check Railway deployment is running |
| Run exits early with no output | Already ran successfully this week | Delete `runs/<product>/<week>/run_log.json` to force a re-run |

---

## Project Structure

```
Playstore/
├── docs/
│   ├── architecture.md             # Full technical architecture
│   ├── implementation_plan.md      # Phase-wise implementation plan
│   ├── edge-cases.md               # Edge case handling
│   └── problemStatement.md         # Scoped problem definition
├── src/
│   ├── agent/                      # Agent pipeline (MCP host/client)
│   │   ├── main.py                 # CLI entrypoint & orchestrator
│   │   ├── config.py               # Config loader (YAML + env vars)
│   │   ├── ingestion.py            # Play Store MCP client + PII L2
│   │   ├── pii_scrubber.py         # Layer 2 PII scrubbing (regex + NER)
│   │   ├── clustering.py           # Embeddings + UMAP + HDBSCAN
│   │   ├── summarizer.py           # Groq LLM summarization
│   │   ├── quote_validator.py      # Verbatim quote verification
│   │   ├── doc_renderer.py         # Google Docs batchUpdate payload builder
│   │   ├── email_renderer.py       # HTML + plain-text email builder
│   │   ├── delivery.py             # MCP delivery (Google Docs + Gmail)
│   │   └── idempotency.py          # Run log read/write
│   └── mcp_servers/
│       └── playstore_reviews/      # Custom Play Store Reviews MCP Server
│           ├── server.py           # MCP server (stdio transport)
│           ├── tools.py            # fetch_reviews, get_app_metadata
│           ├── scraper.py          # google-play-scraper wrapper
│           └── pii.py              # Layer 1 PII scrub (author names)
├── config/
│   ├── config.yaml                 # Runtime configuration
│   └── config.example.yaml         # Template with placeholder values
├── runs/                           # Run logs (git-ignored)
│   └── groww/
│       └── 2026-W23/
│           └── run_log.json
├── tests/                          # Unit tests
├── pyproject.toml                  # Project metadata & dependencies
├── requirements.txt                # Flat dependency list
├── .env                            # Your secrets (never commit)
└── .env.example                    # Environment variable template
```

---

## License

MIT
