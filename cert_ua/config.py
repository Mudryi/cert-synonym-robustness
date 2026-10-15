"""Static configuration: endpoints, politeness settings and data paths."""

from __future__ import annotations

import os
from pathlib import Path

SITE_URL = "https://cert.gov.ua"
API_URL = f"{SITE_URL}/api"
ARTICLE_URL_TEMPLATE = SITE_URL + "/article/{id}"

# Contact is optional; set CERT_UA_CRAWLER_CONTACT to include an e-mail or URL.
_CONTACT = os.environ.get("CERT_UA_CRAWLER_CONTACT", "").strip()
USER_AGENT = "CERT-UA-NLP-Research-Crawler/1.0 (academic NLP research; polite, rate-limited" + (
    f"; contact: {_CONTACT})" if _CONTACT else ")"
)

REQUEST_DELAY_S = 2.0
REQUEST_TIMEOUT_S = 30.0
MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 120.0
MAX_CONSECUTIVE_403 = 3
LISTING_PAGE_SIZE = 5

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CERT_UA_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
STATE_DIR = DATA_DIR / "state"

ARTICLES_PATH = PROCESSED_DIR / "cert_ua_articles.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "cert_ua_text_chunks.jsonl"
STATS_PATH = REPORTS_DIR / "dataset_statistics.json"
FAILED_PATH = REPORTS_DIR / "failed_urls.jsonl"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.md"
CRAWL_LOG_PATH = REPORTS_DIR / "crawl_log.jsonl"
DISCOVERED_PATH = STATE_DIR / "discovered.jsonl"
PROGRESS_PATH = STATE_DIR / "progress.json"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 0


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
