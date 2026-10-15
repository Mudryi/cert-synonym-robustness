import json
from pathlib import Path

import pytest

from cert_ua import config

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(article_id: str) -> dict:
    return json.loads((FIXTURES / f"article_{article_id}.json").read_text(encoding="utf-8"))


def envelope(html: str, article_id: str = "1", title: str = "Тест", date: str = "01.02.2024", **payload) -> dict:
    return {
        "article_id": article_id,
        "request_url": f"https://cert.gov.ua/api/articles/byId?id={article_id}&lang=uk",
        "request_lang": "uk",
        "fetched_at": "2026-10-11T00:00:00Z",
        "payload": {"id": article_id, "title": title, "date": date, "text": html, "description": None,
                    "tags": [{"id": 30, "name": "ШПЗ"}], "langId": "None", **payload},
    }


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point every data path at a temporary directory."""
    for name in ("RAW_DIR", "PROCESSED_DIR", "REPORTS_DIR", "STATE_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "ARTICLES_PATH", tmp_path / "processed_dir" / "cert_ua_articles.jsonl")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "processed_dir" / "chunks.jsonl")
    monkeypatch.setattr(config, "FAILED_PATH", tmp_path / "reports_dir" / "failed_urls.jsonl")
    monkeypatch.setattr(config, "STATS_PATH", tmp_path / "reports_dir" / "stats.json")
    monkeypatch.setattr(config, "VALIDATION_REPORT_PATH", tmp_path / "reports_dir" / "validation_report.md")
    monkeypatch.setattr(config, "CRAWL_LOG_PATH", tmp_path / "reports_dir" / "crawl_log.jsonl")
    monkeypatch.setattr(config, "DISCOVERED_PATH", tmp_path / "state_dir" / "discovered.jsonl")
    monkeypatch.setattr(config, "PROGRESS_PATH", tmp_path / "state_dir" / "progress.json")
    config.ensure_dirs()
    return tmp_path
