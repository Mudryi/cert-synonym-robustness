import httpx
import pytest
from conftest import envelope

from cert_ua import config, storage
from cert_ua.chunker import chunk_article
from cert_ua.crawler import AccessDenied, Crawler, FetchError, NotFound, PoliteClient
from cert_ua.parser import parse_article
from cert_ua.pipeline import build_dataset
from cert_ua.validate import run_validation


def _client(handler):
    return PoliteClient(delay=0, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_upsert_has_no_duplicates(data_dir):
    a = {"article_id": "1", "publication_date": "2024-01-01", "v": 1}
    storage.upsert_articles([a, {"article_id": "2", "publication_date": "2025-01-01"}])
    storage.upsert_articles([{**a, "v": 2}])
    rows = list(storage.read_jsonl(config.ARTICLES_PATH))
    assert [r["article_id"] for r in rows] == ["2", "1"] and rows[1]["v"] == 2


def test_retry_after_and_backoff():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as c:
        assert c.get_json("https://cert.gov.ua/api/x") == {"ok": True}
    assert len(calls) == 3


def test_403_not_retried_and_in_band_errors():
    with _client(lambda req: httpx.Response(403)) as c:
        with pytest.raises(AccessDenied):
            c.get_json("https://cert.gov.ua/api/x")
        assert c.request_count == 1
    body = {"status": {"code": 409, "message": "Article not found with id 9"}}
    with _client(lambda req: httpx.Response(200, json=body)) as c:
        with pytest.raises(NotFound):
            c.get_json("https://cert.gov.ua/api/x")
    with _client(lambda req: httpx.Response(500)) as c:
        c.max_retries = 2
        with pytest.raises(FetchError):
            c.get_json("https://cert.gov.ua/api/x")


def _fake_api(req: httpx.Request) -> httpx.Response:
    path, q = req.url.path, req.url.params
    if path.endswith("/articles/all"):
        page = int(q["page"])
        items = [{"id": 10, "date": "01.01.2024", "title": "A", "tags": []}] if page == 0 else []
        return httpx.Response(200, json={"items": items, "count": 1, "tags": []})
    aid = q["id"]
    if aid == "10":
        html = "<p>" + "Слово " * 60 + "UAC-0001</p>"
        return httpx.Response(200, json={"id": "10", "title": "A", "date": "01.01.2024", "text": html,
                                         "tags": [], "langId": "None", "next": {"id": 11, "name": "B"}})
    if aid == "11":  # reachable only via "next", like the pre-2018 archive
        return httpx.Response(200, json={"id": "11", "title": "B", "date": "02.01.2017", "text": "<p>Короткий текст</p>",
                                         "tags": [], "langId": "None"})
    return httpx.Response(200, json={"status": {"code": 409, "message": "Article not found"}})


def test_crawl_resume_is_idempotent(data_dir):
    with _client(_fake_api) as c:
        cr = Crawler(c)
        cr.discover_listing()
        stats = cr.crawl()
        assert stats["fetched"] == 2 and stats["new_via_links"] == 1
    assert build_dataset() == 2
    first = (config.ARTICLES_PATH).read_text()

    requests = []
    with _client(lambda r: requests.append(r) or _fake_api(r)) as c:
        stats = Crawler(c).crawl()
    assert requests == [] and stats["attempted"] == 0
    assert build_dataset() == 2
    ids = [r["article_id"] for r in storage.read_jsonl(config.ARTICLES_PATH)]
    assert sorted(ids) == ["10", "11"]
    assert len(first.splitlines()) == 2

    summary = run_validation()
    assert summary["total"] == 2 and summary["duplicate_urls"] == []
    assert "short_body" in config.VALIDATION_REPORT_PATH.read_text()


def test_chunks_respect_sections_and_size():
    para = "Речення номер один. " * 30
    html = f"<p><strong>Розділ А</strong></p><p>{para}</p><p><strong>Розділ Б</strong></p><pre>1.2.3.4\n5.6.7.8</pre>"
    r = parse_article(envelope(html))
    chunks = list(chunk_article(r, chunk_size=200))
    assert {c["section"] for c in chunks} == {"Розділ А", "Розділ Б"}
    for c in chunks:
        assert len(c["text"]) <= 200
        assert r["body_text"][c["start_char"]:c["end_char"]] == c["text"]
        assert c["text"].endswith((".", "8"))
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
