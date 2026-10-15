from conftest import envelope, load_fixture

from cert_ua.crawler import strip_data_uris
from cert_ua.parser import normalize_inline, parse_article, parse_date


def test_real_article_with_ioc_blocks():
    r = parse_article(load_fixture("6282946"))
    assert r["article_id"] == "6282946"
    assert r["url"] == "https://cert.gov.ua/article/6282946"
    assert r["publication_date"] == "2025-04-06"
    assert r["categories"] == ["ШПЗ"]
    assert r["language"] == "uk"
    headings = [s["heading"] for s in r["sections"]]
    assert headings[:3] == ["Загальна інформація", "Індикатори кіберзагроз", "Файли:"]
    files = next(s for s in r["sections"] if s["heading"] == "Файли:")
    assert files["level"] == 4 and files["path"] == ["Індикатори кіберзагроз"]
    # IOC <pre> blocks keep their line structure in both renderings.
    assert "4a2ec9f72b910c0a8e3efc4c334f5bad    f8a31715840852e8ef04016b31f909123f2aa864f3850c45eb511fd1885b4037" in r["body_text"]
    assert "```\n(28.02.2025)\n" in r["body_markdown"]
    assert "%PROGRAMDATA%\\Microsoft OneDrive Assistant\\Microsoft OneDrive Assistant" in r["body_text"]
    # The invisible refanged duplicate (<pre class="hidden">) is not part of the visible text.
    assert "tcp://37.120.239.187:6501" not in r["body_text"]
    assert r["hidden_blocks"] and "tcp://37.120.239.187:6501" in r["hidden_blocks"][0]
    assert r["image_count"] == 2 and "[image]" in r["body_markdown"]
    assert "data-removed" not in r["body_text"]
    assert " " not in r["body_text"]


def test_sections_offsets_roundtrip():
    for aid in ("6282946", "2707", "2492"):
        r = parse_article(load_fixture(aid))
        for s in r["sections"]:
            assert r["body_text"][s["content_start_char"]:s["end_char"]] == s["text"]
            if s["heading"]:
                assert r["body_text"][s["start_char"]:].startswith(s["heading"])


def test_old_article_headings_lists_links():
    r = parse_article(load_fixture("2707"))
    assert r["publication_date"] == "2018-12-13"
    rec = next(s for s in r["sections"] if s["heading"] == "Рекомендації:")
    assert "- Перевіряйте відправника повідомлення." in rec["text"]
    assert "### Рекомендації:" in r["body_markdown"]
    assert "[правила кібергігієни](https://cert.gov.ua/recommendation/31)" in r["body_markdown"]
    assert "MD5 71e64237e53f95096710854badd1d348" in r["body_text"]


def test_language_uses_prose_not_quoted_english_lists():
    assert parse_article(load_fixture("2492"))["language"] == "uk"


def test_pseudo_headings():
    html = (
        "<p><strong>Загальна інформація</strong></p><p>Текст абзацу про атаку на органи влади України.</p>"
        "<p><strong>Це довге речення жирним шрифтом, яке не є заголовком розділу статті.</strong></p>"
        "<p><em>Мережеві:</em></p><pre>1.2.3.4</pre>"
        "<p><strong>Payload:</strong> spi.exe</p>"
    )
    r = parse_article(envelope(html))
    assert [(s["heading"], s["level"]) for s in r["sections"]] == [("Загальна інформація", 3), ("Мережеві:", 4)]
    assert "Payload: spi.exe" in r["body_text"]


def test_tables_and_nested_lists():
    html = (
        "<h2>Таблиця</h2><table><tr><th>Тип</th><th>Значення</th></tr>"
        "<tr><td>IP</td><td>10.0.0.1</td></tr><tr><td>Hash</td><td>a|b</td></tr></table>"
        "<ol><li>Перший<ul><li>вкладений</li></ul></li><li>Другий</li></ol>"
        '<ul><li class="ql-indent-1">Quill відступ</li></ul>'
    )
    r = parse_article(envelope(html))
    assert "Тип | Значення\nIP | 10.0.0.1" in r["body_text"]
    assert "| Тип | Значення |\n|---|---|\n| IP | 10.0.0.1 |\n| Hash | a\\|b |" in r["body_markdown"]
    assert "1. Перший\n  - вкладений\n2. Другий" in r["body_text"]
    assert "  - Quill відступ" in r["body_text"]
    assert r["sections"][0]["heading"] == "Таблиця" and r["sections"][0]["level"] == 2


def test_unicode_and_whitespace_normalisation():
    assert normalize_inline("a  b​ c­") == "a b c"
    # Apostrophes and quotes are preserved as written.
    assert normalize_inline("комп’ютер «лапки» об'єкт") == "комп’ютер «лапки» об'єкт"
    # NFC: decomposed й becomes the single code point.
    assert normalize_inline("й") == "й"


def test_dates():
    assert parse_date("15.04.2026") == "2026-04-15"
    assert parse_date("bad") is None
    assert parse_date(None) is None


def test_attachments_and_updated_note():
    html = (
        "<p><em>Оновлено 20.04.2026</em></p><p>Див. <a href='https://cert.gov.ua/files/pdf/Rec.pdf'>PDF</a>, "
        "<a href='https://evil.example/payload.zip'>зовнішній</a> та <a href='https://cert.gov.ua/article/5'>статтю</a>.</p>"
    )
    r = parse_article(envelope(html))
    assert r["attachment_urls"] == ["https://cert.gov.ua/files/pdf/Rec.pdf"]
    assert r["updated_date"] == "2026-04-20"
    assert r["linked_article_ids"] == ["5"]


def test_strip_data_uris():
    html = '<p><img src="data:image/png;base64,aGVsbG8="></p>'
    out, images = strip_data_uris(html)
    assert "base64" not in out
    assert images == [{"mime": "image/png", "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", "bytes": 5}]


def test_empty_body_is_flagged_not_dropped():
    r = parse_article(envelope(""))
    assert r["body_text"] == "" and "empty_body" in r["parse_warnings"]
