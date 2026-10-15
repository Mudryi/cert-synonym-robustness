"""Pilot milestone: a sample spread over years and categories, plus review files.

For each pilot article a Markdown review file is written to data/reports/pilot/
containing the live URL, extracted metadata, sections and the body Markdown, so
it can be compared manually against https://cert.gov.ua/article/{id}.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from . import config, storage

log = logging.getLogger(__name__)


def select_pilot(discovered: dict[str, dict], n: int = 30) -> list[str]:
    """Round-robin over publication years, preferring not-yet-covered categories."""
    by_year: dict[str, list[dict]] = defaultdict(list)
    for rec in discovered.values():
        if "listing" not in rec.get("discovered_via", []):
            continue
        year = (rec.get("listing_date") or "")[-4:] or "unknown"
        by_year[year].append(rec)
    for recs in by_year.values():
        recs.sort(key=lambda r: r.get("listing_position", 0))

    chosen: list[str] = []
    covered_tags: set[str] = set()
    years = sorted(by_year)
    while len(chosen) < n and any(by_year.values()):
        for year in years:
            recs = by_year[year]
            if not recs or len(chosen) >= n:
                continue
            # Prefer an article with a category we have not sampled yet; otherwise spread within the year.
            pick = next((r for r in recs if set(r.get("listing_tags") or []) - covered_tags), None)
            pick = pick or recs[len(recs) // 2]
            recs.remove(pick)
            chosen.append(pick["article_id"])
            covered_tags |= set(pick.get("listing_tags") or [])
    return chosen


def write_review_files(records: list[dict]) -> None:
    out_dir = config.REPORTS_DIR / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = ["# Pilot review\n", "| id | date | lang | words | sections | categories | title |", "|---|---|---|---|---|---|---|"]
    for r in records:
        index.append(
            f"| [{r['article_id']}]({r['article_id']}.md) | {r['publication_date']} | {r['language']} | "
            f"{r['word_count']} | {len(r['sections'])} | {', '.join(r['categories'])} | {r['title']} |"
        )
        counts = {k: len(v) for k, v in r["entities"].items() if isinstance(v, list)}
        lines = [
            f"# {r['title']}",
            "",
            f"- Source: {r['url']}",
            f"- Date: {r['publication_date']}  |  Language: {r['language']} (API: {r.get('source_lang')})",
            f"- Categories: {', '.join(r['categories']) or '—'}",
            f"- Words: {r['word_count']}  |  Inline images: {r.get('image_count', 0)}",
            f"- Entities: {counts}",
            f"- Attachments: {r['attachment_urls'] or '—'}",
            f"- Warnings: {r.get('parse_warnings') or '—'}",
            "",
            "## Sections",
            "",
            *[f"- L{s['level']} {s['heading'] or '(intro)'} — {len(s['text'])} chars" for s in r["sections"]],
            "",
            "## Manual check",
            "",
            "- [ ] title  - [ ] date  - [ ] body complete  - [ ] sections  - [ ] tables/IOCs",
            "",
            "---",
            "",
            r["body_markdown"],
            "",
        ]
        (out_dir / f"{r['article_id']}.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")


def run_pilot(n: int = 30, fetch: bool = True, delay: float = config.REQUEST_DELAY_S) -> list[str]:
    from .pipeline import build_dataset

    discovered = storage.load_discovered()
    if fetch:
        from .crawler import Crawler, PoliteClient

        with PoliteClient(delay=delay) as client:
            crawler = Crawler(client)
            if not crawler.progress.listing_exhausted:
                crawler.discover_listing()
            discovered = crawler.discovered
            ids = select_pilot(discovered, n)
            crawler.crawl(ids=ids, follow_links=False, variants=False)
    else:
        ids = [i for i in select_pilot(discovered, n * 3) if storage.load_raw(i) is not None][:n]

    build_dataset()
    by_id = {r["article_id"]: r for r in storage.read_jsonl(config.ARTICLES_PATH)}
    records = [by_id[i] for i in ids if i in by_id]
    write_review_files(records)
    storage.write_json(config.REPORTS_DIR / "pilot" / "pilot_ids.json", ids)
    return ids
