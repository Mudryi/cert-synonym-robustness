"""Rebuild the processed dataset from the raw cache (no network access)."""

from __future__ import annotations

import logging
import traceback

from . import config, storage
from .parser import parse_article

log = logging.getLogger(__name__)


def build_dataset() -> int:
    """Parse every cached article into data/processed/cert_ua_articles.jsonl.

    The processed file is rewritten from scratch keyed by article_id, so repeated
    runs never create duplicates. Parse failures are recorded, never dropped silently.
    """
    discovered = storage.load_discovered()
    progress = storage.Progress()
    records = []
    for aid in storage.cached_ids():
        envelope = storage.load_raw(aid)
        try:
            records.append(parse_article(envelope, discovered.get(aid)))
            if progress.failed.get(aid, {}).get("stage") == "parse":
                progress.failed.pop(aid)
        except Exception as e:  # noqa: BLE001 - any parser bug must be logged, not crash the run
            log.error("parse failed for %s: %s", aid, e)
            progress.mark_failed(aid, {
                "url": config.ARTICLE_URL_TEMPLATE.format(id=aid),
                "stage": "parse",
                "status": None,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
                "ts": storage.utc_now(),
            })
    progress.save()
    from .crawler import write_failures

    write_failures(progress)
    by_id = {r["article_id"]: r for r in records}
    return storage.write_jsonl(config.ARTICLES_PATH, sorted(by_id.values(), key=storage._sort_key, reverse=True))
