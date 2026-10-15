"""Article discovery and polite fetching via the CERT-UA JSON API.

cert.gov.ua is an Angular SPA: the HTML pages are an empty shell and all content
comes from https://cert.gov.ua/api. We therefore crawl the API directly (no
browser automation needed). Only CERT-UA API endpoints are ever requested;
URLs found inside article bodies are never fetched.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import random
import re
import time
from collections import deque
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.robotparser import RobotFileParser

import httpx

from . import config, storage

log = logging.getLogger(__name__)


class FetchError(Exception):
    def __init__(self, message: str, status: int | None = None, attempts: int = 0):
        super().__init__(message)
        self.status = status
        self.attempts = attempts


class AccessDenied(FetchError):
    """Raised on 403 / robots disallow. Never retried or bypassed."""


class NotFound(FetchError):
    pass


# ---------------------------------------------------------------- HTTP client

class PoliteClient:
    def __init__(
        self,
        delay: float = config.REQUEST_DELAY_S,
        timeout: float = config.REQUEST_TIMEOUT_S,
        max_retries: int = config.MAX_RETRIES,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.delay = delay
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_request = 0.0
        self.request_count = 0
        self.consecutive_403 = 0
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        wait = self._last_request + self.delay - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        value = resp.headers.get("Retry-After")
        if not value:
            return None
        if value.strip().isdigit():
            return float(value.strip())
        try:
            dt = parsedate_to_datetime(value)
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError):
            return None

    def _backoff(self, attempt: int) -> float:
        base = min(config.BACKOFF_MAX_S, config.BACKOFF_BASE_S * (2 ** (attempt - 1)))
        return base + random.uniform(0, base / 2)

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        last_error = "unknown error"
        last_status: int | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            self.request_count += 1
            try:
                resp = self.client.get(url, params=params)
            except httpx.TransportError as e:
                last_error, last_status = f"{type(e).__name__}: {e}", None
                log.warning("transport error %s (attempt %d): %s", url, attempt, last_error)
                self._sleep(self._backoff(attempt))
                continue

            status = resp.status_code
            if status == 403:
                self.consecutive_403 += 1
                raise AccessDenied(f"403 Forbidden for {resp.url}", 403, attempt)
            self.consecutive_403 = 0
            if status == 404:
                raise NotFound(f"404 for {resp.url}", 404, attempt)
            if status == 429 or status >= 500:
                last_error, last_status = f"HTTP {status}", status
                wait = self._retry_after(resp)
                wait = min(wait, config.BACKOFF_MAX_S * 5) if wait is not None else self._backoff(attempt)
                log.warning("HTTP %d for %s; sleeping %.1fs (attempt %d)", status, resp.url, wait, attempt)
                self._sleep(wait)
                continue
            if status >= 400:
                raise FetchError(f"HTTP {status} for {resp.url}", status, attempt)
            return resp
        raise FetchError(f"gave up after {self.max_retries} attempts: {last_error}", last_status, self.max_retries)

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        resp = self.get(url, params)
        try:
            data = resp.json()
        except ValueError as e:
            raise FetchError(f"invalid JSON from {resp.url}: {e}", resp.status_code, 1) from e
        # The API signals errors in-band: HTTP 200 + {"status": {"code": 409, "message": ...}}
        if isinstance(data, dict) and isinstance(data.get("status"), dict) and "code" in data["status"]:
            code = data["status"].get("code")
            msg = data["status"].get("message", "")
            if code in (404, 409) or "not found" in str(msg).lower():
                raise NotFound(f"API {code}: {msg}", code, 1)
            raise FetchError(f"API {code}: {msg}", code, 1)
        return data


# ---------------------------------------------------------------- robots.txt

def check_robots(client: PoliteClient) -> dict[str, Any]:
    url = f"{config.SITE_URL}/robots.txt"
    result: dict[str, Any] = {"event": "robots_check", "url": url, "ts": storage.utc_now()}
    try:
        resp = client.get(url)
    except NotFound:
        result.update(status=404, policy="absent", allowed=True)
        storage.append_jsonl(config.CRAWL_LOG_PATH, result)
        return result
    body = resp.text
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype or body.lstrip().lower().startswith(("<!doctype", "<html")):
        # The SPA serves its index.html for unknown paths; there is no robots policy.
        result.update(status=resp.status_code, policy="absent (SPA shell returned)", allowed=True)
    else:
        rp = RobotFileParser()
        rp.parse(body.splitlines())
        checks = {
            p: rp.can_fetch(config.USER_AGENT, config.SITE_URL + p)
            for p in ("/api/articles/all", "/api/articles/byId", "/article/1", "/articles")
        }
        result.update(
            status=resp.status_code,
            policy="robots.txt parsed",
            checks=checks,
            allowed=all(checks.values()),
            crawl_delay=rp.crawl_delay(config.USER_AGENT),
            body=body[:2000],
        )
    storage.append_jsonl(config.CRAWL_LOG_PATH, result)
    return result


# ---------------------------------------------------------------- images

_DATA_URI_RE = re.compile(r"""(["'])data:([\w.+-]+/[\w.+-]+)?((?:;[\w=.+-]+)*)?,([^"']*)\1""")


def strip_data_uris(html: str | None) -> tuple[str | None, list[dict[str, Any]]]:
    """Replace inline base64 images by a short placeholder (hash + size kept for provenance)."""
    if not html:
        return html, []
    images: list[dict[str, Any]] = []

    def repl(m: re.Match[str]) -> str:
        quote, mime, params, data = m.group(1), m.group(2) or "application/octet-stream", m.group(3) or "", m.group(4)
        raw = data.encode("ascii", "ignore")
        if ";base64" in params:
            try:
                raw = base64.b64decode(raw, validate=False)
            except ValueError:
                pass
        digest = hashlib.sha256(raw).hexdigest()
        images.append({"mime": mime, "sha256": digest, "bytes": len(raw)})
        return f"{quote}data-removed:{mime};sha256={digest};bytes={len(raw)}{quote}"

    return _DATA_URI_RE.sub(repl, html), images


# ---------------------------------------------------------------- API helpers

def _as_id(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s.isdigit() and s != "0" else None


def fetch_listing_page(client: PoliteClient, page: int, lang: str = "uk") -> dict[str, Any]:
    return client.get_json(f"{config.API_URL}/articles/all", {"page": page, "lang": lang})


def fetch_article(client: PoliteClient, article_id: str, lang: str = "uk") -> dict[str, Any]:
    """Fetch an article from the API and return the cache envelope (images stripped)."""
    url = f"{config.API_URL}/articles/byId"
    payload = client.get_json(url, {"id": article_id, "lang": lang})
    if not isinstance(payload, dict) or not payload.get("id"):
        raise FetchError(f"unexpected payload for article {article_id}", None, 1)
    images: list[dict[str, Any]] = []
    for key in ("text", "description"):
        payload[key], found = strip_data_uris(payload.get(key))
        images.extend(found)
    return {
        "article_id": str(payload["id"]),
        "requested_id": article_id,
        "request_url": f"{url}?id={article_id}&lang={lang}",
        "request_lang": lang,
        "fetched_at": storage.utc_now(),
        "inline_images_removed": images,
        "payload": payload,
    }


def linked_ids(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """(article_id, via) pairs referenced by an article: prev/next, sidebar and language variant."""
    out: list[tuple[str, str]] = []
    for key in ("previous", "next"):
        ref = payload.get(key)
        if isinstance(ref, dict) and _as_id(ref.get("id")):
            out.append((_as_id(ref["id"]), key))  # type: ignore[arg-type]
    for ref in payload.get("articles") or []:
        if isinstance(ref, dict) and _as_id(ref.get("id")):
            out.append((_as_id(ref["id"]), "sidebar"))  # type: ignore[arg-type]
    # Links from one CERT-UA article to another (same-site, resolved through the API only).
    for m in _ARTICLE_LINK_RE.finditer(payload.get("text") or ""):
        out.append((m.group(1), "body_link"))
    return out


_ARTICLE_LINK_RE = re.compile(r"""href=["'](?:https?://(?:www\.)?cert\.gov\.ua)?/article/(\d+)""")


# ---------------------------------------------------------------- discovery + crawl

class Crawler:
    def __init__(self, client: PoliteClient, progress: storage.Progress | None = None):
        self.client = client
        self.progress = progress or storage.Progress()
        self.discovered = storage.load_discovered()

    # -- discovery bookkeeping
    def _add(self, article_id: str, via: str, **meta: Any) -> bool:
        """Register an article; returns True if it is new."""
        rec = self.discovered.get(article_id)
        if rec is None:
            self.discovered[article_id] = {
                "article_id": article_id,
                "url": config.ARTICLE_URL_TEMPLATE.format(id=article_id),
                "discovered_via": [via],
                "first_seen": storage.utc_now(),
                **{k: v for k, v in meta.items() if v is not None},
            }
            return True
        if via not in rec["discovered_via"]:
            rec["discovered_via"] = sorted(set(rec["discovered_via"]) | {via})
        for k, v in meta.items():
            if v is not None and rec.get(k) in (None, "", []):
                rec[k] = v
        return False

    def save_discovered(self) -> None:
        storage.save_discovered(self.discovered)

    def discover_listing(self, until_count: int | None = None, start_page: int = 0) -> int:
        """Walk /articles/all pages until exhausted (or until `until_count` IDs are known)."""
        page = start_page
        new_total = 0
        pages_without_new = 0
        while True:
            if until_count is not None and len(self.discovered) >= until_count:
                break
            try:
                data = fetch_listing_page(self.client, page)
            except FetchError as e:
                self.progress.mark_failed(f"listing-page-{page}", {
                    "url": f"{config.API_URL}/articles/all?page={page}&lang=uk",
                    "stage": "discover", "status": e.status, "error": str(e), "ts": storage.utc_now(),
                })
                self.progress.save()
                log.error("listing page %d failed: %s", page, e)
                if isinstance(e, AccessDenied):
                    raise
                break
            storage.write_json(config.RAW_DIR / "listing" / f"page_{page:04d}.json", data)
            items = data.get("items") or []
            if isinstance(data.get("count"), int):
                self.progress.api_reported_count = data["count"]
            self.progress.failed.pop(f"listing-page-{page}", None)
            if not items:
                self.progress.listing_exhausted = True
                log.info("listing exhausted at page %d", page)
                break
            new = 0
            for pos, item in enumerate(items):
                aid = _as_id(item.get("id"))
                if not aid:
                    continue
                is_new = self._add(
                    aid,
                    "listing",
                    source_lang="uk",
                    listing_title=item.get("title"),
                    listing_date=item.get("date"),
                    listing_tags=[t.get("name") for t in item.get("tags") or [] if isinstance(t, dict)],
                    listing_description=item.get("description"),
                    listing_page=page,
                    listing_position=page * config.LISTING_PAGE_SIZE + pos,
                    eng_id=_as_id(item.get("engId")),
                    eng_status=item.get("engStatus"),
                )
                new += is_new
            new_total += new
            pages_without_new = 0 if new else pages_without_new + 1
            self.progress.last_listing_page = page
            log.info("listing page %d: %d items, %d new (total %d)", page, len(items), new, len(self.discovered))
            if pages_without_new >= 2:
                log.info("two consecutive listing pages without new IDs; stopping")
                self.progress.listing_exhausted = True
                break
            page += 1
            self.save_discovered()
            self.progress.save()
        self.save_discovered()
        self.progress.save()
        return new_total

    # -- fetching
    def _queue(self, retry_failed: bool) -> deque[str]:
        def order(aid: str) -> tuple[int, int]:
            rec = self.discovered[aid]
            return (rec.get("listing_position", 10**9), -int(aid))

        q = deque()
        for aid in sorted(self.discovered, key=order):
            if aid in self.progress.done:
                continue
            if aid in self.progress.failed and not retry_failed:
                continue
            q.append(aid)
        return q

    def _register_links(
        self, aid: str, payload: dict[str, Any], follow_links: bool, variants: bool, stats: dict[str, int]
    ) -> list[str]:
        """Register IDs referenced by an article; return the newly discovered ones."""
        rec = self.discovered.setdefault(aid, {
            "article_id": aid, "url": config.ARTICLE_URL_TEMPLATE.format(id=aid),
            "discovered_via": ["cache"], "first_seen": storage.utc_now(),
        })
        lang = rec.get("source_lang") or "uk"
        new: list[str] = []
        if follow_links:
            for linked, via in linked_ids(payload):
                if self._add(linked, via, source_lang=lang, linked_from=aid):
                    new.append(linked)
        if variants:
            other = _as_id(payload.get("langId")) or rec.get("eng_id")
            if other:
                other_lang = "en" if lang == "uk" else "uk"
                rec["variant_id"] = other
                if self._add(other, f"variant:{other_lang}", source_lang=other_lang, variant_id=aid):
                    new.append(other)
        stats["new_via_links"] += len(new)
        return new

    def crawl(
        self,
        limit: int | None = None,
        refresh: bool = False,
        follow_links: bool = True,
        variants: bool = True,
        retry_failed: bool = True,
        ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Fetch every discovered article (and follow prev/next/variant links to new ones)."""
        stats = {"attempted": 0, "fetched": 0, "from_cache": 0, "failed": 0, "not_found": 0, "new_via_links": 0}
        queue = self._queue(retry_failed)
        if refresh:
            queue = deque(sorted(self.discovered, key=lambda a: self.discovered[a].get("listing_position", 10**9)))
        if ids is not None:
            queue = deque(i for i in ids if refresh or i not in self.progress.done)
        elif not refresh:
            # Already-fetched articles are skipped, but their links still count for discovery
            # (e.g. when they were fetched by `pilot` without link following). Cache only, no HTTP.
            for aid in sorted(self.progress.done):
                envelope = storage.load_raw(aid)
                if envelope is not None:
                    queue.extend(self._register_links(aid, envelope["payload"], follow_links, variants, stats))
        seen: set[str] = set()
        while queue:
            if limit is not None and stats["attempted"] >= limit:
                break
            aid = queue.popleft()
            if aid in seen:
                continue
            seen.add(aid)
            rec = self.discovered[aid]
            lang = rec.get("source_lang") or "uk"
            stats["attempted"] += 1

            envelope = None if refresh else storage.load_raw(aid)
            if envelope is not None:
                stats["from_cache"] += 1
            else:
                try:
                    envelope = fetch_article(self.client, aid, lang)
                except FetchError as e:
                    stage_status = e.status
                    info = {"url": rec["url"], "stage": "fetch", "status": stage_status, "error": str(e), "ts": storage.utc_now()}
                    self.progress.mark_failed(aid, info)
                    stats["not_found" if isinstance(e, NotFound) else "failed"] += 1
                    log.error("article %s failed: %s", aid, e)
                    self.progress.save()
                    if isinstance(e, AccessDenied) and self.client.consecutive_403 >= config.MAX_CONSECUTIVE_403:
                        log.error("aborting: %d consecutive 403 responses (not bypassing)", self.client.consecutive_403)
                        break
                    continue
                storage.save_raw(aid, envelope)
                stats["fetched"] += 1

            self.progress.mark_done(aid)
            queue.extend(self._register_links(aid, envelope["payload"], follow_links, variants, stats))
            if stats["attempted"] % 10 == 0:
                self.save_discovered()
                self.progress.save()
                log.info("progress: %s (queue %d)", stats, len(queue))
        self.save_discovered()
        self.progress.save()
        write_failures(self.progress)
        return stats


def write_failures(progress: storage.Progress) -> None:
    """Current (unresolved) failures, one per article/URL."""
    rows = [
        {"article_id": None if key.startswith("listing-") else key, **info}
        for key, info in sorted(progress.failed.items())
    ]
    storage.write_jsonl(config.FAILED_PATH, rows)
