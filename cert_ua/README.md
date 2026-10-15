# CERT-UA research dataset

A reproducible, resumable crawler and parser that turns the public articles of
[CERT-UA](https://cert.gov.ua/articles) into a clean JSONL corpus for Ukrainian NLP
research (semantic retrieval, weak supervision, terminology preservation,
information extraction). It does **data acquisition only**: no models, no labels.

## How the site is accessed

`cert.gov.ua` is an Angular single-page app. The HTML at `/articles` and
`/article/{id}` is an empty shell (`<app-root>`); all content is loaded from a
public JSON API, which this crawler calls directly (no browser automation):

| Endpoint | Use |
|---|---|
| `GET /api/articles/all?page=N&lang=uk` | archive listing, 5 items per page, **0-indexed**, `count` = total reported by the API |
| `GET /api/articles/byId?id=X&lang=uk\|en` | full article: title, date, tags, description, body HTML, `previous`/`next`, `langId` (uk↔en variant) |

The API answers XML by default; we send `Accept: application/json`. Missing
articles return HTTP 200 with `{"status": {"code": 409}}`, which is treated as
"not found".

**Discovery** (article IDs are not sequential, e.g. `2727`, `18419`, `6288271`, so IDs are never enumerated):

1. Page through the listing until an empty page (or two pages without new IDs).
2. Follow `previous`/`next` references of every fetched article — this reaches
   older articles that are no longer in the listing.
3. Follow links from one article body to another `cert.gov.ua/article/{id}`.
4. Fetch language variants referenced by `langId`/`engId` (English versions are separate article IDs).

URLs that appear *inside* articles (indicators of compromise, references) are never requested.

**Politeness**: `robots.txt` is checked first (the site currently returns the SPA
shell instead of a robots file, i.e. no policy — logged to `data/reports/crawl_log.jsonl`);
one request every 2 s (`--delay`); descriptive User-Agent; explicit 30 s timeouts;
exponential backoff with jitter for network errors / 5xx; `Retry-After` honoured
for 429/503; 403 is never retried and three consecutive 403s abort the crawl.
Set `CERT_UA_CRAWLER_CONTACT=you@example.org` to add a contact to the User-Agent.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python ≥ 3.12 is installed automatically).

```bash
uv sync
uv run pytest           # offline parser/entity/storage tests
```

## Usage

```bash
uv run python main.py crawl --limit 50   # attempt 50 unique articles (pages listing only as far as needed)
uv run python main.py crawl --all        # whole archive (re-walks the listing; cached articles are not re-downloaded)
uv run python main.py crawl --resume     # continue an interrupted crawl, retry failures
uv run python main.py parse              # rebuild the processed JSONL from data/raw (offline)
uv run python main.py chunk --chunk-size 1200 --overlap 0
uv run python main.py validate           # writes validation_report.md (+ statistics); exit 1 if < 95% pass
uv run python main.py stats              # writes dataset_statistics.json
uv run python main.py pilot -n 30        # sample across years/categories + review files in data/reports/pilot/
```

Useful flags for `crawl`: `--refresh` (re-download cached articles), `--no-variants`,
`--no-follow` (listing only), `--delay SECONDS`.

The crawl is resumable: progress is saved in `data/state/` after every ten
articles, raw responses are cached in `data/raw/`, and the processed dataset is
always rebuilt from the cache keyed by `article_id`, so re-running never creates
duplicates. Parser changes only need `python main.py parse` — no new requests.

## Outputs

```
data/raw/{id}.json.gz                    cached API response (+ request URL, fetch time)
data/raw/listing/page_NNNN.json          cached listing pages
data/state/discovered.jsonl              every discovered ID with how it was found
data/state/progress.json                 done / failed IDs, listing state
data/processed/cert_ua_articles.jsonl    one record per article
data/processed/cert_ua_text_chunks.jsonl section-aware chunks
data/reports/dataset_statistics.json     corpus statistics
data/reports/validation_report.md        data-quality report
data/reports/validation_results.jsonl    per-record errors / review flags
data/reports/failed_urls.jsonl           unresolved failures with stage and reason
data/reports/pilot/                      pilot review files (one Markdown file per article)
```

### Article record

| Field | Description |
|---|---|
| `article_id` | ID from the source URL (string) |
| `url` | canonical `https://cert.gov.ua/article/{id}` |
| `title` | full title |
| `publication_date` | ISO `YYYY-MM-DD` or `null` |
| `categories` | source tags (e.g. `ШПЗ`, `Phishing`); `[]` if none |
| `summary` | listing preview (API `description`) as text, or `null` |
| `language` | detected language of the running prose (`uk`, `en`, …) |
| `body_text` | clean text; blocks separated by blank lines; IOC blocks keep their line layout |
| `body_markdown` | same content with headings, lists, tables, links and fenced IOC blocks |
| `sections` | `[{heading, level, path, start_char, content_start_char, end_char, text}]` — offsets into `body_text`; first section has `heading: null` if the article starts without one |
| `entities` | see below |
| `attachment_urls` | CERT-UA hosted file links (PDF, DOCX, …) — references only, never downloaded |
| `word_count` | words in `body_text` |
| `content_hash` | SHA-256 of NFC `body_text` |
| `scraped_at` | UTC time the API response was fetched |
| `parser_version` | extraction version |

Extra fields: `source_lang` (language of the API request, `uk`/`en`),
`language_scores` (Cyrillic ratio and per-language prose shares — identifies
bilingual articles), `variant_id` (the other-language version), `updated_date`
("Оновлено DD.MM.YYYY" notes), `char_count`, `image_count`, `outbound_links`,
`linked_article_ids`, `hidden_blocks` (see below), `api_url`, `listing_page`,
`discovered_via`, `parse_warnings`.

### Entities

`entities` holds lists `cert_ua_ids`, `uac_ids`, `cve_ids`, `file_hashes`,
`ipv4_addresses`, `domains`, `urls`, `email_addresses`, plus `counts`. Each
entry groups all mentions of one normalised value:

```json
{"normalized": "37.120.239.187", "value": "37[.]120.239.187", "variants": ["37[.]120.239.187"],
 "defanged": true,
 "mentions": [{"start": 2140, "end": 2156, "field": "body_text", "section": "Мережеві:"}]}
```

`variants` preserve the original spelling (including defanging such as `hXXps`,
`[.]`, `(tcp)://`); `normalized` is refanged/upper-/lower-cased as appropriate
(`CERT-UA#1234`, `UAC-0226`, `CVE-2023-38831`, lowercase hashes with `hash_type`).
`text[start:end]` of the given `field` always reproduces a variant (checked by `validate`).

These are **regex-extracted mentions, not verified threat intelligence**. Known
limitations: version strings shaped like IPs can still be matched (`Chrome/x.y.z.w`
is filtered); domains are accepted only with an IANA TLD, and non-defanged
tokens that look like file names (`status.zip`, `run.sh`) or code (`System.Net.WebClient`)
are rejected, so some genuine plain-text domains with those TLDs are missed.

## Text processing

- Body HTML (Quill editor output) is walked in document order. Headings come from
  `h1–h6` and from CERT-UA's pseudo-headings: a paragraph that is entirely bold and
  short (≤ 8 words or ending with `:`/`?`) → level 3; a short italic label ending
  with `:` (e.g. *Файли:*, *Мережеві:*) → level 4.
- Lists (including nested and Quill `ql-indent-N`) keep `-` / `1.` markers in both
  renderings; tables become `cell | cell` lines in text and pipe tables in Markdown;
  `<pre>` IOC blocks keep line breaks and column spacing.
- Unicode is NFC-normalised; NBSP and other space characters become spaces;
  zero-width characters and soft hyphens are removed; apostrophes, quotes and
  Cyrillic are left untouched. Nothing is translated, summarised or paraphrased.
- Indicator-of-compromise sections are kept.
- Inline base64 images are removed from the cached HTML before storage
  (replaced by `data-removed:{mime};sha256=…;bytes=…`), keeping the cache small
  (one article is ~12 MB with images, ~42 KB without). They appear as `[image]` in Markdown.
- Some articles contain a `<pre class="hidden">` block that is **not displayed** on
  the website (refanged duplicates of the visible IOCs). It is excluded from
  `body_text`/`body_markdown` and kept separately in `hidden_blocks`.
- Language is detected per paragraph with a fixed-seed `langdetect` and weighted by
  length, on running prose only (IOC blocks and quoted English list items are
  excluded), so Ukrainian reports with long English IOC/advisory lists stay `uk`.
- No search-engine or LLM text is ever substituted for failed extractions; failures
  are written to `failed_urls.jsonl` and `progress.json`.

`trafilatura` and Playwright are not used: the API returns the article HTML
without any site chrome, so no boilerplate-removal heuristics are needed.

## Validation

`validate` checks required fields and types, unique canonical URLs, `article_id` ↔
URL consistency, ISO dates, absence of site boilerplate, that the body is not just
the listing preview, entity and section offsets, and surfaces parser warnings.
Short articles (< 50 words), language mismatches and missing dates are **flagged
for review, never dropped**. Statistics include discovery coverage (listing vs.
API-reported count vs. IDs reachable only via links) reported separately from
parsing success, distributions by year/category/language, text lengths, entity
coverage, exact and near-duplicates (5-gram Jaccard ≥ 0.9) and uk/en variant pairs.

## Limitations

- Content comes from the site's JSON API rather than rendered HTML; the raw cache
  therefore stores API JSON. Rendering differences between the API HTML and what
  the browser shows are possible (e.g. the hidden IOC block above).
- The listing reports more articles (`count`) than it returns through paging;
  the gap is reported in the validation report. Articles only reachable from
  other parts of the site (e.g. `/recommendations`) are out of scope.
- Inline images (screenshots, attack-chain diagrams) are not OCR'd; attachments
  (PDF etc.) are listed but not downloaded or extracted.
- Pseudo-heading detection is heuristic; heading levels reflect formatting, not
  necessarily the author's intended hierarchy.
- Entity extraction is purely pattern-based (see above).

## Provenance and licensing

Every record carries the canonical source URL, the exact API request URL,
fetch timestamp, parser version and content hash. All content is © CERT-UA /
State Service of Special Communications and Information Protection of Ukraine.
**Before distributing or publicly releasing the corpus, check CERT-UA's terms of
use and reuse conditions**, and keep attribution with the original URLs. The
dataset contains indicators of compromise (malicious domains, URLs, IPs, hashes)
as published by CERT-UA; treat them as untrusted data and do not visit them.
