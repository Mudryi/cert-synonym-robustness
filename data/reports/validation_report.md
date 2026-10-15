# CERT-UA dataset validation report

Generated: 2026-10-11T20:20:26Z · parser 1.0.0

## Discovery coverage (separate from parsing)

- API-reported listing count: **227**
- IDs found by paging the listing: **222** (97.8% of API count); listing exhausted: True
- Additional IDs found only via previous/next, body links or language variants: **6**
- Discovery sources: {'listing': 222, 'next': 222, 'sidebar': 60, 'previous': 222, 'variant:uk': 6, 'body_link': 18, 'variant:en': 6}
- Total unique IDs discovered: **228**

## Processing

- Raw responses cached: 228
- Parsed records: **228**
- Failed / inaccessible: **0**  (see `failed_urls.jsonl`)

## Validation

- Records passing all rules: **228/228** (100.0%) — PASS (target ≥ 95%)
- Basic title/body validation: 228/228 (100.0%)
- Duplicate canonical URLs: 0; duplicate IDs: 0
- Exact duplicate bodies: 0; near-duplicate pairs (Jaccard ≥ 0.9): 1
- Language variant pairs (uk/en): 6

Rules checked: required fields and types; unique canonical URL; article_id ↔ URL; ISO-8601 dates; no site boilerplate in text; body is not just the listing preview; every entity mention's offsets reproduce the stored value; section offsets reproduce section text; parser warnings surfaced.

### Errors by type

- none

### Review flags by type (records are kept, not dropped)

- language_differs_from_source: 1
- parser:language_mismatch: 1

### Records with errors

- none

### Records flagged for review

- `2720` Деструктивне шкідливе програмне забезпечення VPNFilter: language_differs_from_source:uk->en, parser:language_mismatch:uk->en

## Corpus overview

- By year: {'2018': 16, '2019': 15, '2020': 21, '2021': 11, '2022': 68, '2023': 40, '2024': 34, '2025': 12, '2026': 11}
- By language: {'uk': 221, 'en': 7}
- Words: {'min': 86, 'p05': 137, 'p25': 262, 'median': 429, 'p75': 674, 'p95': 1498, 'max': 8486, 'mean': 632.8, 'total': 144275}
- Articles containing entity type (%): {'cert_ua_ids': 58.3, 'uac_ids': 61.8, 'cve_ids': 18.0, 'file_hashes': 68.4, 'ipv4_addresses': 61.8, 'domains': 84.2, 'urls': 85.5, 'email_addresses': 39.5}
- Missing fields: {'article_id': 0, 'url': 0, 'title': 0, 'publication_date': 0, 'categories': 0, 'language': 0, 'body_text': 0, 'body_markdown': 0, 'sections': 0, 'entities': 0, 'attachment_urls': 0, 'word_count': 0, 'content_hash': 0, 'scraped_at': 0, 'parser_version': 0, 'summary': 0, 'categories_empty': 1, 'sections_without_heading_only': 36}
