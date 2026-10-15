"""Corpus statistics and data-quality validation."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from typing import Any

from . import PARSER_VERSION, config, storage
from .entities import ENTITY_TYPES

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "article_id": str,
    "url": str,
    "title": str,
    "publication_date": (str, type(None)),
    "categories": list,
    "language": str,
    "body_text": str,
    "body_markdown": str,
    "sections": list,
    "entities": dict,
    "attachment_urls": list,
    "word_count": int,
    "content_hash": str,
    "scraped_at": str,
    "parser_version": str,
}
OPTIONAL_FIELDS = ("summary",)

# Site chrome that must never leak into article text (the API returns article HTML only,
# so any hit here indicates a parser regression).
BOILERPLATE_PATTERNS = [
    r"Всі права захищено", r"All rights reserved", r"Підписатися на (?:новини|розсилку)",
    r"Попередня стаття", r"Наступна стаття", r"Схожі статті", r"Останні новини",
    r"Previous article", r"Next article", r"Subscribe to (?:news|newsletter)",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)
MIN_WORDS = 50
NEAR_DUP_THRESHOLD = 0.9
SHINGLE = 5


# ---------------------------------------------------------------- helpers

def _load() -> list[dict[str, Any]]:
    return list(storage.read_jsonl(config.ARTICLES_PATH))


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _shingles(text: str) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {hash(tuple(words[i:i + SHINGLE])) for i in range(max(1, len(words) - SHINGLE + 1))}


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    qs = statistics.quantiles(values, n=20, method="inclusive") if len(values) > 1 else [values[0]] * 19
    return {
        "min": min(values), "p05": round(qs[0]), "p25": round(qs[4]), "median": round(statistics.median(values)),
        "p75": round(qs[14]), "p95": round(qs[18]), "max": max(values), "mean": round(statistics.fmean(values), 1),
        "total": sum(values),
    }


def _histogram(values: list[int], edges: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for lo, hi in zip(edges, edges[1:] + [None]):
        key = f"{lo}-{hi - 1}" if hi else f"{lo}+"
        out[key] = sum(1 for v in values if v >= lo and (hi is None or v < hi))
    return out


def near_duplicates(records: list[dict[str, Any]], threshold: float = NEAR_DUP_THRESHOLD) -> list[dict[str, Any]]:
    sh = {r["article_id"]: _shingles(r["body_text"]) for r in records if r.get("body_text")}
    pairs = []
    for a, b in combinations(sorted(sh), 2):
        sa, sb = sh[a], sh[b]
        if not sa or not sb or min(len(sa), len(sb)) / max(len(sa), len(sb)) < threshold:
            continue
        j = len(sa & sb) / len(sa | sb)
        if j >= threshold:
            pairs.append({"a": a, "b": b, "jaccard": round(j, 3)})
    return pairs


# ---------------------------------------------------------------- statistics

def compute_statistics(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = _load() if records is None else records
    discovered = storage.load_discovered()
    progress = storage.Progress()
    n = len(records)

    via = Counter(v for d in discovered.values() for v in d.get("discovered_via", []))
    listing_ids = {a for a, d in discovered.items() if "listing" in d.get("discovered_via", [])}
    only_links = sorted(a for a, d in discovered.items() if "listing" not in d.get("discovered_via", []))
    failed_by_stage = Counter(f.get("stage") for f in progress.failed.values())
    not_found = [k for k, f in progress.failed.items() if f.get("status") in (404, 409)]

    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_hash[r["content_hash"]].append(r["article_id"])
    exact_dups = [ids for ids in by_hash.values() if len(ids) > 1]
    near = near_duplicates(records)

    words = [r["word_count"] for r in records]
    chars = [len(r["body_text"]) for r in records]
    entity_presence = {t: sum(1 for r in records if r["entities"].get(t)) for t in ENTITY_TYPES}
    entity_mentions = {t: sum(len(g["mentions"]) for r in records for g in r["entities"].get(t, [])) for t in ENTITY_TYPES}
    entity_unique = {t: len({g["normalized"] for r in records for g in r["entities"].get(t, [])}) for t in ENTITY_TYPES}
    hash_types = Counter(g.get("hash_type") for r in records for g in r["entities"].get("file_hashes", []))

    missing: dict[str, int] = {}
    for f in list(REQUIRED_FIELDS) + list(OPTIONAL_FIELDS):
        missing[f] = sum(1 for r in records if r.get(f) in (None, "", []) and f not in ("categories", "attachment_urls", "sections"))
    missing["categories_empty"] = sum(1 for r in records if not r.get("categories"))
    missing["sections_without_heading_only"] = sum(1 for r in records if all(s["heading"] is None for s in r["sections"]))

    ids = {r["article_id"] for r in records}
    variant_pairs = sorted({tuple(sorted((r["article_id"], r["variant_id"]))) for r in records
                            if r.get("variant_id") and r["variant_id"] in ids})
    variant_missing = sorted(r["article_id"] for r in records if r.get("variant_id") and r["variant_id"] not in ids)

    api_count = progress.api_reported_count
    return {
        "generated_at": storage.utc_now(),
        "parser_version": PARSER_VERSION,
        "counts": {
            "discovered": len(discovered),
            "discovered_in_listing": len(listing_ids),
            "discovered_only_via_links": len(only_links),
            "fetched_raw": len(storage.cached_ids()),
            "parsed": n,
            "failed": len(progress.failed),
            "failed_by_stage": dict(failed_by_stage),
            "not_found": len(not_found),
        },
        "discovery_coverage": {
            "api_reported_listing_count": api_count,
            "listing_ids_found": len(listing_ids),
            "listing_coverage_vs_api_count": round(len(listing_ids) / api_count, 4) if api_count else None,
            "listing_exhausted": progress.listing_exhausted,
            "discovered_via": dict(via),
            "ids_found_only_via_links": only_links,
            "note": "The API 'count' refers to the Ukrainian listing; articles reachable only through "
                    "previous/next, body links or language variants are additional. Discovery coverage "
                    "is reported separately from parsing success.",
        },
        "parsing": {
            "parse_success_rate_of_fetched": round(n / max(1, len(storage.cached_ids())), 4),
            "parse_success_rate_of_discovered": round(n / max(1, len(discovered)), 4),
        },
        "by_year": dict(sorted(Counter((r["publication_date"] or "unknown")[:4] for r in records).items())),
        "by_category": dict(Counter(c for r in records for c in r["categories"]).most_common()),
        "by_language": dict(Counter(r["language"] for r in records).most_common()),
        "by_source_lang": dict(Counter(r.get("source_lang") for r in records).most_common()),
        "length": {
            "words": _distribution(words),
            "chars": _distribution(chars),
            "words_histogram": _histogram(words, [0, 50, 100, 200, 400, 800, 1600, 3200]),
        },
        "entities": {
            "articles_with_type": entity_presence,
            "articles_with_type_pct": {t: round(100 * v / max(1, n), 1) for t, v in entity_presence.items()},
            "total_mentions": entity_mentions,
            "unique_values": entity_unique,
            "hash_types": dict(hash_types),
        },
        "duplicates": {
            "exact_duplicate_groups": exact_dups,
            "exact_duplicate_records": sum(len(g) - 1 for g in exact_dups),
            "near_duplicate_pairs": near,
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
        },
        "missing_fields": missing,
        "language_variants": {"pairs": [list(p) for p in variant_pairs], "variant_not_in_dataset": variant_missing},
        "failed_urls": len(progress.failed),
        "images": {"articles_with_inline_images": sum(1 for r in records if r.get("image_count")),
                   "inline_images": sum(r.get("image_count", 0) for r in records)},
        "attachments": {"articles_with_attachments": sum(1 for r in records if r["attachment_urls"]),
                        "attachment_urls": sum(len(r["attachment_urls"]) for r in records)},
    }


# ---------------------------------------------------------------- validation

def validate_record(r: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, review_flags) for a single record."""
    errors: list[str] = []
    flags: list[str] = []
    for f, typ in REQUIRED_FIELDS.items():
        if f not in r:
            errors.append(f"missing_field:{f}")
        elif not isinstance(r[f], typ):
            errors.append(f"bad_type:{f}")
    if "summary" not in r:
        errors.append("missing_field:summary")
    if errors:
        return errors, flags

    if r["url"] != config.ARTICLE_URL_TEMPLATE.format(id=r["article_id"]):
        errors.append("article_id_url_mismatch")
    if not r["title"].strip():
        errors.append("empty_title")
    if not r["body_text"].strip():
        errors.append("empty_body")
    if r["publication_date"] is None:
        flags.append("missing_date")
    elif not _is_iso_date(r["publication_date"]):
        errors.append("invalid_date")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", r["scraped_at"]):
        errors.append("invalid_scraped_at")
    if m := _BOILERPLATE_RE.search(r["body_text"]):
        errors.append(f"boilerplate:{m.group(0)}")

    summary = (r.get("summary") or "").strip()
    body = r["body_text"].strip()
    if r["word_count"] < MIN_WORDS:
        flags.append("short_body")
    if summary and body and len(body) <= len(summary) * 1.1 and (body in summary or summary.startswith(body[:200])):
        flags.append("body_looks_like_preview")
    if r["language"] == "unknown":
        flags.append("language_unknown")
    if r.get("source_lang") and r["language"] != r["source_lang"]:
        flags.append(f"language_differs_from_source:{r['source_lang']}->{r['language']}")

    texts = {"body_text": r["body_text"], "title": r["title"]}
    bad_offsets = 0
    for t in ENTITY_TYPES:
        for g in r["entities"].get(t, []):
            for mnt in g["mentions"]:
                if texts[mnt["field"]][mnt["start"]:mnt["end"]] not in g["variants"]:
                    bad_offsets += 1
    if bad_offsets:
        errors.append(f"entity_offsets_mismatch:{bad_offsets}")

    for s in r["sections"]:
        if r["body_text"][s["content_start_char"]:s["end_char"]] != s["text"]:
            errors.append("section_offsets_mismatch")
            break
    for w in r.get("parse_warnings") or []:
        flags.append(f"parser:{w}")
    return errors, flags


def run_validation() -> dict[str, Any]:
    records = _load()
    stats = compute_statistics(records)
    storage.write_json(config.STATS_PATH, stats)

    url_counts = Counter(r.get("url") for r in records)
    dup_urls = [u for u, c in url_counts.items() if c > 1]
    id_counts = Counter(r.get("article_id") for r in records)
    dup_ids = [i for i, c in id_counts.items() if c > 1]

    results = []
    for r in records:
        errors, flags = validate_record(r)
        if r.get("url") in dup_urls:
            errors.append("duplicate_url")
        results.append({"article_id": r.get("article_id"), "title": r.get("title"), "errors": errors, "flags": flags})

    basic_ok = [x for x in results if not any(e.startswith(("empty_title", "empty_body", "missing_field", "bad_type"))
                                               for e in x["errors"]) and "body_looks_like_preview" not in x["flags"]]
    passed = [x for x in results if not x["errors"]]
    summary = {
        "total": len(results),
        "passed": len(passed),
        "pass_rate": len(passed) / len(results) if results else 0.0,
        "basic_title_body_pass": len(basic_ok),
        "basic_title_body_pass_rate": len(basic_ok) / len(results) if results else 0.0,
        "flagged_for_review": sum(1 for x in results if x["flags"]),
        "duplicate_urls": dup_urls,
        "duplicate_ids": dup_ids,
    }
    storage.write_jsonl(config.REPORTS_DIR / "validation_results.jsonl", results)
    config.VALIDATION_REPORT_PATH.write_text(_render_report(summary, results, stats), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any], results: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    c, cov = stats["counts"], stats["discovery_coverage"]
    err_counts = Counter(e.split(":")[0] for x in results for e in x["errors"])
    flag_counts = Counter(f.split(":")[0] + (":" + f.split(":")[1] if f.startswith("parser:") else "")
                          for x in results for f in x["flags"])
    ok = "PASS" if summary["pass_rate"] >= 0.95 else "FAIL"
    lines = [
        "# CERT-UA dataset validation report",
        "",
        f"Generated: {stats['generated_at']} · parser {stats['parser_version']}",
        "",
        "## Discovery coverage (separate from parsing)",
        "",
        f"- API-reported listing count: **{cov['api_reported_listing_count']}**",
        f"- IDs found by paging the listing: **{cov['listing_ids_found']}** "
        f"({(cov['listing_coverage_vs_api_count'] or 0):.1%} of API count); listing exhausted: {cov['listing_exhausted']}",
        f"- Additional IDs found only via previous/next, body links or language variants: **{c['discovered_only_via_links']}**",
        f"- Discovery sources: {cov['discovered_via']}",
        f"- Total unique IDs discovered: **{c['discovered']}**",
        "",
        "## Processing",
        "",
        f"- Raw responses cached: {c['fetched_raw']}",
        f"- Parsed records: **{c['parsed']}**",
        f"- Failed / inaccessible: **{c['failed']}** {c['failed_by_stage'] or ''} (see `failed_urls.jsonl`)",
        "",
        "## Validation",
        "",
        f"- Records passing all rules: **{summary['passed']}/{summary['total']}** ({summary['pass_rate']:.1%}) — {ok} (target ≥ 95%)",
        f"- Basic title/body validation: {summary['basic_title_body_pass']}/{summary['total']} ({summary['basic_title_body_pass_rate']:.1%})",
        f"- Duplicate canonical URLs: {len(summary['duplicate_urls'])}; duplicate IDs: {len(summary['duplicate_ids'])}",
        f"- Exact duplicate bodies: {stats['duplicates']['exact_duplicate_records']}; near-duplicate pairs "
        f"(Jaccard ≥ {stats['duplicates']['near_duplicate_threshold']}): {len(stats['duplicates']['near_duplicate_pairs'])}",
        f"- Language variant pairs (uk/en): {len(stats['language_variants']['pairs'])}",
        "",
        "Rules checked: required fields and types; unique canonical URL; article_id ↔ URL; ISO-8601 dates; "
        "no site boilerplate in text; body is not just the listing preview; every entity mention's offsets "
        "reproduce the stored value; section offsets reproduce section text; parser warnings surfaced.",
        "",
        "### Errors by type",
        "",
        *([f"- {k}: {v}" for k, v in err_counts.most_common()] or ["- none"]),
        "",
        "### Review flags by type (records are kept, not dropped)",
        "",
        *([f"- {k}: {v}" for k, v in flag_counts.most_common()] or ["- none"]),
        "",
        "### Records with errors",
        "",
        *([f"- `{x['article_id']}` {x['title']}: {', '.join(x['errors'])}" for x in results if x["errors"]] or ["- none"]),
        "",
        "### Records flagged for review",
        "",
        *([f"- `{x['article_id']}` {x['title']}: {', '.join(x['flags'])}" for x in results if x["flags"]] or ["- none"]),
        "",
        "## Corpus overview",
        "",
        f"- By year: {stats['by_year']}",
        f"- By language: {stats['by_language']}",
        f"- Words: {stats['length']['words']}",
        f"- Articles containing entity type (%): {stats['entities']['articles_with_type_pct']}",
        f"- Missing fields: {stats['missing_fields']}",
        "",
    ]
    return "\n".join(lines)
