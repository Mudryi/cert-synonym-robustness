"""S0: freeze the Ukrainian retrieval collection.

- Ukrainian articles only (English records are kept out of the evaluation).
- Near-duplicate articles (dataset_statistics.json) are collapsed: the
  lexicographically smallest id is kept and the others are mapped to it.
- The title and any verbatim repetition of it are stripped from the indexed
  text, so title queries cannot be solved by exact title matching.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from experiments.common import ARTICLES, CHUNKS, OUT, STATS, done, log, read_jsonl, write_json, write_jsonl


def strip_title(text: str, title: str) -> tuple[str, int]:
    if not title:
        return text, 0
    pattern = re.compile(r"\s*".join(re.escape(w) for w in title.split()), re.IGNORECASE)
    new, n = pattern.subn(" ", text)
    return re.sub(r"[ \t]{2,}", " ", new).strip(), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "corpus"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if done(out / "corpus_chunks.jsonl", args.force):
        return

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    canon: dict[str, str] = {}
    for pair in stats["duplicates"]["near_duplicate_pairs"]:
        a, b = sorted([pair["a"], pair["b"]])
        canon[b] = canon.get(a, a)

    arts = [a for a in read_jsonl(ARTICLES) if a["language"] == "uk"]
    docs, dropped, title_hits = [], [], 0
    for a in arts:
        if a["article_id"] in canon:
            dropped.append(a["article_id"])
            continue
        body, n = strip_title(a["body_text"], a["title"])
        title_hits += n
        docs.append({
            "doc_id": a["article_id"],
            "title": a["title"],
            "text": body,
            "publication_date": a["publication_date"],
            "year": (a["publication_date"] or "0000")[:4],
            "categories": a["categories"],
            "summary": a["summary"],
            "word_count": a["word_count"],
            "sections": [{k: s[k] for k in ("heading", "start_char", "end_char")} for s in a["sections"]],
            "entities": {k: [{"value": e["value"], "mentions": [(m["start"], m["end"], m["field"]) for m in e["mentions"]]}
                             for e in a["entities"].get(k, [])]
                         for k in ("cert_ua_ids", "uac_ids", "cve_ids", "file_hashes", "ipv4_addresses", "domains", "urls", "email_addresses")},
        })
    keep = {d["doc_id"] for d in docs}
    titles = {d["doc_id"]: d["title"] for d in docs}

    chunks = []
    for c in read_jsonl(CHUNKS):
        if c["article_id"] not in keep or c["language"] != "uk":
            continue
        text, _ = strip_title(c["text"], titles[c["article_id"]])
        if len(text) < 30:
            continue
        chunks.append({"chunk_id": c["chunk_id"], "doc_id": c["article_id"], "section": c["section"],
                       "start_char": c["start_char"], "end_char": c["end_char"], "text": text})
    docs_with_chunks = {c["doc_id"] for c in chunks}
    missing = sorted(keep - docs_with_chunks)

    write_jsonl(out / "corpus_docs.jsonl", docs)
    write_jsonl(out / "corpus_chunks.jsonl", chunks)
    summary = {
        "uk_articles": len(arts), "docs": len(docs), "chunks": len(chunks),
        "dropped_near_duplicates": dropped, "duplicate_map": canon,
        "title_occurrences_stripped_from_bodies": title_hits,
        "docs_without_uk_chunks": missing,
        "docs_by_year": dict(sorted(Counter(d["year"] for d in docs).items())),
    }
    write_json(out / "corpus_summary.json", summary)
    log(f"corpus: {len(docs)} docs, {len(chunks)} chunks, dropped {dropped}, title strips {title_hits}")


if __name__ == "__main__":
    main()
