"""S1: weakly supervised queries.

Subset A (title): the article title is the query, the article is the positive.
Subset B (section): an LLM writes one question per selected narrative chunk;
the source chunk / article are silver positives.
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from experiments.common import (DEFAULT_LLM, LLM, OUT, SEED, done, has_identifier, log, protected_entities,
                                read_jsonl, write_jsonl)

NON_NARRATIVE = re.compile(r"індикатор|ioc|файли|мережеві|хости|графічн|рекомендац|додатк|посилання|mitre|техніки|ttp|yara|sigma",
                           re.IGNORECASE)

GEN_SYSTEM = ("Ви створюєте пошукові запити для наукового бенчмарку пошуку україномовних звітів CERT-UA з кібербезпеки. "
              "Відповідайте лише JSON.")
GEN_USER = """Фрагмент звіту:
\"\"\"
{passage}
\"\"\"

Напишіть ОДНЕ природне питання українською мовою, яке фахівець з кібербезпеки міг би ввести в пошук, щоб знайти саме цей звіт.
Вимоги:
- 8–20 слів, граматично правильне, відповідь міститься у фрагменті;
- не копіюйте фрагмент: не більше трьох слів поспіль з тексту;
- не використовуйте хеші, IP-адреси, домени, URL, імена файлів;
- назви ШПЗ чи угруповань (UAC-xxxx) — лише якщо без них питання неоднозначне.
Формат: {{"question": "..."}}"""


def ioc_density(doc: dict, chunk: dict) -> float:
    covered = 0
    for ents in doc["entities"].values():
        for e in ents:
            for s, t, field in e["mentions"]:
                if field == "body_text" and s >= chunk["start_char"] and t <= chunk["end_char"]:
                    covered += t - s
    return covered / max(1, chunk["end_char"] - chunk["start_char"])


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(OUT / "corpus"))
    ap.add_argument("--n-title", type=int, default=50)
    ap.add_argument("--n-section", type=int, default=50)
    ap.add_argument("--per-article", type=int, default=1, help="max section queries per article")
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-ioc", type=float, default=0.05)
    ap.add_argument("--max-latin", type=float, default=0.25)
    ap.add_argument("--llm-model", default=DEFAULT_LLM)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if done(out / "queries.jsonl", args.force):
        return
    rng = random.Random(SEED)
    docs = read_jsonl(Path(args.corpus) / "corpus_docs.jsonl")
    chunks = read_jsonl(Path(args.corpus) / "corpus_chunks.jsonl")
    doc_by_id = {d["doc_id"]: d for d in docs}
    indexed_docs = {c["doc_id"] for c in chunks}

    # ---- subset A: titles, stratified by year
    by_year: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        if d["doc_id"] in indexed_docs and len(d["title"].split()) >= 3:
            by_year[d["year"]].append(d)
    pool_size = sum(len(v) for v in by_year.values())
    titles: list[dict] = []
    if args.n_title >= pool_size:
        titles = [d for v in by_year.values() for d in v]
    else:
        for y in sorted(by_year):
            rng.shuffle(by_year[y])
        quotas = {y: max(1, round(args.n_title * len(v) / pool_size)) for y, v in by_year.items()}
        for y in sorted(by_year):
            titles += by_year[y][:quotas[y]]
        rng.shuffle(titles)
        titles = titles[:args.n_title]

    queries = []
    for d in titles:
        ents = protected_entities(d["title"])
        queries.append({"query_id": f"A-{d['doc_id']}", "subset": "title", "article_id": d["doc_id"], "chunk_id": None,
                        "original_query": d["title"], "protected_entities": ents, "has_identifier": has_identifier(ents),
                        "year": d["year"]})

    # ---- subset B: LLM questions from narrative chunks
    eligible: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        d = doc_by_id[c["doc_id"]]
        if (len(c["text"]) >= args.min_chars and not NON_NARRATIVE.search(c["section"] or "")
                and ioc_density(d, c) < args.max_ioc and latin_share(c["text"]) < args.max_latin):
            eligible[c["doc_id"]].append(c)
    art_ids = sorted(eligible)
    rng.shuffle(art_ids)
    picked: list[dict] = []
    for round_i in range(args.per_article):
        for a in art_ids:
            if len(picked) >= args.n_section:
                break
            cands = [c for c in eligible[a] if c not in picked]
            used_sections = {c["section"] for c in picked if c["doc_id"] == a}
            fresh = [c for c in cands if c["section"] not in used_sections] or cands
            if fresh:
                picked.append(rng.choice(fresh))
    log(f"eligible narrative chunks: {sum(map(len, eligible.values()))} in {len(eligible)} articles; picked {len(picked)}")

    llm = LLM(args.llm_model, cache_path=OUT / "llm_cache.jsonl")
    answers = llm.map(lambda c: (GEN_SYSTEM, GEN_USER.format(passage=c["text"][:2500])), picked, desc="question-gen")
    for c, ans in zip(picked, answers):
        q = (ans or {}).get("question", "").strip()
        if not q:
            continue
        ents = protected_entities(q)
        queries.append({"query_id": f"B-{c['chunk_id']}", "subset": "section", "article_id": c["doc_id"],
                        "chunk_id": c["chunk_id"], "original_query": q, "protected_entities": ents,
                        "has_identifier": has_identifier(ents), "year": doc_by_id[c["doc_id"]]["year"],
                        "source_section": c["section"]})
    n = write_jsonl(out / "queries.jsonl", queries)
    log(f"queries: {n} ({sum(q['subset'] == 'title' for q in queries)} title, {sum(q['subset'] == 'section' for q in queries)} section)")


if __name__ == "__main__":
    main()
