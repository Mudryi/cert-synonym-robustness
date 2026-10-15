"""S2 (compose): build perturbed query variants.

Types:
  dict  synonym-dictionary substitutions (pool from ../textfooler_ukr)
  mlm   contextual MLM substitutions (pool from ../bert_attack_uk)
  tech  technical term variation (resources/tech_terms.tsv, pymorphy3 inflection)
  llm   1–2 word synonym substitutions proposed by the LLM

For dict/mlm a variant with n substitutions picks n distinct positions at
random and, at each, a replacement sampled from the 3 single-substitution
candidates with the highest SBERT similarity to the original (contextual
selection, as in the textfooler_ukr pipeline). Protected entities are never
touched; this is asserted for every variant.

Also used to write pool-input files (`--write-pool-input`).
"""
from __future__ import annotations

import argparse
import difflib
import random
import re
from functools import lru_cache
from pathlib import Path

from experiments.common import (DEFAULT_LLM, LLM, OUT, RESOURCES, SEED, TOKEN_RE, apply_subs, done,
                                entities_preserved, log, protected_entities, read_jsonl, sbert_cos, write_jsonl)

CASES = ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct")
NUMBERS = ("sing", "plur")


# ---------------------------------------------------------------- tech terms

@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer(lang="uk")


def _invariant(word: str) -> bool:
    return word.isupper() or bool(re.search(r"[A-Za-z]", word))


def phrase_forms(phrase: str) -> dict[tuple[str, str], str]:
    """Surface form of a lemma phrase for every (case, number)."""
    words = phrase.split()
    morph = _morph()
    # head = first noun; words after it are governed modifiers (орган державної влади) and stay as written
    head_i, head_gender = len(words) - 1, None
    for i, w in enumerate(words):
        if not _invariant(w):
            p = next((p for p in morph.parse(w) if p.tag.POS == "NOUN"), None)
            if p is not None:
                head_i, head_gender = i, p.tag.gender
                break
    forms: dict[tuple[str, str], str] = {}
    for case in CASES:
        for num in NUMBERS:
            out = []
            for i, w in enumerate(words):
                if _invariant(w) or i > head_i:
                    out.append(w)
                    continue
                parses = [p for p in morph.parse(w) if p.tag.POS in ("NOUN", "ADJF")] or morph.parse(w)
                p = parses[0]
                # the uk dictionary marks only plural; singular = case (+ gender for adjectives)
                grams = {case, "plur"} if num == "plur" else {case}
                if p.tag.POS == "ADJF" and num == "sing" and head_gender:
                    grams.add(head_gender)
                inf = p.inflect(grams)
                if inf is None:
                    out = None
                    break
                out.append(inf.word)
            if out:
                forms[(case, num)] = " ".join(out)
    return forms


@lru_cache(maxsize=1)
def tech_table() -> list[dict]:
    rows = []
    for line in (RESOURCES / "tech_terms.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        a, b, kind = line.split("\t")
        fa, fb = phrase_forms(a), phrase_forms(b)
        for src, tgt, fsrc, ftgt in ((a, b, fa, fb), (b, a, fb, fa)):
            rows.append({"src": src, "tgt": tgt, "kind": kind, "fsrc": fsrc, "ftgt": ftgt,
                         "tgt_invariant": all(_invariant(w) for w in tgt.split())})
    return rows


def tech_variants(text: str, protected: list[dict]) -> list[tuple[str, list[list[str]], str]]:
    """All single technical substitutions applicable to `text`."""
    out, seen = [], set()
    for row in tech_table():
        # longest surface forms first so 'шкідливого програмного забезпечення' wins over 'програмного забезпечення'
        for key, surf in sorted(row["fsrc"].items(), key=lambda kv: -len(kv[1])):
            m = re.search(rf"(?<!\w){re.escape(surf)}(?!\w)", text, re.IGNORECASE)
            if not m or any(p["start"] < m.end() and m.start() < p["end"] for p in protected):
                continue
            new = row["tgt"] if row["tgt_invariant"] else row["ftgt"].get(key)
            if not new:
                continue
            if m.group(0)[:1].isupper() and not new[:1].isupper():
                new = new[0].upper() + new[1:]
            variant = text[:m.start()] + new + text[m.end():]
            if variant != text and variant not in seen:
                seen.add(variant)
                out.append((variant, [[m.group(0), new]], row["kind"]))
            break
    return out


# ---------------------------------------------------------------- llm

LLM_SYSTEM = "You create meaning-preserving lexical variants of Ukrainian search queries for a robustness benchmark. Answer with JSON only."
LLM_USER = """Ukrainian query: "{q}"

Create 3 different variants. In each variant replace exactly {n} ordinary word(s) with a contextually appropriate Ukrainian synonym, keeping correct inflection and grammar.
Rules: do not change any other words; do not change identifiers, numbers, Latin-script names, malware or group names (e.g. UAC-0050, CVE-..., SmokeLoader); the variant must keep the same meaning and search intent.
Format: {{"variants": [{{"text": "...", "changed": [["old word", "new word"]]}}]}}"""


def word_diff(a: str, b: str) -> list[list[str]]:
    ta, tb = re.findall(r"[\w'’-]+", a), re.findall(r"[\w'’-]+", b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append([" ".join(ta[i1:i2]), " ".join(tb[j1:j2])])
    return out


# ---------------------------------------------------------------- compose

def pool_variants(q: dict, pool: dict | None, schedule: list[int], rng: random.Random, cos_cache: dict) -> list[tuple[str, list[list[str]], int]]:
    if not pool or not pool["positions"]:
        return []
    tokens = pool["tokens"]
    assert "".join(tokens) == q["original_query"], "pool tokens do not reconstruct the query"
    positions = pool["positions"]
    # top-3 candidates per position by SBERT similarity of the single-substitution query
    best = {}
    for p in positions:
        texts = [apply_subs(tokens, {p["idx"]: c}) for c in p["cands"]]
        scored = sorted(zip(p["cands"], (cos_cache[t] for t in texts)), key=lambda x: -x[1])
        best[p["idx"]] = [c for c, _ in scored[:3]]
    out, used_sets = [], set()
    order = [p["idx"] for p in positions]
    rng.shuffle(order)
    cursor = 0
    for n in schedule:
        if n > len(order):
            continue
        # rotate through positions so single-substitution variants touch different words
        chosen = tuple(sorted((order[(cursor + k) % len(order)] for k in range(n))))
        cursor += n
        if chosen in used_sets:
            continue
        used_sets.add(chosen)
        subs = {i: rng.choice(best[i]) for i in chosen}
        out.append((apply_subs(tokens, subs), [[tokens[i], subs[i]] for i in chosen], n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--types", default="dict,mlm,tech,llm")
    ap.add_argument("--schedule", default="1,1,2", help="number of substitutions of each dict/mlm variant")
    ap.add_argument("--name", default="variants")
    ap.add_argument("--llm-model", default=DEFAULT_LLM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--write-pool-input", action="store_true", help="only write pool_input_queries.jsonl")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    queries = read_jsonl(args.queries or out / "queries.jsonl")

    if args.write_pool_input:
        write_jsonl(out / "pool_input_queries.jsonl",
                    ({"tid": q["query_id"], "text": q["original_query"], "protected": q["protected_entities"]} for q in queries))
        return
    target = out / f"{args.name}.jsonl"
    if done(target, args.force):
        return

    types = args.types.split(",")
    schedule = [int(x) for x in args.schedule.split(",")]
    rng = random.Random(args.seed)
    rows: list[dict] = []

    def add(q: dict, j: int, ptype: str, text: str, changed: list, n_subs: int, kind: str | None = None) -> None:
        if text == q["original_query"]:
            return
        if not entities_preserved(q["protected_entities"], text):
            log(f"drop {ptype} variant of {q['query_id']}: protected entity changed")
            return
        rows.append({"variant_id": f"{q['query_id']}|{ptype}|{j}", "query_id": q["query_id"], "subset": q["subset"],
                     "article_id": q["article_id"], "chunk_id": q["chunk_id"], "original_query": q["original_query"],
                     "synonym_query": text, "perturbation_type": ptype, "tech_kind": kind, "n_subs": n_subs,
                     "changed_words": changed, "protected_entities": q["protected_entities"],
                     "has_identifier": q["has_identifier"]})

    for ptype in ("dict", "mlm"):
        if ptype not in types:
            continue
        pools = {r["tid"]: r for r in read_jsonl(out / f"pool_{ptype}_queries.jsonl")}
        singles = [apply_subs(p["tokens"], {pos["idx"]: c}) for p in pools.values() for pos in p["positions"] for c in pos["cands"]]
        originals = {p["tid"]: p["text"] for p in pools.values()}
        pairs = []
        for p in pools.values():
            for pos in p["positions"]:
                for c in pos["cands"]:
                    pairs.append((originals[p["tid"]], apply_subs(p["tokens"], {pos["idx"]: c})))
        cos_cache = dict(zip((b for _, b in pairs), sbert_cos(pairs)))
        log(f"{ptype}: {len(singles)} single-substitution candidates scored")
        for q in queries:
            for j, (text, changed, n) in enumerate(pool_variants(q, pools.get(q["query_id"]), schedule, rng, cos_cache)):
                add(q, j, ptype, text, changed, n)

    if "tech" in types:
        for q in queries:
            tv = tech_variants(q["original_query"], q["protected_entities"])
            rng.shuffle(tv)
            for j, (text, changed, kind) in enumerate(tv[:3]):
                add(q, j, "tech", text, changed, 1, kind)

    if "llm" in types:
        llm = LLM(args.llm_model, cache_path=OUT / "llm_cache.jsonl")
        n_for = {qq["query_id"]: (1 if i % 2 == 0 else 2) for i, qq in enumerate(queries)}
        answers = llm.map(lambda q: (LLM_SYSTEM, LLM_USER.format(q=q["original_query"], n=n_for[q["query_id"]])), queries, desc="llm-variants")
        for q, ans in zip(queries, answers):
            seen = set()
            for j, v in enumerate((ans or {}).get("variants", [])[:3]):
                text = (v.get("text") or "").strip().strip('"')
                if not text or text in seen:
                    continue
                seen.add(text)
                changed = word_diff(q["original_query"], text)
                if not changed or sum(len(a.split()) for a, _ in changed) > 3:
                    continue  # not a small lexical edit
                add(q, j, "llm", text, changed, len(changed))

    n = write_jsonl(target, rows)
    by_type = {t: sum(r["perturbation_type"] == t for r in rows) for t in types}
    covered = {t: len({r["query_id"] for r in rows if r["perturbation_type"] == t}) for t in types}
    log(f"variants: {n} {by_type}; queries covered {covered} of {len(queries)}")


if __name__ == "__main__":
    main()
