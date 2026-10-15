"""S7: worst-case (adversarial) meaning-preserving substitutions per retriever.

TextFooler-style greedy search where the victim is a retriever: at each step
try every single-word candidate from the dictionary / MLM pools of the
original query (subject to SBERT cos >= --min-cos with the original), keep the
one that minimises the source document's score margin
(gold score − best competing score, level doc_max), up to --max-subs words.
The found variants are written as `<name>.jsonl` in the S2 format so that the
regular S3 judge → S4 retrieval → S6 report pipeline evaluates them (incl.
transfer to the other retrievers). Validity is decided by S3, not here.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments import retrievers as R
from experiments.common import apply_subs, done, entities_preserved, log, read_jsonl, sbert_cos, write_jsonl


def margin_of(row: np.ndarray, gold: int) -> float:
    g = row[gold]
    other = np.max(np.delete(row, gold))
    return float(g - other)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants_worst")
    ap.add_argument("--models", default="bm25,e5_base,bge_m3,hybrid")
    ap.add_argument("--sources", default="dict,mlm")
    ap.add_argument("--max-subs", type=int, default=2)
    ap.add_argument("--min-cos", type=float, default=0.85)
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    target = out / f"{args.name}.jsonl"
    if done(target, args.force):
        return
    corpus = R.Corpus()
    queries = {q["query_id"]: q for q in read_jsonl(out / "queries.jsonl")}
    pools = {s: {r["tid"]: r for r in read_jsonl(out / f"pool_{s}_queries.jsonl")} for s in args.sources.split(",")}

    # admissible single substitutions (pre-filtered by SBERT similarity to the original)
    singles: dict[tuple[str, str], list[tuple[int, str]]] = {}
    pairs, keys = [], []
    for s, pool in pools.items():
        for qid, p in pool.items():
            for pos in p["positions"]:
                for c in pos["cands"][:args.max_cands]:
                    pairs.append((p["text"], apply_subs(p["tokens"], {pos["idx"]: c})))
                    keys.append((s, qid, pos["idx"], c))
    for (s, qid, idx, c), cos in zip(keys, sbert_cos(pairs)):
        if cos >= args.min_cos:
            singles.setdefault((s, qid), []).append((idx, c))
    log(f"admissible single substitutions: {sum(map(len, singles.values()))} (of {len(pairs)})")

    rows = []
    for model in args.models.split(","):
        t0 = time.time()
        r = R.get(model, corpus)
        n_q = 0
        for (s, qid), cands in singles.items():
            q = queries[qid]
            tokens = pools[s][qid]["tokens"]
            gold = corpus.doc_pos[q["article_id"]]
            chosen: dict[int, str] = {}
            prev = margin_of(r.score([q["original_query"]])["doc_max"][0], gold)
            for step in range(args.max_subs):
                options = [(i, c) for i, c in cands if i not in chosen]
                if not options:
                    break
                texts = [apply_subs(tokens, chosen | {i: c}) for i, c in options]
                mat = r.score(texts)["doc_max"]
                margins = [margin_of(mat[j], gold) for j in range(len(texts))]
                j = int(np.argmin(margins))
                if margins[j] >= prev:
                    break  # no further substitution lowers the margin
                prev = margins[j]
                chosen[options[j][0]] = options[j][1]
                text = texts[j]
                if not entities_preserved(q["protected_entities"], text):
                    break
                rows.append({"variant_id": f"{qid}|worst_{s}_{model}|{step + 1}", "query_id": qid, "subset": q["subset"],
                             "article_id": q["article_id"], "chunk_id": q["chunk_id"], "original_query": q["original_query"],
                             "synonym_query": text, "perturbation_type": f"worst_{s}", "attack_model": model,
                             "tech_kind": None, "n_subs": len(chosen),
                             "changed_words": [[tokens[i], c] for i, c in sorted(chosen.items())],
                             "protected_entities": q["protected_entities"], "has_identifier": q["has_identifier"],
                             "attack_margin": margins[j]})
            n_q += 1
        log(f"{model}: worst-case search over {n_q} (query, source) pairs in {time.time() - t0:.0f}s")
        if model not in ("bm25", "bge_m3"):
            R.release_all()
    R.release_all()
    # identical texts found by several attacks are judged/retrieved once per variant id; keep all ids
    write_jsonl(target, rows)
    log(f"worst-case variants: {len(rows)}")


if __name__ == "__main__":
    main()
