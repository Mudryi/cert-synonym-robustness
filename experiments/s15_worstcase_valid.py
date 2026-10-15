"""S15: validity-aware worst-case search (validate -> select).

S7 picks the margin-minimising substitution first and validates afterwards, so
its valid attacks are a lower bound. Here validity constrains the search:

  1. admissible single substitutions exactly as in S7 (pool cands[:max_cands],
     SBERT cos >= min_cos to the original);
  2. every admissible single is judged ONCE with the S3 prompt (shared by all
     attacked retrievers; responses cached, identical prompts reuse S3 verdicts);
  3. per attacked retriever and (query, source): step 1 = the judge-valid single
     that lowers the source document's doc_max margin most; step 2 = combine it
     with a judge-valid single at another position, judging the --step2-judge
     most damaging combinations in order and keeping the first valid one.

Output `<name>.jsonl` in the S2 format (then S3 -> S4 -> S14 as for S7), plus
`<name>_search.json` with the search cost (retriever calls, judge calls).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import retrievers as R
from experiments.common import (DEFAULT_LLM, LLM, OUT, apply_subs, done, entities_preserved, log, read_jsonl, sbert_cos,
                                write_jsonl)
from experiments.s3_judge import JUDGE_SYSTEM, JUDGE_USER
from experiments.s7_worstcase import margin_of


def judge(llm: LLM, rows: list[tuple[str, str, list]], cos_threshold: float, cos: list[float]) -> list[bool]:
    """S3 automatic rule for (original, variant, changed_words) rows: judge says meaning preserved AND cos >= threshold."""
    verdicts = llm.map(lambda r: (JUDGE_SYSTEM, JUDGE_USER.format(
        orig=r[0], new=r[1], changed="; ".join(f"{a} → {b}" for a, b in r[2]))), rows, desc="judge")
    return [bool((v or {}).get("preserves_meaning", False)) and c >= cos_threshold for v, c in zip(verdicts, cos)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants_worstv")
    ap.add_argument("--models", default="bm25,e5_base,bge_m3,hybrid")
    ap.add_argument("--sources", default="dict,mlm")
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--min-cos", type=float, default=0.85)
    ap.add_argument("--cos-threshold", type=float, default=0.80)
    ap.add_argument("--step2-judge", type=int, default=2, help="most damaging step-2 combinations judged per chain")
    ap.add_argument("--limit-queries", type=int, default=0, help="smoke test: only the first N queries")
    ap.add_argument("--llm-model", default=DEFAULT_LLM)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    target = out / f"{args.name}.jsonl"
    if done(target, args.force):
        return
    corpus = R.Corpus()
    queries = {q["query_id"]: q for q in read_jsonl(out / "queries.jsonl")}
    if args.limit_queries:
        queries = dict(list(queries.items())[:args.limit_queries])
    pools = {s: {r["tid"]: r for r in read_jsonl(out / f"pool_{s}_queries.jsonl") if r["tid"] in queries}
             for s in args.sources.split(",")}
    llm = LLM(args.llm_model, cache_path=OUT / "llm_cache.jsonl")

    # 1-2: admissible singles, each judged once
    keys, texts = [], []
    for s, pool in pools.items():
        for qid, p in pool.items():
            for pos in p["positions"]:
                for c in pos["cands"][:args.max_cands]:
                    keys.append((s, qid, pos["idx"], c))
                    texts.append(apply_subs(p["tokens"], {pos["idx"]: c}))
    cos = sbert_cos([(pools[s][qid]["text"], t) for (s, qid, _, _), t in zip(keys, texts)])
    adm = [(k, t, c) for k, t, c in zip(keys, texts, cos) if c >= args.min_cos]
    uniq = {}
    for (s, qid, idx, c), t, cs in adm:
        uniq.setdefault((qid, t), (queries[qid]["original_query"], t, [[pools[s][qid]["tokens"][idx], c]], cs))
    log(f"admissible singles: {len(adm)} (of {len(keys)}), {len(uniq)} unique texts to judge")
    t0 = time.time()
    ok = dict(zip(uniq, judge(llm, [u[:3] for u in uniq.values()], args.cos_threshold, [u[3] for u in uniq.values()])))
    log(f"singles judged in {time.time() - t0:.0f}s: {sum(ok.values())} valid of {len(ok)}")
    singles: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for (s, qid, idx, c), t, _ in adm:
        if ok[(qid, t)]:
            singles[(s, qid)].append((idx, c))

    rows, cost = [], {}
    for model in args.models.split(","):
        t0 = time.time()
        r = R.get(model, corpus)
        plans = []  # (s, qid, step-1 choice, step-1 margin, ranked step-2 combos)
        calls = []
        for (s, qid), cands in singles.items():
            q = queries[qid]
            tokens = pools[s][qid]["tokens"]
            gold = corpus.doc_pos[q["article_id"]]
            prev = margin_of(r.score([q["original_query"]])["doc_max"][0], gold)
            mat = r.score([apply_subs(tokens, {i: c}) for i, c in cands])["doc_max"]
            margins = [margin_of(mat[j], gold) for j in range(len(cands))]
            n_calls = 1 + len(cands)
            j = int(np.argmin(margins))
            if margins[j] >= prev:
                calls.append(n_calls)
                continue  # no valid single lowers the margin
            first = cands[j]
            text1 = apply_subs(tokens, {first[0]: first[1]})
            if not entities_preserved(q["protected_entities"], text1):
                calls.append(n_calls)
                continue
            opts = [(i, c) for i, c in cands if i != first[0]]
            combos = []
            if opts:
                m2 = r.score([apply_subs(tokens, {first[0]: first[1], i: c}) for i, c in opts])["doc_max"]
                n_calls += len(opts)
                mm = [margin_of(m2[k], gold) for k in range(len(opts))]
                combos = [(opts[k], mm[k]) for k in np.argsort(mm) if mm[k] < margins[j]][:args.step2_judge]
            calls.append(n_calls)
            plans.append((s, qid, first, margins[j], combos))
        if model not in ("bm25", "bge_m3"):
            R.release_all()

        # judge the step-2 candidates of all chains in parallel, keep the first valid one per chain
        need = {}
        for s, qid, first, _, combos in plans:
            tokens = pools[s][qid]["tokens"]
            for (i, c), _ in combos:
                subs = dict(sorted({first[0]: first[1], i: c}.items()))
                t = apply_subs(tokens, subs)
                need.setdefault((qid, t), (queries[qid]["original_query"], t, [[tokens[k], v] for k, v in subs.items()]))
        cos2 = sbert_cos([(u[0], u[1]) for u in need.values()]) if need else []
        ok2 = dict(zip(need, judge(llm, list(need.values()), args.cos_threshold, cos2))) if need else {}
        n_judge = []
        for s, qid, first, m1, combos in plans:
            q = queries[qid]
            tokens = pools[s][qid]["tokens"]
            base = {"query_id": qid, "subset": q["subset"], "article_id": q["article_id"], "chunk_id": q["chunk_id"],
                    "original_query": q["original_query"], "perturbation_type": f"worstv_{s}", "attack_model": model,
                    "tech_kind": None, "protected_entities": q["protected_entities"], "has_identifier": q["has_identifier"]}
            rows.append(base | {"variant_id": f"{qid}|worstv_{s}_{model}|1", "synonym_query": apply_subs(tokens, {first[0]: first[1]}),
                                "n_subs": 1, "changed_words": [[tokens[first[0]], first[1]]], "attack_margin": m1})
            judged = 0
            for (i, c), m2 in combos:
                subs = dict(sorted({first[0]: first[1], i: c}.items()))
                t = apply_subs(tokens, subs)
                judged += 1
                if ok2[(qid, t)] and entities_preserved(q["protected_entities"], t):
                    rows.append(base | {"variant_id": f"{qid}|worstv_{s}_{model}|2", "synonym_query": t, "n_subs": 2,
                                        "changed_words": [[tokens[k], v] for k, v in subs.items()], "attack_margin": m2})
                    break
            n_judge.append(judged)
        cost[model] = {"chains": len(singles), "chains_with_valid_step1": len(plans),
                       "retriever_calls_per_chain_mean": float(np.mean(calls)) if calls else 0.0,
                       "retriever_calls_per_chain_max": int(np.max(calls)) if calls else 0,
                       "step2_judge_calls_per_chain_mean": float(np.mean(n_judge)) if n_judge else 0.0}
        log(f"{model}: validity-aware search over {len(singles)} chains in {time.time() - t0:.0f}s; {len(plans)} with a valid "
            f"margin-lowering step 1; {len(need)} step-2 combinations judged")
    R.release_all()
    cost["shared"] = {"admissible_singles": len(adm), "unique_singles_judged": len(uniq), "valid_singles": int(sum(ok.values())),
                      "chains_with_any_valid_single": len(singles), "step2_judge": args.step2_judge,
                      "llm_spent_usd_total": round(llm.spent, 2)}
    (out / f"{args.name}_search.json").write_text(json.dumps(cost, indent=1), encoding="utf-8")
    write_jsonl(target, rows)
    log(f"validity-aware worst-case variants: {len(rows)} -> {target}")


if __name__ == "__main__":
    main()
