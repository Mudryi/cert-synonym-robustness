"""S16: protocol and budget audit of the two worst-case searches (S7 vs S15).

S7 (select -> validate) takes the margin-minimising admissible substitution and
judges it afterwards; S15 (validate -> select) judges every admissible single
first and searches only over valid ones. This script checks that both share
queries, candidate pools and constraints, explains their chain counts, and runs
a budget-matched control on step 1 (single substitutions), where every judge
verdict is already cached:

  backoff-J  rank the margin-lowering admissible singles of a chain by margin
             and judge them in that order, at most J judge calls; keep the
             first valid one (stop if it breaks a protected entity).
             J = 1 is S7's step 1, J = all is S15's step 1.

Both ends are checked against the variants S7 and S15 actually produced, which
shows that the two implementations differ only in where validation happens.
Successes are white-box, auto validity, step 1 only, over all 586 queries.

Output: tier2/report_protocol_audit.md and summary_protocol_audit.json.
Run with EXP_LLM_BUDGET_USD at the current spend so that a cache miss cannot
cost money (a miss then counts as invalid and is reported).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import retrievers as R
from experiments.common import DEFAULT_LLM, LLM, OUT, apply_subs, entities_preserved, log, read_jsonl, sbert_cos
from experiments.s6_report import md_table
from experiments.s7_worstcase import margin_of
from experiments.s15_worstcase_valid import JUDGE_SYSTEM, JUDGE_USER

ATTACKERS = ["bm25", "e5_base", "bge_m3", "hybrid"]
BUDGETS = [1, 2, 4, 8, 16, 0]  # judge calls per chain at step 1; 0 = unlimited


def rank_of(row: np.ndarray, gold: int) -> int:
    return int(1 + (row > row[gold]).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--min-cos", type=float, default=0.85)
    ap.add_argument("--cos-threshold", type=float, default=0.80)
    args = ap.parse_args()
    out = Path(args.out)
    corpus = R.Corpus()
    queries = {q["query_id"]: q for q in read_jsonl(out / "queries.jsonl")}
    pools = {s: {r["tid"]: r for r in read_jsonl(out / f"pool_{s}_queries.jsonl")} for s in ("dict", "mlm")}
    V7 = pd.DataFrame(read_jsonl(out / "variants_worst_judged.jsonl"))
    V15 = pd.DataFrame(read_jsonl(out / "variants_worstv_judged.jsonl"))
    search15 = json.loads((out / "variants_worstv_search.json").read_text(encoding="utf-8"))
    checks = []

    # 1. shared inputs: same queries, same admissible singles (code shared verbatim by S7 and S15)
    for name, V in (("S7", V7), ("S15", V15)):
        checks.append((f"{name} variants use only benchmark queries", bool(V.query_id.isin(queries).all())))
        checks.append((f"{name} keeps protected entities in every variant",
                       bool(all(entities_preserved(queries[q]["protected_entities"], t) for q, t in zip(V.query_id, V.synonym_query)))))
        checks.append((f"{name} uses at most 2 substitutions", bool((V.n_subs <= 2).all())))
    keys, texts = [], []
    for s, pool in pools.items():
        for qid, p in pool.items():
            for pos in p["positions"]:
                for c in pos["cands"][:args.max_cands]:
                    keys.append((s, qid, pos["idx"], c))
                    texts.append(apply_subs(p["tokens"], {pos["idx"]: c}))
    cos = sbert_cos([(pools[s][qid]["text"], t) for (s, qid, _, _), t in zip(keys, texts)])
    adm = [(k, t, c) for k, t, c in zip(keys, texts, cos) if c >= args.min_cos]
    checks.append(("admissible singles equal S15's recorded count", len(adm) == search15["shared"]["admissible_singles"]))

    # 2. judge verdicts for every admissible single (cache only)
    llm = LLM(DEFAULT_LLM, cache_path=OUT / "llm_cache.jsonl")
    spent0 = llm.spent
    uniq = {}
    for (s, qid, idx, c), t, cs in adm:
        uniq.setdefault((qid, t), (queries[qid]["original_query"], t, [[pools[s][qid]["tokens"][idx], c]], cs))
    verdicts = llm.map(lambda r: (JUDGE_SYSTEM, JUDGE_USER.format(orig=r[0], new=r[1], changed="; ".join(f"{a} → {b}" for a, b in r[2]))),
                       [u[:3] for u in uniq.values()], desc="judge (cached)")
    missing = sum(v is None for v in verdicts)
    ok = {k: bool((v or {}).get("preserves_meaning", False)) and u[3] >= args.cos_threshold
          for k, v, u in zip(uniq, verdicts, uniq.values())}
    checks.append((f"all {len(uniq)} single verdicts served from cache (misses: {missing}, new spend ${llm.spent - spent0:.2f})",
                   missing == 0 and llm.spent - spent0 < 0.01))
    chains = defaultdict(list)
    for (s, qid, idx, c), t, _ in adm:
        chains[(s, qid)].append((idx, c, t))
    n_valid_chain = sum(any(ok[(qid, t)] for _, _, t in cands) for (s, qid), cands in chains.items())
    s7_chains = len(json.loads((out / "worst_budget.json").read_text(encoding="utf-8")))
    checks.append((f"S7 chains ({s7_chains}, recomputed by S14) = (query, source) pairs with an admissible single ({len(chains)})",
                   s7_chains == len(chains)))
    checks.append(("S15 chains = pairs with a judge-valid single", n_valid_chain == search15["shared"]["chains_with_any_valid_single"]))

    # 3. budget-matched backoff on step 1, per attacker
    s7_step1 = {(r.query_id, r.perturbation_type[6:], r.attack_model): r.synonym_query for r in V7[V7.n_subs == 1].itertuples()}
    s15_step1 = {(r.query_id, r.perturbation_type[7:], r.attack_model): r.synonym_query for r in V15[V15.n_subs == 1].itertuples()}
    qpos = {q: i for i, q in enumerate(queries)}
    rows, eq_rows, calls_rows = [], [], []
    for m in ATTACKERS:
        t0 = time.time()
        r = R.get(m, corpus)
        succ = {J: np.zeros(len(queries), bool) for J in BUDGETS}
        orig_ok = np.zeros(len(queries), bool)
        judged = {J: [] for J in BUDGETS}
        match = {"S7 (J=1)": [0, 0], "S15 (J=all)": [0, 0]}
        n_calls = []
        for (s, qid), cands in chains.items():
            q = queries[qid]
            gold = corpus.doc_pos[q["article_id"]]
            row0 = r.score([q["original_query"]])["doc_max"][0]
            prev = margin_of(row0, gold)
            orig_ok[qpos[qid]] = rank_of(row0, gold) <= 5
            mat = r.score([t for _, _, t in cands])["doc_max"]
            n_calls.append(1 + len(cands))
            mg = np.array([margin_of(mat[j], gold) for j in range(len(cands))])
            order = [j for j in np.argsort(mg, kind="stable") if mg[j] < prev]  # margin-lowering, most damaging first
            for J in BUDGETS:
                pick, used = None, 0
                for j in order[:J or None]:
                    used += 1
                    if ok[(qid, cands[j][2])]:
                        pick = j if entities_preserved(q["protected_entities"], cands[j][2]) else None
                        break
                judged[J].append(used)
                if pick is not None and rank_of(mat[pick], gold) > 5:
                    succ[J][qpos[qid]] = True
                if J == 1 and order and entities_preserved(q["protected_entities"], cands[order[0]][2]):
                    ref = s7_step1.get((qid, s, m))  # S7 emits step 1 before validation
                    match["S7 (J=1)"][0] += ref == cands[order[0]][2]
                    match["S7 (J=1)"][1] += 1
                if J == 0 and pick is not None:
                    ref = s15_step1.get((qid, s, m))
                    match["S15 (J=all)"][0] += ref == cands[pick][2]
                    match["S15 (J=all)"][1] += 1
        if m not in ("bm25", "bge_m3"):
            R.release_all()
        log(f"{m}: {len(chains)} chains in {time.time() - t0:.0f}s")
        for J in BUDGETS:
            rows.append({"attacked retriever": m, "judge calls / chain (cap)": J or "all",
                         "judge calls / chain (mean used)": f"{np.mean(judged[J]):.2f}",
                         "successful queries (step 1)": int((succ[J] & orig_ok).sum())})
        for k, (a, n) in match.items():
            eq_rows.append({"attacked retriever": m, "reference": k, "chains compared": n, "same step-1 text %": f"{100 * a / max(n, 1):.1f}"})
        calls_rows.append({"attacked retriever": m, "retriever calls / chain, step 1 (all admissible scored)": f"{np.mean(n_calls):.1f}"})
    R.release_all()

    # 4. cost table of the full searches as run
    s14 = {n: json.loads((out / f"summary_worst_verified{sfx}.json").read_text(encoding="utf-8"))
           for n, sfx in (("S7", ""), ("S15", "_variants_worstv"))}
    s7_budget = {b["attacked retriever"]: b for b in s14["S7"]["budget"]}
    auto = {n: {r["attacked"]: r["raw"]["succ"] for r in s["main"] if r["validity"] == "auto"} for n, s in s14.items()}
    cost = []
    for m in ATTACKERS:
        n7 = int((V7.attack_model == m).sum())
        c15 = search15[m]
        cost.append({"attacked retriever": m,
                     "S7 chains": s7_budget[m]["chains searched"], "S15 chains": c15["chains"],
                     "S7 retriever calls / chain": s7_budget[m]["retriever calls / chain (mean)"],
                     "S15 retriever calls / chain": f"{c15['retriever_calls_per_chain_mean']:.1f}",
                     "S7 judge calls (variants judged)": n7,
                     "S15 judge calls (singles, shared ÷ 4 + step 2)":
                         f"{search15['shared']['unique_singles_judged'] / 4 + c15['step2_judge_calls_per_chain_mean'] * c15['chains_with_valid_step1']:.0f}",
                     "S7 successes (auto, 2 steps)": int(auto["S7"][m]), "S15 successes (auto, 2 steps)": int(auto["S15"][m])})

    md = ["# S7 vs S15: protocol and budget audit", "",
          "S7 = select → validate (greedy on margin, judge afterwards); S15 = validate → select (judge every admissible single, "
          "greedy over valid ones). Both use the same 586 queries, the same dictionary/MLM candidate pools (≤ "
          f"{args.max_cands} per position), the same SBERT pre-filter (cos ≥ {args.min_cos}), the same automatic rule (GPT judge + "
          f"SBERT ≥ {args.cos_threshold}), the same protected-entity check and ≤ 2 substitutions.", "",
          "## Consistency checks", ""] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks] + ["",
          f"Chains: S7 searches every (query, source) pair with an admissible single ({len(chains)}); S15 only pairs with at least one "
          f"judge-valid single ({n_valid_chain}). The {len(chains) - n_valid_chain} pairs S15 skips cannot yield an auto-valid "
          "variant in either search, so the difference in chain counts does not favour S15.", "",
          "## Implementation equivalence (step 1)", "",
          "The backoff search with J = 1 should reproduce S7's step-1 variant and with J = all S15's step-1 variant.", "",
          md_table(pd.DataFrame(eq_rows)), "",
          "## Budget-matched control: successes against judge calls (step 1, auto validity, white-box)", "",
          "Each chain ranks its margin-lowering admissible singles by margin and judges them in that order, at most J calls; the "
          "first valid one is the attack. Successful queries: source document leaves the attacked retriever's top 5, out of 586 "
          "(a query counts once over its dictionary and MLM chains).", "", md_table(pd.DataFrame(rows)), "",
          md_table(pd.DataFrame(calls_rows)), "",
          "## Cost of the searches as run", "",
          "Candidate generation (dictionary lookup, MLM proposals, SBERT pre-filter) is shared and identical. S15's judge calls on "
          f"singles ({search15['shared']['unique_singles_judged']} unique texts) are shared by the four attacked retrievers and shown "
          "here divided by four. S7 judges each variant on its greedy path once.", "", md_table(pd.DataFrame(cost)), ""]
    (out / "report_protocol_audit.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary_protocol_audit.json").write_text(json.dumps(
        {"checks": checks, "chains_s7": len(chains), "chains_s15": n_valid_chain, "equivalence": eq_rows, "budget_curve": rows,
         "cost": cost}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"wrote {out / 'report_protocol_audit.md'}")


if __name__ == "__main__":
    main()
