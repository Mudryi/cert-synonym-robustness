"""S14: worst-case robustness on a fixed query denominator, with human validity.

S7 searched, for every (query, source in {dict, mlm}, attacked retriever), a
greedy chain of 1-2 substitutions that lowers the source document's margin;
S3 judged the chain's variants AFTER the search (select -> validate), so the
results are a lower bound on the strongest valid attack. Here every original
query stays in the denominator:

  attacked rank = rank of the attacker's strongest valid variant of the query
                  (worst rank on the ATTACKED retriever, ties by lower margin);
                  the original rank when the query has no valid variant.

Validity levels
  auto       S3 rule (GPT judge + SBERT >= 0.8)
  verified   only variants a human labelled strict-valid count (lower bound)
  estimated  human labels where present; unlabelled variants are drawn valid
             with p ~ Beta(1 + k, 1 + n - k) from audit strata (auto label x
             source x outcome on the attacker); 1000 draws, each combined with
             an article-cluster bootstrap -> mean and 95% interval
  (+ lenient versions of verified / estimated)

Outputs: tier2/report_worst_verified[_<name>].md and summary_worst_verified[_<name>].json
(no suffix for the S7 set `variants_worst`; `--name variants_worstv` for the S15 validity-aware search).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import EXP, apply_subs, log, read_jsonl
from experiments.s6_report import cluster_ci, load_runs, md_table, pct

AUDIT = EXP / "audit"
ATTACKERS = ["bm25", "e5_base", "bge_m3", "hybrid"]
FROZEN = ["bm25", "e5_base", "bge_m3", "hybrid", "e5_large", "m3_all"]
LEVEL = "doc_max"
N_DRAWS = 1000
PRIOR_N = 5  # pseudo-labels pulling each audit cell toward the pooled rate of its automatic label


def outcome(ro: int, rv: int) -> str:
    return "out of top 5" if ro <= 5 < rv else "worsened" if ro <= 10 and rv > ro else "unchanged"


def budget(out: Path, W: pd.DataFrame, max_cands: int, min_cos: float) -> pd.DataFrame:
    """Search cost per attacker, recomputing S7's admissible singles (cached)."""
    cache = out / "worst_budget.json"
    if cache.exists():
        singles = json.loads(cache.read_text(encoding="utf-8"))
    else:
        from experiments.common import sbert_cos
        pairs, keys = [], []
        for s in ("dict", "mlm"):
            for p in read_jsonl(out / f"pool_{s}_queries.jsonl"):
                for pos in p["positions"]:
                    for c in pos["cands"][:max_cands]:
                        pairs.append((p["text"], apply_subs(p["tokens"], {pos["idx"]: c})))
                        keys.append((s, p["tid"], pos["idx"], pos["word"]))
        singles = defaultdict(list)
        for (s, qid, idx, word), cos in zip(keys, sbert_cos(pairs)):
            if cos >= min_cos:
                singles[f"{s}|{qid}"].append([idx, word])
        cache.write_text(json.dumps(singles, ensure_ascii=False), encoding="utf-8")
    rows = []
    for m in ATTACKERS:
        calls, judge, chains = [], [], 0
        for key, cands in singles.items():
            s, qid = key.split("|", 1)
            chains += 1
            ch = W[(W.attack_model == m) & (W.query_id == qid) & (W.perturbation_type == f"worst_{s}")]
            n1 = len(cands)
            c = 1 + n1
            if len(ch):  # a step-2 search over positions other than the step-1 substitution followed
                w1 = ch.sort_values("n_subs").changed_words.iloc[0][0][0]
                c += sum(1 for _, w in cands if w != w1)
            calls.append(c)
            judge.append(len(ch))
        rows.append({"attacked retriever": m, "chains searched": chains, "mean admissible candidates": f"{np.mean([len(v) for v in singles.values()]):.1f}",
                     "retriever calls / chain (mean)": f"{np.mean(calls):.1f}", "retriever calls / chain (max)": int(np.max(calls)),
                     "judged variants / chain": f"{np.mean(judge):.2f}"})
    return pd.DataFrame(rows), singles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants_worst")
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--min-cos", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--draws", type=int, default=N_DRAWS, help="Monte Carlo x bootstrap draws for the estimated levels")
    args = ap.parse_args()
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    Q = pd.DataFrame(read_jsonl(out / "queries.jsonl"))
    W = pd.DataFrame(read_jsonl(out / f"{args.name}_judged.jsonl"))
    W = W[~W.get("judge_missing", False).astype(bool)].reset_index(drop=True)
    runs = load_runs(str(out / "runs" / args.name / "*.jsonl"))
    runs = runs[runs.level == LEVEL]
    ft = sorted(m for m in runs.model.unique() if m.startswith("ft_"))
    ft_groups = defaultdict(list)  # ft_<base>_<cond> -> seeds
    for m in ft:
        ft_groups[re.sub(r"_s\d+$", "", m)].append(m)
    evaluated = [m for m in FROZEN if m in set(runs.model)] + ft
    log(f"{len(Q)} queries, {len(W)} worst-case variants, evaluated: {', '.join(evaluated)}")

    qpos = {q: i for i, q in enumerate(Q.query_id)}
    vpos = {v: i for i, v in enumerate(W.variant_id)}
    R_orig = np.full((len(Q), len(evaluated)), np.nan)
    R_var = np.full((len(W), len(evaluated)), np.nan)
    for j, m in enumerate(evaluated):
        g = runs[runs.model == m]
        o, v = g[g.kind == "orig"], g[g.kind == "variant"]
        R_orig[[qpos[i] for i in o.id], j] = o["rank"].to_numpy()
        keep = v.id.isin(vpos)
        R_var[[vpos[i] for i in v.id[keep]], j] = v["rank"][keep].to_numpy()
    assert not np.isnan(R_orig).any(), "missing original ranks"
    R_var = np.where(np.isnan(R_var), R_orig[[qpos[q] for q in W.query_id]], R_var)
    col = {m: j for j, m in enumerate(evaluated)}

    # per-variant facts on the attacker
    W["q"] = W.query_id.map(qpos)
    W["rank_att"] = [R_var[i, col[m]] for i, m in enumerate(W.attack_model)]
    W["rank_att_orig"] = [R_orig[q, col[m]] for q, m in zip(W.q, W.attack_model)]
    W["outcome"] = [outcome(a, b) for a, b in zip(W.rank_att_orig, W.rank_att)]
    # attacker's preference: worst rank on its own model, then lowest margin
    W = W.sort_values(["attack_model", "q", "rank_att", "attack_margin"], ascending=[True, True, False, True]).reset_index(drop=True)
    R_var = R_var[[vpos[v] for v in W.variant_id]]

    # human labels and audit strata
    # human labels, matched by (query, variant text) so that identical substitutions found by another search inherit them
    vtext = {}
    for f in sorted(out.glob("variants*_judged.jsonl")):
        for v in read_jsonl(f):
            vtext[v["variant_id"]] = (v["query_id"], v["synonym_query"])
    # the same text can carry several independent labels (found by several attacks): it counts as valid only if all agree
    H = defaultdict(list)
    for r in read_jsonl(AUDIT / "human_labels.jsonl"):
        if r["variant_id"] in vtext:
            H[vtext[r["variant_id"]]].append(r)
    conflicts = sum(len({r["human_valid_strict"] for r in rs}) > 1 for rs in H.values())
    log(f"human labels: {len(H)} variant texts, {sum(len(rs) > 1 for rs in H.values())} labelled more than once, "
        f"{conflicts} with conflicting strict verdicts (resolved as invalid)")
    hk = list(zip(W.query_id, W.synonym_query))
    for crit in ("strict", "lenient", "natural"):
        W[f"h_{crit}"] = [float(all(r.get(f"human_valid_{crit}", False) for r in H[k])) if k in H else np.nan for k in hk]
    labelled = W.h_strict.notna().to_numpy()
    W["cell"] = list(zip(W.valid.astype(bool), W.perturbation_type, W.outcome))
    # per-cell label counts, shrunk toward the pooled rate of the same automatic label (prior strength PRIOR_N):
    # cells hold only 2-21 labels, and a flat prior would pull sparse cells toward 50% validity
    cell_stats = {}
    for crit in ("strict", "lenient"):
        lab = W[labelled].drop_duplicates("variant_id")
        h = lab[f"h_{crit}"].astype(float)
        st, pooled = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
        for c, v in zip(lab.cell, h):
            for d, k_ in ((st, c), (pooled, c[0])):
                d[k_][0] += v
                d[k_][1] += 1
        k = np.array([st[c][0] if c in st else 0.0 for c in W.cell])
        n = np.array([st[c][1] if c in st else 0 for c in W.cell], float)
        p0 = np.array([pooled[c[0]][0] / pooled[c[0]][1] for c in W.cell])
        cell_stats[crit] = (k, n, p0)
        log(f"{crit}: audit strata {dict((c, f'{int(a)}/{m}') for c, (a, m) in st.items())}; "
            f"pooled {dict((c, f'{int(a)}/{m}') for c, (a, m) in pooled.items())}")

    att_of = W.attack_model.to_numpy()
    q_of = W.q.to_numpy()
    nQ = len(Q)
    art = Q.article_id.to_numpy()
    art_codes, art_idx = np.unique(art, return_inverse=True)
    has_chain = {m: np.isin(np.arange(nQ), q_of[att_of == m]) for m in ATTACKERS}

    def attacked(valid: np.ndarray) -> dict[str, np.ndarray]:
        """attacker -> index into W of the chosen valid variant per query (-1 = none)."""
        res = {}
        for m in ATTACKERS:
            sel = np.flatnonzero((att_of == m) & valid)
            ch = np.full(nQ, -1)
            qs = q_of[sel]
            first = np.unique(qs, return_index=True)[1]  # W is sorted: first valid row = strongest
            ch[qs[first]] = sel[first]
            res[m] = ch
        return res

    def per_query(ch: np.ndarray, j: int) -> dict[str, np.ndarray]:
        ro = R_orig[:, j]
        ra = np.where(ch >= 0, R_var[np.maximum(ch, 0), j], ro)
        return {"ro": ro, "ra": ra}

    def metrics(pq: dict, w: np.ndarray | None = None) -> dict[str, float]:
        ro, ra = pq["ro"], pq["ra"]
        w = np.ones_like(ro) if w is None else w
        avg = lambda x: float(np.sum(x * w) / np.sum(w))
        ok = (ro <= 5)
        return {"R@1 o": avg(ro <= 1), "R@1 a": avg(ra <= 1), "R@5 o": avg(ok), "R@5 a": avg(ra <= 5),
                "MRR o": avg(1 / ro), "MRR a": avg(1 / ra), "d5": avg(ok & (ra > 5)),
                "succ": float(np.sum((ok & (ra > 5)) * w)), "succ_cond": float(np.sum((ok & (ra > 5)) * w) / np.sum(ok * w))}

    # verified = accepted by the automatic rule AND labelled valid by the human; "incl. rejected" also lets human-valid
    # auto-rejected variants through (sensitivity), "natural" additionally requires fluency "natural"
    auto_v = W.valid.astype(bool).to_numpy()
    levels = {"auto": auto_v,
              "verified (strict)": auto_v & (W.h_strict == 1).to_numpy(),
              "verified (lenient)": auto_v & (W.h_lenient == 1).to_numpy(),
              "verified (natural)": auto_v & (W.h_natural == 1).to_numpy(),
              "verified incl. rejected (strict)": (W.h_strict == 1).to_numpy()}
    det = {lv: attacked(v) for lv, v in levels.items()}
    succ = {lv: {m: int(metrics(per_query(ch[m], col[m]))["succ"]) for m in ATTACKERS} for lv, ch in det.items()}
    for m in ATTACKERS:
        assert succ["verified (natural)"][m] <= succ["verified (strict)"][m] <= succ["verified (lenient)"][m] <= succ["auto"][m], m
    log("white-box successes " + "; ".join(f"{lv}: {d}" for lv, d in succ.items()))

    # Monte Carlo over unlabelled validity x article bootstrap. scope "accepted": unlabelled auto-rejected variants stay
    # invalid (how much of the automatic result survives human validation); "all": they are drawn too (judge false rejects)
    auto = W.valid.astype(bool).to_numpy()
    est = defaultdict(lambda: defaultdict(list))
    est_label = defaultdict(lambda: defaultdict(list))
    for scope in ("accepted", "all"):
        for crit in ("strict", "lenient"):
            if scope == "all" and crit == "lenient":
                continue
            k, n, p0 = cell_stats[crit]
            hv = W[f"h_{crit}"].to_numpy()
            for _ in range(args.draws):
                p = rng.beta(k + PRIOR_N * p0, n - k + PRIOR_N * (1 - p0))
                drawn = rng.random(len(W)) < p
                if scope == "accepted":
                    drawn &= auto
                valid = np.where(labelled, hv == 1, drawn)
                if scope == "accepted":
                    valid &= auto
                ch = attacked(valid)
                wts = np.bincount(rng.integers(0, len(art_codes), len(art_codes)), minlength=len(art_codes))[art_idx].astype(float)
                for m in ATTACKERS:
                    for e in evaluated:
                        est[(scope, crit)][(m, e)].append(metrics(per_query(ch[m], col[e]), wts))
                    # validity uncertainty alone (articles fixed), white-box only: separates it from article sampling
                    est_label[(scope, crit)][m].append(metrics(per_query(ch[m], col[m]))["succ"])

    md = ["# Worst-case robustness on a fixed query denominator, with human validity", "",
          f"Every one of the {nQ} original queries counts for every attacked retriever (doc_max). " + (
              "The attack search (S15) is validity-aware: every admissible single substitution is judged first and the greedy "
              "search runs over judge-valid candidates only (validate → select); validity here is still the automatic judge's, "
              "so human levels below re-check it. " if (out / f"{args.name}_search.json").exists() else
              "The attack search (S7) is greedy and validity-unaware: it picks the margin-minimising substitution first and the "
              "validity judge runs afterwards (select → validate), so a query without a valid variant may still have one outside "
              "the greedy path; all numbers are lower bounds on the strongest valid attack. ") +
          "The attacker uses its strongest valid variant (worst rank on the attacked retriever); "
          "a query without one keeps its original rank.", "",
          "Validity levels: **auto** = GPT judge + SBERT ≥ 0.8; **verified** = only variants a human labelled valid (lower "
          f"bound; {int(labelled.sum())} of {len(W)} variants are labelled); **estimated** = human labels where present, "
          "unlabelled auto-accepted variants drawn valid from Beta posteriors of the audit strata (auto label × source × outcome "
          f"on the attacker, shrunk toward the pooled rate of the automatic label with {PRIOR_N} pseudo-labels), {N_DRAWS} draws "
          "combined with an article bootstrap; **estimated incl. rejected** also draws the auto-rejected variants (judge false "
          "rejections), a sensitivity analysis resting on few labels of rejected variants. The audit over-represents "
          "rank-changing variants, so all estimates are sensitivity analyses rather than population corrections.", ""]
    summary: dict = {"n_queries": nQ, "n_variants": len(W), "n_labelled": int(labelled.sum()), "main": [], "outcomes": [], "transfer": {}}

    # outcome categories (white-box, deterministic levels + estimated means)
    rows = []
    for m in ATTACKERS:
        j = col[m]
        ro = R_orig[:, j]
        for lv, ch in list(det.items()):
            c = ch[m]
            ra = np.where(c >= 0, R_var[np.maximum(c, 0), j], ro)
            fail = ro > 5
            r = {"attacked retriever": m, "validity": lv, "originally failing": int(fail.sum()),
                 "no attack chain": int((~fail & ~has_chain[m]).sum()),
                 "chain, no valid variant": int((~fail & has_chain[m] & (c < 0)).sum()),
                 "valid, source stays in top 5": int((~fail & (c >= 0) & (ra <= 5)).sum()),
                 "successful attack": int((~fail & (c >= 0) & (ra > 5)).sum())}
            assert sum(v for k_, v in r.items() if isinstance(v, int)) == nQ
            r["coverage %"] = pct((r["valid, source stays in top 5"] + r["successful attack"]) / (nQ - r["originally failing"]))
            rows.append(r)
            summary["outcomes"].append(r)
    md += ["## Outcome of the attack per original query (white-box, R@5)", "",
           "Coverage = originally correct queries with at least one valid attack variant.", "", md_table(pd.DataFrame(rows)), ""]

    # main table
    def fmt_row(m, e, lv, mt, ci=None, succ_ci=None):
        return {"attacked": m, "evaluated": e, "validity": lv,
                "R@1 orig → att": f"{pct(mt['R@1 o'])} → {pct(mt['R@1 a'])}", "R@5 orig → att": f"{pct(mt['R@5 o'])} → {pct(mt['R@5 a'])}",
                "MRR orig → att": f"{mt['MRR o']:.3f} → {mt['MRR a']:.3f}",
                "ΔR@5 (pp)": pct(mt["d5"]) + (f" [{pct(ci[0])}, {pct(ci[1])}]" if ci else ""),
                "successful queries": (f"{mt['succ']:.1f}" if succ_ci else f"{int(mt['succ'])}") + (f" [{succ_ci[0]:.0f}, {succ_ci[1]:.0f}]" if succ_ci else ""),
                "success | orig. correct %": pct(mt["succ_cond"])}

    def det_row(m, e, lv):
        pq = per_query(det[lv][m], col[e])
        mt = metrics(pq)
        d = pd.DataFrame({"article_id": art, "d5": ((pq["ro"] <= 5) & (pq["ra"] > 5)).astype(float)})
        return fmt_row(m, e, lv, mt, cluster_ci(d, "d5", rng)), mt

    EST = {"estimated (strict)": ("accepted", "strict"), "estimated (lenient)": ("accepted", "lenient"),
           "estimated incl. rejected (strict)": ("all", "strict")}

    def est_row(m, e, lv):
        draws = est[EST[lv]][(m, e)]
        mt = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}
        d5 = [d["d5"] for d in draws]
        sc = [d["succ"] for d in draws]
        return fmt_row(m, e, lv, mt, (np.percentile(d5, 2.5), np.percentile(d5, 97.5)),
                       (np.percentile(sc, 2.5), np.percentile(sc, 97.5))), mt

    rows = []
    for m in ATTACKERS:
        for lv in ("auto", "estimated (lenient)", "estimated (strict)", "verified (lenient)", "verified (strict)",
                   "verified (natural)", "estimated incl. rejected (strict)", "verified incl. rejected (strict)"):
            r, mt = est_row(m, m, lv) if lv in EST else det_row(m, m, lv)
            if lv in EST:
                lab_only = est_label[EST[lv]][m]
                r["succ. 95% (validity only)"] = f"[{np.percentile(lab_only, 2.5):.0f}, {np.percentile(lab_only, 97.5):.0f}]"
                mt["succ_ci_validity_only"] = [float(np.percentile(lab_only, 2.5)), float(np.percentile(lab_only, 97.5))]
            rows.append(r)
            summary["main"].append(r | {"raw": mt})
        # validity-only draws (articles fixed): the article bootstrap would add sampling noise around the bounds
        sm = {r["validity"]: r["raw"]["succ"] for r in summary["main"] if r["attacked"] == m and r["validity"] in levels}
        est_m = float(np.mean(est_label[EST["estimated (strict)"]][m]))
        assert sm["verified (strict)"] - 1e-9 <= est_m <= sm["auto"] + 1e-9, (m, sm, est_m)
    md += ["## White-box attack: retrieval quality before and after (doc_max, all queries)", "",
           "ΔR@5 = share of all queries whose source leaves the top 5 (95% CI: article bootstrap; for estimated levels the interval "
           "also covers the validity uncertainty). Successful queries: count, with 95% interval for estimated levels (article "
           "bootstrap and unlabelled validity combined); the last column isolates the validity uncertainty (articles fixed). "
           "**verified** requires the automatic rule and the human to accept the variant; **natural** additionally requires "
           "fluency 'natural'; **incl. rejected** also admits human-valid variants the automatic rule rejected.", "",
           md_table(pd.DataFrame(rows).drop(columns=["evaluated"]).fillna("")), ""]

    # by query type, white-box
    rows = []
    for m in ATTACKERS:
        for sub in ("title", "section"):
            mask = (Q.subset == sub).to_numpy()
            for lv in ("auto", "verified (strict)"):
                pq = per_query(det[lv][m], col[m])
                ok, hit = pq["ro"][mask] <= 5, pq["ra"][mask] > 5
                rows.append({"attacked": m, "query type": sub, "validity": lv, "n": int(mask.sum()),
                             "R@5 orig": pct(ok.mean()), "R@5 att": pct((pq["ra"][mask] <= 5).mean()),
                             "successful queries": int((ok & hit).sum())})
    md += ["## White-box attack by query type", "", md_table(pd.DataFrame(rows)), ""]

    # transfer
    for lv in ("auto", "estimated (strict)"):
        tab = []
        for m in ATTACKERS:
            row = {"attacked ↓ / evaluated →": m}
            for e in [x for x in FROZEN if x in col]:
                mt = est_row(m, e, lv)[1] if lv in EST else metrics(per_query(det[lv][m], col[e]))
                row[e] = f"{pct(mt['d5'])} ({mt['succ']:.0f})"
                summary["transfer"].setdefault(lv, {}).setdefault(m, {})[e] = mt["d5"]
            tab.append(row)
        md += [f"## Transfer across retrievers — {lv}", "",
               "Cells: ΔR@5 in pp over all queries (number of successful queries). The attack variant is chosen on the attacked "
               "retriever and evaluated unchanged on the others.", "", md_table(pd.DataFrame(tab)), ""]

    # fine-tuned models under attacks transferred from their frozen base
    if ft_groups:
        rows = []
        for grp, members in sorted(ft_groups.items()):
            base = "e5_base" if "e5_base" in grp else "bge_m3"
            for lv in ("auto", "verified (strict)"):
                vals = [metrics(per_query(det[lv][base], col[mm])) for mm in members]
                agg = lambda k: f"{100 * np.mean([v[k] for v in vals]):.1f} ± {100 * np.std([v[k] for v in vals], ddof=1) if len(vals) > 1 else 0:.1f}"
                rows.append({"model": grp, "seeds": len(members), "attack from": f"frozen {base}", "validity": lv,
                             "R@5 orig": agg("R@5 o"), "R@5 att": agg("R@5 a"), "ΔR@5 (pp)": agg("d5"),
                             "successful queries": f"{np.mean([v['succ'] for v in vals]):.1f}"})
            for lv in ("auto", "verified (strict)"):
                mt = metrics(per_query(det[lv][base], col[base]))
                rows.append({"model": f"{base} (frozen)", "seeds": 1, "attack from": f"frozen {base}", "validity": lv,
                             "R@5 orig": pct(mt["R@5 o"]), "R@5 att": pct(mt["R@5 a"]), "ΔR@5 (pp)": pct(mt["d5"]),
                             "successful queries": f"{mt['succ']:.0f}"})
        rows = pd.DataFrame(rows).drop_duplicates()
        md += ["## Fine-tuned models under worst-case attacks (5-fold by article)", "",
               "Attacks are those found on the frozen base model (transfer to the fine-tuned model, not a re-attack). Mean ± sd over seeds; "
               "the seed also sets the article-to-fold split.", "", md_table(rows), ""]
        summary["finetune"] = rows.to_dict("records")

    # random baseline on the same denominator: one random auto-valid dict/mlm substitution per query
    R0 = pd.DataFrame(read_jsonl(out / "variants_judged.jsonl"))
    R0 = R0[R0.valid.astype(bool) & R0.perturbation_type.isin(["dict", "mlm"])]
    rows = []
    for m in ATTACKERS:
        f = out / "runs" / "variants" / f"{m}.jsonl"
        if not f.exists():
            continue
        g = pd.DataFrame(read_jsonl(f))
        g = g[g.level == LEVEL]
        rv = dict(zip(g[g.kind == "variant"].id, g[g.kind == "variant"]["rank"]))
        by_q = defaultdict(list)
        for vid, qid in zip(R0.variant_id, R0.query_id):
            if vid in rv:
                by_q[qpos[qid]].append(rv[vid])
        ro = R_orig[:, col[m]]
        res = []
        for _ in range(200):
            ra = np.array([rng.choice(by_q[i]) if by_q.get(i) else ro[i] for i in range(nQ)])
            res.append(((ro <= 5) & (ra > 5)).sum())
        wb = metrics(per_query(det["auto"][m], col[m]))
        rows.append({"retriever": m, "queries with a valid random variant": len(by_q),
                     "random: ΔR@5 (pp)": pct(np.mean(res) / nQ), "random: successful queries": f"{np.mean(res):.1f}",
                     "worst-case (auto): ΔR@5 (pp)": pct(wb["d5"]), "worst-case (auto): successful queries": int(wb["succ"])})
    md += ["## Random vs. worst-case on the same denominator (auto validity, doc_max)", "",
           "Random: one auto-valid dictionary or MLM substitution drawn per query (mean of 200 draws); queries without one keep "
           "their original rank. Worst-case: the attacker's strongest valid variant.", "", md_table(pd.DataFrame(rows)), ""]
    summary["random_vs_worst"] = rows

    # budget
    search = out / f"{args.name}_search.json"
    if search.exists():  # S15 validity-aware search records its own cost
        cost = json.loads(search.read_text(encoding="utf-8"))
        sh = cost.pop("shared")
        bt = pd.DataFrame([{"attacked retriever": m} | {k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in c.items()}
                           for m, c in cost.items()])
        md += ["## Attack search budget (validity-aware search)", "",
               f"Every admissible single substitution (≤ {args.max_cands} candidates per position, SBERT cos ≥ {args.min_cos}) is judged "
               f"once before the search and shared by all attacked retrievers: {sh['unique_singles_judged']} judged, "
               f"{sh['valid_singles']} valid. The greedy search uses only valid singles; per chain the {sh['step2_judge']} most "
               "damaging two-word combinations are judged in order and the first valid one is kept.", "", md_table(bt), ""]
    else:
        bt, _ = budget(out, W, args.max_cands, args.min_cos)
        md += ["## Attack search budget", "",
               f"Per (query, source) chain: ≤ 2 substitutions, ≤ {args.max_cands} candidates per position from the dictionary or MLM pool, "
               f"SBERT cos ≥ {args.min_cos} to the original before scoring; one retriever call scores one candidate text; every "
               "variant on the chain is judged once afterwards.", "", md_table(bt), ""]
    summary["budget"] = bt.to_dict("records")

    # verified successful examples
    md += ["## Human-verified successful attacks (white-box)", ""]
    for m in ATTACKERS:
        ch = det["verified (strict)"][m]
        j = col[m]
        for qi in np.flatnonzero(ch >= 0):
            r = W.iloc[ch[qi]]
            if R_orig[qi, j] <= 5 < R_var[ch[qi], j]:
                lab = H[(r.query_id, r.synonym_query)]
                md.append(f"- **{m}** · {Q.subset[qi]} · rank {int(R_orig[qi, j])} → {int(R_var[ch[qi], j])}: "
                          f"«{r.original_query}» → «{r.synonym_query}» ({'; '.join(a + ' → ' + b for a, b in r.changed_words)}) "
                          f"[human: {', '.join(x['meaning'] + '/' + x['fluency'] + ' (b' + str(x.get('batch', '?')) + ')' for x in lab)}]")
    md.append("")

    # per-variant outcome on the attacker and ranks on the main retrievers (input for the batch-3 sampler in S12)
    with open(out / f"variant_outcomes_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(W.itertuples()):
            f.write(json.dumps({"variant_id": r.variant_id, "query_id": r.query_id, "attack_model": r.attack_model,
                                "valid": bool(r.valid), "rank_att_orig": int(r.rank_att_orig), "rank_att": int(r.rank_att),
                                "outcome": r.outcome, "labelled": bool(labelled[i]),
                                "ranks_doc_max": {m: [int(R_orig[r.q, col[m]]), int(R_var[i, col[m]])] for m in ATTACKERS}},
                               ensure_ascii=False) + "\n")

    sfx = "" if args.name == "variants_worst" else f"_{args.name}"
    (out / f"report_worst_verified{sfx}.md").write_text("\n".join(md), encoding="utf-8")
    (out / f"summary_worst_verified{sfx}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"wrote {out / f'report_worst_verified{sfx}.md'}")


if __name__ == "__main__":
    main()
