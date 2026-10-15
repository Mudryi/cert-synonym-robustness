"""S6: metrics, paired robustness drop with clustered bootstrap CIs, failures,
methods comparison (incl. selective activation), go/no-go.

Robustness is always paired: a variant is compared with its own original
query under the same model, level and method. Article-level and chunk-level
results are never mixed in one table.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import OUT, SEED, read_jsonl, write_json

MAIN_MODELS = ["bm25", "e5_base", "bge_m3", "hybrid"]
LEVEL_SUBSETS = [("doc_full", "title"), ("doc_max", "title"), ("doc_full", "section"), ("doc_max", "section"), ("chunk", "section")]
N_BOOT = 1000


def cluster_ci(df: pd.DataFrame, col: str, rng: np.random.Generator) -> tuple[float, float]:
    """95% bootstrap CI of mean(col), resampling source articles."""
    groups = [g[col].to_numpy() for _, g in df.groupby("article_id")]
    if len(groups) < 2:
        return float("nan"), float("nan")
    sums = np.array([g.sum() for g in groups])
    lens = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    means = sums[idx].sum(1) / lens[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def load_runs(pattern: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(pattern)):
        model = Path(f).stem.split("__")[0]
        df = pd.DataFrame(read_jsonl(f))
        if df.empty:
            continue
        df["model"] = model
        if "__" in Path(f).stem:
            df["tag"] = Path(f).stem.split("__")[1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired(runs: pd.DataFrame, queries: pd.DataFrame, variants: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    o = runs[runs.kind == "orig"][keys + ["id", "rank"]].rename(columns={"id": "query_id", "rank": "rank_orig"})
    v = runs[runs.kind == "variant"][keys + ["id", "rank"]].rename(columns={"id": "variant_id", "rank": "rank_var"})
    v = v.merge(variants, on="variant_id").merge(o, on=keys + ["query_id"])
    for k in (1, 5, 10):
        v[f"hit{k}_o"] = (v.rank_orig <= k).astype(float)
        v[f"hit{k}_v"] = (v.rank_var <= k).astype(float)
        v[f"d{k}"] = v[f"hit{k}_o"] - v[f"hit{k}_v"]
    v["rr_o"], v["rr_v"] = 1 / v.rank_orig, 1 / v.rank_var
    v["dmrr"] = v.rr_o - v.rr_v
    v["fallout5"] = ((v.rank_orig <= 5) & (v.rank_var > 5)).astype(float)
    v["gain5"] = ((v.rank_orig > 5) & (v.rank_var <= 5)).astype(float)
    return v


def robustness_rows(p: pd.DataFrame, group_cols: list[str], rng) -> pd.DataFrame:
    rows = []
    for key, g in p.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        lo, hi = cluster_ci(g, "d5", rng)
        rows.append(dict(zip(group_cols, key)) | {
            "n_var": len(g), "n_q": g.query_id.nunique(),
            "R@1 orig": pct(g.hit1_o.mean()), "R@1 pert": pct(g.hit1_v.mean()),
            "R@5 orig": pct(g.hit5_o.mean()), "R@5 pert": pct(g.hit5_v.mean()),
            "ΔR@5": pct(g.d5.mean()), "95% CI": f"[{pct(lo)}, {pct(hi)}]",
            "MRR orig": f"{g.rr_o.mean():.3f}", "MRR pert": f"{g.rr_v.mean():.3f}",
            "fall-out@5 %": pct(g.fallout5.mean()), "gain@5 %": pct(g.gain5.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants")
    ap.add_argument("--title", default="Report")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    rng = np.random.default_rng(SEED)
    queries = pd.DataFrame(read_jsonl(out / "queries.jsonl"))
    variants = pd.DataFrame(read_jsonl(out / f"{args.name}_judged.jsonl"))
    variants["subset"] = variants["subset"].astype(str)
    extra = [c for c in ("attack_model", "local_p_yes") if c in variants.columns]
    vmeta = variants[["variant_id", "query_id", "article_id", "subset", "perturbation_type", "tech_kind", "n_subs",
                      "has_identifier", "valid", "judge_preserves_meaning", "semantic_similarity", "changed_words",
                      "original_query", "synonym_query"] + extra]
    runs = load_runs(str(out / "runs" / args.name / "*.jsonl"))
    md: list[str] = [f"# {args.title}", ""]
    summary: dict = {"name": args.name}

    # ---------------- benchmark statistics
    md += ["## Benchmark", ""]
    qs = queries.groupby("subset").agg(n=("query_id", "size"), with_identifier=("has_identifier", "sum"),
                                       articles=("article_id", "nunique")).reset_index()
    md.append(md_table(qs))
    vs = variants.groupby("perturbation_type").agg(
        n=("variant_id", "size"), queries=("query_id", "nunique"), judge_ok=("judge_preserves_meaning", "mean"),
        valid=("valid", "mean"), mean_cos=("semantic_similarity", "mean")).reset_index()
    for c in ("judge_ok", "valid"):
        vs[c] = vs[c].map(pct)
    vs["mean_cos"] = vs["mean_cos"].map(lambda x: f"{x:.3f}")
    md += ["Variants per perturbation type (judge_ok / valid in %):", "", md_table(vs)]
    summary["validity"] = vs.to_dict("records")
    if (out / f"audit_sample_{args.name}.md").exists():
        md += [f"Manual audit sample: `audit_sample_{args.name}.md`.", ""]

    if runs.empty:
        (out / (args.report or f"report_{args.name}.md")).write_text("\n".join(md + ["_no runs_"]), encoding="utf-8")
        return

    # ---------------- clean retrieval
    md += ["## Clean retrieval (original queries)", "",
           "Pessimistic ranks (ties count against the positive). Titles are stripped from the index.", ""]
    o = runs[runs.kind == "orig"].merge(queries[["query_id", "subset", "has_identifier"]], left_on="id", right_on="query_id")
    clean = []
    for (lv, sub), g in o.groupby(["level", "subset"]):
        for m, gm in g.groupby("model"):
            clean.append({"level": lv, "subset": sub, "model": m, "n": len(gm), "R@1": pct((gm["rank"] <= 1).mean()),
                          "R@5": pct((gm["rank"] <= 5).mean()), "R@10": pct((gm["rank"] <= 10).mean()),
                          "MRR": f"{(1 / gm['rank']).mean():.3f}"})
    clean = pd.DataFrame(clean)
    for lv, sub in LEVEL_SUBSETS:
        t = clean[(clean.level == lv) & (clean.subset == sub)].drop(columns=["level", "subset"])
        if not t.empty:
            md += [f"### {lv} · {sub} queries", "", md_table(t)]

    # ---------------- robustness (paired)
    p_all = paired(runs, queries, vmeta, ["model", "level"])
    p_all = p_all[~((p_all.level == "chunk") & (p_all.subset != "section"))]
    p_valid = p_all[p_all.valid]
    md += ["## Robustness to valid meaning-preserving substitutions", "",
           "Paired: each valid variant vs. its own original. ΔR@5 = R@5(orig) − R@5(perturbed), percentage points, "
           "95% CI from a bootstrap over source articles. fall-out@5 = source was in top-5 for the original but not for the variant.", ""]
    summary["robustness"] = {}
    for lv, sub in LEVEL_SUBSETS:
        g = p_valid[(p_valid.level == lv) & (p_valid.subset == sub)]
        if g.empty:
            continue
        t = robustness_rows(g, ["model"], rng)
        md += [f"### {lv} · {sub} queries — all valid variants", "", md_table(t)]
        summary["robustness"][f"{lv}|{sub}"] = t.to_dict("records")
        t2 = robustness_rows(g, ["model", "perturbation_type"], rng)
        md += [f"#### by perturbation type", "", md_table(t2[["model", "perturbation_type", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI", "fall-out@5 %"]])]
        summary["robustness"][f"{lv}|{sub}|type"] = t2.to_dict("records")
        t3 = robustness_rows(g, ["model", "has_identifier"], rng)
        md += [f"#### with vs. without identifiers in the query", "", md_table(t3[["model", "has_identifier", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI"]])]

    # valid vs invalid sanity: invalid substitutions should hurt more
    md += ["### Valid vs. invalid substitutions (sanity check)", ""]
    vi = robustness_rows(p_all[p_all.level == "doc_max"], ["model", "valid"], rng)
    md.append(md_table(vi[["model", "valid", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI"]]))

    # worst-case transfer: attack found against one retriever, evaluated on all
    if "attack_model" in p_valid.columns:
        md += ["## Worst-case substitutions: transfer across retrievers (doc_max, valid variants)", "",
               "Rows: retriever attacked by the greedy search; columns: evaluated retriever; cells: ΔR@5 (pp).", ""]
        g = p_valid[p_valid.level == "doc_max"]
        tab = g.groupby(["attack_model", "model"]).d5.mean().unstack().map(pct) if not g.empty else pd.DataFrame()
        if not tab.empty:
            md.append(md_table(tab.reset_index()))
            summary["transfer"] = g.groupby(["attack_model", "model"]).d5.mean().unstack().to_dict()
        t = robustness_rows(g[g.attack_model == g.model], ["model", "n_subs"], rng)
        md += ["White-box (attacked = evaluated) by number of substituted words:", "",
               md_table(t[["model", "n_subs", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI", "fall-out@5 %"]]) if not t.empty else "_none_\n"]

    # dose-response (number of substitutions), useful for the dose run
    if p_valid.n_subs.nunique() > 1:
        md += ["## ΔR@5 by number of substituted words (doc_max, valid variants)", ""]
        t = robustness_rows(p_valid[p_valid.level == "doc_max"], ["model", "n_subs"], rng)
        md.append(md_table(t[["model", "n_subs", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI"]]))
        summary["dose"] = t.to_dict("records")

    # second (local) judge agreement
    if "local_p_yes" in variants.columns:
        jv = variants[~variants.get("judge_missing", False).astype(bool)]
        loc = jv.local_p_yes >= 0.5
        api = jv.judge_preserves_meaning.astype(bool)
        po = float((loc == api).mean())
        pe = float(loc.mean() * api.mean() + (1 - loc.mean()) * (1 - api.mean()))
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        best_t = max(np.arange(0.5, 1.0, 0.01), key=lambda t: ((jv.local_p_yes >= t) == api).mean())
        md += ["## Second judge (local MamayLM-12B) agreement with the API judge", "",
               f"- raw agreement at p≥0.5: {100 * po:.1f}%, Cohen's κ = {kappa:.3f} (n={len(jv)})",
               f"- threshold maximising agreement: {best_t:.2f} → {100 * ((jv.local_p_yes >= best_t) == api).mean():.1f}%", ""]
        summary["local_judge"] = {"agreement": po, "kappa": kappa, "best_threshold": float(best_t)}
        both = p_valid[p_valid.local_p_yes >= 0.5] if "local_p_yes" in p_valid.columns else p_valid
        md += ["Robustness restricted to variants accepted by BOTH judges (doc_max):", "",
               md_table(robustness_rows(both[both.level == "doc_max"], ["model"], rng)[["model", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI"]])]

    # ---------------- RQ2: rare words
    one = p_valid[(p_valid.n_subs == 1) & p_valid.perturbation_type.isin(["dict", "mlm", "llm"])].copy()
    if not one.empty:
        from experiments.retrievers import Corpus, analyze

        corpus = Corpus()
        freq = Counter(t for c in corpus.chunks for t in analyze(c["text"], True))

        def new_freq(cw):
            words = " ".join(b for _, b in cw)
            toks = analyze(words, True)
            return min((freq.get(t, 0) for t in toks), default=0)

        one["new_word_corpus_freq"] = one.changed_words.map(new_freq)
        one["freq_bin"] = pd.cut(one.new_word_corpus_freq, [-1, 0, 9, 99, 10**9], labels=["0 (OOV)", "1–9", "10–99", "100+"])
        md += ["## RQ2: corpus frequency of the substituted word (single-substitution valid variants, doc_max)", ""]
        t = robustness_rows(one[one.level == "doc_max"], ["model", "freq_bin"], rng)
        md.append(md_table(t[["model", "freq_bin", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI"]]))
        summary["freq_bins"] = t.to_dict("records")
    tech = p_valid[(p_valid.perturbation_type == "tech") & (p_valid.level == "doc_max")]
    if not tech.empty:
        md += ["### Technical variation by kind (doc_max)", ""]
        t = robustness_rows(tech, ["model", "tech_kind"], rng)
        md.append(md_table(t[["model", "tech_kind", "n_var", "R@5 orig", "R@5 pert", "ΔR@5", "95% CI"]]))

    # ---------------- failures
    fails = p_valid[(p_valid.fallout5 == 1)]
    fails_out = fails[["model", "level", "variant_id", "perturbation_type", "original_query", "synonym_query",
                       "changed_words", "rank_orig", "rank_var"]].sort_values(["model", "level", "rank_var"], ascending=[True, True, False])
    fails_out.to_json(out / f"failures_{args.name}.jsonl", orient="records", lines=True, force_ascii=False)
    p_all.drop(columns=["changed_words"]).to_csv(out / f"per_query_{args.name}.csv", index=False)
    md += ["## Failure cases (valid variant pushes source out of top-5)", "",
           f"All in `failures_{args.name}.jsonl`. Examples (bge_m3, doc_max):", ""]
    ex = fails_out[(fails_out.model == "bge_m3") & (fails_out.level == "doc_max")].head(15)
    for _, r in ex.iterrows():
        md.append(f"- [{r.perturbation_type}] rank {r.rank_orig} → {r.rank_var}: «{r.original_query}» → «{r.synonym_query}» "
                  f"({'; '.join(f'{a}→{b}' for a, b in r.changed_words)})")
    md.append("")

    # ---------------- methods
    meth = load_runs(str(out / "runs_methods" / args.name / "*__*.jsonl"))
    if not meth.empty:
        meth = meth.drop_duplicates(["model", "id", "level", "config"])
        md += ["## Inference-time methods", "",
               "Each method is applied to the incoming query only (original or perturbed). Paired robustness as above, "
               "valid variants only. `sel@p`: expansion (main validated config) only when the base top1–top2 margin is below the "
               "p-th percentile of base margins on original queries (no relevance labels used).", ""]
        lat = {Path(f).stem.replace("latency_", ""): json.load(open(f)) for f in glob.glob(str(out / "runs_methods" / args.name / "latency_*.json"))}
        base = meth[meth.config == "base"]
        val_cfgs = [c for c in meth.config.unique() if c.startswith("val:")]
        main_val = "val:dt:3:0.8" if "val:dt:3:0.8" in val_cfgs else (val_cfgs[0] if val_cfgs else None)
        derived = []
        if main_val:
            for (m, lv), g in base.groupby(["model", "level"]):
                vrows = meth[(meth.model == m) & (meth.level == lv) & (meth.config == main_val)].set_index(["id"])
                gb = g.set_index("id")
                for p in (10, 30, 50):
                    thr = np.percentile(gb[gb.kind == "orig"].margin, p)
                    act = gb.margin < thr
                    d = gb[["kind", "level", "model"]].copy()
                    d["rank"] = np.where(act, vrows.loc[gb.index, "rank"], gb["rank"])
                    d["n_exp"] = np.where(act, vrows.loc[gb.index, "n_exp"], 0)
                    d["activated"] = act.astype(float)
                    d["config"] = f"sel@{p}"
                    derived.append(d.reset_index())
        allm = pd.concat([meth] + derived, ignore_index=True) if derived else meth
        mrows = []
        for (m, lv, cfg), g in allm.groupby(["model", "level", "config"]):
            pp = paired(g.assign(), queries, vmeta, ["model", "level"])
            pp = pp[pp.valid & ~((pp.level == "chunk") & (pp.subset != "section"))]
            if pp.empty:
                continue
            lo, hi = cluster_ci(pp, "d5", rng)
            gorig = g[g.kind == "orig"]
            n_exp = g["n_exp"].mean()
            L = lat.get(m, {})
            est = L.get("retrieval_ms", np.nan) * (1 + n_exp) + (L.get("sbert_validation_ms", 0) * (g["activated"].mean() if "activated" in g and cfg.startswith("sel") else (1 if cfg.startswith("val") else 0)))
            mrows.append({"model": m, "level": lv, "method": cfg, "R@5 clean": pct((gorig["rank"] <= 5).mean()),
                          "R@5 pert": pct(pp.hit5_v.mean()), "ΔR@5": pct(pp.d5.mean()), "95% CI": f"[{pct(lo)}, {pct(hi)}]",
                          "MRR pert": f"{pp.rr_v.mean():.3f}", "retrievals/query": f"{1 + n_exp:.2f}",
                          "est. latency ms": f"{est:.1f}",
                          "reverted %": pct(g.get("reverted", pd.Series(dtype=float)).fillna(False).astype(float).mean()) if "reverted" in g else "–"})
        mt = pd.DataFrame(mrows)
        summary["methods"] = mt.to_dict("records")
        for lv in ("doc_max", "doc_full", "chunk"):
            t = mt[mt.level == lv].drop(columns=["level"])
            if not t.empty:
                md += [f"### {lv}", "", md_table(t)]
        md += ["`reverted %`: share of perturbed inputs whose expansion set contains the exact original query "
               "(the dictionary is symmetric; compare with the `mlm` source, which cannot simply invert a dictionary edit).", ""]

    # ---------------- go / no-go
    md += ["## Go / no-go", ""]
    best_drop, n_fail = 0.0, 0
    lines = []
    for key, rows in summary.get("robustness", {}).items():
        if key.count("|") != 1:
            continue
        for r in rows:
            if r["model"] in MAIN_MODELS:
                d = float(r["ΔR@5"]) if r["ΔR@5"] != "–" else 0.0
                lines.append(f"- {key} {r['model']}: ΔR@5 = {r['ΔR@5']} pp {r['95% CI']}, fall-out {r['fall-out@5 %']}%")
                best_drop = max(best_drop, d)
    n_fail = int(fails[fails.model.isin(MAIN_MODELS)].variant_id.nunique())
    md += lines + ["", f"- largest ΔR@5 among main baselines: **{best_drop:.1f} pp** (criterion ≥ 5)",
                   f"- distinct valid variants causing a top-5 fall-out in some main baseline: **{n_fail}** (criterion ≥ 20)",
                   f"- decision heuristic: **{'GO' if best_drop >= 5 and n_fail >= 20 else 'NOT YET / REVIEW'}** "
                   "(planning heuristic, not a significance test; check the audit sample before trusting it)", ""]
    summary["go"] = {"max_drop_pp": best_drop, "n_fail_variants": n_fail}
    report = out / (args.report or f"report_{args.name}.md")
    report.write_text("\n".join(md), encoding="utf-8")
    write_json(out / f"summary_{args.name}.json", summary)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
