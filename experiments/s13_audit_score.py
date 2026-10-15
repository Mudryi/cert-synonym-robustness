"""S13: score the human validity audit against the automatic judges.

Batch 1 (`audit_key.jsonl`, stratified sample) and batch 2 (`batch2_key.jsonl`,
all remaining auto-accepted worst-case variants) are the first pass and are
scored together. Batch 3 (`batch3_key.jsonl`) is a deliberate second pass over
the variants behind successful attacks (S7 and S15), one item per unique
variant text; its verdicts replace earlier ones for every variant with that
text, and its agreement with the first pass is reported.

Input: one or more verdict files exported from review_tool.html (JSON with a
`verdicts` list) or a JSONL dump of the tool's `verdicts` collection.
Joins them with the audit keys and writes experiments/audit/audit_report.md,
audit/audit_summary.json (headline numbers for S11) and audit/human_labels.jsonl
(variant_id -> human verdict, the input for recomputing robustness on the
manually verified subset).

Human-valid (strict)  = meaning "yes" and fluency not "ungrammatical"
Human-valid (lenient) = meaning "yes" or "partly", fluency not "ungrammatical"
Human-valid (natural) = meaning "yes" and fluency "natural" (sensitivity: awkward
                        wording such as agreement slips does not count)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.common import EXP, read_jsonl
from experiments.s6_report import md_table

AUDIT = EXP / "audit"


def load_verdicts(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        rows += json.loads(text)["verdicts"] if text.lstrip().startswith("{") and '"verdicts"' in text[:300] else \
            [json.loads(l) for l in text.splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    for col, default in (("unsure", False), ("note", ""), ("issues", None)):  # optional fields the tool omits when unused
        if col not in df:
            df[col] = default
    df = df[df.meaning.notna() & df.fluency.notna()].sort_values("t").drop_duplicates(["annotator", "item_id"], keep="last")
    df["strict"] = (df.meaning == "yes") & (df.fluency != "ungrammatical")
    df["lenient"] = df.meaning.isin(["yes", "partly"]) & (df.fluency != "ungrammatical")
    df["natural"] = (df.meaning == "yes") & (df.fluency == "natural")
    return df


def kappa(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(bool), b.astype(bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def rate_rows(D: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Strict and lenient human-valid rates with Wilson CIs per group."""
    rows = []
    for key, g in D.groupby(cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(cols, key)) | {"n": len(g)}
        for name in ("strict", "lenient", "natural"):
            k = int(g[name].sum())
            lo, hi = wilson(k, len(g))
            row[f"{name} %"] = pct(k / len(g))
            row[f"{name} 95% CI"] = f"[{pct(lo)}, {pct(hi)}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts", nargs="+")
    ap.add_argument("--annotator", default=None, help="score one annotator (default: the one with most verdicts)")
    args = ap.parse_args()
    V = load_verdicts(args.verdicts)
    key = [k | {"batch": b} for b, f in ((1, "audit_key.jsonl"), (2, "batch2_key.jsonl"), (3, "batch3_key.jsonl"))
           if (AUDIT / f).exists() for k in read_jsonl(AUDIT / f)]
    K = pd.DataFrame([k for k in key if "pool" in k])
    K3 = K[K.batch == 3].explode("variant_ids") if (K.batch == 3).any() else K.iloc[:0]
    K = K[K.batch < 3]
    text = {}  # item_id -> (original, variant), to find identical substitutions labelled more than once
    for f in ("audit_items.json", "batch2_items.json"):
        if (AUDIT / f).exists():
            text |= {i["id"]: (i["original"], i["variant"]) for i in json.loads((AUDIT / f).read_text(encoding="utf-8"))}
    rep = {k["item_id"]: k["repeat_of"] for k in key if "repeat_of" in k}
    strata = pd.DataFrame(json.loads((AUDIT / "audit_strata.json").read_text(encoding="utf-8")))
    annot = args.annotator or V.annotator.value_counts().index[0]
    A = V[V.annotator == annot]
    md = [f"# Human validity audit — annotator {annot}", "",
          f"{len(A)} of {len(K) + len(rep)} items labelled ({A.unsure.fillna(False).astype(bool).sum()} marked unsure). "
          "Strict = same information need and not ungrammatical; lenient also accepts 'partly'.", ""]

    # self-consistency on the hidden repeats
    a = A.set_index("item_id")
    pairs = [(a.loc[r], a.loc[o]) for r, o in rep.items() if r in a.index and o in a.index]
    summary = {"annotator": annot, "n_labelled": len(A), "repeat_pairs": len(pairs)}
    if pairs:
        summary["self_consistency_strict"] = sum(x.strict == y.strict for x, y in pairs) / len(pairs)
        summary["self_consistency_meaning"] = sum(x.meaning == y.meaning for x, y in pairs) / len(pairs)
        md += ["## Self-consistency (repeated items)", "",
               f"- {len(pairs)} pairs; same meaning label {pct(sum(x.meaning == y.meaning for x, y in pairs) / len(pairs))}%, "
               f"same strict verdict {pct(sum(x.strict == y.strict for x, y in pairs) / len(pairs))}%", ""]

    D = K.merge(A[~A.item_id.isin(rep)], on="item_id")
    if D.empty:
        print("no labelled items")
        return
    # identical variant texts reached by several searches were labelled independently: a second consistency check
    D["text"] = D.item_id.map(text)
    dup = D.groupby("text").filter(lambda g: len(g) > 1).groupby("text")
    if dup.ngroups:
        same = dup.strict.nunique().eq(1)
        summary["duplicate_texts"] = {"n": int(dup.ngroups), "strict_agree": float(same.mean()), "conflicts": int((~same).sum())}
        md += [f"- {dup.ngroups} variant texts were labelled more than once as separate items (found by several attacks); "
               f"same strict verdict for {pct(same.mean())}%, {int((~same).sum())} conflicts. Downstream (S14) a text counts "
               "as valid only if every label says so.", ""]
    D["local_valid"] = D.local_p_yes.fillna(0) >= 0.5
    D["outcome"] = ["out of top 5" if f else "worsened" if w else "unchanged" for f, w in zip(D.fell_out_top5, D.rank_worse)]
    D["auto label"] = D.auto_valid.map({True: "valid", False: "invalid"})

    md += ["## Human validity rates with 95% Wilson intervals", "",
           "Raw rates in the audit sample. The sample deliberately over-represents rank-changing variants, so rates for whole pools "
           "are not population estimates; the weighted table below is a sensitivity analysis, not a population correction.", "",
           "### By pool and substitution source", "", md_table(rate_rows(D, ["pool", "perturbation_type"])), "",
           "### By pool and automatic label", "", md_table(rate_rows(D, ["pool", "auto label"])), "",
           "### By attack outcome (auto-valid only)", "",
           "Outcome on the attacked retriever (worst-case) or any main retriever (random, dose): `out of top 5` = source left the "
           "top 5; `worsened` = source rank dropped, starting from the top 10; `unchanged` otherwise.", "",
           md_table(rate_rows(D[D.auto_valid], ["pool", "outcome"])), ""]

    # automatic rule vs human, per pool
    rows = []
    for (pool, auto), g in D.groupby(["pool", "auto_valid"]):
        rows.append({"pool": pool, "auto label": "valid" if auto else "invalid", "n": len(g),
                     "human strict-valid %": pct(g.strict.mean()), "human lenient-valid %": pct(g.lenient.mean())})
    md += ["## Automatic label vs. human verdict", "",
           "For auto-valid rows the strict-valid % is the precision of the automatic rule; for auto-invalid rows it is the share wrongly rejected.", "",
           md_table(pd.DataFrame(rows)), ""]

    # census: batch 2 completes the labels of every auto-accepted worst-case variant
    census = D[(D.pool == "worst") & D.auto_valid]
    if K[(K.pool == "worst") & K.auto_valid].variant_id.isin(census.variant_id).all() and (K.batch == 2).any():
        k_s, k_l, n = int(census.strict.sum()), int(census.lenient.sum()), len(census)
        k_n = int(census.natural.sum())
        summary["worst_census"] = {"n": n, "strict": k_s / n, "strict_ci": wilson(k_s, n), "lenient": k_l / n, "lenient_ci": wilson(k_l, n),
                                   "natural": k_n / n, "natural_ci": wilson(k_n, n)}
        summary["worst_outcome"] = {o: {"n": len(g), "strict": float(g.strict.mean()), "strict_ci": wilson(int(g.strict.sum()), len(g))}
                                    for o, g in census.groupby("outcome")}
        md += ["## Census: every automatically accepted worst-case variant", "",
               f"All {n} auto-accepted worst-case variants are labelled (first pass), so the rates are exact for these {n} cases; the "
               "Wilson interval only expresses uncertainty when generalising to similar generated attacks.", "",
               "### By attack source", "", md_table(rate_rows(census, ["perturbation_type"])), "",
               "### By attacked retriever", "", md_table(rate_rows(census, ["attack_model"])), "",
               "### By attack outcome on the attacked retriever", "", md_table(rate_rows(census, ["outcome"])), ""]

    # population-weighted estimate of the share of auto-valid variants that are truly valid (batch 1: stratified sample)
    D["cell"] = list(zip(D.pool, D.group, D.auto_valid, D.rank_worse))
    pop = {(r.pool, r.group, r.auto_valid, r.rank_worse): r.n for r in strata.itertuples()}
    rows = []
    for pool, g in D[D.auto_valid & (D.batch == 1)].groupby("pool"):
        c = g.groupby("cell")[["strict", "lenient"]].mean()
        w = pd.Series({cell: pop.get(cell, 0) for cell in c.index})
        rows.append({"pool": pool, "labelled": len(g), "population": int(w.sum()),
                     "weighted strict-valid %": pct((c.strict * w).sum() / w.sum()),
                     "weighted lenient-valid %": pct((c.lenient * w).sum() / w.sum())})
    md += ["## Estimated precision of the automatic rule (weighted by stratum size, batch 1)", "",
           "The stratified batch-1 sample over-represents rank-changing variants; weights restore each pool's real mix. "
           "For the worst-case pool the census above supersedes this estimate.", "",
           md_table(pd.DataFrame(rows)), ""]

    rows = []
    summary["agreement"] = {}
    for scope, S in (("batch 1 (stratified)", D[D.batch == 1]), ("all labelled", D)):
        for name, col in [("GPT judge (meaning)", "judge_preserves_meaning"), ("automatic rule (GPT + SBERT ≥ 0.8)", "auto_valid"),
                          ("local MamayLM p ≥ 0.5", "local_valid")]:
            m = S[col].notna()
            r = {"items": scope, "judge": name, "n": int(m.sum()), "agreement % (strict)": pct((S[m][col].astype(bool) == S[m].strict).mean()),
                 "κ (strict)": f"{kappa(S[m][col], S[m].strict):.3f}", "κ (lenient)": f"{kappa(S[m][col], S[m].lenient):.3f}"}
            rows.append(r)
            summary["agreement"].setdefault(scope, {})[name] = {"agreement": float((S[m][col].astype(bool) == S[m].strict).mean()),
                                                               "kappa_strict": kappa(S[m][col], S[m].strict),
                                                               "kappa_lenient": kappa(S[m][col], S[m].lenient)}
    md += ["## Agreement of automatic judges with the human verdict", "",
           "Batch 1 is the stratified sample (accepted and rejected variants, all pools) and is the fair basis for κ; batch 2 "
           "adds only accepted worst-case variants, which shifts κ toward the precision of the accepted set.", "",
           md_table(pd.DataFrame(rows)), ""]
    summary["source_strict"] = {t: float(g.strict.mean()) for t, g in D[(D.pool == "random") & (D.batch == 1)].groupby("perturbation_type")}

    rows = []
    for (pool, worse), g in D.groupby(["pool", "rank_worse"]):
        rows.append({"pool": pool, "rank got worse": worse, "n": len(g), "human strict-valid %": pct(g.strict.mean()),
                     "auto-valid %": pct(g.auto_valid.mean())})
    md += ["## Do ranking-changing substitutions tend to be invalid?", "",
           "`rank got worse`: a main retriever (the attacked one for worst-case) ranked the source lower, starting from the top 10.", "",
           md_table(pd.DataFrame(rows)), ""]

    rows = []
    for t, g in D.groupby(["pool", "perturbation_type"]):
        rows.append({"pool": t[0], "type": t[1], "n": len(g), "human strict-valid %": pct(g.strict.mean()),
                     "auto-valid %": pct(g.auto_valid.mean())})
    md += ["## By perturbation type", "", md_table(pd.DataFrame(rows)), ""]

    issues = D.explode("issues").issues.dropna().value_counts()
    if len(issues):
        md += ["## Issues ticked", "", md_table(issues.rename_axis("issue").reset_index(name="n")), ""]
    dis = D[D.auto_valid & ~D.strict]
    if len(dis):
        md += ["## Auto-valid but human-rejected (examples)", ""]
        for r in dis.head(25).itertuples():
            note = r.note if isinstance(r.note, str) else ""
            md.append(f"- `{r.variant_id}` ({r.meaning}/{r.fluency}) {note}".rstrip())
        md.append("")

    # batch 3: second pass over the variants behind successful attacks; replaces first-pass labels of the same text
    final = D.copy()
    final["batch_used"] = final.batch
    if len(K3):
        B3 = K3.drop(columns=["variant_id"], errors="ignore").rename(columns={"variant_ids": "variant_id"}).merge(A, on="item_id")
        if len(B3):
            first = D.drop_duplicates("variant_id").set_index("variant_id")
            both = B3[B3.variant_id.isin(first.index)].drop_duplicates("item_id")
            if len(both):
                f1 = first.loc[both.variant_id]
                summary["recheck"] = {"n_items": int(B3.item_id.nunique()), "n_previously_labelled": len(both),
                                      "strict_agree": float((f1.strict.to_numpy() == both.strict.to_numpy()).mean()),
                                      "meaning_agree": float((f1.meaning.to_numpy() == both.meaning.to_numpy()).mean()),
                                      "strict_to_invalid": int((f1.strict.to_numpy() & ~both.strict.to_numpy()).sum()),
                                      "invalid_to_strict": int((~f1.strict.to_numpy() & both.strict.to_numpy()).sum())}
                rc = summary["recheck"]
                md += ["## Batch 3: second pass over successful attacks", "",
                       f"{rc['n_items']} unique variant texts; {len(both)} had a first-pass label. Same strict verdict "
                       f"{pct(rc['strict_agree'])}%, same meaning label {pct(rc['meaning_agree'])}%; {rc['strict_to_invalid']} "
                       f"changed from strict-valid to not, {rc['invalid_to_strict']} the other way. Batch-3 verdicts replace the "
                       "first pass downstream.", "", md_table(rate_rows(B3.drop_duplicates("item_id"), ["source"])), ""]
            final = pd.concat([final[~final.variant_id.isin(B3.variant_id)], B3.assign(batch_used=3)], ignore_index=True)

    (AUDIT / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    (AUDIT / "audit_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(AUDIT / "human_labels.jsonl", "w", encoding="utf-8") as f:
        for r in final.drop_duplicates("variant_id").itertuples():
            f.write(json.dumps({"variant_id": r.variant_id, "annotator": annot, "batch": int(r.batch_used), "meaning": r.meaning,
                                "fluency": r.fluency, "issues": r.issues if isinstance(r.issues, list) else [],
                                "unsure": bool(r.unsure) if r.unsure == r.unsure else False, "human_valid_strict": bool(r.strict),
                                "human_valid_lenient": bool(r.lenient), "human_valid_natural": bool(r.natural)}, ensure_ascii=False) + "\n")
    print(f"wrote {AUDIT / 'audit_report.md'} and human_labels.jsonl ({final.variant_id.nunique()} variants)")


if __name__ == "__main__":
    main()
