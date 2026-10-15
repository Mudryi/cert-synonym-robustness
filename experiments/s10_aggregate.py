"""S10: stability across tier-7 repetitions.

(a) independent perturbation samples (variants_seed*, local judge): ΔR@5 per
    retriever, mean ± sd / min–max over samples;
(b) fine-tuning seeds (ft_<model>_<cond>_s<seed>): clean R@5 and paired ΔR@5 on
    valid variants (main API-judged set), mean ± sd over seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import read_jsonl
from experiments.s6_report import md_table, paired

LS = [("doc_max", "title"), ("doc_max", "section"), ("chunk", "section")]


def msd(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.1f} ± {x.std(ddof=1) if len(x) > 1 else 0:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    md = ["# Stability across repetitions (tier 7)", ""]

    # (a) perturbation samples
    per = defaultdict(list)
    files = sorted(glob.glob(str(out / "summary_variants_seed*.json")))
    for f in files:
        S = json.loads(Path(f).read_text(encoding="utf-8"))
        for key, rows in S.get("robustness", {}).items():
            if key.count("|") != 1:
                continue
            for r in rows:
                if r["ΔR@5"] != "–":
                    per[(key, r["model"])].append(float(r["ΔR@5"]))
    rows = [{"level|subset": k, "model": m, "samples": len(v), "ΔR@5 mean ± sd": msd(v),
             "min": f"{min(v):.1f}", "max": f"{max(v):.1f}"} for (k, m), v in sorted(per.items()) if m != "bm25_titled"]
    md += [f"## (a) {len(files)} independent perturbation samples (dict/mlm/tech, local judge at calibrated threshold)", "",
           "ΔR@5 in pp on valid variants; each sample re-draws substitutions for the same queries.", "",
           md_table(pd.DataFrame(rows))]

    # (b) fine-tuning seeds, evaluated on the main (API-judged) variant set
    queries = pd.DataFrame(read_jsonl(out / "queries.jsonl"))
    variants = pd.DataFrame(read_jsonl(out / "variants_judged.jsonl"))
    vmeta = variants[["variant_id", "query_id", "article_id", "subset", "perturbation_type", "valid"]]
    sub_of = dict(zip(queries.query_id, queries.subset))
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(out / "runs" / "variants" / "ft_*_s*.jsonl"))):
        m = re.match(r"ft_(.+)_(ft_orig|ft_aug)_s(\d+)", Path(f).stem)
        if not m:
            continue
        df = pd.DataFrame(read_jsonl(f))
        df["model"] = "x"
        p = paired(df, queries, vmeta, ["model", "level"])
        p = p[p.valid]
        o = df[df.kind == "orig"].copy()
        o["subset"] = o.id.map(sub_of)
        for lv, sub in LS:
            oo = o[(o.level == lv) & (o.subset == sub)]
            pp = p[(p.level == lv) & (p.subset == sub)]
            if len(oo):
                acc[(m.group(1), m.group(2), lv, sub)]["clean"].append(100 * (oo["rank"] <= 5).mean())
                acc[(m.group(1), m.group(2), lv, sub)]["delta"].append(100 * pp.d5.mean())
    base = {}
    for name in ("e5_base", "bge_m3"):
        f = out / "runs" / "variants" / f"{name}.jsonl"
        df = pd.DataFrame(read_jsonl(f))
        o = df[df.kind == "orig"].copy()
        o["subset"] = o.id.map(sub_of)
        for lv, sub in LS:
            oo = o[(o.level == lv) & (o.subset == sub)]
            base[(name, lv, sub)] = 100 * (oo["rank"] <= 5).mean()
    rows = []
    for (model, cond, lv, sub), d in sorted(acc.items()):
        rows.append({"model": model, "condition": cond, "level|subset": f"{lv}|{sub}", "seeds": len(d["clean"]),
                     "frozen R@5": f"{base[(model, lv, sub)]:.1f}", "R@5 clean mean ± sd": msd(d["clean"]),
                     "ΔR@5 mean ± sd": msd(d["delta"])})
    md += ["## (b) Article-level 5-fold fine-tuning across seeds", "",
           "Clean R@5 on original queries and paired ΔR@5 on valid (API-judged) variants.", "",
           md_table(pd.DataFrame(rows))]
    (out / "report_stability.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'report_stability.md'}")


if __name__ == "__main__":
    main()
