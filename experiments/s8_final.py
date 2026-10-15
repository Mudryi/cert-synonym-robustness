"""S8: paper figures (PNG + PDF) and a merged index report.

Figures (static, light background, fixed colour per retriever):
  fig_robustness_<level>_<subset>  ΔR@5 per retriever with 95% CI (valid variants)
  fig_by_type                      ΔR@5 per perturbation type (doc_max, section)
  fig_dose                         ΔR@5 vs number of substituted words
  fig_methods                      perturbed R@5 vs retrievals per query
  fig_transfer                     worst-case transfer heatmap (per-variant, superseded)
  fig_worst_searches               successful queries out of 586: random vs S7 vs S15, auto vs human-verified
  fig_judge_budget                 step-1 successes vs judge calls per chain (S16 control)
  fig_transfer_s15                 S15 transfer heatmap on the fixed denominator (auto validity)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical slots (validated reference palette), fixed per entity, never cycled
COLORS = {"bm25": "#2a78d6", "e5_base": "#eb6834", "bge_m3": "#1baf7a", "hybrid": "#eda100",
          "e5_large": "#e87ba4", "m3_all": "#008300"}
OTHER = "#8a8a85"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def num(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def ci(s: str) -> tuple[float, float]:
    m = re.findall(r"-?\d+\.?\d*", s or "")
    return (float(m[0]), float(m[1])) if len(m) == 2 else (np.nan, np.nan)


def save(fig, figdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bar_ci(rows: list[dict], key: str, title: str, figdir: Path, name: str) -> None:
    rows = [r for r in rows if not np.isnan(num(r["ΔR@5"])) and r[key] != "bm25_titled"]
    if not rows:
        return
    labels = [str(r[key]) for r in rows]
    vals = [num(r["ΔR@5"]) for r in rows]
    los, his = zip(*(ci(r["95% CI"]) for r in rows))
    err = np.array([[v - lo for v, lo in zip(vals, los)], [hi - v for v, hi in zip(vals, his)]])
    fig, ax = plt.subplots(figsize=(max(3.2, 0.55 * len(rows) + 1), 2.6))
    x = np.arange(len(rows))
    ax.bar(x, vals, width=0.6, color=[COLORS.get(l, OTHER) for l in labels], edgecolor="white", linewidth=2)
    ax.errorbar(x, vals, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("ΔR@5 (pp), higher = less robust")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    save(fig, figdir, name)


ATT = ["bm25", "e5_base", "bge_m3", "hybrid"]


def worst_figures(out: Path, figdir: Path) -> None:
    """Fixed-denominator worst-case figures from the S14 summaries and the S16 protocol audit."""
    f7, f15 = out / "summary_worst_verified.json", out / "summary_worst_verified_variants_worstv.json"
    if f7.exists() and f15.exists():
        S7, S15 = (json.loads(f.read_text(encoding="utf-8")) for f in (f7, f15))
        get = lambda S, lv: {r["attacked"]: r["raw"]["succ"] for r in S["main"] if r["validity"] == lv}
        rand = {r["retriever"]: float(r["random: successful queries"]) for r in S15["random_vs_worst"]}
        bars = [("random (1 per query)", rand, "#c9c8c0", None), ("S7 auto", get(S7, "auto"), "#9fb7d6", None),
                ("S7 verified", get(S7, "verified (strict)"), "#2a78d6", None), ("S15 auto", get(S15, "auto"), "#f2b48f", None),
                ("S15 verified", get(S15, "verified (strict)"), "#eb6834", None)]
        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        x = np.arange(len(ATT))
        w = 0.8 / len(bars)
        for k, (lab, d, c, _) in enumerate(bars):
            v = [d.get(m, np.nan) for m in ATT]
            ax.bar(x - 0.4 + w * (k + 0.5), v, width=w, color=c, edgecolor="white", linewidth=1, label=lab)
        ax.set_xticks(x, ATT)
        ax.set_ylabel("successful queries (of 586)")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
        ax.set_title("Source leaves the attacked retriever's top 5 · doc_max · white-box", fontsize=9, loc="left")
        save(fig, figdir, "fig_worst_searches")

        T = S15.get("transfer", {}).get("auto")
        if T:
            evals = [e for e in ["bm25", "e5_base", "bge_m3", "hybrid", "e5_large", "m3_all"] if e in T[ATT[0]]]
            M = np.array([[100 * T[a][e] for e in evals] for a in ATT])
            fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(evals), 3.0))
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(5, M.max()))
            ax.grid(False)
            ax.set_xticks(range(len(evals)), evals, rotation=30, ha="right")
            ax.set_yticks(range(len(ATT)), ATT)
            ax.set_xlabel("evaluated retriever")
            ax.set_ylabel("attacked retriever")
            for i in range(len(ATT)):
                for j in range(len(evals)):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.6 * M.max() else INK)
            fig.colorbar(im, ax=ax, label="ΔR@5 (pp, all 586 queries)", shrink=0.8)
            ax.set_title("S15 attack transfer · doc_max · auto validity", fontsize=9, loc="left")
            save(fig, figdir, "fig_transfer_s15")

    fp = out / "summary_protocol_audit.json"
    if fp.exists():
        C = json.loads(fp.read_text(encoding="utf-8"))["budget_curve"]
        fig, ax = plt.subplots(figsize=(3.8, 2.7))
        for m in ATT:
            pts = [(float(r["judge calls / chain (mean used)"]), r["successful queries (step 1)"], r["judge calls / chain (cap)"])
                   for r in C if r["attacked retriever"] == m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=COLORS[m], linewidth=2, markersize=5, label=m,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set_xlabel("judge calls per chain (mean, step 1)")
        ax.set_ylabel("successful queries (of 586)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Validation budget inside the search\nleft end = S7, right end = S15 (step 1)", fontsize=9, loc="left")
        save(fig, figdir, "fig_judge_budget")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="main (full-benchmark) tier dir")
    args = ap.parse_args()
    out = Path(args.out)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    md = ["# Final summary", ""]

    s = out / "summary_variants.json"
    if s.exists():
        S = json.loads(s.read_text(encoding="utf-8"))
        for key, rows in S.get("robustness", {}).items():
            if key.count("|") == 1:
                lv, sub = key.split("|")
                bar_ci(rows, "model", f"Robustness drop · {lv} · {sub} queries (valid variants, 95% CI)",
                       figdir, f"fig_robustness_{lv}_{sub}")
        bt = S.get("robustness", {}).get("doc_max|section|type", [])
        if bt:
            types = sorted({r["perturbation_type"] for r in bt})
            models = [m for m in COLORS if any(r["model"] == m for r in bt)]
            fig, ax = plt.subplots(figsize=(1.0 + 1.1 * len(types), 2.6))
            w = 0.8 / max(1, len(models))
            for i, m in enumerate(models):
                vals = [next((num(r["ΔR@5"]) for r in bt if r["model"] == m and r["perturbation_type"] == t), np.nan) for t in types]
                ax.bar(np.arange(len(types)) + i * w - 0.4 + w / 2, vals, width=w, color=COLORS[m], label=m,
                       edgecolor="white", linewidth=1.5)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xticks(np.arange(len(types)), types)
            ax.set_ylabel("ΔR@5 (pp)")
            ax.legend(frameon=False, fontsize=8, ncol=2)
            ax.set_title("ΔR@5 by perturbation type · doc_max · section queries", fontsize=9, loc="left")
            save(fig, figdir, "fig_by_type")
        meth = S.get("methods", [])
        if meth:
            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            for m in COLORS:
                pts = [(num(r["retrievals/query"]), num(r["R@5 pert"]), r["method"]) for r in meth
                       if r["model"] == m and r["level"] == "doc_max"]
                if not pts:
                    continue
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=COLORS[m], markersize=6, label=m,
                        markeredgecolor="white", markeredgewidth=1.5)
                base = next((p for p in pts if p[2] == "base"), None)
                if base:
                    ax.annotate("base", (base[0], base[1]), textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
            ax.set_xlabel("retrievals per query (cost)")
            ax.set_ylabel("R@5 on perturbed queries (%)")
            ax.legend(frameon=False, fontsize=8)
            ax.set_title("Quality vs. cost of inference-time methods · doc_max", fontsize=9, loc="left")
            save(fig, figdir, "fig_methods")
        md += [f"- go/no-go: {S.get('go')}", f"- validity: see report_variants.md", ""]

    d = out / "summary_variants_dose.json"
    if d.exists():
        D = json.loads(d.read_text(encoding="utf-8")).get("dose", [])
        if D:
            fig, ax = plt.subplots(figsize=(3.6, 2.6))
            for m in COLORS:
                pts = sorted((int(r["n_subs"]), num(r["ΔR@5"])) for r in D if r["model"] == m)
                if pts:
                    ax.plot(*zip(*pts), "-o", color=COLORS[m], linewidth=2, markersize=6, label=m,
                            markeredgecolor="white", markeredgewidth=1.5)
            ax.set_xlabel("substituted words")
            ax.set_ylabel("ΔR@5 (pp)")
            ax.legend(frameon=False, fontsize=8)
            ax.set_title("Dose–response · doc_max · valid variants", fontsize=9, loc="left")
            save(fig, figdir, "fig_dose")

    w = out / "summary_variants_worst.json"
    if w.exists():
        T = json.loads(w.read_text(encoding="utf-8")).get("transfer")
        if T:
            evals = list(T.keys())
            attacks = sorted({a for col in T.values() for a in col})
            M = np.array([[100 * T[e].get(a, np.nan) for e in evals] for a in attacks])
            fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(evals), 1.0 + 0.5 * len(attacks)))
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(5, np.nanmax(M)))
            ax.grid(False)
            ax.set_xticks(range(len(evals)), evals, rotation=30, ha="right")
            ax.set_yticks(range(len(attacks)), attacks)
            ax.set_xlabel("evaluated retriever")
            ax.set_ylabel("attacked retriever")
            for i in range(len(attacks)):
                for j in range(len(evals)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if M[i, j] > 0.6 * np.nanmax(M) else INK)
            fig.colorbar(im, ax=ax, label="ΔR@5 (pp)", shrink=0.8)
            ax.set_title("Worst-case substitution transfer · doc_max", fontsize=9, loc="left")
            save(fig, figdir, "fig_transfer")

    worst_figures(out, figdir)

    md += ["## Reports", ""] + [f"- `{p.name}`" for p in sorted(out.glob("report_*.md"))]
    md += ["", "## Figures", ""] + [f"- `figures/{p.name}`" for p in sorted(figdir.glob("*.png"))]
    (out / "report_final.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"figures: {len(list(figdir.glob('*.png')))} in {figdir}")


if __name__ == "__main__":
    main()
