"""S11: one curated results file (experiments/out/RESULTS.md).

Pulls sections from the tier reports, keeps table rows for the frozen
retrievers (and fine-tuning seed 13 only; all seeds are summarised in the
stability section), and adds the setup, headline findings and a file index.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.common import EXP, OUT

T2 = OUT / "tier2"
MAIN = {"bm25", "e5_base", "bge_m3", "hybrid"}
ATT = ["bm25", "e5_base", "bge_m3", "hybrid"]
ALL_FROZEN = MAIN | {"bm25_surface", "bm25_titled", "e5_large", "m3_sparse", "m3_colbert", "m3_all", "hybrid_e5",
                     "ukr_mpnet", "para_mpnet"}
FT13 = {f"ft_{m}_{c}_s13" for m in ("e5_base", "bge_m3") for c in ("ft_orig", "ft_aug")}


def sections(path: Path) -> list[tuple[str, list[str]]]:
    """[(heading line, body lines)] for every markdown heading."""
    out, cur, body = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if cur is not None:
                out.append((cur, body))
            cur, body = line, []
        else:
            body.append(line)
    if cur is not None:
        out.append((cur, body))
    return out


def get(path: Path, heading: str, nth: int = 0, models: set[str] | None = None, demote: int = 0, exact: bool = False) -> str:
    """Section whose heading contains `heading` (nth match), with table rows filtered by first-column model."""
    hits = [(h, b) for h, b in sections(path) if (h.strip() == heading if exact else heading in h)]
    if len(hits) <= nth:
        return f"_section '{heading}' not found in {path.name}_\n"
    h, body = hits[nth]
    keep = []
    for line in body:
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        is_data = cells and not set(line) <= set("|-: ") and cells[0] not in ("model", "attack_model", "perturbation_type", "subset", "level|subset")
        if models is not None and is_data and cells[0] not in models:
            continue
        keep.append(line)
    level = len(h) - len(h.lstrip("#")) + demote
    return "#" * level + h.lstrip("#") + "\n" + "\n".join(keep).strip() + "\n"


def worstv_section(path: Path) -> list[str]:
    """RQ2 main result: the validity-aware search (S15), when it has been run."""
    if not path.exists():
        return []
    has = lambda h: h in path.read_text(encoding="utf-8")
    return ["### 4.1 Validity-aware attack search (validate → select, S15) — main result", "",
            "The greedy search only considers substitutions the automatic judge accepted, so it finds the strongest "
            "automatically-valid attack rather than validating the strongest attack afterwards. Human validity: labels where the "
            "variant coincides with an audited one (verified), otherwise drawn from the audit strata (estimated).", "",
            get(path, "## Outcome of the attack per original query", 0, None, demote=2),
            get(path, "## White-box attack: retrieval quality", 0, None, demote=2),
            get(path, "## Random vs. worst-case on the same denominator", 0, None, demote=2),
            get(path, "## White-box attack by query type", 0, None, demote=2),
            get(path, "## Transfer across retrievers — auto", 0, None, demote=2),
            get(path, "## Transfer across retrievers — estimated (strict)", 0, None, demote=2),
            get(path, "## Attack search budget", 0, None, demote=2),
            get(path, "## Fine-tuned models under worst-case attacks", 0, None, demote=2) if has("Fine-tuned models") else "",
            get(path, "## Human-verified successful attacks", 0, None, demote=2)]


def protocol_section(path: Path) -> list[str]:
    if not path.exists():
        return []
    return ["### 4.3 Why the two searches differ: protocol and budget audit (S16)", "",
            get(path, "## Consistency checks", 0, None, demote=2),
            get(path, "## Implementation equivalence", 0, None, demote=2),
            get(path, "## Budget-matched control", 0, None, demote=2),
            get(path, "## Cost of the searches as run", 0, None, demote=2)]


def relevance_section(path: Path) -> list[str]:
    if not path.exists():
        return []
    return ["### 4.4 Are attack failures real? Relevance spot-check (S17)", "",
            "\n".join(get(path, "# Relevance spot-check", 0, None).splitlines()[1:]) + "\n"]


def main() -> None:
    rv, rw, rd = T2 / "report_variants.md", T2 / "report_variants_worst.md", T2 / "report_variants_dose.md"
    stab = T2 / "report_stability.md"
    q = [json.loads(l) for l in open(T2 / "queries.jsonl", encoding="utf-8")]
    corpus = json.loads((OUT / "corpus" / "corpus_summary.json").read_text(encoding="utf-8"))
    spend = sum(json.loads(l)["usd"] for l in open(OUT / "llm_spend.jsonl"))
    shown = ALL_FROZEN | FT13

    wv = T2 / "report_worst_verified.md"
    wvv = T2 / "report_worst_verified_variants_worstv.md"
    audit = EXP / "audit" / "audit_report.md"
    AS = json.loads((EXP / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    M4 = ("bm25", "e5_base", "bge_m3", "hybrid")

    def load_wv(f):
        S = json.loads(f.read_text(encoding="utf-8"))
        return S, {(r["attacked"], r["validity"]): r for r in S["main"]}, {r["retriever"]: r for r in S.get("random_vs_worst", [])}

    WV, main_rows, rnd = load_wv(T2 / "summary_worst_verified.json")
    has_v = (T2 / "summary_worst_verified_variants_worstv.json").exists()
    WVV, vrows, _ = load_wv(T2 / "summary_worst_verified_variants_worstv.json") if has_v else (None, {}, {})
    cnt = lambda rows, lv: ", ".join(f"{m} {rows[(m, lv)]['successful queries'].split(' [')[0]}" for m in M4)
    d5 = lambda rows, lv: [float(rows[(m, lv)]["ΔR@5 (pp)"].split(" ")[0]) for m in M4]
    rng_ = lambda xs: f"{min(xs):.1f}–{max(xs):.1f}"
    rand_d5 = [float(rnd[m]["random: ΔR@5 (pp)"]) for m in M4]
    succ_n = lambda rows, lv: [float(rows[(m, lv)]["successful queries"].split(" ")[0]) for m in M4]
    r1 = lambda rows, lv: ", ".join(f"{m} {rows[(m, lv)]['R@1 orig → att']}" for m in M4)
    ratio = lambda a, b: f"{min(x / y for x, y in zip(a, b)):.1f}–{max(x / y for x, y in zip(a, b)):.1f}×"
    ft_src = WVV if has_v and WVV.get("finetune") else WV
    ft_line = ""
    if ft_src.get("finetune"):
        ft = [r for r in ft_src["finetune"] if r["validity"] == "auto" and r["model"].startswith("ft_")]
        ver = {r["model"]: r["ΔR@5 (pp)"] for r in ft_src["finetune"] if r["validity"] == "verified (strict)"}
        ft_line = "; ".join(f"{r['model']} ({r['seeds']} seed{'s' if r['seeds'] > 1 else ''}) ΔR@5 {r['ΔR@5 (pp)']} pp "
                            f"(verified {ver.get(r['model'], '?')})" for r in ft)
    ft_frozen = ", ".join(f"{r['model']} {r['ΔR@5 (pp)']} pp" for r in ft_src.get("finetune", [])
                          if r["validity"] == "auto" and "(frozen)" in r["model"]) + " auto; " + ", ".join(
        f"{r['model']} {r['ΔR@5 (pp)']} pp" for r in ft_src.get("finetune", []) if r["validity"] == "verified (strict)" and "(frozen)" in r["model"]) + " verified"
    census, oc = AS.get("worst_census", {}), AS.get("worst_outcome", {})
    ag = AS["agreement"]["batch 1 (stratified)"]
    src = AS["source_strict"]
    pc = lambda x: f"{100 * x:.0f}%"

    PA = json.loads((T2 / "summary_protocol_audit.json").read_text(encoding="utf-8")) if (T2 / "summary_protocol_audit.json").exists() else {}
    RS = json.loads((EXP / "audit" / "relevance_summary.json").read_text(encoding="utf-8")) if (EXP / "audit" / "relevance_summary.json").exists() else {}
    curve = defaultdict(dict)
    for r in PA.get("budget_curve", []):
        curve[r["attacked retriever"]][str(r["judge calls / chain (cap)"])] = r["successful queries (step 1)"]
    pa_line = (" A budget-matched control on single substitutions (same candidates, judge verdicts reused) shows the two "
               "searches are the two ends of one procedure: judging only the most damaging candidate reproduces S7 "
               f"({', '.join(str(curve[m]['1']) for m in ATT)} successful queries for {', '.join(ATT)}), two judge calls per "
               f"chain give {', '.join(str(curve[m]['2']) for m in ATT)}, and judging all margin-lowering candidates (S15's step 1) "
               f"gives {', '.join(str(curve[m]['all']) for m in ATT)}. The gain comes from spending judge calls inside the search, "
               "at about 4× S7's judge calls.") if curve else ""
    rq2 = (f"2. **RQ2: human-verified targeted attacks are {ratio(d5(vrows, 'verified (strict)'), rand_d5)} more damaging than "
           f"random substitutions ({ratio(d5(vrows, 'auto'), rand_d5)} under auto validity), but affect a small share of queries.** Over all 586 queries (doc_max), a validity-aware greedy attack (only judge-accepted "
           f"substitutions) pushes the source out of the attacked retriever's top 5 for {cnt(vrows, 'auto')} queries "
           f"(ΔR@5 {rng_(d5(vrows, 'auto'))} pp). Every variant behind a successful attack was labelled by the human (batch 3), "
           f"so the human-verified counts are exact: {cnt(vrows, 'verified (strict)')} strictly valid (ΔR@5 "
           f"{rng_(d5(vrows, 'verified (strict)'))} pp, {ratio(d5(vrows, 'verified (strict)'), rand_d5)} random), "
           f"{cnt(vrows, 'verified (natural)')} if awkward wording is also rejected. The attack also moves R@1 "
           f"({r1(vrows, 'auto')} under auto validity). One random valid substitution per query costs "
           f"{rng_(rand_d5)} pp. The same attack WITHOUT validity in the search (select, then validate) finds only "
           f"{cnt(main_rows, 'auto')} (human-verified census: {cnt(main_rows, 'verified (strict)')}), i.e. no more than random: "
           "a validity-unaware attack spends its search on substitutions that are then rejected." + pa_line) if has_v else (
           "2. **RQ2:** validity-unaware greedy attack: " + cnt(main_rows, "auto") + " successful queries (auto).")
    md = [
        "# Results: validity-aware robustness of Ukrainian document retrieval to synonym substitution (CERT-UA)",
        "",
        f"Overnight runs 2026-10-12/13, human validity audit ({AS['n_labelled']} verdicts in "
        f"{'three passes' if AS.get('recheck') else 'two batches'}) and fixed-denominator "
        "worst-case analysis. OpenAI spend $%.2f. Generated by `experiments/s11_results.py`."
        "",
        "## Key findings",
        "",
        "1. **RQ1: random meaning-preserving substitutions barely affect retrieval.** ΔR@5 on valid variants ≤ 2.5 pp for every "
        "retriever (BM25 2.5 ± 0.3, e5-base 1.8 ± 0.3, BGE-M3 0.7 ± 0.2, hybrid 0.1 ± 0.2 pp, section queries, 20 resamples).",
        rq2,
        "3. **RQ3: how validity is handled decides the robustness conclusion.** Of all "
        f"{census.get('n', '?')} automatically accepted worst-case variants, a human judged {pc(census.get('strict', 0))} strictly "
        f"and {pc(census.get('lenient', 0))} leniently valid (census; {pc(census.get('natural', 0))} if awkward wording is also "
        "rejected). Strict validity is "
        f"{pc(oc.get('out of top 5', {}).get('strict', 0))} for successful attacks, {pc(oc.get('worsened', {}).get('strict', 0))} "
        f"for rank-worsening and {pc(oc.get('unchanged', {}).get('strict', 0))} for harmless ones. On the stratified sample, "
        f"GPT-4.1-mini agrees with the human at κ = {ag['GPT judge (meaning)']['kappa_strict']:.2f} and MamayLM-12B at "
        f"κ = {ag['local MamayLM p ≥ 0.5']['kappa_strict']:.2f}." + (
            f" A blind second pass over the {AS['recheck']['n_items']} texts behind successful attacks agreed with the first "
            f"pass on {pc(AS['recheck']['strict_agree'])} of the {AS['recheck']['n_previously_labelled']} previously labelled "
            f"ones ({AS['recheck']['strict_to_invalid']} downgraded, {AS['recheck']['invalid_to_strict']} upgraded)."
            if AS.get("recheck") else "") + " Building validity into the search instead of filtering "
        f"afterwards multiplies successful attacks {ratio(succ_n(vrows, 'auto'), succ_n(main_rows, 'auto'))} (auto validity)." if has_v else "",
        f"4. **Substitution source matters.** Strict human validity (random pool): curated technical terms {pc(src.get('tech', 0))}, "
        f"LLM {pc(src.get('llm', 0))}, MLM {pc(src.get('mlm', 0))}, dictionary {pc(src.get('dict', 0))}; only 14% of all "
        "admissible dictionary/MLM single substitutions pass the automatic judge.",
        "5. **Mitigation.** Hybrid BM25+BGE-M3 is the most robust retriever under every attack; lemmatisation is essential for BM25 "
        "(section R@5 94.5 vs 83.2 without); fine-tuning improves clean retrieval, and synonym augmentation prevents e5-base "
        "from overfitting." + (f" Under worst-case attacks transferred from the frozen model ({ft_frozen}; a transfer, not a "
                               f"re-attack, so not evidence of robustness to an attacker targeting the fine-tuned model): {ft_line}."
                               if ft_line else ""),
        "6. **Inference-time query expansion gives no measurable gain** (appendix).",
        "",
        f"**Caveats.** Single annotator (self-consistency {pc(AS.get('self_consistency_strict', 0))} on "
        f"{AS.get('repeat_pairs', 0)} hidden repeats; no inter-annotator κ); silver source-document labels"
        + (f" (relevance spot-check: the rank-1 document answers the query for {pc(RS['attack']['false_failure'])} of "
           f"{RS['attack']['n']} attack-induced failures, at most partly for {pc(RS['attack']['yes_or_partly'])}; and for "
           f"{pc(RS['clean_fail']['false_failure'])} of {RS['clean_fail']['n']} clean hybrid failures, partly for "
           f"{pc(RS['clean_fail']['yes_or_partly'])}, mostly near-identical report series)" if RS.get("attack") else "") +
        "; greedy (not exhaustive) attacks; successful attacks are human-labelled in both searches, while the validity of S15 "
        "variants that did not succeed is estimated from the audit (it affects R@1/MRR at the estimated level, not success "
        "counts); fine-tuned models are evaluated on transferred attacks.",
        "",
        "## Setup",
        "",
        f"- Corpus: {corpus['docs']} Ukrainian CERT-UA articles ({corpus['chunks']} chunks), near-duplicate {corpus['dropped_near_duplicates']} "
        f"removed, titles stripped from indexed text ({corpus['title_occurrences_stripped_from_bodies']} occurrences).",
        f"- Queries: {sum(x['subset'] == 'title' for x in q)} title queries (title → article) and {sum(x['subset'] == 'section' for x in q)} "
        "section questions (GPT-4.1-mini question from a narrative chunk → article / chunk). Silver labels = source.",
        "- Perturbations: `dict` (synonym dictionary + morphology, ../textfooler_ukr), `mlm` (XLM-R-large fill-mask, ../bert_attack_uk), "
        "`tech` (40 curated technical term pairs), `llm` (GPT synonym substitution). Protected identifiers (CVE/UAC/CERT-UA IDs, "
        "hashes, IPs, domains, Latin names) are never changed.",
        "- Automatic validity: GPT-4.1-mini judge says the information need is preserved AND SBERT (lang-uk mpnet) cosine ≥ 0.80. "
        "Human validity: blind audit by one annotator in passes — batch 1, a stratified sample of 254 substitutions (+15 repeats) over "
        "random/worst-case/dose pools; batch 2, the remaining 437 auto-accepted S7 worst-case variants (+10 repeats, census of 509); "
        "batch 3, a re-check of every variant text behind a successful S7 or S15 attack with an explicit grammar rule (its verdicts "
        "replace earlier ones). Strict = same need and not ungrammatical; lenient also accepts 'partly'; natural (sensitivity) also "
        "rejects awkward wording. Identical texts labelled more than once count as valid only if every label says so.",
        "- Worst-case, two searches with identical candidates and constraints (≤ 2 words, ≤ 8 candidates per position, SBERT ≥ 0.85, "
        "protected entities): S15 (main) judges every admissible single first and searches over judge-valid ones (validate → select); "
        "S7 (comparison) searches all admissible candidates and judges the result (select → validate). A query without a valid "
        "attack keeps its original rank (fixed denominator of 586). Verified = accepted by the automatic rule and by the human.",
        "- Retrievers (frozen): BM25 (pymorphy3 lemmas), multilingual-e5-base/large, BGE-M3 (dense, sparse, ColBERT, all), "
        "Ukrainian paraphrase mpnet, multilingual mpnet; hybrid = RRF(BM25, BGE-M3).",
        "- Levels: `doc_max` (article ranked by its best chunk), `doc_full` (whole-article embedding), `chunk` (section queries → source chunk).",
        "- Metrics: R@1/R@5/MRR, pessimistic ranks; 95% CIs by bootstrap over articles.",
        "",
        "## 1. Benchmark",
        "",
        get(rv, "## Benchmark", demote=1),
        "## 2. Clean retrieval (original queries)",
        "",
        get(rv, "### doc_max · title queries", 0, ALL_FROZEN, demote=0),
        get(rv, "### doc_max · section queries", 0, ALL_FROZEN, demote=0),
        get(rv, "### chunk · section queries", 0, ALL_FROZEN, demote=0),
        "## 3. RQ1 — random valid substitutions",
        "",
        "ΔR@5 is averaged over valid variants (paired with their own original).",
        "",
        get(rv, "### doc_max · section queries — all valid variants", 0, ALL_FROZEN),
        get(rv, "### doc_max · title queries — all valid variants", 0, ALL_FROZEN),
        "### By perturbation type (doc_max · section queries)\n",
        "\n".join(get(rv, "#### by perturbation type", 3, MAIN).splitlines()[1:]) + "\n",
        "### With vs. without identifiers in the query (doc_max · section queries)\n",
        "\n".join(get(rv, "#### with vs. without identifiers", 3, MAIN).splitlines()[1:]) + "\n",
        get(rv, "### Technical variation by kind", 0, MAIN),
        get(rv, "## RQ2: corpus frequency", 0, MAIN, demote=1).replace("RQ2: corpus frequency", "Corpus frequency of the substituted word"),
        "### Stability over 20 independent perturbation samples\n",
        "\n".join(get(stab, "## (a)", 0, None).splitlines()[1:]) + "\n",
        "### Number of substituted words (dose-response)\n",
        "Caveat from the audit: only 47% of auto-valid 3–4-word variants are strictly valid.\n",
        "\n".join(get(rd, "## ΔR@5 by number of substituted words", 0, MAIN).splitlines()[1:]) + "\n",
        "## 4. RQ2 — worst-case substitutions on a fixed query denominator",
        "",
        "\n".join(get(wv, "# Worst-case robustness on a fixed", 0, None).splitlines()[1:]) + "\n",
        *worstv_section(wvv),
        "### 4.2 Validity-unaware greedy search (select → validate, S7), for comparison",
        "",
        get(wv, "## Outcome of the attack per original query", 0, None, demote=2),
        get(wv, "## White-box attack: retrieval quality", 0, None, demote=2),
        get(wv, "## Random vs. worst-case on the same denominator", 0, None, demote=2),
        get(wv, "## White-box attack by query type", 0, None, demote=2),
        get(wv, "## Transfer across retrievers — auto", 0, None, demote=2),
        get(wv, "## Attack search budget", 0, None, demote=2),
        get(wv, "## Human-verified successful attacks", 0, None, demote=2),
        *protocol_section(T2 / "report_protocol_audit.md"),
        *relevance_section(EXP / "audit" / "relevance_report.md"),
        "## 5. RQ3 — validity of substitutions (human audit)",
        "",
        "\n".join(get(audit, "# Human validity audit", 0, None).splitlines()[1:]) + "\n",
        get(audit, "## Self-consistency", 0, None, demote=1),
        get(audit, "### By pool and substitution source", 0, None),
        get(audit, "### By pool and automatic label", 0, None),
        get(audit, "### By attack outcome", 0, None),
        get(audit, "## Estimated precision of the automatic rule", 0, None, demote=1),
        get(audit, "## Agreement of automatic judges", 0, None, demote=1),
        get(audit, "## Issues ticked", 0, None, demote=1),
        get(rv, "### Valid vs. invalid substitutions", 0, MAIN),
        get(rv, "## Second judge", 0, MAIN, demote=1),
        "## 6. Mitigation: fine-tuning",
        "",
        "\n".join(get(stab, "## (b)", 0, None).splitlines()[1:]) + "\n",
        get(wv, "## Fine-tuned models under worst-case attacks", 0, None, demote=1) if "Fine-tuned models" in wv.read_text(encoding="utf-8") else "",
        "## Appendix A. Per-variant worst-case averages (superseded by section 4)",
        "",
        "These average ΔR@5 over the auto-valid attack variants only, so queries without a valid attack drop out of the denominator.",
        "",
        get(rw, "## Worst-case substitutions: transfer", 0, None, demote=1),
        get(rw, "## Failure cases", 0, None, demote=1),
        "## Appendix B. Inference-time methods (query expansion + fusion)",
        "",
        get(rv, "## Inference-time methods", 0, None, demote=1).split("### doc_max")[0],
        get(rv, "### doc_max", 0, MAIN, exact=True).replace("### doc_max", "### doc_max (all configurations incl. ablations)"),
        "## Appendix C. Chunk-level robustness and pilot go/no-go",
        "",
        get(rv, "### chunk · section queries — all valid variants", 0, ALL_FROZEN),
        "\n".join(get(rv, "## Go / no-go", 0, None).splitlines()[1:]) + "\n",
        "## Appendix D. Figures",
        "",
    ]
    for f in sorted((T2 / "figures").glob("*.png")):
        md += [f"### {f.stem}", "", f"![{f.stem}](tier2/figures/{f.name})", ""]
    md += [
        "## 11. Files",
        "",
        "| What | Path (relative to `experiments/out/`) |",
        "|---|---|",
        "| Full report, all tables and fine-tuning seeds | `tier2/report_variants.md` |",
        "| Worst-case report | `tier2/report_variants_worst.md` |",
        "| Dose-response report | `tier2/report_variants_dose.md` |",
        "| Stability across samples and seeds | `tier2/report_stability.md` |",
        "| Pilot (go/no-go) | `report_tier1.md` |",
        "| Fixed-denominator worst-case report | `tier2/report_worst_verified.md`, `tier2/summary_worst_verified.json` |",
        "| S7 vs S15 protocol and budget audit | `tier2/report_protocol_audit.md`, `tier2/summary_protocol_audit.json` |",
        "| Relevance spot-check of silver labels | `../audit/relevance_report.md`, `../audit/relevance_key.jsonl` |",
        "| Human audit: report, summary, labels | `../audit/audit_report.md`, `../audit/AUDIT_SUMMARY.md`, `../audit/human_labels.jsonl` |",
        "| Manual audit samples (superseded by `experiments/audit/`) | `tier2/audit_sample_variants.md`, `tier2/audit_sample_variants_worst.md`, `tier2/audit_sample_variants_dose.md` |",
        "| Blind human audit: sample, key, tool | `../audit/` |",
        "| Per-query ranks (CSV) | `tier2/per_query_variants*.csv` |",
        "| Failure cases (JSONL) | `tier2/failures_variants*.jsonl` |",
        "| Queries and judged variants | `tier2/queries.jsonl`, `tier2/variants*_judged.jsonl` |",
        "| Figures (PNG + PDF) | `tier2/figures/` |",
        "| Run status and logs | `STATUS.md`, `logs/` |",
        "",
    ]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(md))
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'RESULTS.md'} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
