"""S3: meaning-preservation validity of every variant.

valid = LLM judge says the meaning / search intent is preserved
        AND SBERT cosine (lang-uk paraphrase mpnet) >= --cos-threshold.
The threshold is fixed in advance and reported, not tuned. Also writes a
stratified sample for manual audit and the pool-input file for variants
(needed by S5 query expansion).
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from experiments.common import (DEFAULT_LLM, LLM, OUT, SEED, done, log, protected_entities, read_jsonl, sbert_cos,
                                write_jsonl)

JUDGE_SYSTEM = ("You are a careful Ukrainian linguist validating a search-robustness benchmark for cybersecurity reports. "
                "Answer with JSON only.")
JUDGE_USER = """Original Ukrainian search query:
"{orig}"

Modified query (some words were replaced):
"{new}"

Replaced: {changed}

Judge the modified query:
1. preserves_meaning: true if a user typing it has the SAME information need, i.e. it should retrieve the same document. Synonyms, abbreviations and spelling variants are fine; a change of entity, action, object, scope or a word used in a wrong sense is not.
2. grammatical: true if it is grammatical, natural Ukrainian (minor stylistic awkwardness is acceptable).
Format: {{"preserves_meaning": true/false, "grammatical": true/false, "reason": "<= 15 words"}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants")
    ap.add_argument("--cos-threshold", type=float, default=0.80)
    ap.add_argument("--audit-n", type=int, default=30)
    ap.add_argument("--llm-model", default=DEFAULT_LLM)
    ap.add_argument("--backend", choices=["openai", "local"], default="openai",
                    help="local = MamayLM-12B P(yes) >= --local-threshold (zero API cost)")
    ap.add_argument("--local-threshold", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    target = out / f"{args.name}_judged.jsonl"
    if done(target, args.force):
        return
    rows = read_jsonl(out / f"{args.name}.jsonl")
    cos = sbert_cos([(r["original_query"], r["synonym_query"]) for r in rows])
    if args.backend == "local":
        from experiments.local_judge import p_yes

        ps = p_yes(rows)
        verdicts = [{"preserves_meaning": p >= args.local_threshold, "grammatical": None,
                     "reason": f"local p_yes={p:.3f}"} for p in ps]
        for r, p in zip(rows, ps):
            r["judge_p_yes"] = round(p, 4)  # not `local_p_yes`: that name marks a SECOND judge in S6
            r["judge_backend"] = "local"
    else:
        llm = LLM(args.llm_model, cache_path=OUT / "llm_cache.jsonl")
        verdicts = llm.map(lambda r: (JUDGE_SYSTEM, JUDGE_USER.format(
            orig=r["original_query"], new=r["synonym_query"],
            changed="; ".join(f"{a} → {b}" for a, b in r["changed_words"]))), rows, desc="judge")
    for r, c, v in zip(rows, cos, verdicts):
        v = v or {}
        r["semantic_similarity"] = round(c, 4)
        r["judge_preserves_meaning"] = bool(v.get("preserves_meaning", False))
        r["judge_grammatical"] = bool(v.get("grammatical", False))
        r["judge_reason"] = v.get("reason")
        r["judge_missing"] = not v
        r["valid"] = r["judge_preserves_meaning"] and c >= args.cos_threshold
    write_jsonl(target, rows)

    # validity summary per type
    for t in sorted({r["perturbation_type"] for r in rows}):
        sub = [r for r in rows if r["perturbation_type"] == t]
        log(f"{t}: {len(sub)} variants, judge-ok {sum(r['judge_preserves_meaning'] for r in sub)}, "
            f"cos-ok {sum(r['semantic_similarity'] >= args.cos_threshold for r in sub)}, valid {sum(r['valid'] for r in sub)}")

    # stratified audit sample
    rng = random.Random(SEED)
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["perturbation_type"], []).append(r)
    per = max(1, args.audit_n // max(1, len(by_type)))
    sample = [r for t in sorted(by_type) for r in rng.sample(by_type[t], min(per, len(by_type[t])))]
    lines = ["# Manual validity audit", "",
             "Mark each pair: **Y** meaning preserved / **N** not. Compare with the automatic label.", "",
             f"Automatic rule: judge preserves_meaning AND SBERT cos >= {args.cos_threshold}.", ""]
    for i, r in enumerate(sample, 1):
        lines += [f"## {i}. `{r['variant_id']}` ({r['perturbation_type']})", "",
                  f"- original: {r['original_query']}", f"- variant:  {r['synonym_query']}",
                  f"- changed: {'; '.join(f'{a} → {b}' for a, b in r['changed_words'])}",
                  f"- auto: valid={r['valid']} judge={r['judge_preserves_meaning']} grammatical={r['judge_grammatical']} "
                  f"cos={r['semantic_similarity']} — {r['judge_reason']}", "- manual: ☐ Y ☐ N", ""]
    (out / f"audit_sample_{args.name}.md").write_text("\n".join(lines), encoding="utf-8")

    # pool input for S5 expansion of perturbed queries
    write_jsonl(out / f"pool_input_{args.name}.jsonl",
                ({"tid": r["variant_id"], "text": r["synonym_query"], "protected": protected_entities(r["synonym_query"])}
                 for r in rows))
    log(f"judged {len(rows)} variants; valid {sum(r['valid'] for r in rows)}; types {Counter(r['perturbation_type'] for r in rows)}")


if __name__ == "__main__":
    main()
