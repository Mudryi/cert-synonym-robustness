"""S12: blind human validity audit sample (Day 1 of the publication plan).

Draws a stratified sample of substitutions from the random (`variants`),
worst-case (`variants_worst`) and dose (`variants_dose`, 3-4 words) pools.
Strata cover the automatic verdict (judge-valid / judge-invalid) and the
ranking effect (some main retriever's gold rank got worse vs. unchanged), so
the audit contains examples that changed rankings and examples that did not.

Two outputs, deliberately separated:
  experiments/audit/audit_items.json  what the reviewer sees: queries, changed
                                      words, source passage. No model output.
  experiments/audit/audit_key.jsonl   hidden metadata (pool, automatic judges,
                                      rank effect) joined back in S13.
  experiments/audit/audit_strata.json population size of every sampling
                                      stratum, for weighting estimates in S13.
and experiments/audit/review_tool.html, the template with the items inlined.
A few items are repeated later in the order to measure self-consistency.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from experiments.common import EXP, OUT, SEED, read_jsonl

T2 = OUT / "tier2"
AUDIT = EXP / "audit"
MAIN = ["bm25", "e5_base", "bge_m3", "hybrid"]


def rank_effect(csv: Path, whitebox: bool) -> dict[str, dict]:
    """variant_id -> {worse, fell_out_top5, ranks}: doc_max ranks of the main retrievers (attacked one only if whitebox)."""
    cols = ["model", "level", "variant_id", "rank_orig", "rank_var"] + (["attack_model"] if whitebox else [])
    out: dict[str, dict] = {}
    for df in pd.read_csv(csv, usecols=cols, chunksize=500_000):
        df = df[(df.level == "doc_max") & df.model.isin(MAIN)]
        if whitebox:
            df = df[df.model == df.attack_model]
        for r in df.itertuples(index=False):
            e = out.setdefault(r.variant_id, {"worse": False, "fell_out_top5": False, "ranks": {}})
            e["ranks"][r.model] = [int(r.rank_orig), int(r.rank_var)]
            e["worse"] |= r.rank_orig <= 10 and r.rank_var > r.rank_orig  # ignore reshuffles deep in the list
            e["fell_out_top5"] |= r.rank_orig <= 5 < r.rank_var
    return out


def draw(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n rows; half rank-worsening, half unchanged when both exist."""
    worse = [r for r in rows if r["_eff"]["worse"]]
    same = [r for r in rows if not r["_eff"]["worse"]]
    k = min(len(worse), n // 2)
    pick = rng.sample(worse, k) + rng.sample(same, min(len(same), n - k))
    if len(pick) < n:  # top up from whichever stratum has more
        rest = [r for r in rows if r not in pick]
        pick += rng.sample(rest, min(len(rest), n - len(pick)))
    return pick


def records(picked: list[dict], prefix: str) -> tuple[list[dict], list[dict]]:
    """Blind items (what the reviewer sees) and hidden key rows for sampled variants."""
    docs = {d["doc_id"]: d for d in read_jsonl(OUT / "corpus" / "corpus_docs.jsonl")}
    chunks = {c["chunk_id"]: c for c in read_jsonl(OUT / "corpus" / "corpus_chunks.jsonl")}
    items, key = [], []
    for i, v in enumerate(picked, 1):
        iid = f"{prefix}{i:03d}"
        ctx = chunks[v["chunk_id"]]["text"] if v.get("chunk_id") in chunks else docs[v["article_id"]]["text"]
        items.append({"id": iid, "subset": v["subset"], "original": v["original_query"], "variant": v["synonym_query"],
                      "changed": v["changed_words"], "context": ctx[:900]})
        group = v["perturbation_type"] if v["_pool"] != "worst" else f"{v['perturbation_type']}|{v['attack_model']}"
        key.append({"item_id": iid, "variant_id": v["variant_id"], "pool": v["_pool"], "group": group, "perturbation_type": v["perturbation_type"],
                    "attack_model": v.get("attack_model"), "n_subs": v.get("n_subs", len(v["changed_words"])),
                    "subset": v["subset"], "auto_valid": bool(v["valid"]), "judge_preserves_meaning": v["judge_preserves_meaning"],
                    "judge_grammatical": v.get("judge_grammatical"), "semantic_similarity": v["semantic_similarity"],
                    "local_p_yes": v.get("local_p_yes"), "rank_worse": v["_eff"]["worse"],
                    "fell_out_top5": v["_eff"]["fell_out_top5"], "ranks_doc_max": v["_eff"]["ranks"]})
    return items, key


def add_repeats(items: list[dict], key: list[dict], n: int, prefix: str, rng: random.Random) -> None:
    """Hidden copies placed at least 40 positions after their original, for self-consistency."""
    for src in rng.sample(items[: len(items) - 40], n):
        rep = dict(src, id=f"{prefix}{src['id'][1:]}")
        items.insert(rng.randint(items.index(src) + 40, len(items)), rep)
        key.append({"item_id": rep["id"], "repeat_of": src["id"]})


def write_batch(name: str, items: list[dict], key: list[dict]) -> None:
    AUDIT.mkdir(exist_ok=True)
    (AUDIT / f"{name}_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")
    with open(AUDIT / f"{name}_key.jsonl", "w", encoding="utf-8") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    tpl = (AUDIT / "review_tool.template.html").read_text(encoding="utf-8")
    html = "review_tool.html" if name == "audit" else f"review_tool_{name}.html"
    if name != "audit":
        tpl = tpl.replace("Substitution Validity Audit", "Worst-Case Validity Audit" if name == "batch2" else "Successful-Attack Re-check")
    if name == "batch3":  # spell out the grammar rule that batch 1/2 left implicit
        tpl = tpl.replace("<li>Fluency is separate: a variant can keep the meaning and still be awkward or ungrammatical.</li>",
                          "<li>Fluency is separate: a variant can keep the meaning and still be awkward or ungrammatical. A wrong case, "
                          "gender or number agreement (e.g. «новий знаряддя», «при підозрінню») or a non-existent word form is "
                          "<b>Ungrammatical</b>, not Awkward.</li>")
    # escape "</" so that source passages quoting HTML (e.g. "</script>") cannot end the page's script early
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    (AUDIT / html).write_text(tpl.replace("/*__ITEMS__*/[]", data), encoding="utf-8")


def worst_rest(rng: random.Random, n_repeat: int) -> None:
    """Batch 2: every auto-accepted worst-case variant not yet in the audit. Variants that push the source out of the
    top 5 of any frozen retriever come first, shuffled with an equal number of others so the block is not all failures."""
    eff = rank_effect(T2 / "per_query_variants_worst.csv", True)
    done_ids = {k["variant_id"] for k in read_jsonl(AUDIT / "audit_key.jsonl") if "variant_id" in k}
    df = pd.read_csv(T2 / "per_query_variants_worst.csv", usecols=["model", "level", "variant_id", "rank_orig", "rank_var"])
    df = df[(df.level == "doc_max") & (df.rank_orig <= 5) & (df.rank_var > 5)]
    any_fail = set(df.variant_id)
    rest = [dict(v, _pool="worst", _eff=eff[v["variant_id"]]) for v in read_jsonl(T2 / "variants_worst_judged.jsonl")
            if v["valid"] and not v.get("judge_missing") and v["variant_id"] not in done_ids and v["variant_id"] in eff]
    fail = [v for v in rest if v["variant_id"] in any_fail]
    other = [v for v in rest if v["variant_id"] not in any_fail]
    rng.shuffle(other)
    head = fail + other[:len(fail)]
    rng.shuffle(head)
    picked = head + other[len(fail):]
    items, key = records(picked, "w")
    add_repeats(items, key, n_repeat, "x", rng)
    write_batch("batch2", items, key)
    print(f"batch2: {len(items)} items ({len(picked)} unique, {len(fail)} failing on some retriever in the first "
          f"{len(head)}, {n_repeat} repeats)")


def successes(rng: random.Random, n_repeat: int) -> None:
    """Batch 3: one item per unique variant text behind a white-box success of either search (S7 `variants_worst`, S15
    `variants_worstv`, auto-accepted), labelled before or not, plus texts whose earlier labels conflict. Needs S14's
    variant_outcomes_<name>.jsonl. Every variant with the same (query, text) is listed in the key and gets the verdict."""
    by_text: dict[tuple, list[dict]] = defaultdict(list)  # (query_id, text) -> variant records of all worst-case searches
    src: dict[tuple, set] = defaultdict(set)
    eff: dict[str, dict] = {}
    for name, tag in (("variants_worst", "S7"), ("variants_worstv", "S15")):
        oc = {o["variant_id"]: o for o in read_jsonl(T2 / f"variant_outcomes_{name}.jsonl")}
        for v in read_jsonl(T2 / f"{name}_judged.jsonl"):
            if v.get("judge_missing") or v["variant_id"] not in oc:
                continue
            o, k = oc[v["variant_id"]], (v["query_id"], v["synonym_query"])
            by_text[k].append(v)
            eff[v["variant_id"]] = {"worse": o["outcome"] != "unchanged", "fell_out_top5": o["outcome"] == "out of top 5",
                                    "ranks": o["ranks_doc_max"]}
            if v["valid"] and o["outcome"] == "out of top 5":
                src[k].add(tag)
    # conflicting first-pass labels of the same text
    vtext = {v["variant_id"]: k for k, vs in by_text.items() for v in vs}
    labels = defaultdict(set)
    for r in read_jsonl(AUDIT / "human_labels.jsonl"):
        if r["variant_id"] in vtext and r.get("batch", 1) < 3:
            labels[vtext[r["variant_id"]]].add(r["human_valid_strict"])
    for k, ls in labels.items():
        if len(ls) > 1:
            src[k].add("conflict")
    keys = sorted(src)
    rng.shuffle(keys)
    picked = [dict(by_text[k][0], _pool="recheck", _eff=eff[by_text[k][0]["variant_id"]]) for k in keys]
    items, key = records(picked, "c")
    for kr, k in zip(key, keys):
        kr["variant_ids"] = sorted({v["variant_id"] for v in by_text[k]})
        kr["source"] = "+".join(sorted(src[k]))
        kr["previously_labelled"] = k in labels
    add_repeats(items, key, n_repeat, "y", rng)
    write_batch("batch3", items, key)
    n = defaultdict(int)
    for k in keys:
        n["+".join(sorted(src[k]))] += 1
    print(f"batch3: {len(items)} items ({len(keys)} unique texts covering {sum(len(by_text[k]) for k in keys)} variants, "
          f"{sum(k in labels for k in keys)} previously labelled, {n_repeat} repeats); by source {dict(n)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=28, help="per perturbation type (dict, mlm, tech, llm)")
    ap.add_argument("--n-worst", type=int, default=14, help="per (attack type, attacked retriever)")
    ap.add_argument("--n-dose", type=int, default=15, help="per perturbation type (dict, mlm), 3-4 words")
    ap.add_argument("--valid-share", type=float, default=0.67, help="share of judge-valid items per stratum")
    ap.add_argument("--n-repeat", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--worst-rest", action="store_true", help="build batch 2: all remaining auto-accepted worst-case variants")
    ap.add_argument("--successes", action="store_true", help="build batch 3: re-check of every successful S7/S15 attack text")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.worst_rest:
        return worst_rest(rng, 10)
    if args.successes:
        return successes(rng, 6)

    pools = {
        "random": (read_jsonl(T2 / "variants_judged.jsonl"), rank_effect(T2 / "per_query_variants.csv", False)),
        "worst": (read_jsonl(T2 / "variants_worst_judged.jsonl"), rank_effect(T2 / "per_query_variants_worst.csv", True)),
        "dose": ([v for v in read_jsonl(T2 / "variants_dose_judged.jsonl") if v["n_subs"] >= 3],
                 rank_effect(T2 / "per_query_variants_dose.csv", False)),
    }
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for pool, (rows, eff) in pools.items():
        for v in rows:
            if v.get("judge_missing") or v["variant_id"] not in eff:
                continue
            v = dict(v, _pool=pool, _eff=eff[v["variant_id"]])
            group = v["perturbation_type"] if pool != "worst" else f"{v['perturbation_type']}|{v['attack_model']}"
            strata[(pool, group, bool(v["valid"]))].append(v)

    population = [{"pool": p, "group": g, "auto_valid": ok, "rank_worse": w, "n": sum(r["_eff"]["worse"] == w for r in rows)}
                  for (p, g, ok), rows in sorted(strata.items()) for w in (False, True)]
    quota = {"random": args.n_random, "worst": args.n_worst, "dose": args.n_dose}
    picked = []
    for (pool, group, valid), rows in sorted(strata.items()):
        n = round(quota[pool] * (args.valid_share if valid else 1 - args.valid_share))
        picked += draw(rows, n, rng)
    rng.shuffle(picked)

    items, key = records(picked, "v")
    add_repeats(items, key, args.n_repeat, "r", rng)

    write_batch("audit", items, key)
    (AUDIT / "audit_strata.json").write_text(json.dumps(population, indent=1), encoding="utf-8")

    k = pd.DataFrame([x for x in key if "pool" in x])
    print(f"{len(items)} items ({len(k)} unique + {args.n_repeat} repeats)")
    print(k.groupby(["pool", "auto_valid", "rank_worse"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
