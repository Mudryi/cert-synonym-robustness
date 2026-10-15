"""S17: spot-check of the silver relevance labels (one source article per query).

A retrieval "failure" may be a false failure when the document ranked first
also answers the query (another CERT-UA report on the same campaign, malware or
group). Two blind samples:
  attack      queries whose source left the attacked retriever's top 5 under an
              auto-valid S7/S15 attack; shown: the attacked query and the
              attacked retriever's rank-1 document
  clean_fail  original queries that the hybrid retriever already misses (source
              outside the top 5); shown: the query and hybrid's rank-1 document
The reviewer sees the query, the rank-1 document (title + summary) and the
source document, and answers whether the rank-1 document answers the query.

  build:  python -m experiments.s17_relevance_check --out experiments/out/tier2
  score:  python -m experiments.s17_relevance_check --out experiments/out/tier2 --score <verdicts>

The page reuses audit/review_tool.template.html with relabelled rubric rows:
meaning = does the rank-1 document answer the query (yes / partly / no),
fluency = how it relates to the source (same report / related / different).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from experiments.common import EXP, OUT, SEED, read_jsonl
from experiments.s6_report import md_table
from experiments.s13_audit_score import load_verdicts, pct, wilson

AUDIT = EXP / "audit"
LEVEL = "doc_max"
RELATION = {"natural": "same report", "awkward": "related report", "ungrammatical": "different"}

TEMPLATE_EDITS = [
    ("<title>Substitution Validity Audit</title>", "<title>Relevance Spot-Check</title>"),
    ("<h1>Substitution Validity Audit</h1>", "<h1>Relevance Spot-Check</h1>"),
    ('<span class="lab">Original</span>', '<span class="lab">Query</span>'),
    ('<span class="lab">Variant</span>', '<span class="lab">Rank 1</span>'),
    ('<details id="ctxBox"><summary>Source passage (optional context)</summary>',
     '<details id="ctxBox" open><summary>Rank-1 document and the source document</summary>'),
    ('<div class="lab">Same information need?<small>Would an analyst get the same documents?</small></div>',
     '<div class="lab">Does rank 1 answer the query?<small>Judged from the rank-1 document</small></div>'),
    ('<div class="lab">Fluency<small>As Ukrainian text</small></div>',
     '<div class="lab">Rank 1 vs. source<small>Same report, related, or different</small></div>'),
    ('<kbd>A</kbd>Natural', '<kbd>A</kbd>Same report'),
    ('<kbd>S</kbd>Awkward', '<kbd>S</kbd>Related report'),
    ('<kbd>D</kbd>Ungrammatical', '<kbd>D</kbd>Different topic'),
    ('$("#itemSubs").textContent = `${it.changed.length} word${it.changed.length > 1 ? "s" : ""} changed`;',
     '$("#itemSubs").textContent = "";'),
]
GUIDE = ("<ul>\n"
         "        <li>The query was generated from the source document. Decide whether the <b>rank-1</b> document would satisfy "
         "an analyst who typed the query: <b>Yes</b> if it answers it as well as the source would, <b>Partly</b> if it covers "
         "the same campaign, malware or group but misses the specific request, <b>No</b> otherwise.</li>\n"
         "        <li>Second row: is the rank-1 document the same report as the source (e.g. an update or a duplicate), a "
         "related report (same actor, malware or campaign), or about something different?</li>\n"
         "        <li>The query may contain a substituted synonym; judge the information need, not the wording.</li>\n"
         "      </ul>")


def runs(out: Path, name: str, model: str) -> dict[tuple[str, str], dict]:
    return {(r["id"], r["kind"]): r for r in read_jsonl(out / "runs" / name / f"{model}.jsonl") if r["level"] == LEVEL}


def build(out: Path, n_attack: int, n_clean: int, rng: random.Random) -> None:
    docs = {d["doc_id"]: d for d in read_jsonl(OUT / "corpus" / "corpus_docs.jsonl")}
    chunks = {c["chunk_id"]: c for c in read_jsonl(OUT / "corpus" / "corpus_chunks.jsonl")}
    Q = {q["query_id"]: q for q in read_jsonl(out / "queries.jsonl")}
    cand = defaultdict(list)  # query -> successful (search, variant)
    for name in ("variants_worstv", "variants_worst"):
        V = {v["variant_id"]: v for v in read_jsonl(out / f"{name}_judged.jsonl")}
        for o in read_jsonl(out / f"variant_outcomes_{name}.jsonl"):
            if o["valid"] and o["outcome"] == "out of top 5":
                cand[o["query_id"]].append((name, o["attack_model"], V[o["variant_id"]]))
    picked = []
    for qid in rng.sample(sorted(cand), min(n_attack, len(cand))):
        name, m, v = rng.choice(cand[qid])
        r = runs(out, name, m)[(v["variant_id"], "variant")]
        picked.append({"kind": "attack", "query_id": qid, "search": name, "model": m, "variant_id": v["variant_id"],
                       "query": v["synonym_query"], "rank": r["rank"], "top1": r["top10"][0]})
    hyb = runs(out, "variants_worstv", "hybrid")
    fails = [qid for qid in Q if hyb[(qid, "orig")]["rank"] > 5]
    for qid in rng.sample(fails, min(n_clean, len(fails))):
        r = hyb[(qid, "orig")]
        picked.append({"kind": "clean_fail", "query_id": qid, "search": None, "model": "hybrid", "variant_id": None,
                       "query": Q[qid]["original_query"], "rank": r["rank"], "top1": r["top10"][0]})
    rng.shuffle(picked)

    def doc_block(d: dict, passage: str | None = None) -> str:
        body = passage or d.get("summary") or d["text"]
        return f"{d['title']}\n{body[:700]}"

    items, key = [], []
    for i, p in enumerate(picked, 1):
        q = Q[p["query_id"]]
        src = docs[q["article_id"]]
        top = docs[p["top1"]]
        passage = chunks[q["chunk_id"]]["text"] if q.get("chunk_id") in chunks else None
        iid = f"r{i:03d}"
        items.append({"id": iid, "subset": q["subset"], "original": p["query"], "variant": top["title"], "changed": [],
                      "context": f"RANK 1 ({top['publication_date']})\n{doc_block(top)}\n\n──── SOURCE ({src['publication_date']}), "
                                 f"the query was generated from this ────\n{doc_block(src, passage)}"})
        key.append({"item_id": iid, "source_doc": q["article_id"]} | p)
    tpl = (AUDIT / "review_tool.template.html").read_text(encoding="utf-8")
    for a, b in TEMPLATE_EDITS:
        assert a in tpl, a
        tpl = tpl.replace(a, b)
    s, e = tpl.index('<details class="guide" open>'), tpl.index('<details class="guide" id="ioBox">')
    g = tpl[s:e]
    tpl = tpl[:s] + g[:g.index("<ul>")] + GUIDE + g[g.index("</ul>") + 5:] + tpl[e:]
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    (AUDIT / "relevance_tool.html").write_text(tpl.replace("/*__ITEMS__*/[]", data), encoding="utf-8")
    with open(AUDIT / "relevance_key.jsonl", "w", encoding="utf-8") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"relevance check: {len(items)} items ({sum(p['kind'] == 'attack' for p in picked)} attack, "
          f"{sum(p['kind'] == 'clean_fail' for p in picked)} clean failures) -> {AUDIT / 'relevance_tool.html'}")


def score(paths: list[str]) -> None:
    V = load_verdicts(paths)
    K = pd.DataFrame(read_jsonl(AUDIT / "relevance_key.jsonl"))
    D = K.merge(V, on="item_id")
    D["relation"] = D.fluency.map(RELATION)
    rows, summary = [], {}
    for kind, g in list(D.groupby("kind")) + [("all", D)]:
        k_yes, k_any = int((g.meaning == "yes").sum()), int(g.meaning.isin(["yes", "partly"]).sum())
        lo, hi = wilson(k_yes, len(g))
        rows.append({"sample": kind, "n": len(g), "rank 1 answers the query (yes) %": f"{pct(k_yes / len(g))} [{pct(lo)}, {pct(hi)}]",
                     "yes or partly %": pct(k_any / len(g)), "same report %": pct((g.relation == "same report").mean()),
                     "related report %": pct((g.relation == "related report").mean())})
        summary[kind] = {"n": len(g), "false_failure": k_yes / len(g), "false_failure_ci": [lo, hi], "yes_or_partly": k_any / len(g)}
    md = ["# Relevance spot-check of the silver labels", "",
          "Share of retrieval failures where the document ranked first also answers the query (a false failure of the "
          "single-source label). `attack`: failures caused by an auto-valid S7/S15 attack on the attacked retriever; "
          "`clean_fail`: original queries the hybrid retriever misses. 95% Wilson intervals.", "", md_table(pd.DataFrame(rows)), ""]
    (AUDIT / "relevance_report.md").write_text("\n".join(md), encoding="utf-8")
    (AUDIT / "relevance_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-attack", type=int, default=24)
    ap.add_argument("--n-clean", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--score", nargs="*", help="verdict files exported from the relevance tool or dumped from its db")
    args = ap.parse_args()
    if args.score:
        return score(args.score)
    build(Path(args.out), args.n_attack, args.n_clean, random.Random(args.seed))


if __name__ == "__main__":
    main()
