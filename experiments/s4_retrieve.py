"""S4: retrieve with frozen models for original queries and all variants.

Writes one row per (item, level) to runs/<name>/<model>.jsonl with the
pessimistic rank of the silver positive, the top-10 ids and the top1-top2
score margin. Doc levels use the source article; the chunk level is used only
for section queries (source chunk). Each model is resumable on its own.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import retrievers as R
from experiments.common import OUT, log, read_jsonl


def items_for(out: Path, name: str) -> list[dict]:
    queries = read_jsonl(out / "queries.jsonl")
    items = [{"id": q["query_id"], "kind": "orig", "text": q["original_query"], "article_id": q["article_id"],
              "chunk_id": q["chunk_id"]} for q in queries]
    vpath = out / f"{name}_judged.jsonl"
    if vpath.exists():
        items += [{"id": v["variant_id"], "kind": "variant", "text": v["synonym_query"], "article_id": v["article_id"],
                   "chunk_id": v["chunk_id"]} for v in read_jsonl(vpath)]
    return items


def run_model(model: str, items: list[dict], corpus: R.Corpus, path: Path, levels: list[str]) -> None:
    t0 = time.time()
    r = R.get(model, corpus)
    log(f"{model}: ready in {time.time() - t0:.0f}s; scoring {len(items)} texts")
    t1 = time.time()
    rows = rank_items(r, items, corpus, levels)
    dt = time.time() - t1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.with_suffix(".tmp").replace(path)
    log(f"{model}: {len(rows)} rows in {dt:.0f}s ({1000 * dt / max(1, len(items)):.1f} ms/query batched)")


def rank_items(r, items: list[dict], corpus: R.Corpus, levels: list[str]) -> list[dict]:
    texts = [it["text"] for it in items]
    rows = []
    for start, scores in R.score_batched(r, texts):
        for lv, mat in scores.items():
            if lv not in levels:
                continue
            ids = corpus.chunk_ids if lv == "chunk" else corpus.doc_ids
            for j in range(mat.shape[0]):
                it = items[start + j]
                if lv == "chunk":
                    if not it["chunk_id"] or it["chunk_id"] not in corpus.chunk_pos:
                        continue
                    gold = corpus.chunk_pos[it["chunk_id"]]
                else:
                    gold = corpus.doc_pos[it["article_id"]]
                row = mat[j]
                top = np.argpartition(-row, 10)[:10]
                top = top[np.argsort(-row[top])]
                rows.append({"id": it["id"], "kind": it["kind"], "level": lv, "rank": R.gold_rank(row, gold),
                             "top10": [ids[k] for k in top], "margin": float(row[top[0]] - row[top[1]])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants")
    ap.add_argument("--models", default="bm25,bm25_titled,e5_base,bge_m3,hybrid")
    ap.add_argument("--levels", default=",".join(R.LEVELS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    corpus = R.Corpus()
    items = items_for(out, args.name)
    missing = {it["article_id"] for it in items} - set(corpus.doc_pos)
    assert not missing, f"source documents missing from the index: {sorted(missing)[:5]}"
    failed = []
    for model in args.models.split(","):
        path = out / "runs" / args.name / f"{model}.jsonl"
        if path.exists() and not args.force:
            log(f"[skip] {path}")
            continue
        try:
            run_model(model, items, corpus, path, args.levels.split(","))
        except Exception:
            failed.append(model)
            log(f"{model} FAILED:\n{traceback.format_exc()}")
        if model not in ("bm25", "bge_m3", "e5_base") or model == args.models.split(",")[-1]:
            R.release_all()
    if failed:
        raise SystemExit(f"failed models: {failed}")


if __name__ == "__main__":
    main()
