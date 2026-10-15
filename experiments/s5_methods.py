"""S5: inference-time robustness methods (no training).

For every input query x (originals and perturbed variants), single-word lexical
variants of x are generated from the candidate pools of x itself (the method
never sees the original query of a perturbed variant). Configurations:

  naive:<src>:<K>          K random single-substitution variants, RRF with x
  val:<src>:<K>:<tau>      contextual admissibility: keep variants with SBERT
                           cos(x, x') >= tau, top-K by cos, RRF with x
  src: dt (dictionary + technical terms) | mlm | all

Selective activation (expansion only when the base top1-top2 margin is low)
is derived in S6 from the stored base margins, so any percentile can be
evaluated without re-running. Also writes single-query latency measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from experiments import retrievers as R
from experiments.common import OUT, apply_subs, log, protected_entities, read_jsonl, sbert_cos, write_json
from experiments.s2_compose import tech_variants

MAX_PER_POS = 5


def parse_configs(spec: str) -> list[dict]:
    out = []
    for c in spec.split(","):
        p = c.split(":")
        cfg = {"name": c, "sel": p[0], "src": p[1], "K": int(p[2])}
        if p[0] == "val":
            cfg["tau"] = float(p[3])
        out.append(cfg)
    return out


def load_pools(out: Path, name: str, kind: str) -> dict:
    pools = {}
    for f in (out / f"pool_{kind}_queries.jsonl", out / f"pool_{kind}_{name}.jsonl"):
        if f.exists():
            pools.update({r["tid"]: r for r in read_jsonl(f)})
        else:
            log(f"warning: missing pool {f}")
    return pools


def candidates(item: dict, pools: dict[str, dict]) -> list[tuple[str, str]]:
    """(text, source) single-substitution variants of the input query."""
    out: list[tuple[str, str]] = []
    for kind in ("dict", "mlm"):
        p = pools[kind].get(item["id"])
        if not p:
            continue
        for pos in p["positions"]:
            for c in pos["cands"][:MAX_PER_POS]:
                out.append((apply_subs(p["tokens"], {pos["idx"]: c}), kind))
    for text, _, _ in tech_variants(item["text"], protected_entities(item["text"])):
        out.append((text, "tech"))
    seen, uniq = set(), []
    for t, s in out:
        if t != item["text"] and t not in seen:
            seen.add(t)
            uniq.append((t, s))
    return uniq


SRC = {"dt": {"dict", "tech"}, "mlm": {"mlm"}, "all": {"dict", "tech", "mlm"}}


def select(item: dict, cands: list[tuple[str, str, float]], cfg: dict) -> list[str]:
    pool = [c for c in cands if c[1] in SRC[cfg["src"]]]
    if cfg["sel"] == "naive":
        seed = int(hashlib.md5((item["id"] + cfg["name"]).encode()).hexdigest()[:8], 16)
        return [c[0] for c in random.Random(seed).sample(pool, min(cfg["K"], len(pool)))]
    ok = sorted((c for c in pool if c[2] >= cfg["tau"]), key=lambda c: -c[2])
    return [c[0] for c in ok[:cfg["K"]]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants")
    ap.add_argument("--models", default="bm25,e5_base,bge_m3,hybrid")
    ap.add_argument("--configs", default="naive:dt:3,val:dt:3:0.8")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    corpus = R.Corpus()
    configs = parse_configs(args.configs)

    queries = read_jsonl(out / "queries.jsonl")
    variants = read_jsonl(out / f"{args.name}_judged.jsonl")
    originals = {q["query_id"]: q["original_query"] for q in queries}
    items = [{"id": q["query_id"], "kind": "orig", "text": q["original_query"], "article_id": q["article_id"],
              "chunk_id": q["chunk_id"], "original": q["original_query"]} for q in queries]
    items += [{"id": v["variant_id"], "kind": "variant", "text": v["synonym_query"], "article_id": v["article_id"],
               "chunk_id": v["chunk_id"], "original": originals[v["query_id"]]} for v in variants]
    pools = {k: load_pools(out, args.name, k) for k in ("dict", "mlm")}

    # candidate generation + contextual admissibility scores (shared by all models)
    cand_path = out / f"expansion_candidates_{args.name}.jsonl"
    if cand_path.exists():
        cand = {r["id"]: [tuple(c) for c in r["cands"]] for r in read_jsonl(cand_path)}
    else:
        t0 = time.time()
        raw = {it["id"]: candidates(it, pools) for it in items}
        pairs = [(it["text"], t) for it in items for t, _ in raw[it["id"]]]
        cos = iter(sbert_cos(pairs))
        cand = {it["id"]: [(t, s, round(next(cos), 4)) for t, s in raw[it["id"]]] for it in items}
        with open(cand_path, "w", encoding="utf-8") as f:
            for k, v in cand.items():
                f.write(json.dumps({"id": k, "cands": v}, ensure_ascii=False) + "\n")
        log(f"expansion candidates: {len(pairs)} for {len(items)} inputs in {time.time() - t0:.0f}s")

    for model in args.models.split(","):
        path = out / "runs_methods" / args.name / f"{model}__{args.tag}.jsonl"
        if path.exists() and not args.force:
            log(f"[skip] {path}")
            continue
        try:
            r = R.get(model, corpus)
            sbert_cos([(items[0]["text"], items[0]["text"])])  # warm-up: exclude one-off SBERT loading from timing
            # single-query latency (batch of 1), on texts the encoder has not cached
            probe = [it["text"] + " ." for it in items[:40]]
            t0 = time.time()
            for t in probe:
                r.score([t])
            t_ret = (time.time() - t0) / len(probe)
            t0 = time.time()
            for it in items[:40]:
                sbert_cos([(it["text"], c[0]) for c in cand[it["id"]]] or [(it["text"], it["text"])])
            t_sbert = (time.time() - t0) / 40
            write_json(out / "runs_methods" / args.name / f"latency_{model}.json",
                       {"retrieval_ms": 1000 * t_ret, "sbert_validation_ms": 1000 * t_sbert,
                        "mean_candidates": float(np.mean([len(cand[it["id"]]) for it in items]))})

            rows = []
            t0 = time.time()
            for b in range(0, len(items), 128):
                batch = items[b:b + 128]
                chosen = {(it["id"], cfg["name"]): select(it, cand[it["id"]], cfg) for it in batch for cfg in configs}
                texts = list(dict.fromkeys([it["text"] for it in batch] + [t for v in chosen.values() for t in v]))
                pos = {t: i for i, t in enumerate(texts)}
                scores = {}
                for s0, sc in R.score_batched(r, texts):
                    for lv, m in sc.items():
                        scores.setdefault(lv, []).append(m)
                scores = {lv: np.concatenate(ms) for lv, ms in scores.items()}
                for it in batch:
                    for lv, mat in scores.items():
                        if lv == "chunk":
                            if not it["chunk_id"]:
                                continue
                            gold = corpus.chunk_pos[it["chunk_id"]]
                        else:
                            gold = corpus.doc_pos[it["article_id"]]
                        base = mat[pos[it["text"]]]
                        top2 = np.partition(-base, 1)[:2]
                        rows.append({"id": it["id"], "kind": it["kind"], "level": lv, "config": "base",
                                     "rank": R.gold_rank(base, gold), "margin": float(top2[1] - top2[0]), "n_exp": 0})
                        for cfg in configs:
                            exp = chosen[(it["id"], cfg["name"])]
                            fused = R.rrf([base[None, :]] + [mat[pos[t]][None, :] for t in exp])[0] if exp else base
                            rows.append({"id": it["id"], "kind": it["kind"], "level": lv, "config": cfg["name"],
                                         "rank": R.gold_rank(fused, gold), "n_exp": len(exp),
                                         "reverted": it["kind"] == "variant" and it["original"] in exp})
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.with_suffix(".tmp"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            path.with_suffix(".tmp").replace(path)
            log(f"{model}: methods {[c['name'] for c in configs]} on {len(items)} inputs in {time.time() - t0:.0f}s")
        except Exception:
            import traceback
            log(f"{model} FAILED:\n{traceback.format_exc()}")
        if model not in ("bm25", "bge_m3"):
            R.release_all()
    R.release_all()


if __name__ == "__main__":
    main()
