"""S9 (exploratory, GPU-heavy): training-time defence vs. inference-time methods.

K-fold cross-validation split by source ARTICLE (a test article's queries and
variants are never seen in training). For each fold, a dense encoder is
fine-tuned with in-batch-negative contrastive loss (MultipleNegativesRanking)
on (query, source chunk) pairs from the training articles:

  ft_orig  original queries only
  ft_aug   original queries + their valid synonym variants (synonym-augmented
           training, cf. adversarial augmentation for classification)

Test-fold items (originals + all variants of every --eval-names set, e.g.
variants and variants_worst) are ranked with the fold model and written to
runs/<eval name>/ft_<model>_<cond>_s<seed>.jsonl, so S6/S14 report them next to
the frozen baselines; only missing files are written. The seed also sets the
article -> fold assignment, saved to runs/folds_s<seed>.json. With ~200 articles this is small-data fine-tuning:
treat as exploratory and compare against the frozen model on the same items.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

from experiments import retrievers as R
from experiments.common import OUT, log, read_jsonl
from experiments.s4_retrieve import items_for, rank_items

BASES = {"e5_base": ("intfloat/multilingual-e5-base", "query: ", "passage: ", 512, 32),
         "bge_m3": ("BAAI/bge-m3", "", "", 8192, 16)}


def train_fold(base: str, pairs: list[tuple[str, str]], seed: int, epochs: int, outdir: Path) -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    model_id, qp, dp, _, bs = BASES[base]
    model = SentenceTransformer(model_id, device="cuda")
    model.max_seq_length = 256
    ds = Dataset.from_dict({"anchor": [qp + a for a, _ in pairs], "positive": [dp + p for _, p in pairs]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(outdir / "trainer"), num_train_epochs=epochs, per_device_train_batch_size=bs,
        learning_rate=2e-5, warmup_ratio=0.1, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=50, report_to="none", seed=seed)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(outdir / "model"))
    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants")
    ap.add_argument("--models", default="e5_base,bge_m3")
    ap.add_argument("--conds", default="ft_orig,ft_aug")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--eval-names", default=None, help="variant sets to evaluate (default: --name)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    eval_names = (args.eval_names or args.name).split(",")
    corpus = R.Corpus()
    chunk_text = {c["chunk_id"]: c["text"] for c in corpus.chunks}
    first_chunk = {}
    for c in corpus.chunks:
        first_chunk.setdefault(c["doc_id"], c["text"])
    queries = read_jsonl(out / "queries.jsonl")
    variants = read_jsonl(out / f"{args.name}_judged.jsonl")
    sets = {n: items_for(out, n) for n in eval_names}
    items = list({it["id"]: it for its in sets.values() for it in its}.values())  # originals shared across sets
    member = {n: {it["id"] for it in its} for n, its in sets.items()}

    arts = sorted({q["article_id"] for q in queries})
    random.Random(args.seed).shuffle(arts)
    fold_of = {a: i % args.folds for i, a in enumerate(arts)}
    assert all(it["article_id"] in fold_of for it in items), "test item from an article without a fold"
    folds_path = out / "runs" / f"folds_s{args.seed}.json"
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds_path.write_text(json.dumps({"seed": args.seed, "folds": args.folds, "fold_of": fold_of}, indent=0), encoding="utf-8")

    def positive(rec):
        return chunk_text.get(rec["chunk_id"]) if rec.get("chunk_id") else first_chunk[rec["article_id"]]

    for base in args.models.split(","):
        for cond in args.conds.split(","):
            paths = {n: out / "runs" / n / f"ft_{base}_{cond}_s{args.seed}.jsonl" for n in eval_names}
            todo = {n: p for n, p in paths.items() if args.force or not p.exists()}
            if not todo:
                log(f"[skip] ft_{base}_{cond}_s{args.seed}: all outputs exist")
                continue
            t0 = time.time()
            rows = []
            for k in range(args.folds):
                train = [q for q in queries if fold_of[q["article_id"]] != k]
                if cond == "ft_aug":
                    train += [v for v in variants if v["valid"] and fold_of[v["article_id"]] != k]
                pairs = [(r.get("synonym_query") or r["original_query"], positive(r)) for r in train]
                train_arts = {r["article_id"] for r in train}
                tmp = Path(tempfile.mkdtemp(prefix=f"ft_{base}_{cond}_{k}_", dir=OUT / "cache"))
                try:
                    train_fold(base, pairs, args.seed + k, args.epochs, tmp)
                    _, qp, dp, doc_len, _ = BASES[base]
                    r = R.Dense(corpus, f"ft_{base}_{cond}_s{args.seed}_f{k}", str(tmp / "model"), qp, dp, doc_len)
                    test = [it for it in items if fold_of.get(it["article_id"]) == k]
                    assert not train_arts & {it["article_id"] for it in test}, "article in both train and test"
                    rows += rank_items(r, test, corpus, list(R.LEVELS))
                    r.close()
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                    for f in (OUT / "cache").glob(f"emb_ft_{base}_{cond}_s{args.seed}_f{k}_*.npy"):
                        f.unlink()
                log(f"ft {base} {cond} fold {k}: {len(pairs)} training pairs, {len(test)} test items")
            for n, path in todo.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    for row in rows:
                        if row["id"] in member[n]:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"ft {base} {cond}: done in {(time.time() - t0) / 60:.1f} min -> {', '.join(map(str, todo.values()))}")


if __name__ == "__main__":
    main()
