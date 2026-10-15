"""S2 (MLM pool): contextual candidates from ../bert_attack_uk.

Runs with ../bert_attack_uk/bert_attack_env/bin/python. Uses the BERT-Attack
recipe of that repo: xlm-roberta-large fill-mask candidates for each content
word, filtered by `filter_not_words`, `has_foreign_letters`, fastText cbow
similarity (`is_semantic_near`) and a different lemma (`compare_normal_forms`).
Candidates are not guaranteed synonyms: meaning preservation is decided later
by the S3 judge. Self-contained: does not import experiments.common.

input/output format: see s2_pool_dict.py
"""
import argparse
import json
import os
import sys

REPO = os.environ.get("BERT_ATTACK_UK", os.path.join(os.path.dirname(__file__), "..", "..", "bert_attack_uk"))

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--max-cands", type=int, default=15)
ap.add_argument("--top-k", type=int, default=64)
ap.add_argument("--min-score", type=float, default=0.005)
ap.add_argument("--ft-threshold", type=float, default=0.33)
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()
args.input, args.output = os.path.abspath(args.input), os.path.abspath(args.output)

os.chdir(REPO)  # main.py loads fasttext_uk_cbow/cbow.uk.300.bin by relative path
sys.path.insert(0, REPO)
import re  # noqa: E402

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

import main as ba  # noqa: E402  (bert_attack_uk/main.py; loads fastText + pymorphy2 at import)

CYR_WORD = re.compile(r"^[а-щьюяґєіїА-ЩЬЮЯҐЄІЇ']+$")
unmasker = pipeline("fill-mask", model="FacebookAI/xlm-roberta-large", device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16)


def perturbable(tok, start, protected):
    if not CYR_WORD.match(tok) or tok.isupper() or ba.filter_not_words(tok):
        return False
    end = start + len(tok)
    return not any(p["start"] < end and start < p["end"] for p in protected)


rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
jobs = []  # (row_index, token_index, masked_text)
token_lists = []
for r_i, row in enumerate(rows):
    tokens = ba.tokenize_with_whitespace(row["text"])
    token_lists.append(tokens)
    offset = 0
    for i, tok in enumerate(tokens):
        start = offset
        offset += len(tok)
        if perturbable(tok, start, row.get("protected", [])):
            jobs.append((r_i, i, "".join("<mask>" if j == i else t for j, t in enumerate(tokens))))
print(f"mlm pool: {len(rows)} texts, {len(jobs)} masked positions", flush=True)

positions = [[] for _ in rows]
with torch.no_grad():
    for b in range(0, len(jobs), args.batch_size):
        batch = jobs[b:b + args.batch_size]
        preds = unmasker([j[2] for j in batch], top_k=args.top_k, batch_size=args.batch_size)
        if len(batch) == 1:
            preds = [preds]
        for (r_i, i, _), pred in zip(batch, preds):
            word = token_lists[r_i][i]
            cands = []
            for p in pred:
                if p["score"] < args.min_score:
                    break
                new = p["token_str"].strip()
                if word[:1].isupper() and new:
                    new = new[0].upper() + new[1:]
                if (not new or not CYR_WORD.match(new) or ba.has_foreign_letters(new) or ba.filter_not_words(new, word)
                        or new.lower() == word.lower() or not ba.is_semantic_near(word.lower(), new.lower(), args.ft_threshold)
                        or ba.compare_normal_forms(word.lower(), new.lower())):
                    continue
                cands.append(new)
                if len(cands) >= args.max_cands:
                    break
            if cands:
                positions[r_i].append({"idx": i, "word": word, "cands": list(dict.fromkeys(cands))})
        if (b // args.batch_size) % 50 == 0:
            print(f"  {b + len(batch)}/{len(jobs)}", flush=True)

with open(args.output + ".tmp", "w", encoding="utf-8") as f:
    for row, tokens, pos in zip(rows, token_lists, positions):
        f.write(json.dumps({"tid": row["tid"], "text": row["text"], "tokens": tokens, "positions": pos}, ensure_ascii=False) + "\n")
os.replace(args.output + ".tmp", args.output)
print(f"mlm pool: {sum(map(len, positions))} positions, {sum(len(p['cands']) for ps in positions for p in ps)} candidates", flush=True)
