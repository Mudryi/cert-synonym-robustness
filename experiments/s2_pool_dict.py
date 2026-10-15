"""S2 (dictionary pool): synonym candidates from ../textfooler_ukr.

Runs with ../textfooler_ukr/textfooler_env/bin/python (Python 3.10, pymorphy2).
For every input text and every content-word position, emit the inflected
single-word replacements obtained with the textfooler_ukr synonym dictionary
(synonimy.info + hand-parsed entries, antonyms removed) and its morphology-
aware `replace_word`. Self-contained: does not import experiments.common.

input  jsonl: {"tid", "text", "protected": [{"start", "end"}]}
output jsonl: {"tid", "text", "tokens", "positions": [{"idx", "word", "cands": [..]}]}
"""
import argparse
import json
import os
import re
import sys
from functools import lru_cache

REPO = os.environ.get("TEXTFOOLER_UKR", os.path.join(os.path.dirname(__file__), "..", "..", "textfooler_ukr"))
DICT_PATH = os.environ.get("SYNONYM_DICT", os.path.join(REPO, "synonimy_info_clean.json"))

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--max-cands", type=int, default=15)
args = ap.parse_args()
args.input, args.output = os.path.abspath(args.input), os.path.abspath(args.output)

os.chdir(REPO)  # the repo loads hand_parsed_top_100.json by relative path
sys.path.insert(0, REPO)
import pymorphy2  # noqa: E402
import criteria  # noqa: E402
from prepare_synonym_dict import read_and_clean_synonym_dict  # noqa: E402
from synonym_replacement import replace_word, tokenize_ukrainian  # noqa: E402

_morph = pymorphy2.MorphAnalyzer(lang="uk")


class CachedMorph:
    """pymorphy2 analyser with memoised parse (replace_word re-parses every token)."""

    @lru_cache(maxsize=500_000)
    def parse(self, word):
        return _morph.parse(word)


morph = CachedMorph()
STOP = criteria.get_stopwords()
synonym_dict = read_and_clean_synonym_dict(DICT_PATH, use_antonyms=True, used_hand_parse=True)
CYR_WORD = re.compile(r"^[а-щьюяґєіїА-ЩЬЮЯҐЄІЇ']+$")


def all_synonyms(word):
    lw = word.lower()
    nf = morph.parse(lw)[0].normal_form
    syns = synonym_dict.get(lw) or synonym_dict.get(nf) or []
    return [s for s in dict.fromkeys(syns) if " " not in s and s.lower() not in (lw, nf.lower())]


def perturbable(tok, start, protected):
    if not CYR_WORD.match(tok) or len(tok) < 3 or tok.lower() in STOP:
        return False
    if tok.isupper():  # abbreviations (ШПЗ, СБУ) belong to the technical-term table
        return False
    end = start + len(tok)
    return not any(p["start"] < end and start < p["end"] for p in protected)


n_in = n_pos = n_cands = 0
with open(args.input, encoding="utf-8") as fin, open(args.output + ".tmp", "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        text = row["text"]
        tokens = tokenize_ukrainian(text)
        positions, offset = [], 0
        for i, tok in enumerate(tokens):
            start = offset
            offset += len(tok)
            if not perturbable(tok, start, row.get("protected", [])):
                continue
            orig_pos = criteria.get_pos([tok])[0]
            cands = []
            for syn in all_synonyms(tok):
                new_tokens = replace_word(text, tok, syn, morph)
                if not new_tokens or len(new_tokens) != len(tokens):
                    continue
                form = new_tokens[i]
                if form == tok or not CYR_WORD.match(form):
                    continue
                if not criteria.pos_filter(orig_pos, criteria.get_pos([form]))[0]:
                    continue
                cands.append(form)
                if len(cands) >= args.max_cands:
                    break
            if cands:
                positions.append({"idx": i, "word": tok, "cands": list(dict.fromkeys(cands))})
                n_pos += 1
                n_cands += len(cands)
        fout.write(json.dumps({"tid": row["tid"], "text": text, "tokens": tokens, "positions": positions}, ensure_ascii=False) + "\n")
        n_in += 1
os.replace(args.output + ".tmp", args.output)
print(f"dict pool: {n_in} texts, {n_pos} positions, {n_cands} candidates", flush=True)
