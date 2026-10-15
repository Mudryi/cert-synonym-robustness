# Validity-aware synonym robustness of Ukrainian document retrieval (CERT-UA)

Code for the paper *Beyond Average Robustness: Validity-Aware Evaluation of Synonym
Attacks on Ukrainian Document Retrieval*.

We build a weakly supervised Ukrainian cybersecurity retrieval benchmark from the
public articles of [CERT-UA](https://cert.gov.ua/articles), perturb queries with
meaning-preserving word substitutions (synonym dictionary, MLM, LLM, curated
terms), and measure how 12 retrievers (BM25, multilingual-e5, BGE-M3, hybrid, …)
react to random, dose-response and worst-case (greedy) substitutions. Validity of
every substitution is checked by an LLM judge, a second local judge, and a blind
human audit (887 verdicts), and results are reported with automatic,
human-estimated and human-verified validity.

All reported numbers are in [`experiments/out/RESULTS.md`](experiments/out/RESULTS.md);
figures are in `experiments/out/tier2/figures/`.

## Repository layout

| path | content |
|---|---|
| `cert_ua/` | crawler and parser for the CERT-UA archive ([details](cert_ua/README.md)) |
| `main.py` | CLI for the crawler (`crawl`, `parse`, `chunk`, `validate`, `stats`) |
| `data/processed/` | the parsed corpus: `cert_ua_articles.jsonl`, `cert_ua_text_chunks.jsonl` |
| `experiments/` | the experiment pipeline, stages `s0`–`s17` ([details](experiments/README.md)) |
| `experiments/audit/` | blind human-audit tool, sampling keys and human verdicts |
| `experiments/out/` | `RESULTS.md`, final reports and figures |
| `tests/` | offline tests for the parser |

Intermediate experiment files (queries, substitution pools, judged variants,
per-query ranks, LLM cache) are released as a dataset on Hugging Face (link TBA).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically).

```bash
uv sync
uv run pytest
```

LLM stages read `OPENAI_API_KEY` from `.env`. The two candidate-pool stages run
in the environments of separate repositories:

| stage | repository | environment variable |
|---|---|---|
| `s2_pool_dict` | [textfooler-ukr](https://github.com/Mudryi/textfooler-ukr) | `TEXTFOOLER_UKR`, `SYNONYM_DICT` |
| `s2_pool_mlm` | bert_attack_uk | `BERT_ATTACK_UK` |

By default they are expected next to this repository (`../textfooler_ukr`, `../bert_attack_uk`).

## Reproducing

```bash
# corpus (optional: data/processed/ is already included)
uv run python main.py crawl --all && uv run python main.py parse && uv run python main.py chunk

# full pipeline, detached (see experiments/README.md for tiers and knobs)
tmux new -d -s certua './run_overnight.sh'
./run_overnight2.sh                                   # validity-aware worst-case search (S15)

# tables from the human audit and fixed-denominator analysis
uv run python -m experiments.s13_audit_score experiments/audit/verdicts/A1*.jsonl
uv run python -m experiments.s14_worstcase_verified --out experiments/out/tier2
uv run python -m experiments.s11_results              # -> experiments/out/RESULTS.md
```

## Data and licence

The code is released under the MIT licence. The corpus consists of public
articles by CERT-UA (the Computer Emergency Response Team of Ukraine); all
rights to the texts remain with their authors, and the source URL of every
article is kept in the data. The articles contain **real indicators of
compromise** (malicious domains, URLs, IP addresses, e-mail addresses and file
hashes); do not open or resolve them.

## Citation

See [`CITATION.cff`](CITATION.cff).
