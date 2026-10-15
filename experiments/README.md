# Synonym-robust retrieval experiments (CERT-UA)

Pipeline for the paper: a weakly supervised Ukrainian retrieval benchmark
with meaning-preserving substitutions, frozen retrievers, inference-time
robustness methods and a fine-tuning comparison.

## Run

```bash
tmux new -d -s certua './run_overnight.sh'     # detached; survives closing the SSH session
tmux attach -t certua                          # look at it (Ctrl-b d to detach again)
touch experiments/out/STOP                     # stop after the current stage
SMOKE=1 HOURS=2 ./run_overnight.sh             # tiny end-to-end check
```

Environment knobs: `HOURS` (deadline, default 10), `EXP_LLM_BUDGET_USD` (hard
OpenAI cap across all stages, default 8; spend is logged in `out/llm_spend.jsonl`),
`EXP_LLM_MODEL` (default `gpt-4.1-mini`). The OpenAI key is read from `.env`.

Progress: `experiments/out/STATUS.md` (one row per stage, with exit code),
logs in `experiments/out/logs/`, GPU usage in `logs/gpu.log`.

## Tiers

| tier | dir | what | main output |
|---|---|---|---|
| 1 | `out/tier1` | pilot, 50 title + 50 section queries | `out/report_tier1.md` (go/no-go) |
| 2 | `out/tier2` | full benchmark (~220 title + ~370 section queries) | `out/report_tier2.md` |
| 3 | `out/tier2` | extra retrievers (e5-large, BGE-M3 sparse/ColBERT/all, uk mpnet, …) + method ablations | `out/report_tier3.md` |
| 4 | `out/tier2` | dose-response (1–4 words), worst-case greedy substitutions + transfer | `out/report_tier4_*.md`, `tier2/figures/` |
| 5 | `out/tier2` | second judge (local MamayLM-12B, 4-bit): agreement, both-judges robustness | `out/report_tier5.md` |
| 6 | `out/tier2` | article-level 5-fold fine-tuning, with/without synonym augmentation | `out/report_tier6.md` |
| 7 | `out/tier2` | until deadline: new perturbation samples (local judge) + fine-tuning seeds | `tier2/report_variants_seed*.md` |

## Stages

`s0_corpus` → `s1_queries` → `s2_pool_dict` (runs in `../textfooler_ukr` venv) /
`s2_pool_mlm` (runs in `../bert_attack_uk` venv) → `s2_compose` → `s3_judge` →
`s4_retrieve` → `s5_methods` → `s6_report`; `s7_worstcase`, `s8_final` (figures),
`s9_finetune`, `local_judge`, `s10_aggregate` (stability across samples and seeds),
`s11_results` (curated `out/RESULTS.md`). Every stage skips work whose output exists
(`--force` to rebuild); LLM responses are cached in `out/llm_cache.jsonl`.

## Notes on validity

- Silver labels: a query's positive is the article/chunk it was generated from.
- Titles are stripped from indexed text; `bm25_titled` (titles kept) is a sanity check only.
- A variant is *valid* if the API judge says the information need is preserved
  AND SBERT cos ≥ 0.80. Check `audit_sample_*.md` by hand before trusting numbers.
- Robustness is paired (variant vs. its own original), CIs bootstrap over articles.
- Dictionary-based expansion can exactly undo a dictionary perturbation
  (`reverted %` column); the `mlm` source is the leakage-free comparison.

## Human validity audit

```bash
uv run python -m experiments.s12_audit_sample          # stratified blind sample -> experiments/audit/
# label in experiments/audit/review_tool.html (or the published copy), then Export
uv run python -m experiments.s13_audit_score audit_verdicts_A1.json   # -> audit/audit_report.md, human_labels.jsonl
```

| file | content |
|---|---|
| `audit/audit_items.json` | what the reviewer sees (no model output) |
| `audit/audit_key.jsonl` | hidden: pool, judge verdicts, ranks; do not open before finishing |
| `audit/audit_strata.json` | population size of each sampling stratum (for weighted estimates) |
| `audit/review_tool.template.html` | the tool; `review_tool.html` is the template with items inlined |

## Fixed-denominator worst-case analysis and fine-tuning under attack

```bash
# fine-tuned models, also evaluated on the worst-case variants (writes only missing run files; ~18 min per seed)
uv run python -m experiments.s9_finetune --out experiments/out/tier2 --models e5_base,bge_m3 --folds 5 --epochs 2 \
    --seed 13 --eval-names variants,variants_worst
# every query in the denominator; auto / human-estimated / human-verified validity; transfer; budget
uv run python -m experiments.s14_worstcase_verified --out experiments/out/tier2   # -> tier2/report_worst_verified.md
uv run python -m experiments.s11_results                                          # -> out/RESULTS.md
```

`s9` also writes `runs/folds_s<seed>.json` (article → fold; the seed sets the split)
and asserts that no article is in both training and test pairs. `s14` reads
`audit/human_labels.jsonl`; the estimated level draws unlabelled auto-accepted
variants from Beta posteriors of the audit strata (shrunk toward the pooled
rate with 5 pseudo-labels) and combines 1000 draws with an article bootstrap.

Validity-aware worst-case search (validate → select): `run_overnight2.sh` runs
`s15_worstcase_valid` (all admissible singles judged once, greedy search over
valid ones only) → `s3_judge` → `s4_retrieve` → `s14 --name variants_worstv` →
`s9 --eval-names …,variants_worstv` → `s11`. Second audit batch (all remaining
auto-accepted worst-case variants): `s12_audit_sample --worst-rest` →
`audit/review_tool_batch2.html`.

Pre-freeze checks:

```bash
# batch 3: re-check every variant text behind a successful S7/S15 attack (needs s14's variant_outcomes_*.jsonl)
uv run python -m experiments.s12_audit_sample --successes                         # -> audit/review_tool_batch3.html
# S7 vs S15: shared inputs, implementation equivalence, successes vs judge calls (cached verdicts only)
EXP_LLM_BUDGET_USD=6.40 uv run python -m experiments.s16_protocol_audit --out experiments/out/tier2
# silver-label spot-check: does the rank-1 document of a failure answer the query?
uv run python -m experiments.s17_relevance_check --out experiments/out/tier2     # -> audit/relevance_tool.html
uv run python -m experiments.s17_relevance_check --out experiments/out/tier2 --score audit/verdicts/relevance_A1.jsonl
```

`s13` treats batch 3 as a second pass whose verdicts replace earlier ones and
reports the re-check agreement. `s14` counts an identical text labelled more
than once as valid only if every label agrees, requires the automatic rule and
the human for **verified**, and adds a **natural** level (awkward wording
rejected) and a validity-only interval (`--draws` sets the number of draws).

Keys in the tool: `1 2 3` meaning, `A S D` fluency, `U` unsure, `← →` move,
`J` next unlabelled. Changing the annotator name starts a separate set of
verdicts (use it for a second annotator).
