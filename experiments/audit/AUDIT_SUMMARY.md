# Human validity audit: summary (annotator A1, 2026-10-12/13)

**887 verdicts in three passes, none marked unsure:**
- **Batch 1** (`audit_key.jsonl`): a stratified, blind sample of 254 substitutions plus 15 hidden repeats. It covers the random, worst-case and dose pools, both accepted and rejected variants, and rank-changing as well as harmless ones.
- **Batch 2** (`batch2_key.jsonl`): all 437 automatically accepted worst-case variants that batch 1 did not cover, plus 10 repeats. Together with batch 1, every one of the 509 auto-accepted worst-case variants is labelled (a census).
- **Batch 3** (`batch3_key.jsonl`): a blind second pass over the 165 unique variant texts behind every successful S7 or S15 attack, plus 6 repeats, with an explicit grammar rule. Its verdicts replace earlier ones for every variant with the same text.

**Files:**
- `audit_report.md`: every table, with Wilson 95% intervals.
- `audit_summary.json`: the headline numbers used by `s11_results.py`.
- `human_labels.jsonl`: one label per variant.
- `verdicts/A1*.jsonl`: raw verdicts.

**Definitions:**
- **Strict**: same information need ("yes") and not ungrammatical.
- **Lenient**: also accepts "partly", i.e. a recognisably same request with a small shift in scope, register or term.

## Findings

1. **Annotations are consistent.** On 25 hidden repeats the strict verdict agreed 96% of the time and the exact meaning label 88%. Agreement between two annotators is not measured.
2. **The automatic rule (GPT judge + SBERT ≥ 0.8) is reliable leniently, less so strictly.** Census of the 509 accepted worst-case variants:

   | | Strict-valid | Lenient-valid |
   |---|---|---|
   | All accepted worst-case variants | 81.7% [78.1, 84.8] | 99.0% [97.7, 99.6] |
   | Dictionary attacks | 77.9% | 98.6% |
   | MLM attacks | 90.8% | 100% |

   Weighted batch-1 estimate for the random pool: 82% strict, 96.5% lenient. Almost every disagreement is a "partly" (register, terminology, scope), not a "no".
3. **Damaging attacks are only slightly less valid** (census, by outcome on the attacked retriever):

   | Outcome | Strict-valid |
   |---|---|
   | Source pushed out of the top 5 (n = 31) | 80.6% [63.7, 90.8] |
   | Rank worsened (n = 104) | 74.0% [64.9, 81.5] |
   | Unchanged (n = 374) | 84.0% [79.9, 87.3] |

   The contrast from batch 1 alone (53% vs 83%) was largely sampling noise in small cells.
4. **The automatic judges agree only moderately with the human.** On the stratified batch 1:

   | Judge | κ (strict) | κ (lenient) |
   |---|---|---|
   | GPT-4.1-mini | 0.45 | 0.48 |
   | MamayLM-12B | 0.30 | 0.54 |

   The GPT judge is too conservative on curated-term and LLM substitutions: 78% of the ones it rejected were strictly valid.
5. **Substitution source matters a lot.** Strict validity in the random pool:

   | Source | Strict-valid |
   |---|---|
   | Curated terms | 89% |
   | LLM | 86% |
   | MLM | 43% |
   | Dictionary | 29% |

   Main issues ticked: technical term changed (59), wrong word sense (47), broader/narrower (40), register (30).

## Batch 3: re-check of successful attacks

- 45 of the 165 texts had a first-pass label; the strict verdict agreed on 76% (7 downgraded, 4 upgraded). Hidden repeats over all batches: 94% (31 pairs).
- Strict validity of the re-checked texts:

  | Texts found by | Strict-valid |
  |---|---|
  | S15 only (n = 130) | 70.8% |
  | both searches (n = 23) | 87.0% |
  | S7 only (n = 5) | 80% |
  | earlier conflicts (n = 7) | 14% |

  Lenient validity is 100% except for the conflicts.
- Identical texts labelled more than once count as valid only if every label agrees.

## How the labels enter the robustness results

`s14_worstcase_verified.py` matches labels to variants by query and text. **Verified** requires the automatic rule and the human to accept the variant. Every variant behind a successful attack is labelled in both searches, so the verified success counts are exact.

| Retriever | S7 verified | S15 auto | S15 verified (strict) | S15 verified (natural) |
|---|---|---|---|---|
| BM25 | 11 | 32 | 30 | 27 |
| e5-base | 6 | 27 | 26 | 26 |
| BGE-M3 | 3 | 24 | 18 | 16 |
| hybrid | 4 | 17 | 13 | 12 |

Cells are queries whose source document leaves the attacked retriever's top 5, out of all 586 queries. The earlier audit-based estimates for S15 (29.2 / 25.5 / 21.5 / 14.3) were within 1–3.5 queries of these.

## Relevance spot-check (`relevance_report.md`)

36 blind items (`s17_relevance_check.py`): the rank-1 document answers the query for 0 of 24 attack-induced failures (3 partly) and 1 of 12 clean hybrid failures (5 partly, mostly near-identical report series such as the Armageddon CERT-UA#4378/#4434 titles).
