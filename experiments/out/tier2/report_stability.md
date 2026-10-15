# Stability across repetitions (tier 7)

## (a) 20 independent perturbation samples (dict/mlm/tech, local judge at calibrated threshold)

ΔR@5 in pp on valid variants; each sample re-draws substitutions for the same queries.

| level|subset | model | samples | ΔR@5 mean ± sd | min | max |
|---|---|---|---|---|---|
| chunk|section | bge_m3 | 20 | 1.0 ± 0.2 | 0.5 | 1.5 |
| chunk|section | bm25 | 20 | 2.9 ± 0.3 | 2.3 | 3.5 |
| chunk|section | e5_base | 20 | 1.1 ± 0.4 | 0.3 | 1.9 |
| chunk|section | e5_large | 20 | 0.8 ± 0.3 | 0.4 | 1.5 |
| chunk|section | hybrid | 20 | -0.0 ± 0.2 | -0.3 | 0.2 |
| doc_full|section | bge_m3 | 20 | 0.3 ± 0.3 | -0.3 | 0.8 |
| doc_full|section | bm25 | 20 | 2.4 ± 0.4 | 1.8 | 3.5 |
| doc_full|section | e5_base | 20 | 1.5 ± 0.3 | 1.1 | 2.0 |
| doc_full|section | e5_large | 20 | 0.2 ± 0.3 | -0.4 | 0.7 |
| doc_full|section | hybrid | 20 | 0.4 ± 0.4 | -0.2 | 1.1 |
| doc_full|title | bge_m3 | 20 | 0.3 ± 0.6 | -0.7 | 1.3 |
| doc_full|title | bm25 | 20 | 0.9 ± 0.4 | 0.2 | 1.5 |
| doc_full|title | e5_base | 20 | 0.7 ± 0.6 | -0.4 | 2.2 |
| doc_full|title | e5_large | 20 | -0.3 ± 0.5 | -1.3 | 0.5 |
| doc_full|title | hybrid | 20 | -0.4 ± 0.4 | -1.3 | 0.5 |
| doc_max|section | bge_m3 | 20 | 0.7 ± 0.2 | 0.3 | 1.1 |
| doc_max|section | bm25 | 20 | 2.5 ± 0.3 | 2.1 | 3.1 |
| doc_max|section | e5_base | 20 | 1.8 ± 0.3 | 1.3 | 2.5 |
| doc_max|section | e5_large | 20 | 0.2 ± 0.3 | -0.2 | 0.9 |
| doc_max|section | hybrid | 20 | 0.1 ± 0.2 | -0.3 | 0.4 |
| doc_max|title | bge_m3 | 20 | -0.1 ± 0.6 | -1.1 | 1.3 |
| doc_max|title | bm25 | 20 | 0.4 ± 0.6 | -1.1 | 1.2 |
| doc_max|title | e5_base | 20 | 0.7 ± 0.6 | -0.4 | 1.8 |
| doc_max|title | e5_large | 20 | 0.7 ± 0.4 | -0.2 | 1.3 |
| doc_max|title | hybrid | 20 | -0.8 ± 0.5 | -2.0 | 0.0 |

## (b) Article-level 5-fold fine-tuning across seeds

Clean R@5 on original queries and paired ΔR@5 on valid (API-judged) variants.

| model | condition | level|subset | seeds | frozen R@5 | R@5 clean mean ± sd | ΔR@5 mean ± sd |
|---|---|---|---|---|---|---|
| bge_m3 | ft_aug | chunk|section | 21 | 93.7 | 95.0 ± 0.7 | 0.3 ± 0.2 |
| bge_m3 | ft_aug | doc_max|section | 21 | 93.7 | 96.0 ± 0.6 | 0.3 ± 0.3 |
| bge_m3 | ft_aug | doc_max|title | 21 | 80.8 | 90.6 ± 1.0 | 0.1 ± 0.5 |
| bge_m3 | ft_orig | chunk|section | 21 | 93.7 | 95.2 ± 0.6 | 0.6 ± 0.4 |
| bge_m3 | ft_orig | doc_max|section | 21 | 93.7 | 96.2 ± 0.5 | 0.5 ± 0.3 |
| bge_m3 | ft_orig | doc_max|title | 21 | 80.8 | 89.3 ± 1.1 | -0.2 ± 0.8 |
| e5_base | ft_aug | chunk|section | 21 | 88.0 | 88.6 ± 0.9 | 0.7 ± 0.3 |
| e5_base | ft_aug | doc_max|section | 21 | 90.7 | 90.3 ± 0.6 | 0.5 ± 0.4 |
| e5_base | ft_aug | doc_max|title | 21 | 72.1 | 87.5 ± 1.1 | 0.4 ± 0.6 |
| e5_base | ft_orig | chunk|section | 21 | 88.0 | 74.1 ± 1.0 | 0.6 ± 0.6 |
| e5_base | ft_orig | doc_max|section | 21 | 90.7 | 81.1 ± 1.0 | 0.9 ± 0.5 |
| e5_base | ft_orig | doc_max|title | 21 | 72.1 | 78.9 ± 0.7 | -0.1 ± 0.6 |
