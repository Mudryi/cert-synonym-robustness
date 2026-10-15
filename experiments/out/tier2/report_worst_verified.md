# Worst-case robustness on a fixed query denominator, with human validity

Every one of the 586 original queries counts for every attacked retriever (doc_max). The attack search (S7) is greedy and validity-unaware: it picks the margin-minimising substitution first and the validity judge runs afterwards (select → validate), so a query without a valid variant may still have one outside the greedy path; all numbers are lower bounds on the strongest valid attack. The attacker uses its strongest valid variant (worst rank on the attacked retriever); a query without one keeps its original rank.

Validity levels: **auto** = GPT judge + SBERT ≥ 0.8; **verified** = only variants a human labelled valid (lower bound; 557 of 7791 variants are labelled); **estimated** = human labels where present, unlabelled auto-accepted variants drawn valid from Beta posteriors of the audit strata (auto label × source × outcome on the attacker, shrunk toward the pooled rate of the automatic label with 5 pseudo-labels), 1000 draws combined with an article bootstrap; **estimated incl. rejected** also draws the auto-rejected variants (judge false rejections), a sensitivity analysis resting on few labels of rejected variants. The audit over-represents rank-changing variants, so all estimates are sensitivity analyses rather than population corrections.

## Outcome of the attack per original query (white-box, R@5)

Coverage = originally correct queries with at least one valid attack variant.

| attacked retriever | validity | originally failing | no attack chain | chain, no valid variant | valid, source stays in top 5 | successful attack | coverage % |
|---|---|---|---|---|---|---|---|
| bm25 | auto | 48 | 12 | 363 | 152 | 11 | 30.3 |
| bm25 | verified (strict) | 48 | 12 | 386 | 129 | 11 | 26.0 |
| bm25 | verified (lenient) | 48 | 12 | 363 | 152 | 11 | 30.3 |
| bm25 | verified (natural) | 48 | 12 | 393 | 123 | 10 | 24.7 |
| bm25 | verified incl. rejected (strict) | 48 | 12 | 383 | 131 | 12 | 26.6 |
| e5_base | auto | 95 | 3 | 444 | 37 | 7 | 9.0 |
| e5_base | verified (strict) | 95 | 3 | 455 | 27 | 6 | 6.7 |
| e5_base | verified (lenient) | 95 | 3 | 445 | 36 | 7 | 8.8 |
| e5_base | verified (natural) | 95 | 3 | 455 | 27 | 6 | 6.7 |
| e5_base | verified incl. rejected (strict) | 95 | 3 | 454 | 28 | 6 | 6.9 |
| bge_m3 | auto | 65 | 3 | 463 | 51 | 4 | 10.6 |
| bge_m3 | verified (strict) | 65 | 3 | 478 | 37 | 3 | 7.7 |
| bge_m3 | verified (lenient) | 65 | 3 | 464 | 50 | 4 | 10.4 |
| bge_m3 | verified (natural) | 65 | 3 | 480 | 35 | 3 | 7.3 |
| bge_m3 | verified incl. rejected (strict) | 65 | 3 | 478 | 37 | 3 | 7.7 |
| hybrid | auto | 48 | 51 | 369 | 114 | 4 | 21.9 |
| hybrid | verified (strict) | 48 | 51 | 384 | 99 | 4 | 19.1 |
| hybrid | verified (lenient) | 48 | 51 | 369 | 114 | 4 | 21.9 |
| hybrid | verified (natural) | 48 | 51 | 388 | 96 | 3 | 18.4 |
| hybrid | verified incl. rejected (strict) | 48 | 51 | 384 | 99 | 4 | 19.1 |


## White-box attack: retrieval quality before and after (doc_max, all queries)

ΔR@5 = share of all queries whose source leaves the top 5 (95% CI: article bootstrap; for estimated levels the interval also covers the validity uncertainty). Successful queries: count, with 95% interval for estimated levels (article bootstrap and unlabelled validity combined); the last column isolates the validity uncertainty (articles fixed). **verified** requires the automatic rule and the human to accept the variant; **natural** additionally requires fluency 'natural'; **incl. rejected** also admits human-valid variants the automatic rule rejected.

| attacked | validity | R@1 orig → att | R@5 orig → att | MRR orig → att | ΔR@5 (pp) | successful queries | success | orig. correct % | succ. 95% (validity only) |
|---|---|---|---|---|---|---|---|---|
| bm25 | auto | 75.3 → 69.5 | 91.8 → 89.9 | 0.824 → 0.787 | 1.9 [0.8, 3.1] | 11 | 2.0 |  |
| bm25 | estimated (lenient) | 75.2 → 69.4 | 91.8 → 89.9 | 0.824 → 0.787 | 1.9 [0.9, 3.1] | 11.0 [5, 18] | 2.1 | [11, 11] |
| bm25 | estimated (strict) | 75.2 → 70.6 | 91.8 → 89.9 | 0.823 → 0.794 | 1.9 [0.9, 3.0] | 11.0 [5, 18] | 2.0 | [11, 11] |
| bm25 | verified (lenient) | 75.3 → 69.5 | 91.8 → 89.9 | 0.824 → 0.787 | 1.9 [0.9, 3.0] | 11 | 2.0 |  |
| bm25 | verified (strict) | 75.3 → 70.6 | 91.8 → 89.9 | 0.824 → 0.794 | 1.9 [0.9, 3.1] | 11 | 2.0 |  |
| bm25 | verified (natural) | 75.3 → 71.0 | 91.8 → 90.1 | 0.824 → 0.796 | 1.7 [0.7, 2.8] | 10 | 1.9 |  |
| bm25 | estimated incl. rejected (strict) | 75.2 → 65.9 | 91.8 → 87.1 | 0.824 → 0.757 | 4.8 [2.8, 7.0] | 28.1 [17, 41] | 5.2 | [22, 35] |
| bm25 | verified incl. rejected (strict) | 75.3 → 70.6 | 91.8 → 89.8 | 0.824 → 0.794 | 2.0 [0.9, 3.4] | 12 | 2.2 |  |
| e5_base | auto | 65.4 → 64.0 | 83.8 → 82.6 | 0.736 → 0.722 | 1.2 [0.4, 2.1] | 7 | 1.4 |  |
| e5_base | estimated (lenient) | 65.3 → 64.0 | 83.8 → 82.6 | 0.736 → 0.722 | 1.2 [0.4, 2.1] | 7.0 [2, 12] | 1.4 | [7, 7] |
| e5_base | estimated (strict) | 65.3 → 64.3 | 83.8 → 82.7 | 0.736 → 0.725 | 1.0 [0.3, 1.9] | 6.0 [2, 11] | 1.2 | [6, 6] |
| e5_base | verified (lenient) | 65.4 → 64.0 | 83.8 → 82.6 | 0.736 → 0.722 | 1.2 [0.3, 2.1] | 7 | 1.4 |  |
| e5_base | verified (strict) | 65.4 → 64.3 | 83.8 → 82.8 | 0.736 → 0.725 | 1.0 [0.3, 1.9] | 6 | 1.2 |  |
| e5_base | verified (natural) | 65.4 → 64.3 | 83.8 → 82.8 | 0.736 → 0.725 | 1.0 [0.3, 1.9] | 6 | 1.2 |  |
| e5_base | estimated incl. rejected (strict) | 65.3 → 57.0 | 83.8 → 77.1 | 0.736 → 0.664 | 6.8 [4.3, 9.6] | 39.7 [25, 56] | 8.1 | [30, 49] |
| e5_base | verified incl. rejected (strict) | 65.4 → 64.2 | 83.8 → 82.8 | 0.736 → 0.724 | 1.0 [0.3, 1.9] | 6 | 1.2 |  |
| bge_m3 | auto | 69.8 → 68.8 | 88.9 → 88.2 | 0.781 → 0.774 | 0.7 [0.2, 1.4] | 4 | 0.8 |  |
| bge_m3 | estimated (lenient) | 69.8 → 68.8 | 88.9 → 88.2 | 0.781 → 0.774 | 0.7 [0.2, 1.4] | 4.0 [1, 8] | 0.8 | [4, 4] |
| bge_m3 | estimated (strict) | 69.8 → 68.9 | 88.9 → 88.4 | 0.781 → 0.776 | 0.5 [0.0, 1.2] | 3.0 [0, 7] | 0.6 | [3, 3] |
| bge_m3 | verified (lenient) | 69.8 → 68.8 | 88.9 → 88.2 | 0.781 → 0.774 | 0.7 [0.2, 1.5] | 4 | 0.8 |  |
| bge_m3 | verified (strict) | 69.8 → 68.9 | 88.9 → 88.4 | 0.781 → 0.776 | 0.5 [0.0, 1.2] | 3 | 0.6 |  |
| bge_m3 | verified (natural) | 69.8 → 68.9 | 88.9 → 88.4 | 0.781 → 0.776 | 0.5 [0.0, 1.1] | 3 | 0.6 |  |
| bge_m3 | estimated incl. rejected (strict) | 69.8 → 63.8 | 88.9 → 84.5 | 0.781 → 0.733 | 4.5 [2.5, 6.7] | 26.2 [15, 40] | 5.0 | [18, 34] |
| bge_m3 | verified incl. rejected (strict) | 69.8 → 68.9 | 88.9 → 88.4 | 0.781 → 0.776 | 0.5 [0.0, 1.2] | 3 | 0.6 |  |
| hybrid | auto | 75.9 → 72.9 | 91.8 → 91.1 | 0.834 → 0.814 | 0.7 [0.2, 1.4] | 4 | 0.7 |  |
| hybrid | estimated (lenient) | 75.9 → 72.8 | 91.8 → 91.1 | 0.834 → 0.815 | 0.7 [0.2, 1.4] | 4.0 [1, 8] | 0.7 | [4, 4] |
| hybrid | estimated (strict) | 75.9 → 73.9 | 91.8 → 91.1 | 0.834 → 0.820 | 0.7 [0.2, 1.4] | 4.0 [1, 8] | 0.7 | [4, 4] |
| hybrid | verified (lenient) | 75.9 → 72.9 | 91.8 → 91.1 | 0.834 → 0.815 | 0.7 [0.2, 1.4] | 4 | 0.7 |  |
| hybrid | verified (strict) | 75.9 → 73.9 | 91.8 → 91.1 | 0.834 → 0.820 | 0.7 [0.0, 1.4] | 4 | 0.7 |  |
| hybrid | verified (natural) | 75.9 → 73.9 | 91.8 → 91.3 | 0.834 → 0.821 | 0.5 [0.0, 1.2] | 3 | 0.6 |  |
| hybrid | estimated incl. rejected (strict) | 75.9 → 69.3 | 91.8 → 88.2 | 0.834 → 0.784 | 3.6 [1.8, 5.7] | 21.2 [11, 33] | 3.9 | [15, 28] |
| hybrid | verified incl. rejected (strict) | 75.9 → 73.9 | 91.8 → 91.1 | 0.834 → 0.820 | 0.7 [0.2, 1.4] | 4 | 0.7 |  |


## White-box attack by query type

| attacked | query type | validity | n | R@5 orig | R@5 att | successful queries |
|---|---|---|---|---|---|---|
| bm25 | title | auto | 219 | 84.9 | 84.0 | 2 |
| bm25 | title | verified (strict) | 219 | 84.9 | 84.0 | 2 |
| bm25 | section | auto | 367 | 95.9 | 93.5 | 9 |
| bm25 | section | verified (strict) | 367 | 95.9 | 93.5 | 9 |
| e5_base | title | auto | 219 | 72.1 | 72.1 | 0 |
| e5_base | title | verified (strict) | 219 | 72.1 | 72.1 | 0 |
| e5_base | section | auto | 367 | 90.7 | 88.8 | 7 |
| e5_base | section | verified (strict) | 367 | 90.7 | 89.1 | 6 |
| bge_m3 | title | auto | 219 | 80.8 | 80.8 | 0 |
| bge_m3 | title | verified (strict) | 219 | 80.8 | 80.8 | 0 |
| bge_m3 | section | auto | 367 | 93.7 | 92.6 | 4 |
| bge_m3 | section | verified (strict) | 367 | 93.7 | 92.9 | 3 |
| hybrid | title | auto | 219 | 85.8 | 85.8 | 0 |
| hybrid | title | verified (strict) | 219 | 85.8 | 85.8 | 0 |
| hybrid | section | auto | 367 | 95.4 | 94.3 | 4 |
| hybrid | section | verified (strict) | 367 | 95.4 | 94.3 | 4 |


## Transfer across retrievers — auto

Cells: ΔR@5 in pp over all queries (number of successful queries). The attack variant is chosen on the attacked retriever and evaluated unchanged on the others.

| attacked ↓ / evaluated → | bm25 | e5_base | bge_m3 | hybrid | e5_large | m3_all |
|---|---|---|---|---|---|---|
| bm25 | 1.9 (11) | 0.3 (2) | 0.9 (5) | 0.7 (4) | 0.3 (2) | 0.0 (0) |
| e5_base | 0.3 (2) | 1.2 (7) | 0.7 (4) | 0.2 (1) | 0.3 (2) | 0.3 (2) |
| bge_m3 | 0.3 (2) | 1.2 (7) | 0.7 (4) | 0.2 (1) | 0.2 (1) | 0.2 (1) |
| hybrid | 0.9 (5) | 0.3 (2) | 1.0 (6) | 0.7 (4) | 0.7 (4) | 0.5 (3) |


## Transfer across retrievers — estimated (strict)

Cells: ΔR@5 in pp over all queries (number of successful queries). The attack variant is chosen on the attacked retriever and evaluated unchanged on the others.

| attacked ↓ / evaluated → | bm25 | e5_base | bge_m3 | hybrid | e5_large | m3_all |
|---|---|---|---|---|---|---|
| bm25 | 1.9 (11) | 0.3 (2) | 0.7 (4) | 0.7 (4) | 0.2 (1) | 0.0 (0) |
| e5_base | 0.3 (2) | 1.0 (6) | 0.7 (4) | 0.2 (1) | 0.2 (1) | 0.3 (2) |
| bge_m3 | 0.3 (2) | 1.0 (6) | 0.5 (3) | 0.2 (1) | 0.2 (1) | 0.2 (1) |
| hybrid | 0.9 (5) | 0.3 (2) | 1.0 (6) | 0.7 (4) | 0.7 (4) | 0.5 (3) |


## Fine-tuned models under worst-case attacks (5-fold by article)

Attacks are those found on the frozen base model (transfer to the fine-tuned model, not a re-attack). Mean ± sd over seeds; the seed also sets the article-to-fold split.

| model | seeds | attack from | validity | R@5 orig | R@5 att | ΔR@5 (pp) | successful queries |
|---|---|---|---|---|---|---|---|
| ft_bge_m3_ft_aug | 5 | frozen bge_m3 | auto | 94.0 ± 0.4 | 93.8 ± 0.4 | 0.2 ± 0.2 | 1.4 |
| ft_bge_m3_ft_aug | 5 | frozen bge_m3 | verified (strict) | 94.0 ± 0.4 | 93.9 ± 0.3 | 0.2 ± 0.2 | 1.0 |
| bge_m3 (frozen) | 1 | frozen bge_m3 | auto | 88.9 | 88.2 | 0.7 | 4 |
| bge_m3 (frozen) | 1 | frozen bge_m3 | verified (strict) | 88.9 | 88.4 | 0.5 | 3 |
| ft_bge_m3_ft_orig | 5 | frozen bge_m3 | auto | 93.5 ± 0.5 | 93.2 ± 0.4 | 0.3 ± 0.2 | 2.0 |
| ft_bge_m3_ft_orig | 5 | frozen bge_m3 | verified (strict) | 93.5 ± 0.5 | 93.3 ± 0.4 | 0.3 ± 0.1 | 1.6 |
| ft_e5_base_ft_aug | 5 | frozen e5_base | auto | 89.2 ± 0.7 | 88.8 ± 0.5 | 0.5 ± 0.2 | 2.8 |
| ft_e5_base_ft_aug | 5 | frozen e5_base | verified (strict) | 89.2 ± 0.7 | 88.9 ± 0.6 | 0.3 ± 0.1 | 1.8 |
| e5_base (frozen) | 1 | frozen e5_base | auto | 83.8 | 82.6 | 1.2 | 7 |
| e5_base (frozen) | 1 | frozen e5_base | verified (strict) | 83.8 | 82.8 | 1.0 | 6 |
| ft_e5_base_ft_orig | 5 | frozen e5_base | auto | 80.1 ± 0.7 | 79.9 ± 0.8 | 0.2 ± 0.1 | 1.2 |
| ft_e5_base_ft_orig | 5 | frozen e5_base | verified (strict) | 80.1 ± 0.7 | 80.0 ± 0.7 | 0.1 ± 0.1 | 0.6 |


## Random vs. worst-case on the same denominator (auto validity, doc_max)

Random: one auto-valid dictionary or MLM substitution drawn per query (mean of 200 draws); queries without one keep their original rank. Worst-case: the attacker's strongest valid variant.

| retriever | queries with a valid random variant | random: ΔR@5 (pp) | random: successful queries | worst-case (auto): ΔR@5 (pp) | worst-case (auto): successful queries |
|---|---|---|---|---|---|
| bm25 | 422 | 1.1 | 6.3 | 1.9 | 11 |
| e5_base | 422 | 1.3 | 7.8 | 1.2 | 7 |
| bge_m3 | 422 | 1.0 | 6.0 | 0.7 | 4 |
| hybrid | 422 | 0.8 | 4.9 | 0.7 | 4 |


## Attack search budget

Per (query, source) chain: ≤ 2 substitutions, ≤ 8 candidates per position from the dictionary or MLM pool, SBERT cos ≥ 0.85 to the original before scoring; one retriever call scores one candidate text; every variant on the chain is judged once afterwards.

| attacked retriever | chains searched | mean admissible candidates | retriever calls / chain (mean) | retriever calls / chain (max) | judged variants / chain |
|---|---|---|---|---|---|
| bm25 | 1147 | 25.5 | 45.0 | 142 | 1.61 |
| e5_base | 1147 | 25.5 | 45.5 | 143 | 1.88 |
| bge_m3 | 1147 | 25.5 | 45.5 | 146 | 1.87 |
| hybrid | 1147 | 25.5 | 43.7 | 139 | 1.43 |


## Human-verified successful attacks (white-box)

- **bm25** · title · rank 4 → 7: «Цільові кібератаки UAC-0185 у відношенні Сил оборони та підприємств ОПК України (CERT-UA#12414)» → «Цільові кібератаки UAC-0185 у відношенні Сил оборони та виробництв ОПК України (CERT-UA#12414)» (підприємств → виробництв) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · title · rank 2 → 19: «Кібератака на сайти державних органів» → «Кібератака на сайти офіційних установ» (державних → офіційних; органів → установ) [human: yes/natural (b3)]
- **bm25** · section · rank 2 → 9: «Які види діяльності здійснює угрупування UAC-0050 за даними CERT-UA?» → «Які види робіт здійснює організація UAC-0050 за даними CERT-UA?» (діяльності → робіт; угрупування → організація) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 5 → 7: «Які методи поширення SMOKELOADER використовує угруповання UAC-0006 у травні 2024?» → «Які методи поширення SMOKELOADER застосовує угруповання UAC-0006 у травні 2024?» (використовує → застосовує) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 5 → 8: «Як шкідлива програма OUTSTEEL передає дані на сервер управління?» → «Як шкідлива платформа OUTSTEEL передає дані на сервер керування?» (програма → платформа; управління → керування) [human: yes/natural (b3)]
- **bm25** · section · rank 5 → 7: «Які методи поширення використовує шкідлива програма MerlinAgent в Україні?» → «Які засоби поширення використовує шкідлива програма MerlinAgent в Україні?» (методи → засоби) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 4 → 7: «Які ознаки та наслідки атаки з використанням шкідливої програми JesterStealer?» → «Які ознаки та результати атаки з використанням шкідливої програми JesterStealer?» (наслідки → результати) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 3 → 7: «Як безпечно перевіряти електронні листи з посиланнями від державних установ?» → «Як безпечно перевіряти електронні листи з посиланнями від державних організацій?» (установ → організацій) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 2 → 37: «Які повсякденні заходи безпеки рекомендують для захисту від кібератак?» → «Які повсякденні засоби захисту рекомендують для захисту від кібератак?» (заходи → засоби; безпеки → захисту) [human: yes/natural (b3)]
- **bm25** · section · rank 2 → 7: «Як діяти при підозрі на кібератаку з використанням легітимних облікових записів?» → «Як діяти при підозрінню на кібератаку з використанням легітимних облікових записів?» (підозрі → підозрінню) [human: yes/awkward (b3), yes/awkward (b3), yes/awkward (b3), yes/awkward (b3), yes/awkward (b3)]
- **bm25** · section · rank 3 → 10: «Як захистити ЕОМ від запуску шкідливих утиліт через Microsoft Office?» → «Як захистити ЕОМ від запуску шкідливих програм через Microsoft Office?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 3 → 25: «Які методи використовувалися у кібератаках з ідентифікатором UAC-0219 у 2024 році?» → «Які стратегії використовувалися у кібератаках з ідентифікатором UAC-0219 у 2024 році?» (методи → стратегії) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 3 → 12: «Як виявити шкідливий VBS-скрипт для рекурсивного пошуку і викрадення файлів?» → «Як помітити шкідливий VBS-скрипт для рекурсивного пошуку і викрадення файлів?» (виявити → помітити) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 3 → 18: «Які заходи вжито CERT-UA для аналізу шахрайських доменів з 2021 року?» → «Які засоби вжито CERT-UA для дослідження шахрайських доменів з 2021 року?» (заходи → засоби; аналізу → дослідження) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 5 → 8: «Які методи атаки використовує угруповання UAC-0006 із SmokeLoader у липні 2023?» → «Які технології атаки використовує угруповання UAC-0006 із SmokeLoader у липні 2023?» (методи → технології) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 1 → 17: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 4 → 9: «Які методи використання PowerShell для інфікування комп'ютерів описані у звіті CERT-UA 2026?» → «Які методи використання PowerShell для інфікування комп'ютерів описані у рапорті CERT-UA 2026?» (звіті → рапорті) [human: yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 2 → 11: «Які вектори компрометації використовує угруповання UAC-0063 у червні 2024 року?» → «Які технології компрометації використовує угруповання UAC-0063 у червні 2024 року?» (вектори → технології) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 4 → 14: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 4 → 9: «Які методи використано у кібератаці на Сили оборони України через месенджер Signal?» → «Які методи використовують у кібератаці на Сили оборони України через додаток Signal?» (використано → використовують; месенджер → додаток) [human: yes/natural (b3)]
- **hybrid** · section · rank 5 → 8: «Як правильно налаштувати двофакторну автентифікацію для захисту облікових записів?» → «Як правильно налаштувати двофакторну автентифікацію для оборони облікових записів?» (захисту → оборони) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 1 → 8: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 3 → 16: «Які заходи проводить CERT-UA для протидії кіберзагрозам в Україні?» → «Які дії проводить CERT-UA для запобігання кіберзагрозам в Україні?» (заходи → дії; протидії → запобігання) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 2 → 7: «Як діяти при підозрі на кібератаку з використанням легітимних облікових записів?» → «Як діяти при підозрінню на кібератаку з використанням законних облікових записів?» (підозрі → підозрінню; легітимних → законних) [human: yes/awkward (b3), yes/awkward (b3)]
