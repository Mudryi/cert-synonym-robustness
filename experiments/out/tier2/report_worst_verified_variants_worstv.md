# Worst-case robustness on a fixed query denominator, with human validity

Every one of the 586 original queries counts for every attacked retriever (doc_max). The attack search (S15) is validity-aware: every admissible single substitution is judged first and the greedy search runs over judge-valid candidates only (validate → select); validity here is still the automatic judge's, so human levels below re-check it. The attacker uses its strongest valid variant (worst rank on the attacked retriever); a query without one keeps its original rank.

Validity levels: **auto** = GPT judge + SBERT ≥ 0.8; **verified** = only variants a human labelled valid (lower bound; 971 of 4168 variants are labelled); **estimated** = human labels where present, unlabelled auto-accepted variants drawn valid from Beta posteriors of the audit strata (auto label × source × outcome on the attacker, shrunk toward the pooled rate of the automatic label with 5 pseudo-labels), 1000 draws combined with an article bootstrap; **estimated incl. rejected** also draws the auto-rejected variants (judge false rejections), a sensitivity analysis resting on few labels of rejected variants. The audit over-represents rank-changing variants, so all estimates are sensitivity analyses rather than population corrections.

## Outcome of the attack per original query (white-box, R@5)

Coverage = originally correct queries with at least one valid attack variant.

| attacked retriever | validity | originally failing | no attack chain | chain, no valid variant | valid, source stays in top 5 | successful attack | coverage % |
|---|---|---|---|---|---|---|---|
| bm25 | auto | 48 | 117 | 0 | 389 | 32 | 78.3 |
| bm25 | verified (strict) | 48 | 117 | 246 | 145 | 30 | 32.5 |
| bm25 | verified (lenient) | 48 | 117 | 219 | 170 | 32 | 37.5 |
| bm25 | verified (natural) | 48 | 117 | 256 | 138 | 27 | 30.7 |
| bm25 | verified incl. rejected (strict) | 48 | 117 | 246 | 145 | 30 | 32.5 |
| e5_base | auto | 95 | 57 | 0 | 407 | 27 | 88.4 |
| e5_base | verified (strict) | 95 | 57 | 318 | 90 | 26 | 23.6 |
| e5_base | verified (lenient) | 95 | 57 | 298 | 109 | 27 | 27.7 |
| e5_base | verified (natural) | 95 | 57 | 323 | 85 | 26 | 22.6 |
| e5_base | verified incl. rejected (strict) | 95 | 57 | 318 | 90 | 26 | 23.6 |
| bge_m3 | auto | 65 | 58 | 0 | 439 | 24 | 88.9 |
| bge_m3 | verified (strict) | 65 | 58 | 344 | 101 | 18 | 22.8 |
| bge_m3 | verified (lenient) | 65 | 58 | 317 | 122 | 24 | 28.0 |
| bge_m3 | verified (natural) | 65 | 58 | 349 | 98 | 16 | 21.9 |
| bge_m3 | verified incl. rejected (strict) | 65 | 58 | 344 | 101 | 18 | 22.8 |
| hybrid | auto | 48 | 184 | 0 | 337 | 17 | 65.8 |
| hybrid | verified (strict) | 48 | 184 | 200 | 141 | 13 | 28.6 |
| hybrid | verified (lenient) | 48 | 184 | 171 | 166 | 17 | 34.0 |
| hybrid | verified (natural) | 48 | 184 | 207 | 135 | 12 | 27.3 |
| hybrid | verified incl. rejected (strict) | 48 | 184 | 200 | 141 | 13 | 28.6 |


## White-box attack: retrieval quality before and after (doc_max, all queries)

ΔR@5 = share of all queries whose source leaves the top 5 (95% CI: article bootstrap; for estimated levels the interval also covers the validity uncertainty). Successful queries: count, with 95% interval for estimated levels (article bootstrap and unlabelled validity combined); the last column isolates the validity uncertainty (articles fixed). **verified** requires the automatic rule and the human to accept the variant; **natural** additionally requires fluency 'natural'; **incl. rejected** also admits human-valid variants the automatic rule rejected.

| attacked | validity | R@1 orig → att | R@5 orig → att | MRR orig → att | ΔR@5 (pp) | successful queries | success | orig. correct % | succ. 95% (validity only) |
|---|---|---|---|---|---|---|---|---|
| bm25 | auto | 75.3 → 59.7 | 91.8 → 86.5 | 0.824 → 0.717 | 5.5 [3.8, 7.4] | 32 | 5.9 |  |
| bm25 | estimated (lenient) | 75.3 → 59.7 | 91.8 → 86.5 | 0.824 → 0.717 | 5.5 [3.7, 7.4] | 32.0 [22, 43] | 5.9 | [32, 32] |
| bm25 | estimated (strict) | 75.3 → 61.7 | 91.8 → 86.8 | 0.824 → 0.731 | 5.1 [3.4, 7.0] | 30.0 [20, 41] | 5.6 | [30, 30] |
| bm25 | verified (lenient) | 75.3 → 68.4 | 91.8 → 86.3 | 0.824 → 0.773 | 5.5 [3.8, 7.3] | 32 | 5.9 |  |
| bm25 | verified (strict) | 75.3 → 69.8 | 91.8 → 86.7 | 0.824 → 0.782 | 5.1 [3.4, 7.1] | 30 | 5.6 |  |
| bm25 | verified (natural) | 75.3 → 70.1 | 91.8 → 87.2 | 0.824 → 0.785 | 4.6 [3.0, 6.5] | 27 | 5.0 |  |
| bm25 | estimated incl. rejected (strict) | 75.3 → 61.7 | 91.8 → 86.9 | 0.824 → 0.730 | 5.1 [3.4, 7.0] | 30.0 [20, 41] | 5.6 | [30, 30] |
| bm25 | verified incl. rejected (strict) | 75.3 → 69.8 | 91.8 → 86.7 | 0.824 → 0.782 | 5.1 [3.5, 6.9] | 30 | 5.6 |  |
| e5_base | auto | 65.4 → 55.1 | 83.8 → 79.4 | 0.736 → 0.659 | 4.6 [3.0, 6.4] | 27 | 5.5 |  |
| e5_base | estimated (lenient) | 65.3 → 55.1 | 83.8 → 79.3 | 0.736 → 0.659 | 4.6 [3.0, 6.4] | 27.0 [17, 38] | 5.5 | [27, 27] |
| e5_base | estimated (strict) | 65.4 → 56.0 | 83.8 → 79.5 | 0.736 → 0.666 | 4.4 [2.8, 6.2] | 26.0 [16, 36] | 5.3 | [26, 26] |
| e5_base | verified (lenient) | 65.4 → 63.0 | 83.8 → 79.4 | 0.736 → 0.711 | 4.6 [2.9, 6.4] | 27 | 5.5 |  |
| e5_base | verified (strict) | 65.4 → 63.7 | 83.8 → 79.5 | 0.736 → 0.716 | 4.4 [2.8, 6.3] | 26 | 5.3 |  |
| e5_base | verified (natural) | 65.4 → 63.8 | 83.8 → 79.5 | 0.736 → 0.717 | 4.4 [2.7, 6.1] | 26 | 5.3 |  |
| e5_base | estimated incl. rejected (strict) | 65.3 → 55.9 | 83.8 → 79.5 | 0.736 → 0.666 | 4.4 [2.8, 6.2] | 26.0 [16, 36] | 5.3 | [26, 26] |
| e5_base | verified incl. rejected (strict) | 65.4 → 63.7 | 83.8 → 79.5 | 0.736 → 0.716 | 4.4 [2.8, 6.4] | 26 | 5.3 |  |
| bge_m3 | auto | 69.8 → 65.0 | 88.9 → 85.2 | 0.781 → 0.740 | 4.1 [2.6, 5.8] | 24 | 4.6 |  |
| bge_m3 | estimated (lenient) | 69.8 → 65.0 | 88.9 → 85.1 | 0.781 → 0.740 | 4.1 [2.6, 5.7] | 24.0 [15, 34] | 4.6 | [24, 24] |
| bge_m3 | estimated (strict) | 69.8 → 65.8 | 88.9 → 86.2 | 0.781 → 0.747 | 3.1 [1.7, 4.5] | 18.0 [10, 26] | 3.5 | [18, 18] |
| bge_m3 | verified (lenient) | 69.8 → 67.7 | 88.9 → 85.0 | 0.781 → 0.761 | 4.1 [2.6, 5.7] | 24 | 4.6 |  |
| bge_m3 | verified (strict) | 69.8 → 68.4 | 88.9 → 86.0 | 0.781 → 0.767 | 3.1 [1.8, 4.4] | 18 | 3.5 |  |
| bge_m3 | verified (natural) | 69.8 → 68.4 | 88.9 → 86.3 | 0.781 → 0.768 | 2.7 [1.5, 4.1] | 16 | 3.1 |  |
| bge_m3 | estimated incl. rejected (strict) | 69.8 → 65.7 | 88.9 → 86.2 | 0.781 → 0.746 | 3.1 [1.8, 4.5] | 18.0 [10, 26] | 3.5 | [18, 18] |
| bge_m3 | verified incl. rejected (strict) | 69.8 → 68.4 | 88.9 → 86.0 | 0.781 → 0.767 | 3.1 [1.8, 4.5] | 18 | 3.5 |  |
| hybrid | auto | 75.9 → 65.9 | 91.8 → 88.9 | 0.834 → 0.767 | 2.9 [1.6, 4.2] | 17 | 3.2 |  |
| hybrid | estimated (lenient) | 75.9 → 65.9 | 91.8 → 88.9 | 0.834 → 0.767 | 2.9 [1.6, 4.3] | 16.9 [9, 26] | 3.1 | [17, 17] |
| hybrid | estimated (strict) | 75.9 → 67.5 | 91.8 → 89.6 | 0.834 → 0.778 | 2.2 [1.0, 3.5] | 13.0 [6, 21] | 2.4 | [13, 13] |
| hybrid | verified (lenient) | 75.9 → 70.5 | 91.8 → 88.9 | 0.834 → 0.795 | 2.9 [1.4, 4.4] | 17 | 3.2 |  |
| hybrid | verified (strict) | 75.9 → 71.7 | 91.8 → 89.6 | 0.834 → 0.804 | 2.2 [1.0, 3.6] | 13 | 2.4 |  |
| hybrid | verified (natural) | 75.9 → 72.0 | 91.8 → 89.8 | 0.834 → 0.807 | 2.0 [1.0, 3.1] | 12 | 2.2 |  |
| hybrid | estimated incl. rejected (strict) | 75.9 → 67.5 | 91.8 → 89.6 | 0.834 → 0.778 | 2.2 [1.0, 3.6] | 13.0 [6, 21] | 2.4 | [13, 13] |
| hybrid | verified incl. rejected (strict) | 75.9 → 71.7 | 91.8 → 89.6 | 0.834 → 0.804 | 2.2 [1.0, 3.6] | 13 | 2.4 |  |


## White-box attack by query type

| attacked | query type | validity | n | R@5 orig | R@5 att | successful queries |
|---|---|---|---|---|---|---|
| bm25 | title | auto | 219 | 84.9 | 82.6 | 6 |
| bm25 | title | verified (strict) | 219 | 84.9 | 82.2 | 6 |
| bm25 | section | auto | 367 | 95.9 | 88.8 | 26 |
| bm25 | section | verified (strict) | 367 | 95.9 | 89.4 | 24 |
| e5_base | title | auto | 219 | 72.1 | 67.6 | 10 |
| e5_base | title | verified (strict) | 219 | 72.1 | 68.0 | 9 |
| e5_base | section | auto | 367 | 90.7 | 86.4 | 17 |
| e5_base | section | verified (strict) | 367 | 90.7 | 86.4 | 17 |
| bge_m3 | title | auto | 219 | 80.8 | 78.1 | 7 |
| bge_m3 | title | verified (strict) | 219 | 80.8 | 79.0 | 5 |
| bge_m3 | section | auto | 367 | 93.7 | 89.4 | 17 |
| bge_m3 | section | verified (strict) | 367 | 93.7 | 90.2 | 13 |
| hybrid | title | auto | 219 | 85.8 | 83.6 | 5 |
| hybrid | title | verified (strict) | 219 | 85.8 | 84.5 | 3 |
| hybrid | section | auto | 367 | 95.4 | 92.1 | 12 |
| hybrid | section | verified (strict) | 367 | 95.4 | 92.6 | 10 |


## Transfer across retrievers — auto

Cells: ΔR@5 in pp over all queries (number of successful queries). The attack variant is chosen on the attacked retriever and evaluated unchanged on the others.

| attacked ↓ / evaluated → | bm25 | e5_base | bge_m3 | hybrid | e5_large | m3_all |
|---|---|---|---|---|---|---|
| bm25 | 5.5 (32) | 2.4 (14) | 1.7 (10) | 1.4 (8) | 1.2 (7) | 0.7 (4) |
| e5_base | 2.6 (15) | 4.6 (27) | 2.6 (15) | 1.5 (9) | 1.9 (11) | 0.7 (4) |
| bge_m3 | 2.4 (14) | 2.7 (16) | 4.1 (24) | 1.7 (10) | 1.5 (9) | 1.4 (8) |
| hybrid | 4.9 (29) | 1.9 (11) | 2.7 (16) | 2.9 (17) | 1.2 (7) | 1.2 (7) |


## Transfer across retrievers — estimated (strict)

Cells: ΔR@5 in pp over all queries (number of successful queries). The attack variant is chosen on the attacked retriever and evaluated unchanged on the others.

| attacked ↓ / evaluated → | bm25 | e5_base | bge_m3 | hybrid | e5_large | m3_all |
|---|---|---|---|---|---|---|
| bm25 | 5.1 (30) | 2.3 (14) | 1.7 (10) | 1.2 (7) | 1.3 (8) | 0.7 (4) |
| e5_base | 2.5 (15) | 4.4 (26) | 2.5 (14) | 1.3 (8) | 1.8 (11) | 0.8 (5) |
| bge_m3 | 2.2 (13) | 2.5 (15) | 3.1 (18) | 1.4 (8) | 1.4 (8) | 1.2 (7) |
| hybrid | 4.5 (27) | 1.9 (11) | 2.4 (14) | 2.2 (13) | 1.2 (7) | 1.4 (8) |


## Fine-tuned models under worst-case attacks (5-fold by article)

Attacks are those found on the frozen base model (transfer to the fine-tuned model, not a re-attack). Mean ± sd over seeds; the seed also sets the article-to-fold split.

| model | seeds | attack from | validity | R@5 orig | R@5 att | ΔR@5 (pp) | successful queries |
|---|---|---|---|---|---|---|---|
| ft_bge_m3_ft_aug | 5 | frozen bge_m3 | auto | 94.4 ± 0.2 | 93.2 ± 0.4 | 1.3 ± 0.3 | 7.6 |
| ft_bge_m3_ft_aug | 5 | frozen bge_m3 | verified (strict) | 94.4 ± 0.2 | 93.7 ± 0.4 | 0.7 ± 0.3 | 4.0 |
| bge_m3 (frozen) | 1 | frozen bge_m3 | auto | 88.9 | 85.2 | 4.1 | 24 |
| bge_m3 (frozen) | 1 | frozen bge_m3 | verified (strict) | 88.9 | 86.0 | 3.1 | 18 |
| ft_bge_m3_ft_orig | 5 | frozen bge_m3 | auto | 93.3 ± 0.4 | 91.8 ± 0.8 | 1.6 ± 0.5 | 9.4 |
| ft_bge_m3_ft_orig | 5 | frozen bge_m3 | verified (strict) | 93.3 ± 0.4 | 92.5 ± 0.5 | 0.8 ± 0.2 | 4.6 |
| ft_e5_base_ft_aug | 5 | frozen e5_base | auto | 89.2 ± 0.6 | 88.2 ± 0.9 | 1.4 ± 0.3 | 8.0 |
| ft_e5_base_ft_aug | 5 | frozen e5_base | verified (strict) | 89.2 ± 0.6 | 88.6 ± 0.6 | 0.7 ± 0.1 | 4.2 |
| e5_base (frozen) | 1 | frozen e5_base | auto | 83.8 | 79.4 | 4.6 | 27 |
| e5_base (frozen) | 1 | frozen e5_base | verified (strict) | 83.8 | 79.5 | 4.4 | 26 |
| ft_e5_base_ft_orig | 5 | frozen e5_base | auto | 80.2 ± 0.7 | 78.8 ± 0.5 | 1.7 ± 0.7 | 10.2 |
| ft_e5_base_ft_orig | 5 | frozen e5_base | verified (strict) | 80.2 ± 0.7 | 79.6 ± 0.6 | 0.7 ± 0.1 | 4.0 |


## Random vs. worst-case on the same denominator (auto validity, doc_max)

Random: one auto-valid dictionary or MLM substitution drawn per query (mean of 200 draws); queries without one keep their original rank. Worst-case: the attacker's strongest valid variant.

| retriever | queries with a valid random variant | random: ΔR@5 (pp) | random: successful queries | worst-case (auto): ΔR@5 (pp) | worst-case (auto): successful queries |
|---|---|---|---|---|---|
| bm25 | 422 | 1.1 | 6.3 | 5.5 | 32 |
| e5_base | 422 | 1.3 | 7.8 | 4.6 | 27 |
| bge_m3 | 422 | 1.0 | 5.9 | 4.1 | 24 |
| hybrid | 422 | 0.8 | 4.8 | 2.9 | 17 |


## Attack search budget (validity-aware search)

Every admissible single substitution (≤ 8 candidates per position, SBERT cos ≥ 0.85) is judged once before the search and shared by all attacked retrievers: 28955 judged, 4013 valid. The greedy search uses only valid singles; per chain the 2 most damaging two-word combinations are judged in order and the first valid one is kept.

| attacked retriever | chains | chains_with_valid_step1 | retriever_calls_per_chain_mean | retriever_calls_per_chain_max | step2_judge_calls_per_chain_mean |
|---|---|---|---|---|---|
| bm25 | 991 | 690 | 7.0 | 31 | 0.4 |
| e5_base | 991 | 844 | 7.1 | 32 | 0.5 |
| bge_m3 | 991 | 841 | 7.0 | 36 | 0.5 |
| hybrid | 991 | 607 | 6.7 | 32 | 0.3 |


## Human-verified successful attacks (white-box)

- **bm25** · title · rank 4 → 7: «Цільові кібератаки UAC-0185 у відношенні Сил оборони та підприємств ОПК України (CERT-UA#12414)» → «Цільові кібератаки UAC-0185 у відношенні Сил оборони та виробництв ОПК України (CERT-UA#12414)» (підприємств → виробництв) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · title · rank 4 → 17: «Зведена інформація щодо діяльності угрупування UAC-0010 станом на липень 2023 року» → «Зведена інформація щодо роботи угрупування UAC-0010 станом на липень 2023 року» (діяльності → роботи) [human: yes/natural (b3)]
- **bm25** · title · rank 3 → 7: «Кібератака на державні організації України з використанням шкідливої програми IcedID (CERT-UA#4464)» → «Кібератака на державні організації України з використанням небезпечної програми IcedID (CERT-UA#4464)» (шкідливої → небезпечної) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · title · rank 2 → 10: «Кібератака на організації та установи України з використанням програми Remote Utilities (CERT-UA#3751)» → «Кібератака на організації та установи України з використанням системи Remote Utilities (CERT-UA#3751)» (програми → системи) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · title · rank 2 → 7: «Кібератака на сайти державних органів» → «Кібератака на сайти державних структур» (органів → структур) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · title · rank 1 → 13: «Робоча нарада з представниками енергетичного сектору» → «Робоча засідання з делегатами енергетичного сектору» (нарада → засідання; представниками → делегатами) [human: yes/awkward (b3), yes/awkward (b3)]
- **bm25** · section · rank 2 → 9: «Які види діяльності здійснює угрупування UAC-0050 за даними CERT-UA?» → «Які види робіт здійснює організація UAC-0050 за даними CERT-UA?» (діяльності → робіт; угрупування → організація) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 5 → 12: «Як працює фішингова атака з підробкою reCAPTCHA для викрадення даних браузерів?» → «Як працює фішингова атака з підробкою reCAPTCHA для викрадення інформації браузерів?» (даних → інформації) [human: yes/natural (b3)]
- **bm25** · section · rank 4 → 6: «Які методи кібератак використовує CERT-UA проти Сил оборони України у 2025 році?» → «Які технології кібератак використовує CERT-UA проти Сил захисту України у 2025 році?» (методи → технології; оборони → захисту) [human: yes/natural (b3)]
- **bm25** · section · rank 3 → 10: «Як виявити шкідливий VBS-скрипт для рекурсивного пошуку і викрадення файлів?» → «Як виявити шкідливий VBS-скрипт для рекурсивного розшукування і викрадення файлів?» (пошуку → розшукування) [human: yes/awkward (b3), yes/awkward (b3)]
- **bm25** · section · rank 5 → 7: «Які методи поширення SMOKELOADER використовує угруповання UAC-0006 у травні 2024?» → «Які методи поширення SMOKELOADER застосовує угруповання UAC-0006 у травні 2024?» (використовує → застосовує) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 5 → 13: «Як програма на Python забезпечує персистентність через реєстр Windows?» → «Як програма на Python забезпечить персистентність через реєстр Windows?» (забезпечує → забезпечить) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 5 → 6: «Які методи розповсюдження використовує група UAC-0010 для кібератак у 2022 році?» → «Які технології розповсюдження використовує група UAC-0010 для кібератак у 2022 році?» (методи → технології) [human: yes/natural (b3)]
- **bm25** · section · rank 5 → 7: «Як шкідлива програма OUTSTEEL передає дані на сервер управління?» → «Як небезпечна програма OUTSTEEL передає дані на сервер керування?» (шкідлива → небезпечна; управління → керування) [human: yes/natural (b3)]
- **bm25** · section · rank 2 → 8: «Які методи використовують для розповсюдження шкідливих програм через месенджер Signal?» → «Які методи застосовують для розповсюдження шкідливих програм через додаток Signal?» (використовують → застосовують; месенджер → додаток) [human: yes/natural (b3)]
- **bm25** · section · rank 5 → 7: «Які методи поширення використовує шкідлива програма MerlinAgent в Україні?» → «Які засоби поширення використовує шкідлива програма MerlinAgent в Україні?» (методи → засоби) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 3 → 12: «Які методи поширення шкідливого ПЗ використовує група UAC-0056 в Україні?» → «Які засоби поширення шкідливого ПЗ використовує організація UAC-0056 в Україні?» (методи → засоби; група → організація) [human: yes/natural (b3)]
- **bm25** · section · rank 4 → 6: «Які шкідливі програми пов’язані з атакою групи UAC-0056 на державні органи України?» → «Які шкідливі програми пов’язані з атакою групи UAC-0056 на державні служби України?» (органи → служби) [human: yes/natural (b3)]
- **bm25** · section · rank 1 → 6: «Як працює шкідливий інсталятор, що імітує додаток Армія+, і які його функції?» → «Як працює шкідливий інсталятор, що імітує програму Армія+, і які його дії?» (додаток → програму; функції → дії) [human: yes/natural (b3)]
- **bm25** · section · rank 4 → 7: «Які ознаки та наслідки атаки з використанням шкідливої програми JesterStealer?» → «Які ознаки та результати атаки з використанням шкідливої програми JesterStealer?» (наслідки → результати) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 3 → 13: «Як безпечно перевіряти електронні листи з посиланнями від державних установ?» → «Як безпечно перевіряти електронні листи з посиланнями від офіційних організацій?» (державних → офіційних; установ → організацій) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 2 → 6: «Які повсякденні заходи безпеки рекомендують для захисту від кібератак?» → «Які повсякденні засоби безпеки рекомендують для захисту від кібератак?» (заходи → засоби) [human: yes/natural (b3)]
- **bm25** · section · rank 1 → 11: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 4 → 8: «Які методи використано у кібератаці на Сили оборони України через месенджер Signal?» → «Які засоби використано у кібератаці на Сили оборони України через додаток Signal?» (методи → засоби; месенджер → додаток) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 2 → 17: «Які методи прихованого управління використовує угруповання APT28 у атаках на Україну?» → «Які засоби прихованого контролю використовує угруповання APT28 у атаках на Україну?» (методи → засоби; управління → контролю) [human: yes/natural (b3)]
- **bm25** · section · rank 1 → 7: «Які методи розповсюдження шкідливого ПЗ використовували для атаки на оборонне відомство України?» → «Які методи поширення шкідливого ПЗ використовували для атаки на оборонне агентство України?» (розповсюдження → поширення; відомство → агентство) [human: yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 5 → 9: «Які основні виклики державної системи кіберзахисту в Україні?» → «Які чільні виклики державної системи кіберзахисту в Україні?» (основні → чільні) [human: yes/natural (b3)]
- **bm25** · section · rank 2 → 7: «Як діяти при підозрі на кібератаку з використанням легітимних облікових записів?» → «Як діяти при підозрінню на кібератаку з використанням легітимних облікових записів?» (підозрі → підозрінню) [human: yes/awkward (b3), yes/awkward (b3), yes/awkward (b3), yes/awkward (b3), yes/awkward (b3)]
- **bm25** · section · rank 3 → 10: «Як захистити ЕОМ від запуску шкідливих утиліт через Microsoft Office?» → «Як захистити ЕОМ від запуску шкідливих програм через Microsoft Office?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bm25** · section · rank 1 → 7: «Які особливості віддаленого керування через програмний засіб на C# з динамічною компіляцією команд?» → «Які особливості віддаленого управління через програмний продукт на C# з динамічною компіляцією команд?» (керування → управління; засіб → продукт) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · title · rank 5 → 10: «Месенджери та сайти знайомств - нові способи проведення атак UAC-0184 (CERT-UA#9474)» → «Месенджери та сайти знайомств - нові методи здійснення атак UAC-0184 (CERT-UA#9474)» (способи → методи; проведення → здійснення) [human: yes/natural (b3)]
- **e5_base** · title · rank 5 → 6: «Щодо обстановки в сфері кібер на 23-24 лютого 2024 року» → «Щодо обстановки в царині кібер на 23-24 лютого 2024 року» (сфері → царині) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · title · rank 4 → 11: «UAC-0184: Цільові атаки у відношенні українських військовослужбовців з використанням тематики рекрутингу до 3 ОШБр та ЦАХАЛ (CERT-UA#8386)» → «UAC-0184: Цільові атаки у напрямку українських військовослужбовців з використанням елементів рекрутингу до 3 ОШБр та ЦАХАЛ (CERT-UA#8386)» (відношенні → напрямку; тематики → елементів) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · title · rank 3 → 6: «Цільові атаки Turla (UAC-0024, UAC-0003) з використанням шкідливих програм CAPIBAR та KAZUAR (CERT-UA#6981)» → «Цільові атаки Turla (UAC-0024, UAC-0003) з використанням шкідливих засобів CAPIBAR та KAZUAR (CERT-UA#6981)» (програм → засобів) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · title · rank 2 → 10: «Онлайн-шахрайство з використанням тематики "грошової допомоги від країн ЄС" (CERT-UA#4492)» → «Онлайн-афера з використанням тематики "грошової допомоги від країн ЄС" (CERT-UA#4492)» (шахрайство → афера) [human: yes/natural (b3)]
- **e5_base** · title · rank 5 → 10: «Попередження щодо можливих кібератак» → «Попередження щодо імовірних кібератак» (можливих → імовірних) [human: yes/natural (b3)]
- **e5_base** · title · rank 5 → 7: «Довідкова інформація з питань діяльності CERT-UA за фактами впливу на стан кібербезпеки у 2022 році» → «Довідкова інформація з питань роботи CERT-UA за фактами впливу на стан кібербезпеки у 2022 році» (діяльності → роботи) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · title · rank 3 → 6: «Масштабна фішингова атака на державні установи України 19.01.2021» → «Велика фішингова атака на державні організації України 19.01.2021» (Масштабна → Велика; установи → організації) [human: yes/natural (b3)]
- **e5_base** · title · rank 5 → 8: «Популярні вразливості в українському сегменті мережі Інтернет» → «Відомі вразливості в українському сегменті мережі Інтернет» (Популярні → Відомі) [human: yes/natural (b3)]
- **e5_base** · section · rank 4 → 16: «Як розпізнати фішингові листи з шкідливими вкладеннями для крадіжки облікових даних?» → «Як розпізнати фішингові повідомлення з шкідливими вкладеннями для крадіжки облікових даних?» (листи → повідомлення) [human: yes/natural (b3)]
- **e5_base** · section · rank 4 → 11: «Як працює фішингова атака з підробкою reCAPTCHA для викрадення даних браузерів?» → «Як відбувається фішингова атака з підробкою reCAPTCHA для викрадення даних браузерів?» (працює → відбувається) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 3 → 25: «Які методи використовувалися у кібератаках з ідентифікатором UAC-0219 у 2024 році?» → «Які стратегії використовувалися у кібератаках з ідентифікатором UAC-0219 у 2024 році?» (методи → стратегії) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 3 → 12: «Як виявити шкідливий VBS-скрипт для рекурсивного пошуку і викрадення файлів?» → «Як помітити шкідний VBS-скрипт для рекурсивного пошуку і викрадення файлів?» (виявити → помітити; шкідливий → шкідний) [human: yes/awkward (b3)]
- **e5_base** · section · rank 3 → 18: «Які заходи вжито CERT-UA для аналізу шахрайських доменів з 2021 року?» → «Які засоби вжито CERT-UA для дослідження шахрайських доменів з 2021 року?» (заходи → засоби; аналізу → дослідження) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 5 → 8: «Які методи атаки використовує угруповання UAC-0006 із SmokeLoader у липні 2023?» → «Які технології атаки використовує угруповання UAC-0006 із SmokeLoader у липні 2023?» (методи → технології) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 5 → 7: «Які ознаки активності групи UAC-0057 у звіті CERT-UA про шкідливі архіви?» → «Які види активності групи UAC-0057 у звіті CERT-UA про шкідливі архіви?» (ознаки → види) [human: yes/natural (b3)]
- **e5_base** · section · rank 4 → 10: «Які ознаки компрометації пов’язані з розсиланням листів через уразливість CVE-2022-30190?» → «Які особливості компрометації пов’язані з розсиланням листів через уразливість CVE-2022-30190?» (ознаки → особливості) [human: yes/natural (b3)]
- **e5_base** · section · rank 5 → 7: «Які особливості фішингової атаки на державні установи України у січні 2021 року?» → «Які ознаки фішингової атаки на державні організації України у січні 2021 року?» (особливості → ознаки; установи → організації) [human: yes/natural (b3)]
- **e5_base** · section · rank 3 → 7: «Які методи використано для розповсюдження шкідливого ПЗ через фішингові листи податкової тематики?» → «Які засоби використано для поширення шкідливого ПЗ через фішингові листи податкової тематики?» (методи → засоби; розповсюдження → поширення) [human: yes/natural (b3)]
- **e5_base** · section · rank 5 → 14: «Які вектори компрометації використовує угруповання UAC-0063 у червні 2024 року?» → «Які схеми компрометації використовує угруповання UAC-0063 у червні 2024 року?» (вектори → схеми) [human: yes/natural (b3)]
- **e5_base** · section · rank 1 → 17: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 2 → 6: «Які основні помилки допускають власники систем у захисті від кібератак?» → «Які основні хиби допускають господарі систем у захисті від кібератак?» (помилки → хиби; власники → господарі) [human: yes/natural (b3)]
- **e5_base** · section · rank 5 → 10: «Які методи використано у кібератаці на Сили оборони України через месенджер Signal?» → «Які засоби використано у кібератаці на Сили захисту України через месенджер Signal?» (методи → засоби; оборони → захисту) [human: yes/natural (b3)]
- **e5_base** · section · rank 5 → 6: «Які загрози пов'язані з шкідливими програмами, що змінюють мережеві налаштування і реєстр?» → «Які загрози пов'язані з шкідливими програмами, що змінюють мережеві настройки і реєстр?» (налаштування → настройки) [human: yes/natural (b3), yes/natural (b3)]
- **e5_base** · section · rank 1 → 7: «Які методи використовувала група UAC-0082 для атаки на Укрінформ у 2023 році?» → «Які схеми використовувала кампанія UAC-0082 для атаки на Укрінформ у 2023 році?» (методи → схеми; група → кампанія) [human: yes/natural (b3)]
- **e5_base** · section · rank 4 → 9: «Які методи використання PowerShell для інфікування комп'ютерів описані у звіті CERT-UA 2026?» → «Які методи використання PowerShell для інфікування комп'ютерів описані у повідомленні CERT-UA 2026?» (звіті → повідомленні) [human: yes/natural (b3), yes/natural (b3)]
- **bge_m3** · title · rank 5 → 8: «Кібератака UAC-0020 (Vermin) з використанням тематики військовополонених на Курському напрямку; новий інструмент FIRMACHAGENT (CERT-UA#10742)» → «Кібератака UAC-0020 (Vermin) з використанням інформації військовополонених на Курському напрямку; новий ресурс FIRMACHAGENT (CERT-UA#10742)» (тематики → інформації; інструмент → ресурс) [human: yes/awkward (b3), yes/awkward (b3), yes/awkward (b3)]
- **bge_m3** · title · rank 5 → 6: «Листи нібито від Держспецзв'язку та ДСНС - атака UAC-0050 з використанням RemoteUtilities (CERT-UA#8654)» → «Повідомлення нібито від Держспецзв'язку та ДСНС - атака UAC-0050 з використанням RemoteUtilities (CERT-UA#8654)» (Листи → Повідомлення) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · title · rank 4 → 14: «APT28: від первинного ураження до створення загроз для контролеру домену за годину (CERT-UA#8399)» → «APT28: від першого ураження до появи загроз для контролеру домену за годину (CERT-UA#8399)» (первинного → першого; створення → появи) [human: yes/natural (b3), yes/natural (b3)]
- **bge_m3** · title · rank 4 → 20: «Активність групи UAC-0114 (Winter Vivern) у відношенні державних органів України та Польщі (CERT-UA#5909)» → «Активність команди UAC-0114 (Winter Vivern) у відношенні державних структур України та Польщі (CERT-UA#5909)» (групи → команди; органів → структур) [human: yes/natural (b3)]
- **bge_m3** · title · rank 3 → 14: «Онлайн-шахрайство з використанням тематики "допомоги від Червоного Хреста" (CERT-UA#5063)» → «Онлайн-ошуканство з використанням тематики "підтримки від Червоного Хреста" (CERT-UA#5063)» (шахрайство → ошуканство; допомоги → підтримки) [human: yes/natural (b3)]
- **bge_m3** · section · rank 5 → 11: «Які методи поширення шкідливого ПЗ використовує фейковий лист від імені CERT-UA?» → «Які стратегії поширення шкідливого ПЗ використовує фейковий лист від представника CERT-UA?» (методи → стратегії; імені → представника) [human: yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 2 → 11: «Які заходи вживає CERT-UA при компрометації VPN і використанні шифрованого вебшелу?» → «Які стратегії здійснює CERT-UA при компрометації VPN і використанні шифрованого вебшелу?» (заходи → стратегії; вживає → здійснює) [human: yes/natural (b3)]
- **bge_m3** · section · rank 5 → 6: «Які методи кібератак використовують для доступу до комп'ютерів ЗСУ через Signal?» → «Які засоби кібератак використовують для доступу до комп'ютерів ЗСУ через Signal?» (методи → засоби) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 2 → 8: «Які ознаки масового розповсюдження шкідливих листів від імені Ради нацбезпеки України?» → «Які властивості масового розповсюдження шкідних листів від імені Ради нацбезпеки України?» (ознаки → властивості; шкідливих → шкідних) [human: yes/awkward (b3), yes/awkward (b3)]
- **bge_m3** · section · rank 5 → 16: «Які інструменти використовують зловмисники для горизонтального переміщення в локальній мережі?» → «Які методи застосовують зловмисники для горизонтального переміщення в локальній мережі?» (інструменти → методи; використовують → застосовують) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 3 → 6: «Які нові методи зараження використовує кластер UAC-0099 у 2026 році?» → «Які нові стратегії зараження використовує кластер UAC-0099 у 2026 році?» (методи → стратегії) [human: yes/natural (b3)]
- **bge_m3** · section · rank 2 → 11: «Які вектори компрометації використовує угруповання UAC-0063 у червні 2024 року?» → «Які технології компрометації використовує угруповання UAC-0063 у червні 2024 року?» (вектори → технології) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 4 → 7: «Які заходи рекомендує CERT-UA при розповсюдженні шкідливих листів з макросами?» → «Які засоби порекомендує CERT-UA при розповсюдженні шкідливих листів з макросами?» (заходи → засоби; рекомендує → порекомендує) [human: yes/awkward (b3)]
- **bge_m3** · section · rank 3 → 19: «Які шкідливі програми використовувалися в атаці на Укрінформ у січні 2023?» → «Які шкідливі засоби використовувалися в атаці на Укрінформ у січні 2023?» (програми → засоби) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 4 → 14: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 1 → 7: «Як безпечно обробляти електронні листи щоб уникнути шкідливого ПЗ?» → «Як безпечно обробляти електронні повідомлення щоб уникнути шкідливого ПЗ?» (листи → повідомлення) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 4 → 7: «Які методи використано у кібератаці на Сили оборони України через месенджер Signal?» → «Які методи використано у кібератаці на Сили оборони України через додаток Signal?» (месенджер → додаток) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **bge_m3** · section · rank 3 → 8: «Як захистити комп'ютер від шкідливих додатків у листах електронної пошти?» → «Як захистити комп'ютер від шкідливих програм у листах електронної пошти?» (додатків → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · title · rank 3 → 6: «APT28: від первинного ураження до створення загроз для контролеру домену за годину (CERT-UA#8399)» → «APT28: від першого ураження до появи загроз для контролеру домену за годину (CERT-UA#8399)» (первинного → першого; створення → появи) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · title · rank 5 → 6: «Повернення UAC-0006: масове розповсюдження SmokeLoader з використанням тематики "рахунків" (CERT-UA#6613)» → «Повернення UAC-0006: масове поширення SmokeLoader з використанням тематики "рахунків" (CERT-UA#6613)» (розповсюдження → поширення) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · title · rank 5 → 6: «Онлайн-шахрайство з використанням тематики "грошової компенсації" (CERT-UA#4964)» → «Інтернет-шахрайство з використанням тематики "грошової компенсації" (CERT-UA#4964)» (Онлайн → Інтернет) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 2 → 7: «Як програма на Python забезпечує персистентність через реєстр Windows?» → «Як додаток на Python підтримує персистентність через реєстр Windows?» (програма → додаток; забезпечує → підтримує) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 5 → 10: «Які методи розповсюдження використовували для шкідливого документа з CVE-2022-30190?» → «Які методи розповсюдження використовували для шкідливого матеріалу з CVE-2022-30190?» (документа → матеріалу) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 5 → 8: «Як правильно налаштувати двофакторну автентифікацію для захисту облікових записів?» → «Як правильно налаштувати двофакторну автентифікацію для оборони облікових записів?» (захисту → оборони) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 1 → 6: «Які шкідливі програми використовувалися в атаці на Укрінформ у січні 2023?» → «Які шкідливі засоби використовувалися в атаці на Укрінформ у січні 2023?» (програми → засоби) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 1 → 8: «Як заборонити запуск небезпечних утиліт користувачам у Windows для кібербезпеки?» → «Як заборонити запуск небезпечних програм користувачам у Windows для кібербезпеки?» (утиліт → програм) [human: yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 2 → 6: «Які методи використано у кібератаці на Сили оборони України через месенджер Signal?» → «Які засоби використано у кібератаці на Сили оборони України через додаток Signal?» (методи → засоби; месенджер → додаток) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 5 → 9: «Які методи прихованого управління використовує угруповання APT28 у атаках на Україну?» → «Які технології прихованого контролю використовує угруповання APT28 у атаках на Україну?» (методи → технології; управління → контролю) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 3 → 13: «Які методи кібератак використовувалися проти фінансових установ у лютому 2022 року?» → «Які технології кібератак використовувалися проти фінансових організацій у лютому 2022 року?» (методи → технології; установ → організацій) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 3 → 16: «Які заходи проводить CERT-UA для протидії кіберзагрозам в Україні?» → «Які дії проводить CERT-UA для запобігання кіберзагрозам в Україні?» (заходи → дії; протидії → запобігання) [human: yes/natural (b3), yes/natural (b3)]
- **hybrid** · section · rank 2 → 7: «Як діяти при підозрі на кібератаку з використанням легітимних облікових записів?» → «Як діяти при підозрінню на кібератаку з використанням законних облікових записів?» (підозрі → підозрінню; легітимних → законних) [human: yes/awkward (b3), yes/awkward (b3)]
