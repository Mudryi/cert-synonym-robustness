# Human validity audit — annotator A1

887 of 722 items labelled (0 marked unsure). Strict = same information need and not ungrammatical; lenient also accepts 'partly'.

## Self-consistency (repeated items)

- 31 pairs; same meaning label 87.1%, same strict verdict 93.5%

- 52 variant texts were labelled more than once as separate items (found by several attacks); same strict verdict for 82.7%, 9 conflicts. Downstream (S14) a text counts as valid only if every label says so.

## Human validity rates with 95% Wilson intervals

Raw rates in the audit sample. The sample deliberately over-represents rank-changing variants, so rates for whole pools are not population estimates; the weighted table below is a sensitivity analysis, not a population correction.

### By pool and substitution source

| pool | perturbation_type | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|---|
| dose | dict | 15 | 26.7 | [10.9, 52.0] | 66.7 | [41.7, 84.8] | 0.0 | [0.0, 20.4] |
| dose | mlm | 15 | 33.3 | [15.2, 58.3] | 66.7 | [41.7, 84.8] | 33.3 | [15.2, 58.3] |
| random | dict | 28 | 28.6 | [15.3, 47.1] | 82.1 | [64.4, 92.1] | 21.4 | [10.2, 39.5] |
| random | llm | 28 | 85.7 | [68.5, 94.3] | 96.4 | [82.3, 99.4] | 82.1 | [64.4, 92.1] |
| random | mlm | 28 | 42.9 | [26.5, 60.9] | 75.0 | [56.6, 87.3] | 42.9 | [26.5, 60.9] |
| random | tech | 28 | 89.3 | [72.8, 96.3] | 92.9 | [77.4, 98.0] | 82.1 | [64.4, 92.1] |
| worst | worst_dict | 377 | 74.3 | [69.6, 78.4] | 96.6 | [94.2, 98.0] | 67.4 | [62.5, 71.9] |
| worst | worst_mlm | 172 | 80.8 | [74.3, 86.0] | 93.0 | [88.2, 96.0] | 77.9 | [71.1, 83.5] |


### By pool and automatic label

| pool | auto label | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|---|
| dose | invalid | 10 | 0.0 | [0.0, 27.8] | 20.0 | [5.7, 51.0] | 0.0 | [0.0, 27.8] |
| dose | valid | 20 | 45.0 | [25.8, 65.8] | 90.0 | [69.9, 97.2] | 25.0 | [11.2, 46.9] |
| random | invalid | 36 | 38.9 | [24.8, 55.1] | 66.7 | [50.3, 79.8] | 30.6 | [18.0, 46.9] |
| random | valid | 76 | 72.4 | [61.4, 81.2] | 96.1 | [89.0, 98.6] | 69.7 | [58.7, 78.9] |
| worst | invalid | 40 | 7.5 | [2.6, 19.9] | 50.0 | [35.2, 64.8] | 7.5 | [2.6, 19.9] |
| worst | valid | 509 | 81.7 | [78.1, 84.8] | 99.0 | [97.7, 99.6] | 75.6 | [71.7, 79.2] |


### By attack outcome (auto-valid only)

Outcome on the attacked retriever (worst-case) or any main retriever (random, dose): `out of top 5` = source left the top 5; `worsened` = source rank dropped, starting from the top 10; `unchanged` otherwise.

| pool | outcome | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|---|
| dose | out of top 5 | 3 | 66.7 | [20.8, 93.9] | 100.0 | [43.8, 100.0] | 33.3 | [6.1, 79.2] |
| dose | unchanged | 10 | 70.0 | [39.7, 89.2] | 90.0 | [59.6, 98.2] | 40.0 | [16.8, 68.7] |
| dose | worsened | 7 | 0.0 | [0.0, 35.4] | 85.7 | [48.7, 97.4] | 0.0 | [0.0, 35.4] |
| random | out of top 5 | 8 | 75.0 | [40.9, 92.9] | 100.0 | [67.6, 100.0] | 75.0 | [40.9, 92.9] |
| random | unchanged | 40 | 70.0 | [54.6, 81.9] | 95.0 | [83.5, 98.6] | 65.0 | [49.5, 77.9] |
| random | worsened | 28 | 75.0 | [56.6, 87.3] | 96.4 | [82.3, 99.4] | 75.0 | [56.6, 87.3] |
| worst | out of top 5 | 31 | 80.6 | [63.7, 90.8] | 100.0 | [89.0, 100.0] | 80.6 | [63.7, 90.8] |
| worst | unchanged | 374 | 84.0 | [79.9, 87.3] | 98.9 | [97.3, 99.6] | 78.3 | [73.9, 82.2] |
| worst | worsened | 104 | 74.0 | [64.9, 81.5] | 99.0 | [94.8, 99.8] | 64.4 | [54.9, 73.0] |


## Automatic label vs. human verdict

For auto-valid rows the strict-valid % is the precision of the automatic rule; for auto-invalid rows it is the share wrongly rejected.

| pool | auto label | n | human strict-valid % | human lenient-valid % |
|---|---|---|---|---|
| dose | invalid | 10 | 0.0 | 20.0 |
| dose | valid | 20 | 45.0 | 90.0 |
| random | invalid | 36 | 38.9 | 66.7 |
| random | valid | 76 | 72.4 | 96.1 |
| worst | invalid | 40 | 7.5 | 50.0 |
| worst | valid | 509 | 81.7 | 99.0 |


## Census: every automatically accepted worst-case variant

All 509 auto-accepted worst-case variants are labelled (first pass), so the rates are exact for these 509 cases; the Wilson interval only expresses uncertainty when generalising to similar generated attacks.

### By attack source

| perturbation_type | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|
| worst_dict | 357 | 77.9 | [73.3, 81.9] | 98.6 | [96.8, 99.4] | 70.6 | [65.7, 75.1] |
| worst_mlm | 152 | 90.8 | [85.1, 94.4] | 100.0 | [97.5, 100.0] | 87.5 | [81.3, 91.8] |


### By attacked retriever

| attack_model | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|
| bge_m3 | 77 | 76.6 | [66.0, 84.7] | 97.4 | [91.0, 99.3] | 66.2 | [55.1, 75.8] |
| bm25 | 213 | 85.0 | [79.6, 89.2] | 100.0 | [98.2, 100.0] | 79.3 | [73.4, 84.2] |
| e5_base | 67 | 70.1 | [58.3, 79.8] | 97.0 | [89.8, 99.2] | 62.7 | [50.7, 73.3] |
| hybrid | 152 | 84.9 | [78.3, 89.7] | 99.3 | [96.4, 99.9] | 80.9 | [73.9, 86.4] |


### By attack outcome on the attacked retriever

| outcome | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|
| out of top 5 | 31 | 80.6 | [63.7, 90.8] | 100.0 | [89.0, 100.0] | 80.6 | [63.7, 90.8] |
| unchanged | 374 | 84.0 | [79.9, 87.3] | 98.9 | [97.3, 99.6] | 78.3 | [73.9, 82.2] |
| worsened | 104 | 74.0 | [64.9, 81.5] | 99.0 | [94.8, 99.8] | 64.4 | [54.9, 73.0] |


## Estimated precision of the automatic rule (weighted by stratum size, batch 1)

The stratified batch-1 sample over-represents rank-changing variants; weights restore each pool's real mix. For the worst-case pool the census above supersedes this estimate.

| pool | labelled | population | weighted strict-valid % | weighted lenient-valid % |
|---|---|---|---|---|
| dose | 20 | 67 | 46.6 | 88.7 |
| random | 76 | 2449 | 82.0 | 96.5 |
| worst | 72 | 509 | 74.5 | 97.2 |


## Agreement of automatic judges with the human verdict

Batch 1 is the stratified sample (accepted and rejected variants, all pools) and is the fair basis for κ; batch 2 adds only accepted worst-case variants, which shifts κ toward the precision of the accepted set.

| items | judge | n | agreement % (strict) | κ (strict) | κ (lenient) |
|---|---|---|---|---|---|
| batch 1 (stratified) | GPT judge (meaning) | 254 | 73.2 | 0.455 | 0.476 |
| batch 1 (stratified) | automatic rule (GPT + SBERT ≥ 0.8) | 254 | 73.2 | 0.455 | 0.476 |
| batch 1 (stratified) | local MamayLM p ≥ 0.5 | 254 | 66.5 | 0.303 | 0.537 |
| all labelled | GPT judge (meaning) | 691 | 79.5 | 0.387 | 0.547 |
| all labelled | automatic rule (GPT + SBERT ≥ 0.8) | 691 | 79.5 | 0.387 | 0.547 |
| all labelled | local MamayLM p ≥ 0.5 | 691 | 77.0 | 0.244 | 0.547 |


## Do ranking-changing substitutions tend to be invalid?

`rank got worse`: a main retriever (the attacked one for worst-case) ranked the source lower, starting from the top 10.

| pool | rank got worse | n | human strict-valid % | auto-valid % |
|---|---|---|---|---|
| dose | False | 16 | 43.8 | 62.5 |
| dose | True | 14 | 14.3 | 71.4 |
| random | False | 60 | 61.7 | 66.7 |
| random | True | 52 | 61.5 | 69.2 |
| worst | False | 398 | 79.4 | 94.0 |
| worst | True | 151 | 68.2 | 89.4 |


## By perturbation type

| pool | type | n | human strict-valid % | auto-valid % |
|---|---|---|---|---|
| dose | dict | 15 | 26.7 | 66.7 |
| dose | mlm | 15 | 33.3 | 66.7 |
| random | dict | 28 | 28.6 | 67.9 |
| random | llm | 28 | 85.7 | 67.9 |
| random | mlm | 28 | 42.9 | 67.9 |
| random | tech | 28 | 89.3 | 67.9 |
| worst | worst_dict | 377 | 74.3 | 94.7 |
| worst | worst_mlm | 172 | 80.8 | 88.4 |


## Issues ticked

| issue | n |
|---|---|
| term_changed | 59 |
| wrong_sense | 47 |
| scope_shift | 40 |
| register | 30 |
| morphology | 16 |
| other | 6 |


## Auto-valid but human-rejected (examples)

- `B-2498-0000|mlm|2` (partly/natural)
- `B-14958-0001|worst_dict_bm25|2` (partly/awkward)
- `B-3761023-0000|mlm|0` (partly/natural)
- `B-5160737-0000|mlm|2` (partly/natural)
- `B-405538-0000|tech|0` (partly/awkward)
- `B-6282517-0000|worst_dict_hybrid|2` (partly/awkward)
- `B-25-0003|worst_dict_hybrid|1` (partly/natural)
- `B-22-0000|worst_dict_hybrid|1` (partly/natural)
- `B-40125-0000|worst_dict_bm25|1` (partly/natural)
- `B-6279561-0000|worst_mlm_hybrid|1` (partly/natural)
- `B-6280129-0002|worst_mlm_bge_m3|1` (partly/natural)
- `A-2660|mlm|1` (partly/natural)
- `B-6276584-0002|dict|3` (partly/awkward)
- `B-6277822-0003|worst_dict_e5_base|1` (partly/awkward)
- `B-4501891-0000|worst_mlm_hybrid|1` (partly/natural)
- `B-17899-0001|dict|1` (partly/awkward)
- `B-6277849-0003|dict|0` (no/ungrammatical)
- `A-2697|llm|2` (no/awkward)
- `A-6286219|worst_dict_bge_m3|1` (no/ungrammatical)
- `B-5702579-0001|dict|2` (partly/awkward)
- `B-4697016-0001|dict|2` (partly/awkward)
- `B-6288047-0001|worst_dict_bm25|1` (partly/natural)
- `B-6032734-0001|worst_mlm_bge_m3|1` (partly/awkward)
- `B-6280129-0002|worst_mlm_hybrid|1` (partly/natural)
- `B-14958-0001|dict|2` (partly/natural)

## Batch 3: second pass over successful attacks

165 unique variant texts; 45 had a first-pass label. Same strict verdict 75.6%, same meaning label 75.6%; 7 changed from strict-valid to not, 4 the other way. Batch-3 verdicts replace the first pass downstream.

| source | n | strict % | strict 95% CI | lenient % | lenient 95% CI | natural % | natural 95% CI |
|---|---|---|---|---|---|---|---|
| S15 | 130 | 70.8 | [62.4, 77.9] | 100.0 | [97.1, 100.0] | 63.8 | [55.3, 71.6] |
| S15+S7 | 23 | 87.0 | [67.9, 95.5] | 100.0 | [85.7, 100.0] | 78.3 | [58.1, 90.3] |
| S15+conflict | 1 | 0.0 | [0.0, 79.3] | 100.0 | [20.7, 100.0] | 0.0 | [0.0, 79.3] |
| S7 | 5 | 80.0 | [37.6, 96.4] | 100.0 | [56.6, 100.0] | 80.0 | [37.6, 96.4] |
| conflict | 6 | 16.7 | [3.0, 56.4] | 83.3 | [43.6, 97.0] | 16.7 | [3.0, 56.4] |

