# S7 vs S15: protocol and budget audit

S7 = select → validate (greedy on margin, judge afterwards); S15 = validate → select (judge every admissible single, greedy over valid ones). Both use the same 586 queries, the same dictionary/MLM candidate pools (≤ 8 per position), the same SBERT pre-filter (cos ≥ 0.85), the same automatic rule (GPT judge + SBERT ≥ 0.8), the same protected-entity check and ≤ 2 substitutions.

## Consistency checks

- ✅ S7 variants use only benchmark queries
- ✅ S7 keeps protected entities in every variant
- ✅ S7 uses at most 2 substitutions
- ✅ S15 variants use only benchmark queries
- ✅ S15 keeps protected entities in every variant
- ✅ S15 uses at most 2 substitutions
- ✅ admissible singles equal S15's recorded count
- ✅ all 28955 single verdicts served from cache (misses: 0, new spend $0.00)
- ✅ S7 chains (1147, recomputed by S14) = (query, source) pairs with an admissible single (1147)
- ✅ S15 chains = pairs with a judge-valid single

Chains: S7 searches every (query, source) pair with an admissible single (1147); S15 only pairs with at least one judge-valid single (991). The 156 pairs S15 skips cannot yield an auto-valid variant in either search, so the difference in chain counts does not favour S15.

## Implementation equivalence (step 1)

The backoff search with J = 1 should reproduce S7's step-1 variant and with J = all S15's step-1 variant.

| attacked retriever | reference | chains compared | same step-1 text % |
|---|---|---|---|
| bm25 | S7 (J=1) | 1041 | 100.0 |
| bm25 | S15 (J=all) | 690 | 100.0 |
| e5_base | S7 (J=1) | 1132 | 100.0 |
| e5_base | S15 (J=all) | 845 | 99.3 |
| bge_m3 | S7 (J=1) | 1134 | 100.0 |
| bge_m3 | S15 (J=all) | 839 | 99.6 |
| hybrid | S7 (J=1) | 991 | 100.0 |
| hybrid | S15 (J=all) | 607 | 99.8 |


## Budget-matched control: successes against judge calls (step 1, auto validity, white-box)

Each chain ranks its margin-lowering admissible singles by margin and judges them in that order, at most J calls; the first valid one is the attack. Successful queries: source document leaves the attacked retriever's top 5, out of 586 (a query counts once over its dictionary and MLM chains).

| attacked retriever | judge calls / chain (cap) | judge calls / chain (mean used) | successful queries (step 1) |
|---|---|---|---|
| bm25 | 1 | 0.91 | 8 |
| bm25 | 2 | 1.61 | 13 |
| bm25 | 4 | 2.76 | 16 |
| bm25 | 8 | 4.40 | 19 |
| bm25 | 16 | 5.97 | 22 |
| bm25 | all | 6.66 | 22 |
| e5_base | 1 | 0.99 | 6 |
| e5_base | 2 | 1.92 | 6 |
| e5_base | 4 | 3.57 | 9 |
| e5_base | 8 | 6.00 | 18 |
| e5_base | 16 | 8.54 | 24 |
| e5_base | all | 9.69 | 24 |
| bge_m3 | 1 | 0.99 | 3 |
| bge_m3 | 2 | 1.91 | 6 |
| bge_m3 | 4 | 3.51 | 11 |
| bge_m3 | 8 | 5.93 | 15 |
| bge_m3 | 16 | 8.41 | 17 |
| bge_m3 | all | 9.57 | 17 |
| hybrid | 1 | 0.86 | 3 |
| hybrid | 2 | 1.56 | 5 |
| hybrid | 4 | 2.66 | 7 |
| hybrid | 8 | 4.11 | 9 |
| hybrid | 16 | 5.21 | 11 |
| hybrid | all | 5.56 | 12 |


| attacked retriever | retriever calls / chain, step 1 (all admissible scored) |
|---|---|
| bm25 | 26.5 |
| e5_base | 26.5 |
| bge_m3 | 26.5 |
| hybrid | 26.5 |


## Cost of the searches as run

Candidate generation (dictionary lookup, MLM proposals, SBERT pre-filter) is shared and identical. S15's judge calls on singles (28955 unique texts) are shared by the four attacked retrievers and shown here divided by four. S7 judges each variant on its greedy path once.

| attacked retriever | S7 chains | S15 chains | S7 retriever calls / chain | S15 retriever calls / chain | S7 judge calls (variants judged) | S15 judge calls (singles, shared ÷ 4 + step 2) | S7 successes (auto, 2 steps) | S15 successes (auto, 2 steps) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 1147 | 991 | 45.0 | 7.0 | 1844 | 7514 | 11 | 32 |
| e5_base | 1147 | 991 | 45.5 | 7.1 | 2153 | 7640 | 7 | 27 |
| bge_m3 | 1147 | 991 | 45.5 | 7.0 | 2149 | 7626 | 4 | 24 |
| hybrid | 1147 | 991 | 43.7 | 6.7 | 1645 | 7428 | 4 | 17 |

