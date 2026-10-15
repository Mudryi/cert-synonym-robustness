# Relevance spot-check of the silver labels

Share of retrieval failures where the document ranked first also answers the query (a false failure of the single-source label). `attack`: failures caused by an auto-valid S7/S15 attack on the attacked retriever; `clean_fail`: original queries the hybrid retriever misses. 95% Wilson intervals.

| sample | n | rank 1 answers the query (yes) % | yes or partly % | same report % | related report % |
|---|---|---|---|---|---|
| attack | 24 | 0.0 [0.0, 13.8] | 12.5 | 0.0 | 25.0 |
| clean_fail | 12 | 8.3 [1.5, 35.4] | 50.0 | 0.0 | 75.0 |
| all | 36 | 2.8 [0.5, 14.2] | 25.0 | 0.0 | 41.7 |

