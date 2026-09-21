# Oracle reranker investigation — September 21, 2026

The benchmark is working with the fixed delay removed. On this machine, the revised
in-database scoring path costs **1.06–1.24 times** the application path across the six tested
configurations. This is a local validation result, not proof of universal performance equivalence.

## What caused the operational problem

The delay is reproducibly associated with invoking the embedding and reranking models on
the **same database session**. Moving query embedding to a different statement on that same
session does not fix it. Keeping embedding on a dedicated session and binding its returned
vector into retrieval does fix it. No model rebuild, database restart or memory-setting change
was needed.

The underlying Oracle implementation remains unconfirmed. These experiments support a
model-switching explanation; they do not establish cache eviction, a specific memory limit,
or an Oracle defect. A-Rows and plan hashes alone cannot count scalar model invocations.

Single-candidate probe, query q01; median of three warm executions after discarding the first:

| Variant | Vector (ms) | Hybrid (ms) |
|---|---:|---:|
| Original inline embedding + reranking | 2,098.6 | 2,084.1 |
| Precomputed vector, reranking only | 80.9 | 121.2 |
| Separate embedding statement, same session; both calls timed | 2,488.1 | 2,521.0 |
| Dedicated embedding session; both calls timed | **93.4** | **129.5** |
| Two embedding calls using the same model | 68.0 | 100.6 |
| Inline embedding + LENGTH control | 15.6 | 17.8 |

All reranking variants returned identical IDs and scores. The bound-vector-only case excludes
embedding time and is a diagnostic, not the production latency claim. The dedicated-session
case includes fresh embedding and both database round trips. Two embedding calls do different
work from a cross-encoder; that control tests whether any two model calls trigger the penalty.

Evidence: [probe observations](results/model-cost-probe/raw.json),
[exact SQL](results/model-cost-probe/sql.json), [metadata](results/model-cost-probe/metadata.json).
Reproduce with `npm run probe:model-cost`, while no other benchmark is running.

## Fair comparison after the fix

Default `ORACLE_QUERY_EMBEDDING=separate-session` applies to both arms. Oracle's original
embedding model computes every query vector without caching; no replacement embedding model
or approximate retrieval was introduced. Candidate text remains inside Oracle in the in-database
arm. The additional embedding session and its resource use are part of this architecture.

The validation run used all 16 queries, all three retrieval strategies, N=10 and N=20,
top-K=10, two warmup passes and three measured passes. Oracle Free 23.26.3.0.0 reported two
CPUs; the q8 application reranker used two intra-op threads. All 96 arm-parity checks passed.
All 288 measured in-database candidate counts matched their application counterparts.
A separate check compared inline versus bound-vector retrieval across N=10/20/40/80:
all **192 candidate lists matched**, including order, scores and text.

Paired median treatment-minus-control scoring costs:

| Retrieval | N | In database (ms) | Application (ms) | DB/app |
|---|---:|---:|---:|---:|
| Vector | 10 | 723.9 | 601.5 | 1.20× |
| Vector | 20 | 1,439.4 | 1,276.7 | 1.13× |
| Lexical | 10 | 718.7 | 580.1 | 1.24× |
| Lexical | 20 | 1,137.4 | 1,011.5 | 1.12× |
| Hybrid | 10 | 651.3 | 593.1 | 1.10× |
| Hybrid | 20 | 1,270.0 | 1,193.6 | 1.06× |

These ratios describe scoring increments, not ratios of fitted slopes. The short two-depth
run is not a replacement for the original 20-iteration, four-depth publication sweep. Old
results remain valid observations of the original inline architecture and were not overwritten.

Evidence: [full report with intervals](results/2026-09-21T19-01-51-753Z/summary.md),
[raw data](results/2026-09-21T19-01-51-753Z/raw.json),
[inspection](results/2026-09-21T19-01-51-753Z/inspection.md).

## Has reranking earned its place?

For vector and lexical retrieval, reranking improved this synthetic corpus's nDCG@10.
For hybrid retrieval, it still did not provide a meaningful measured quality gain:

| Hybrid configuration | p50 end-to-end (ms) | nDCG@10 |
|---|---:|---:|
| No reranker | 23.5 | 0.8317 |
| In database, N=10 | 671.2 | 0.8316 |
| Application, N=10 | 614.2 | 0.8324 |
| In database, N=20 | 1,290.2 | 0.8289 |
| Application, N=20 | 1,215.8 | 0.8294 |

The supported conclusion is that Oracle reranking can be much closer to application performance
than the old headline suggested, provided the models use separate sessions. Whether the
reranker itself is worth deploying is a separate question; this hybrid workload still offers
no demonstrated quality benefit for its added latency. A deployment decision needs the real
query distribution, latency budget and concurrent database load.

## Tooling and reporting repairs

- Candidate counts are measured in SQL before scoring and top-K truncation. An explicit
  marker lets the inspector distinguish new observations from legacy requested-depth counts.
- Query embedding time is included, its vector payload is counted, and embedding mode is saved.
- LENGTH controls show no relevance metrics in Markdown or CSV.
- Top-K membership differences are no longer described as necessarily substantive relevance
  disagreements; near-tied scores at the cutoff can cause them.
- The transfer batch is correctly labelled a **control latency difference**. Its statements
  differ in sorting, projection and returned row counts. It does not isolate network transfer
  cost, even if its confidence interval excludes zero. The old 0.56 ms claim should not be used
  as a precise estimate of the cost of moving text.
- Explain output distinguishes returned text from the full candidate pool and no longer treats
  row-source counts or cumulative plan time as direct scalar-inference measurements.

Validation: TypeScript check and all **70 tests** passed; live benchmark and diagnostic assertions
passed. `npm run bench` now uses the revised path by default. Set
`ORACLE_QUERY_EMBEDDING=inline` in the environment to reproduce the original architecture.
