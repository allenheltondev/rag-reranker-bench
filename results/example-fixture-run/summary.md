# Reranker placement benchmark

Run started 2026-09-17T17:04:50.918Z, finished 2026-09-17T17:04:54.578Z.

> **This run used the fixture backend.** Retrieval is hashed TF-IDF and BM25 over a
> local corpus, and the "reranker" is a scoring function, not a model. These numbers
> validate the harness. They are not results and must not be quoted as any.

## Environment

| | |
|---|---|
| Node | v22.22.2 |
| Platform | linux/x64 |
| CPU | Intel(R) Xeon(R) Processor @ 2.10GHz (4 vCPU) |
| Memory | 16096 MB |
| App rerank model | (fixture reranker, no model) |
| App execution | fixture, dtype n/a, intra-op threads default |

Queries: 16 · iterations: 5 · warmup: 2 · repeats: 1 · top-K: 10 · RRF k: 60

Isolation batches, in order, with a reset between each: `baseline` → `app`. Quiesce 200 ms.

## Latency and quality by stage

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | nDCG@10 | Recall@10 | MRR@10 | Bytes from DB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector (no rerank) | 0.1 | 0.1 | 0.2 | 0.520 | 0.573 | 0.519 | 3.1 KB |
| Lexical (no rerank) | 0.3 | 0.4 | 0.6 | 0.657 | 0.740 | 0.638 | 2.8 KB |
| Hybrid RRF (no rerank) | 0.3 | 0.6 | 0.7 | 0.671 | 0.719 | 0.676 | 3.1 KB |
| Vector + application rerank (N=10) | 1.7 | 1.9 | 2.1 | 0.578 | 0.573 | 0.627 | 3.1 KB |
| Vector + application control, no scoring (N=10) | 0.1 | 0.2 | 0.2 | 0.520 | 0.573 | 0.519 | 3.1 KB |
| Vector + application rerank (N=80) | 9.1 | 13.1 | 13.3 | 0.666 | 0.740 | 0.693 | 17.5 KB |
| Vector + application control, no scoring (N=80) | 0.1 | 0.3 | 0.3 | 0.520 | 0.573 | 0.519 | 17.5 KB |
| Lexical + application rerank (N=10) | 1.9 | 2.1 | 2.2 | 0.682 | 0.740 | 0.702 | 2.8 KB |
| Lexical + application control, no scoring (N=10) | 0.3 | 0.5 | 0.6 | 0.657 | 0.740 | 0.638 | 2.8 KB |
| Lexical + application rerank (N=80) | 3.7 | 8.1 | 8.4 | 0.666 | 0.740 | 0.693 | 7.5 KB |
| Lexical + application control, no scoring (N=80) | 0.3 | 0.5 | 0.6 | 0.657 | 0.740 | 0.638 | 7.5 KB |
| Hybrid RRF + application rerank (N=10) | 2.0 | 2.2 | 2.3 | 0.680 | 0.719 | 0.710 | 3.1 KB |
| Hybrid RRF + application control, no scoring (N=10) | 0.4 | 0.7 | 0.7 | 0.671 | 0.719 | 0.676 | 3.1 KB |
| Hybrid RRF + application rerank (N=80) | 9.5 | 13.4 | 13.8 | 0.666 | 0.740 | 0.693 | 17.6 KB |
| Hybrid RRF + application control, no scoring (N=80) | 0.5 | 0.7 | 0.9 | 0.687 | 0.740 | 0.688 | 17.6 KB |

## Scope violations

No stage returned a row belonging to another tenant, another user, or an expired fact. 16 queries x 15 stages checked.

## Cost of scoring as the candidate pool grows

Each row is a treatment minus its control. The control is the same pipeline at the same
depth with the scoring step removed and nothing else changed, run interleaved with the
treatment. "Scoring" is the median of the per-observation differences, with a 95%
bootstrap interval. Quality deltas are against the no-rerank baseline at top-K.

| Retrieval | N | Where | Control p50 (ms) | Treatment p50 (ms) | Scoring, median Δ (ms) | 95% CI | Pairs | nDCG@10 | Δ nDCG |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| vector | 10 | application | 0.1 | 1.7 | **1.6** | [1.6, 1.6] | 80 | 0.578 | +0.057 |
| vector | 80 | application | 0.1 | 9.1 | **8.9** | [8.6, 9.6] | 80 | 0.666 | +0.145 |
| lexical | 10 | application | 0.3 | 1.9 | **1.6** | [1.6, 1.6] | 80 | 0.682 | +0.025 |
| lexical | 80 | application | 0.3 | 3.7 | **3.4** | [2.5, 3.7] | 80 | 0.666 | +0.009 |
| hybrid-rrf | 10 | application | 0.4 | 2.0 | **1.6** | [1.6, 1.6] | 80 | 0.680 | +0.008 |
| hybrid-rrf | 80 | application | 0.5 | 9.5 | **9.0** | [8.5, 9.5] | 80 | 0.666 | -0.005 |

### Is subtraction a valid way to measure this?

The application path can be timed both ways: by subtraction, exactly as the in-database
path has to be, and directly with clocks around tokenize, infer and sort. If the two
agree, the subtraction method is sound and the in-database figures above can be trusted
to the same degree. If they do not, the gap is measurement error and it applies to every
subtracted number in this report.

| Retrieval | N | By subtraction (ms) | Measured directly (ms) | Gap (ms) | Gap as % of direct |
|---|---:|---:|---:|---:|---:|
| vector | 10 | 1.61 | 1.61 | -0.00 | -0.0% |
| vector | 80 | 8.91 | 8.95 | -0.04 | -0.5% |
| lexical | 10 | 1.59 | 1.61 | -0.01 | -0.8% |
| lexical | 80 | 3.35 | 3.36 | -0.01 | -0.2% |
| hybrid-rrf | 10 | 1.60 | 1.61 | -0.01 | -0.8% |
| hybrid-rrf | 80 | 8.97 | 9.01 | -0.04 | -0.4% |

## Where the application path spends its time

The in-database path is a single statement and cannot be broken down from outside the
database, so it is absent from this table by construction rather than by omission.

| Stage | Candidates p50 (ms) | Tokenize p50 (ms) | Infer p50 (ms) | Sort p50 (ms) | Total p50 (ms) |
|---|---:|---:|---:|---:|---:|
| Vector + application rerank (N=10) | 0.1 | 0.1 | 1.5 | 0.01 | 1.7 |
| Vector + application rerank (N=80) | 0.1 | 0.3 | 8.6 | 0.02 | 9.1 |
| Lexical + application rerank (N=10) | 0.3 | 0.1 | 1.5 | 0.00 | 1.9 |
| Lexical + application rerank (N=80) | 0.3 | 0.1 | 3.2 | 0.01 | 3.7 |
| Hybrid RRF + application rerank (N=10) | 0.4 | 0.1 | 1.5 | 0.00 | 2.0 |
| Hybrid RRF + application rerank (N=80) | 0.4 | 0.3 | 8.7 | 0.02 | 9.5 |

## Per-query detail

Where the best-graded chunk ended up, per stage. This is where a reranker earning or
not earning its place becomes visible on individual queries rather than in an average.

| Query | Kind | vector | lexical | hybrid-rrf | vector+rerank-app@10 | vector+rerank-app@80 | lexical+rerank-app@10 | lexical+rerank-app@80 | hybrid-rrf+rerank-app@10 | hybrid-rrf+rerank-app@80 |
|---|---|---|---|---|---|---|---|---|---|---|
| q01 | exact-identifier | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q02 | exact-identifier | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q03 | conceptual | — | — | — | — | — | — | — | — | — |
| q04 | conceptual | 9 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 |
| q05 | scoped | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q06 | scoped | — | 5 | 9 | — | 1 | 1 | 1 | 1 | 1 |
| q07 | mixed | 2 | 1 | 1 | 5 | 8 | 7 | 8 | 6 | 8 |
| q08 | mixed | 2 | 3 | 2 | 4 | 5 | 5 | 5 | 4 | 5 |
| q09 | mixed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q10 | exact-identifier | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q11 | scoped | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q12 | conceptual | — | 3 | 6 | — | 10 | 7 | 10 | 6 | 10 |
| q13 | mixed | 2 | 2 | 2 | 3 | 3 | 3 | 3 | 3 | 3 |
| q14 | scoped | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q15 | conceptual | 2 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| q16 | mixed | — | 3 | 5 | — | — | 9 | — | 5 | — |

`—` means the best-graded chunk for that query never reached the context window.

## Stability

Run-to-run noise, measured within each query: the coefficient of variation of a query's
end-to-end time across iterations, summarised across queries. It is computed this way so
that queries which legitimately cost different amounts do not read as instability. Stages
under a millisecond will show large percentages from timer resolution alone; read those in
absolute terms.

| Stage | p50 (ms) | Median within-query CV | Worst query | Its CV |
|---|---:|---:|---|---:|
| vector | 0.1 | 12.6% | q15 | 28.5% |
| lexical | 0.3 | 10.6% | q06 | 49.3% |
| hybrid-rrf | 0.3 | 10.5% | q10 | 40.6% |
| vector+rerank-app@10 | 1.7 | 2.8% | q09 | 10.1% |
| vector+control-app@10 | 0.1 | 11.1% | q12 | 76.2% |
| vector+rerank-app@80 | 9.1 | 1.6% | q01 | 3.9% |
| vector+control-app@80 | 0.1 | 19.9% | q10 | 80.3% |
| lexical+rerank-app@10 | 1.9 | 1.9% | q14 | 9.5% |
| lexical+control-app@10 | 0.3 | 8.5% | q09 | 47.4% |
| lexical+rerank-app@80 | 3.7 | 1.8% | q04 | 14.9% |
| lexical+control-app@80 | 0.3 | 9.5% | q14 | 37.1% |
| hybrid-rrf+rerank-app@10 | 2.0 | 2.6% | q09 | 7.6% |
| hybrid-rrf+control-app@10 | 0.4 | 9.6% | q14 | 37.5% |
| hybrid-rrf+rerank-app@80 | 9.5 | 1.5% | q01 | 6.5% |
| hybrid-rrf+control-app@80 | 0.5 | 7.0% | q05 | 30.1% |

No stage over a millisecond exceeds 25% within-query variation; the intervals above are as tight as the iteration count allows.
