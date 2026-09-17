# Reranker placement benchmark

Run started 2026-09-17T16:10:04.155Z, finished 2026-09-17T16:10:07.767Z.

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

Queries: 16 · iterations: 3 · warmup: 1 · top-K: 10 · RRF k: 60

## Latency and quality by stage

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | nDCG@10 | Recall@10 | MRR@10 | Bytes from DB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector (no rerank) | 0.1 | 0.2 | 0.2 | 0.520 | 0.573 | 0.519 | 3.1 KB |
| Vector + application rerank (N=10) | 1.8 | 2.1 | 2.3 | 0.578 | 0.573 | 0.627 | 3.1 KB |
| Vector + application rerank (N=20) | 3.4 | 3.6 | 3.7 | 0.661 | 0.688 | 0.698 | 6.1 KB |
| Vector + application rerank (N=40) | 6.6 | 7.0 | 7.0 | 0.669 | 0.740 | 0.696 | 11.8 KB |
| Vector + application rerank (N=80) | 9.3 | 13.2 | 13.4 | 0.666 | 0.740 | 0.693 | 17.5 KB |
| Lexical (no rerank) | 0.3 | 0.5 | 0.8 | 0.657 | 0.740 | 0.638 | 2.8 KB |
| Lexical + application rerank (N=10) | 2.0 | 2.3 | 2.3 | 0.682 | 0.740 | 0.702 | 2.8 KB |
| Lexical + application rerank (N=20) | 3.5 | 3.8 | 3.9 | 0.666 | 0.740 | 0.693 | 4.9 KB |
| Lexical + application rerank (N=40) | 3.7 | 6.8 | 6.9 | 0.666 | 0.740 | 0.693 | 7.0 KB |
| Lexical + application rerank (N=80) | 3.7 | 8.0 | 8.5 | 0.666 | 0.740 | 0.693 | 7.5 KB |
| Hybrid RRF (no rerank) | 0.4 | 0.6 | 0.9 | 0.671 | 0.719 | 0.676 | 3.1 KB |
| Hybrid RRF + application rerank (N=10) | 2.0 | 2.3 | 2.5 | 0.680 | 0.719 | 0.710 | 3.1 KB |
| Hybrid RRF + application rerank (N=20) | 3.6 | 3.8 | 3.9 | 0.677 | 0.740 | 0.700 | 6.1 KB |
| Hybrid RRF + application rerank (N=40) | 6.9 | 7.2 | 7.3 | 0.666 | 0.740 | 0.693 | 11.8 KB |
| Hybrid RRF + application rerank (N=80) | 9.4 | 13.4 | 13.4 | 0.666 | 0.740 | 0.693 | 17.6 KB |

## Scope violations

No stage returned a row belonging to another tenant, another user, or an expired fact. 16 queries x 15 stages checked.

## Cost of reranking as the candidate pool grows

Added latency is measured against the same retrieval strategy with no reranking, so it
isolates the scoring stage from candidate generation.

| Retrieval | N | Where | p50 (ms) | p95 (ms) | Added p50 vs no rerank | nDCG@10 | Δ nDCG |
|---|---:|---|---:|---:|---:|---:|---:|
| vector | 10 | application | 1.8 | 2.1 | 1.6 | 0.578 | +0.057 |
| vector | 20 | application | 3.4 | 3.6 | 3.2 | 0.661 | +0.141 |
| vector | 40 | application | 6.6 | 7.0 | 6.4 | 0.669 | +0.148 |
| vector | 80 | application | 9.3 | 13.2 | 9.1 | 0.666 | +0.145 |
| lexical | 10 | application | 2.0 | 2.3 | 1.7 | 0.682 | +0.025 |
| lexical | 20 | application | 3.5 | 3.8 | 3.2 | 0.666 | +0.009 |
| lexical | 40 | application | 3.7 | 6.8 | 3.4 | 0.666 | +0.009 |
| lexical | 80 | application | 3.7 | 8.0 | 3.3 | 0.666 | +0.009 |
| hybrid-rrf | 10 | application | 2.0 | 2.3 | 1.7 | 0.680 | +0.008 |
| hybrid-rrf | 20 | application | 3.6 | 3.8 | 3.3 | 0.677 | +0.006 |
| hybrid-rrf | 40 | application | 6.9 | 7.2 | 6.5 | 0.666 | -0.005 |
| hybrid-rrf | 80 | application | 9.4 | 13.4 | 9.1 | 0.666 | -0.005 |

## Where the application path spends its time

The in-database path is a single statement and cannot be broken down from outside the
database, so it is absent from this table by construction rather than by omission.

| Stage | Candidates (ms) | Tokenize (ms) | Infer (ms) | Sort (ms) | Total p50 (ms) |
|---|---:|---:|---:|---:|---:|
| Vector + application rerank (N=10) | 0.1 | 0.1 | 1.6 | 0.01 | 1.8 |
| Vector + application rerank (N=20) | 0.1 | 0.1 | 3.1 | 0.01 | 3.4 |
| Vector + application rerank (N=40) | 0.2 | 0.2 | 5.8 | 0.01 | 6.6 |
| Vector + application rerank (N=80) | 0.2 | 0.3 | 8.5 | 0.02 | 9.3 |
| Lexical + application rerank (N=10) | 0.4 | 0.1 | 1.5 | 0.01 | 2.0 |
| Lexical + application rerank (N=20) | 0.4 | 0.1 | 2.5 | 0.01 | 3.5 |
| Lexical + application rerank (N=40) | 0.4 | 0.2 | 3.4 | 0.01 | 3.7 |
| Lexical + application rerank (N=80) | 0.4 | 0.1 | 3.6 | 0.01 | 3.7 |
| Hybrid RRF + application rerank (N=10) | 0.4 | 0.1 | 1.6 | 0.00 | 2.0 |
| Hybrid RRF + application rerank (N=20) | 0.4 | 0.1 | 3.1 | 0.01 | 3.6 |
| Hybrid RRF + application rerank (N=40) | 0.5 | 0.2 | 5.8 | 0.02 | 6.9 |
| Hybrid RRF + application rerank (N=80) | 0.5 | 0.3 | 8.5 | 0.02 | 9.4 |

## Per-query detail

Where the best-graded chunk ended up, per stage. This is where a reranker earning or
not earning its place becomes visible on individual queries rather than in an average.

| Query | Kind | vector | vector+rerank-app@10 | vector+rerank-app@20 | vector+rerank-app@40 | vector+rerank-app@80 | lexical | lexical+rerank-app@10 | lexical+rerank-app@20 | lexical+rerank-app@40 | lexical+rerank-app@80 | hybrid-rrf | hybrid-rrf+rerank-app@10 | hybrid-rrf+rerank-app@20 | hybrid-rrf+rerank-app@40 | hybrid-rrf+rerank-app@80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q01 | exact-identifier | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q02 | exact-identifier | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q03 | conceptual | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| q04 | conceptual | 9 | 2 | 2 | 2 | 2 | 3 | 2 | 2 | 2 | 2 | 3 | 2 | 2 | 2 | 2 |
| q05 | scoped | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q06 | scoped | — | — | 1 | 1 | 1 | 5 | 1 | 1 | 1 | 1 | 9 | 1 | 1 | 1 | 1 |
| q07 | mixed | 2 | 5 | 6 | 6 | 8 | 1 | 7 | 8 | 8 | 8 | 1 | 6 | 8 | 8 | 8 |
| q08 | mixed | 2 | 4 | 5 | 5 | 5 | 3 | 5 | 5 | 5 | 5 | 2 | 4 | 5 | 5 | 5 |
| q09 | mixed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q10 | exact-identifier | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q11 | scoped | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q12 | conceptual | — | — | 6 | 9 | 10 | 3 | 7 | 10 | 10 | 10 | 6 | 6 | 7 | 10 | 10 |
| q13 | mixed | 2 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 |
| q14 | scoped | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q15 | conceptual | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 |
| q16 | mixed | — | — | — | — | — | 3 | 9 | — | — | — | 5 | 5 | 10 | — | — |

`—` means the best-graded chunk for that query never reached the context window.
