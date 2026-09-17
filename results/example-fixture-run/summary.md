# Reranker placement benchmark

Run started 2026-09-17T16:40:56.996Z, finished 2026-09-17T16:41:01.907Z.

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

Queries: 16 · iterations: 4 · warmup: 1 · repeats: 2 · top-K: 10 · RRF k: 60

Isolation batches, in order, with a reset between each: `baseline` → `app`. Quiesce 300 ms.

## Latency and quality by stage

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | nDCG@10 | Recall@10 | MRR@10 | Bytes from DB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector (no rerank) | 0.1 | 0.2 | 0.2 | 0.520 | 0.573 | 0.519 | 3.1 KB |
| Lexical (no rerank) | 0.3 | 0.6 | 1.0 | 0.657 | 0.740 | 0.638 | 2.8 KB |
| Hybrid RRF (no rerank) | 0.4 | 0.6 | 1.0 | 0.671 | 0.719 | 0.676 | 3.1 KB |
| Vector + application rerank (N=10) | 1.7 | 2.0 | 2.1 | 0.578 | 0.573 | 0.627 | 3.1 KB |
| Vector + application control, no scoring (N=10) | 0.1 | 0.2 | 0.3 | 0.520 | 0.573 | 0.519 | 3.1 KB |
| Vector + application rerank (N=40) | 6.5 | 6.8 | 6.9 | 0.669 | 0.740 | 0.696 | 11.8 KB |
| Vector + application control, no scoring (N=40) | 0.1 | 0.2 | 0.4 | 0.520 | 0.573 | 0.519 | 11.8 KB |
| Lexical + application rerank (N=10) | 1.9 | 2.1 | 2.3 | 0.682 | 0.740 | 0.702 | 2.8 KB |
| Lexical + application control, no scoring (N=10) | 0.3 | 0.5 | 0.7 | 0.657 | 0.740 | 0.638 | 2.8 KB |
| Lexical + application rerank (N=40) | 3.6 | 6.9 | 7.2 | 0.666 | 0.740 | 0.693 | 7.0 KB |
| Lexical + application control, no scoring (N=40) | 0.3 | 0.5 | 0.5 | 0.657 | 0.740 | 0.638 | 7.0 KB |
| Hybrid RRF + application rerank (N=10) | 2.0 | 2.3 | 2.4 | 0.680 | 0.719 | 0.710 | 3.1 KB |
| Hybrid RRF + application control, no scoring (N=10) | 0.4 | 0.6 | 0.8 | 0.671 | 0.719 | 0.676 | 3.1 KB |
| Hybrid RRF + application rerank (N=40) | 6.8 | 7.1 | 7.2 | 0.666 | 0.740 | 0.693 | 11.8 KB |
| Hybrid RRF + application control, no scoring (N=40) | 0.4 | 0.6 | 0.7 | 0.687 | 0.740 | 0.688 | 11.8 KB |

## Scope violations

No stage returned a row belonging to another tenant, another user, or an expired fact. 16 queries x 15 stages checked.

## Cost of scoring as the candidate pool grows

Each row is a treatment minus its control. The control is the same pipeline at the same
depth with the scoring step removed and nothing else changed, run interleaved with the
treatment. "Scoring" is the median of the per-observation differences, with a 95%
bootstrap interval. Quality deltas are against the no-rerank baseline at top-K.

| Retrieval | N | Where | Control p50 (ms) | Treatment p50 (ms) | Scoring, median Δ (ms) | 95% CI | Pairs | nDCG@10 | Δ nDCG |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| vector | 10 | application | 0.1 | 1.7 | **1.6** | [1.6, 1.6] | 128 | 0.578 | +0.057 |
| vector | 40 | application | 0.1 | 6.5 | **6.4** | [6.4, 6.4] | 128 | 0.669 | +0.148 |
| lexical | 10 | application | 0.3 | 1.9 | **1.6** | [1.6, 1.6] | 128 | 0.682 | +0.025 |
| lexical | 40 | application | 0.3 | 3.6 | **3.3** | [2.6, 3.7] | 128 | 0.666 | +0.009 |
| hybrid-rrf | 10 | application | 0.4 | 2.0 | **1.6** | [1.6, 1.6] | 128 | 0.680 | +0.008 |
| hybrid-rrf | 40 | application | 0.4 | 6.8 | **6.4** | [6.4, 6.4] | 128 | 0.666 | -0.005 |

### Is subtraction a valid way to measure this?

The application path can be timed both ways: by subtraction, exactly as the in-database
path has to be, and directly with clocks around tokenize, infer and sort. If the two
agree, the subtraction method is sound and the in-database figures above can be trusted
to the same degree. If they do not, the gap is measurement error and it applies to every
subtracted number in this report.

| Retrieval | N | By subtraction (ms) | Measured directly (ms) | Gap (ms) | Gap as % of direct |
|---|---:|---:|---:|---:|---:|
| vector | 10 | 1.61 | 1.62 | -0.01 | -0.5% |
| vector | 40 | 6.39 | 6.41 | -0.02 | -0.3% |
| lexical | 10 | 1.58 | 1.60 | -0.02 | -1.0% |
| lexical | 40 | 3.33 | 3.35 | -0.02 | -0.6% |
| hybrid-rrf | 10 | 1.61 | 1.61 | -0.01 | -0.4% |
| hybrid-rrf | 40 | 6.37 | 6.40 | -0.04 | -0.6% |

## Where the application path spends its time

The in-database path is a single statement and cannot be broken down from outside the
database, so it is absent from this table by construction rather than by omission.

| Stage | Candidates (ms) | Tokenize (ms) | Infer (ms) | Sort (ms) | Total p50 (ms) |
|---|---:|---:|---:|---:|---:|
| Vector + application rerank (N=10) | 0.1 | 0.1 | 1.6 | 0.00 | 1.7 |
| Vector + application rerank (N=40) | 0.1 | 0.2 | 5.8 | 0.01 | 6.5 |
| Lexical + application rerank (N=10) | 0.3 | 0.1 | 1.5 | 0.00 | 1.9 |
| Lexical + application rerank (N=40) | 0.3 | 0.1 | 3.5 | 0.01 | 3.6 |
| Hybrid RRF + application rerank (N=10) | 0.4 | 0.1 | 1.6 | 0.00 | 2.0 |
| Hybrid RRF + application rerank (N=40) | 0.4 | 0.2 | 5.8 | 0.01 | 6.8 |

## Per-query detail

Where the best-graded chunk ended up, per stage. This is where a reranker earning or
not earning its place becomes visible on individual queries rather than in an average.

| Query | Kind | vector | lexical | hybrid-rrf | vector+rerank-app@10 | vector+rerank-app@40 | lexical+rerank-app@10 | lexical+rerank-app@40 | hybrid-rrf+rerank-app@10 | hybrid-rrf+rerank-app@40 |
|---|---|---|---|---|---|---|---|---|---|---|
| q01 | exact-identifier | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q02 | exact-identifier | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q03 | conceptual | — | — | — | — | — | — | — | — | — |
| q04 | conceptual | 9 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 |
| q05 | scoped | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q06 | scoped | — | 5 | 9 | — | 1 | 1 | 1 | 1 | 1 |
| q07 | mixed | 2 | 1 | 1 | 5 | 6 | 7 | 8 | 6 | 8 |
| q08 | mixed | 2 | 3 | 2 | 4 | 5 | 5 | 5 | 4 | 5 |
| q09 | mixed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q10 | exact-identifier | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q11 | scoped | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q12 | conceptual | — | 3 | 6 | — | 9 | 7 | 10 | 6 | 10 |
| q13 | mixed | 2 | 2 | 2 | 3 | 3 | 3 | 3 | 3 | 3 |
| q14 | scoped | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| q15 | conceptual | 2 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| q16 | mixed | — | 3 | 5 | — | — | 9 | — | 5 | — |

`—` means the best-graded chunk for that query never reached the context window.

## Repeatability across repeats

The scoring cost recomputed from each full repetition of the protocol on its own. The
spread between repeats is what the number would do if you ran the benchmark again.

| Retrieval | N | Where | Repeat 1 (ms) | Repeat 2 (ms) | Spread (ms) | Spread % |
|---|---:|---|---:|---:|---:|---:|
| vector | 10 | application | 1.63 | 1.60 | 0.04 | 2.2% |
| vector | 40 | application | 6.41 | 6.35 | 0.05 | 0.8% |
| lexical | 10 | application | 1.61 | 1.57 | 0.03 | 2.1% |
| lexical | 40 | application | 3.32 | 3.35 | 0.03 | 0.9% |
| hybrid-rrf | 10 | application | 1.62 | 1.59 | 0.02 | 1.5% |
| hybrid-rrf | 40 | application | 6.36 | 6.38 | 0.02 | 0.3% |

## Stability

Coefficient of variation (stddev / mean) of the end-to-end time per stage. Highest: 63.1% on lexical.

These stages exceed 25% and their intervals should be read with that in mind — more iterations, or a quieter machine, will tighten them:

- `lexical`: 63.1%
- `hybrid-rrf`: 32.0%
- `vector+control-app@10`: 25.8%
- `vector+control-app@40`: 28.6%
- `lexical+control-app@10`: 26.9%
- `lexical+rerank-app@40`: 49.1%
