# rag-reranker-bench

**Where should the cross-encoder run?**

A controlled benchmark comparing reranking inside Oracle AI Database against reranking in
application code — same model, same candidates, same queries, different execution location.

The question it exists to answer is not "is reranking good" and not "does Oracle win". It is:
*what does keeping the reranker next to the data actually cost, and what does it buy?*

---

## What is being measured

Retrieval is held constant. One statement does scope filtering, lexical retrieval, vector
retrieval and RRF fusion, and produces a candidate list. That identical candidate list is then
scored two ways:

```
                          ┌─ lexical ─┐
scope filter ── query ────┤           ├── RRF ──┬── in-database cross-encoder ──┐
                          └─ vector ──┘         │                               ├── top K
                                                └── candidates to the app ──────┘
                                                        └─ app-side cross-encoder
```

Both paths run `BAAI/bge-reranker-base`. The benchmark verifies before every run that both
paths saw byte-identical candidate sets; if they did not, it says so and the comparison is
void.

### What the run optimizes for

Fairness over flattery, and an honest tradeoff over a winner. Concretely:

1. **Only one variable changes.** Same model, same candidates, same fusion, same top-K. The
   candidate-parity check is run as a preflight and reported, so "we held retrieval constant"
   is a measurement rather than a claim.
2. **Added latency at equal quality is the headline number.** Not total pipeline latency,
   which is dominated by whatever else is in the stage. Every reranked stage is differenced
   against the same retrieval strategy with no reranking.
3. **Quality is allowed to say no.** Every stage reports nDCG@10, Recall@10 and MRR@10 against
   graded judgments, plus a per-query table of where the best chunk landed. If reranking makes
   a query worse, that shows up as a row, not as a rounding error inside an average.
4. **Bytes leaving the database is a first-class metric,** because it is the architectural
   difference, not a footnote. The in-database path returns identifiers and scores. The
   application path returns every candidate's full text so it can be scored.
5. **Nothing is reported more precisely than it was measured.** The in-database path is one
   SQL statement and cannot be decomposed from outside the database, so it reports a total and
   no sub-phases. Inventing a tokenize/infer split for it would be fiction.

### Metrics collected

| Metric | Why it is here |
|---|---|
| p50 / p95 / p99 end-to-end | The number a user feels. p95 matters more than p50 for a reranker. |
| Added p50 vs no rerank | Isolates the scoring stage from candidate generation. |
| nDCG@10, Recall@10, MRR@10 | Whether the reordering was worth paying for. |
| Bytes from DB per query | The architectural cost of scoring somewhere else. |
| Top-K Jaccard and Kendall tau | Do the two paths agree? Same model, same input — they should. |
| Scope violations | Rows from another tenant, another user, or an expired fact. Any non-zero value is a bug, not a quality score. |
| Candidate sweep (N = 10/20/40/80) | Cross-encoders score every pair independently, so this curve is the cost model. |

---

## Quick start, with no database

The fixture backend runs the entire harness locally with no Oracle and no model weights. Use
it to see the shape of the output and to check the harness works before setting anything up.

```bash
npm install
npm run corpus
npm run bench -- --backend fixture --iterations 5 --warmup 2 --candidates 10,40
```

> The fixture backend is a **self-test, not a simulation**. Its "vector" arm is hashed TF-IDF
> and its "reranker" is a scoring function. The numbers it produces validate the plumbing and
> mean nothing about retrieval. Every report from it is stamped to that effect.

## Running it for real

### 1. A database

```bash
cp .env.example .env          # fill in ORACLE_PASSWORD
docker compose up -d
docker compose logs -f oracle # wait for "DATABASE IS READY TO USE!"
```

Check the image tag: in-database reranking needs a release that can load an ONNX reranking
model. The report records the version you actually ran.

### 2. The models

Both sides must run the same checkpoint, packaged two ways — a plain ONNX export for the
application, and an augmented export with the tokenizer in the graph for the database. See
[`sql/README.md`](sql/README.md).

```bash
scripts/export-reranker-onnx.sh          # application copy -> ./models/bge-reranker-base
# put the augmented export and the embedding model in ./models/oracle, then:
# sqlplus> @sql/03_load_models.sql
```

### 3. Load and check

```bash
npm run load
npm run doctor
```

`doctor` checks each moving part separately and names the one that is not ready. It is worth
running before every session; the failure it most often catches is the scoring expression
silently returning a class instead of a score.

### 4. Run

```bash
npm run bench
```

Results land in `results/<timestamp>/` as `raw.json`, `summary.md` and `summary.csv`.

```bash
npm run bench -- --retrievals hybrid-rrf --candidates 10,20,40,80 --iterations 30
npm run bench -- --dump-sql        # print the exact SQL every stage runs
npm run report -- --run results/<timestamp>/raw.json
```

---

## The corpus

480 chunks of operational and agent-memory content for a fictional company, with 16 queries
carrying graded relevance judgments (0–3). Regenerate with `npm run corpus`; it is
deterministic for a seed.

The queries are built around four failure modes, because they fail differently and a reranker
helps with only some of them:

| Kind | What it stresses |
|---|---|
| `exact-identifier` | `INC-4821`, `ORA-00060`, `us-east-2`. Embeddings blur these; lexical nails them. |
| `conceptual` | The answer shares almost no vocabulary with the question. |
| `scoped` | The right text also exists for another tenant, another user, or with an expiry. Only metadata separates them. |
| `mixed` | A dense topical neighbourhood where fusion ranks the answer mid-pack. This is the reranker's case. |

Judgments are **derived by construction**: a chunk written to answer a query is graded 2–3, a
chunk written to be *almost* right is graded 0–1. That is honest only because the distractors
are real topical neighbours rather than filler — but it is still a synthetic corpus, and it
measures relative behaviour between pipelines, not absolute retrieval quality. Do not present
a number from it as a leaderboard result.

## Honest limitations

- **Synthetic corpus, construction-derived judgments.** See above. Swap `data/corpus.json` and
  `data/queries.json` for your own data in the same shape and the whole harness works on it.
- **One machine, one database.** A reranker on the database's CPUs competes with query traffic;
  an application reranker does not. A single-tenant benchmark on an idle box flatters the
  in-database path. Run it under load before drawing a capacity conclusion.
- **`bytesFromDb` is payload, not wire bytes.** It does not include protocol overhead or
  compression, and it is measured, not modelled.
- **The two paths are not required to agree perfectly.** Tokenizer truncation and float
  handling can differ. Disagreement is reported as a number to investigate, never as evidence
  that one side ranks better.
- **Thread counts move the numbers more than anything else here,** which is why the app-side
  execution provider, dtype and intra-op thread count are recorded in every report.

## Layout

```
data/       generated corpus and queries (committed, deterministic)
sql/        every statement the benchmark runs, with a README on the version-sensitive parts
scripts/    model export
src/
  corpus/   scenario definitions and the generator
  db/       connection, schema, loading, SQL template rendering
  retrieval/candidate generation (Oracle and fixture)
  rerank/   in-database, application, and fixture rerankers
  bench/    harness, metrics, reporting
test/       metrics, corpus invariants, SQL rendering
```

```bash
npm test        # unit tests, no database required
npm run typecheck
```
