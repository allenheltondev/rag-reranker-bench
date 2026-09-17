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
2. **The cost of scoring is isolated by subtraction against a matched control,** measured
   pairwise. See the next section — this is the number the article is about, and it is
   calculated, not read off a clock.
3. **Quality is allowed to say no.** Every stage reports nDCG@10, Recall@10 and MRR@10 against
   graded judgments, plus a per-query table of where the best chunk landed. If reranking makes
   a query worse, that shows up as a row, not as a rounding error inside an average.
4. **Bytes leaving the database is a first-class metric,** because it is the architectural
   difference, not a footnote. The in-database path returns identifiers and scores. The
   application path returns every candidate's full text so it can be scored.
5. **Nothing is reported more precisely than it was measured.** The in-database path is one
   SQL statement and cannot be decomposed from outside the database, so it reports a total and
   no sub-phases. Its scoring cost comes from the subtraction below, with an interval.

### How the reranking cost is calculated

The in-database path is one statement. There is no point between "query sent" and "rows back"
where application code can read a clock, so the cost of the cross-encoder cannot be observed
directly. It can be calculated, if the calculation is set up carefully. Three things make it
technically sound:

**A matched control.** For every treatment at depth N there is a control: the *same
statement* with the cross-encoder expression replaced by a cheap one that reads the same text
(`LENGTH(:qtext || TITLE || '. ' || CONTENT)`). Same scope filter, same retrieval arms, same
fusion, same N-row sort, same identifier-and-score projection, same binds. The only work
removed is inference. `npm run bench -- --dump-sql` prints both so the claim can be checked
by diffing them, and a unit test asserts they differ in that expression alone.

The application path gets the same treatment: its control is the candidate fetch at depth N
with no scoring, which is exactly what the app-side pipeline does before it scores.

**Paired, interleaved measurement — within an arm.** A treatment and its control are not run
as separate batches. Within each iteration every query goes through the treatment and the
control back to back, and the order rotates between iterations. Each observation of the
treatment therefore has a partner observation of the control taken moments earlier under
the same load, cache state and clock speed. The quantity reported is

```
scoring(N) = median over (query, repeat, iteration) of [ T_treatment − T_control ]
```

which is the median of per-observation differences. It is **not** `p50(treatment) −
p50(control)`; that is a different quantity, and drift over a run biases it in whichever
direction the drift happened to go. A seeded bootstrap (2,000 resamples) gives a 95%
interval on the median, so re-rendering a report from the same raw data reproduces the
interval exactly.

**Isolation — between arms.** Nothing is subtracted across the in-database and application
arms, so they do not interleave. They run as separate batches with a full reset between:
connection pool closed, model disposed, heap collected, an optional shell command (a
container restart, if you want a true teardown), then a quiesce period. Every unit warms up
again afterwards. Batch order is `baseline → in-db → app → transfer`. What this removes is
carryover: ONNX Runtime threads spin-wait briefly after inference, the database's own
runtime does likewise, CPU boost state and buffer caches persist. None of that can now land
in the other arm's measurements.

**Repeats.** `--repeats R` runs the entire protocol R times, resets included, and the report
recomputes every scoring cost from each repeat alone. The spread between repeats is the
repeatability figure — what the number would do if you ran the benchmark again.

**A built-in validity check.** The application path can be measured *both* ways: by
subtraction, exactly as the in-database path has to be, and directly, with clocks around
tokenize, infer and sort. The report prints both side by side. If the method is sound the
two agree, and whatever gap exists is the measurement error to apply to every subtracted
number in the report. On the fixture backend they agree within 1.5%.

The same machinery gives one more calculable number. The `transfer` batch holds only the two
controls, run interleaved — they do identical work up to the projection (one returns
identifiers and a number, the other identifiers and every candidate's full text) and neither
runs inference, so there is nothing to carry over between them. Their paired difference is the
cost of moving the text out of the database and nothing else:

```
transfer(N) = median over (query, repeat, iteration) of [ T_app_control − T_indb_control ]
```

What this does not remove: the treatment and control are different SQL text, so they are
different cursors and could in principle get different plans. They should not — the plan is
identical up to the final projection — but for a result you intend to publish, confirm it
with `EXPLAIN PLAN` on both statements from `--dump-sql`.

### Metrics collected

| Metric | Why it is here |
|---|---|
| p50 / p95 / p99 end-to-end | The number a user feels. p95 matters more than p50 for a reranker. |
| Scoring cost, median Δ with 95% CI | Treatment minus matched control, paired. Isolates inference from everything else. |
| Transfer cost, median Δ with 95% CI | App control minus in-DB control, paired. The cost of the text leaving the database. |
| Subtraction vs direct, app path | Validates the subtraction method against clocks where clocks exist. |
| CV per stage | Run-to-run stability. Flags stages whose intervals should not be trusted yet. |
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

Two targets. **Local** (Oracle Free in a container, app on the same machine) proves the
pipeline and gives you the compute comparison. **Oracle Cloud** (Autonomous Database plus a
separate VM, [`infra/oci`](infra/oci/README.md)) is what a published number should come from:
with the database and the application on different hosts, the "bytes leaving the database"
measurement crosses a real network instead of a loopback interface, and the locality argument
becomes something the transfer batch can actually measure. The steps below are the local
path; the cloud README maps each one onto its equivalent.

### 1. A database

```bash
cp .env.example .env          # set ORACLE_SYS_PASSWORD and ORACLE_PASSWORD
docker compose up -d
docker compose logs -f oracle # wait for "DATABASE IS READY TO USE!" (several minutes first time)
```

Check the image tag: in-database reranking needs a release that can load an ONNX reranking
model. The report records the version you actually ran.

### 2. The user

```bash
npm run bootstrap
```

Creates the benchmark user with the password from `.env`, grants it what the rest of the repo
needs, and points the `ONNX_DIR` directory object at the mounted `models/oracle` folder. It is
the only command that uses `ORACLE_SYS_PASSWORD`, and it is safe to re-run — an existing user
has its password reset to match `.env` rather than erroring.

### 3. The models

Both sides must run the same checkpoint, packaged two ways — a plain ONNX export for the
application, and an augmented export with the tokenizer in the graph for the database. See
[`sql/README.md`](sql/README.md).

```bash
scripts/export-reranker-onnx.sh          # application copy -> ./models/bge-reranker-base
```

The two database copies go in `./models/oracle` (mounted into the container at
`/opt/oracle/onnx`). They load independently, which matters because they are not equally
easy to obtain:

```bash
npm run models -- --only embed     # the embedding model
npm run models -- --only rerank    # the augmented cross-encoder
npm run models                     # both
```

The **embedding model** is a plain load of a prepared ONNX file. The **cross-encoder** must be
the augmented export with the tokenizer inside the graph, because `PREDICTION()` hands it raw
text — the `optimum` export above is not that, and Oracle's OML4Py utility is what produces it.

You do not have to wait for the cross-encoder to get real numbers out of Oracle. With only the
embedding model loaded, retrieval, fusion, metadata filtering and application-side reranking
all run against the database:

```bash
npm run load
npm run doctor                                  # reports the reranker as skipped, not failed
npm run bench -- --rerankers none,app
```

That gives you genuine Oracle retrieval quality and the application arm's cost. Adding the
cross-encoder later unlocks the in-database arm and the transfer comparison, with no other
changes.

### 4. Load and check

```bash
npm run load
npm run doctor
```

`doctor` checks each moving part separately and names the one that is not ready. It is worth
running before every session; the failure it most often catches is the scoring expression
silently returning a class instead of a score.

### 5. Run

```bash
npm run bench
```

Results land in `results/<timestamp>/` as `raw.json`, `summary.md`, `summary.csv` and
`scoring-costs.csv`. Re-rendering a report from `raw.json` reproduces every number,
intervals included.

```bash
npm run bench -- --retrievals hybrid-rrf --candidates 10,20,40,80 --iterations 30
npm run bench -- --repeats 3       # full protocol three times; report shows between-repeat spread
npm run bench -- --reset-cmd "docker compose restart oracle && sleep 90"   # true teardown between arms
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
