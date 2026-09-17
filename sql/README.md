# The SQL, and which parts of it to double-check

Everything version-sensitive lives in this directory rather than being scattered through the
TypeScript, because the parts of Oracle this benchmark leans on are the parts most likely to
differ between your release and the one an article was written against.

Run `npm run bench -- --dump-sql` to print the exact statement every stage executes, with all
placeholders filled in.

## Files

| File | What it does |
|---|---|
| `00_user.sql` | Creates the benchmark user and its grants. Run once as ADMIN/SYS from SQLcl; the only script not run through `npm`. |
| `01_schema.sql` | The chunk table: text, metadata, and a `VECTOR` column in one row. Creates the Oracle Text index used by the lexical arm. |
| `02_vector_index.sql` | Optional approximate vector index. Not used by default (see below). |
| `03_load_models.sql` | Loads both models from a directory object (container). `npm run models`. |
| `03_load_models_adb.sql` | Loads both models from Object Storage (Autonomous). `ORACLE_TARGET=adb npm run models`. **Unverified — see below.** |
| `query_candidates.sql` | Scope filter → lexical + vector retrieval → RRF fusion. The candidate set both rerankers score. |
| `rerank_indb_prediction.sql` | The same pipeline with cross-encoder scoring appended, as one statement. Also the control: with `${SCORE_EXPR}` swapped for `LENGTH(:qtext \|\| TITLE \|\| '. ' \|\| CONTENT)` it does everything but inference. |
| `rerank_indb_utl.sql` | Alternative in-database path via `DBMS_VECTOR.UTL_TO_RERANK`. **Unverified — see below.** |
| `99_teardown.sql` | Drops the table. |

## The control statement

The cost of in-database scoring is calculated as treatment minus control, paired per query
and iteration (README, "How the reranking cost is calculated"). The control is
`rerank_indb_prediction.sql` rendered with the scoring expression replaced by
`LENGTH(:qtext || TITLE || '. ' || CONTENT)`. That expression is chosen so the control still
reads the same CLOB text the model reads, still binds `:qtext`, and is still evaluated for all
N rows (it is in the `ORDER BY`). Override with `ORACLE_INDB_CONTROL_EXPR` if your treatment
expression reads different columns — the two must touch the same data or the subtraction
measures I/O, not inference.

Before publishing, confirm the two statements plan identically up to the projection:

```sql
EXPLAIN PLAN FOR <treatment from --dump-sql>;   SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
EXPLAIN PLAN FOR <control from --dump-sql>;     SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

## Three things to verify before quoting a number

**1. The scoring expression.** `rerank_indb_prediction.sql` scores candidates with
`PREDICTION(model USING :qtext AS FIRST_INPUT, ... AS SECOND_INPUT)`. Whether that returns a
useful relevance score depends on how your ONNX export loaded: a regression export scores with
`PREDICTION()`, a classification export needs `PREDICTION_PROBABILITY()`. Ordering by a
predicted *class* ranks nothing, and it fails silently — every row gets the same value and the
tie-break decides the order. `npm run doctor` catches this by scoring an obviously-matching
pair against an obviously-irrelevant one.

Override with `ORACLE_INDB_SCORE_EXPR` if you need the probability form.

**2. `rerank_indb_utl.sql` and `03_load_models_adb.sql` were written without a live instance to check against.** The
signature of `DBMS_VECTOR.UTL_TO_RERANK`, its params JSON, and the shape of its JSON result
have moved between releases. It is included because it is the interesting API — the same call
routes to an in-database ONNX model or to Cohere, Vertex AI, or OCI Generative AI by changing
one field — but treat the template as a starting point, not as tested code. The default path
is `PREDICTION()`.

**3. `DBMS_HYBRID_VECTOR.SEARCH` is not used here, deliberately.** It has native RRF and
would be the idiomatic way to write hybrid retrieval. This benchmark spells the fusion out by
hand so that the application-side path and the in-database path provably fuse identically —
if fusion were a black box on one side only, a ranking difference could be fusion rather than
the reranker. If you want to measure the built-in fusion, that is a separate and worthwhile
experiment; do not fold it into this one.

Note also that `DBMS_HYBRID_VECTOR`'s `search_fusion: "RERANK"` mode is a different thing from
the cross-encoder reranking measured here. It re-orders text-search results by their vector
score. Calling both "reranking" is the single easiest way to confuse a reader.

## Why exact vector search by default

`02_vector_index.sql` is not run unless you pass `--vector-index` to `npm run load`. An
approximate index introduces a recall/accuracy knob, and this benchmark is trying to isolate
one variable: where the cross-encoder runs. With exact search, both paths see identical
candidates and the parity check can prove it. Turn the index on when you want to measure
retrieval scaling — just don't change two things at once.

## Two exports of the same model

The application and the database both run `BAAI/bge-reranker-base`, but they need it packaged
differently:

- **Application:** a plain ONNX export plus `tokenizer.json`. `scripts/export-reranker-onnx.sh`
  produces this with `optimum-cli`, and ONNX Runtime is fed pre-tokenized input.
- **Database:** an *augmented* export with the tokenizer embedded in the graph, because
  `PREDICTION()` is handed raw text and has to tokenize it itself. Oracle's OML4Py client
  provides the utility that produces this form.

Both must come from the same checkpoint. If they do not, the benchmark is comparing two models
and the latency numbers are meaningless. The report records the model name and path on each
side so a run can be audited after the fact.
