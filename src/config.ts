import 'dotenv/config';
import { resolve } from 'node:path';
import type { RerankerId, RunConfig, Stage } from './types.js';

const num = (v: string | undefined, fallback: number): number => {
  if (v === undefined || v.trim() === '') return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`Expected a number, got ${JSON.stringify(v)}`);
  return n;
};

const list = (v: string | undefined, fallback: number[]): number[] =>
  v === undefined || v.trim() === '' ? fallback : v.split(',').map((s) => num(s.trim(), NaN));

export const oracle = {
  user: process.env.ORACLE_USER ?? '',
  password: process.env.ORACLE_PASSWORD ?? '',
  connectString: process.env.ORACLE_CONNECT_STRING ?? 'localhost:1521/FREEPDB1',
  /** Thin mode needs no Instant Client. Set ORACLE_CLIENT_LIB_DIR to force thick mode. */
  clientLibDir: process.env.ORACLE_CLIENT_LIB_DIR ?? '',
  poolMin: num(process.env.ORACLE_POOL_MIN, 1),
  poolMax: num(process.env.ORACLE_POOL_MAX, 4),
  /** Name of the ONNX embedding model loaded into the DB via DBMS_VECTOR.LOAD_ONNX_MODEL. */
  embedModel: process.env.ORACLE_EMBED_MODEL ?? 'DOC_EMBEDDER',
  /** Name of the ONNX cross-encoder loaded into the DB. */
  rerankModel: process.env.ORACLE_RERANK_MODEL ?? 'BGE_RERANKER',
  /** Dimensionality of the embedding model; must match the VECTOR column in sql/01_schema.sql. */
  embedDims: num(process.env.ORACLE_EMBED_DIMS, 384),
  /**
   * Which documented API to use for in-database reranking:
   *   'prediction'     -> PREDICTION(model USING :q AS FIRST_INPUT, text AS SECOND_INPUT)
   *   'utl_to_rerank'  -> DBMS_VECTOR.UTL_TO_RERANK(:q, json_of_docs, json('{"provider":"database",...}'))
   * Both are exercised by the same harness; see sql/README.md for the tradeoff.
   */
  indbRerankApi: (process.env.ORACLE_INDB_RERANK_API ?? 'prediction') as 'prediction' | 'utl_to_rerank',
  /**
   * The SQL expression that produces a relevance score for one candidate. Overridable because
   * how an exported cross-encoder surfaces in SQL depends on how it was loaded: a regression
   * export scores with PREDICTION(), a classification export needs PREDICTION_PROBABILITY().
   * `npm run doctor` checks which one your model answers to.
   */
  indbScoreExpr: process.env.ORACLE_INDB_SCORE_EXPR ?? '',
  schemaPrefix: process.env.ORACLE_SCHEMA_PREFIX ?? 'BENCH',
} as const;

export const app = {
  /**
   * Local directory holding the exported cross-encoder (config.json, tokenizer.json,
   * onnx/model.onnx). Produced by scripts/export-reranker-onnx.sh so that the app side
   * runs the *same weights* the database does.
   */
  modelPath: resolve(process.env.APP_RERANK_MODEL_PATH ?? './models/bge-reranker-base'),
  /** 'fp32' keeps parity with the in-database model. Quantized variants change the comparison. */
  dtype: process.env.APP_RERANK_DTYPE ?? 'fp32',
  device: process.env.APP_RERANK_DEVICE ?? 'cpu',
  /** ONNX Runtime intra-op threads. Recorded in the report because it moves the numbers a lot. */
  intraOpThreads: process.env.APP_RERANK_THREADS ? num(process.env.APP_RERANK_THREADS, 1) : undefined,
  maxLength: num(process.env.APP_RERANK_MAX_LENGTH, 512),
  batchSize: num(process.env.APP_RERANK_BATCH_SIZE, 16),
} as const;

export const paths = {
  corpus: resolve(process.env.CORPUS_PATH ?? './data/corpus.json'),
  queries: resolve(process.env.QUERIES_PATH ?? './data/queries.json'),
  results: resolve(process.env.RESULTS_DIR ?? './results'),
} as const;

export function runConfigFromEnv(overrides: Partial<RunConfig> = {}): RunConfig {
  const base: RunConfig = {
    backend: (process.env.BENCH_BACKEND ?? 'oracle') as RunConfig['backend'],
    queries: num(process.env.BENCH_QUERIES, 0), // 0 = all
    iterations: num(process.env.BENCH_ITERATIONS, 20),
    warmup: num(process.env.BENCH_WARMUP, 5),
    topK: num(process.env.BENCH_TOP_K, 10),
    candidateCounts: list(process.env.BENCH_CANDIDATES, [10, 20, 40, 80]),
    rerankers: (process.env.BENCH_RERANKERS ?? 'none,in-db,app').split(',') as RerankerId[],
    retrievals: (process.env.BENCH_RETRIEVALS ?? 'vector,lexical,hybrid-rrf').split(',') as RunConfig['retrievals'],
    rrfK: num(process.env.BENCH_RRF_K, 60),
    corpusSize: num(process.env.CORPUS_SIZE, 480),
    seed: num(process.env.CORPUS_SEED, 20250917),
  };
  return { ...base, ...overrides };
}

/**
 * Expand a run config into the concrete stages to measure.
 *
 * Unreranked retrieval is measured once per strategy (candidateCount == topK, because
 * without a reranker a deeper candidate list is just latency you throw away). Reranked
 * stages are measured across the candidate sweep, which is where the interesting curve is.
 */
export function buildStages(cfg: RunConfig): Stage[] {
  const stages: Stage[] = [];
  const rerankers = cfg.rerankers.filter((r) => r !== 'none');

  for (const retrieval of cfg.retrievals) {
    if (cfg.rerankers.includes('none')) {
      stages.push({
        id: `${retrieval}`,
        label: `${labelFor(retrieval)} (no rerank)`,
        retrieval,
        reranker: 'none',
        candidateCount: cfg.topK,
        topK: cfg.topK,
      });
    }
    // Reranking only earns its place on top of a real candidate pool, so sweep N here.
    for (const reranker of rerankers) {
      for (const n of cfg.candidateCounts) {
        if (n < cfg.topK) continue;
        stages.push({
          id: `${retrieval}+rerank-${reranker}@${n}`,
          label: `${labelFor(retrieval)} + ${labelFor(reranker)} rerank (N=${n})`,
          retrieval,
          reranker,
          candidateCount: n,
          topK: cfg.topK,
        });
      }
    }
  }
  return stages;
}

function labelFor(key: string): string {
  switch (key) {
    case 'vector': return 'Vector';
    case 'lexical': return 'Lexical';
    case 'hybrid-rrf': return 'Hybrid RRF';
    case 'in-db': return 'in-database';
    case 'app': return 'application';
    case 'fixture': return 'fixture';
    default: return key;
  }
}
