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
  /** Elevated credentials, used only by `npm run bootstrap` to create the benchmark user. */
  sysUser: process.env.ORACLE_SYS_USER ?? 'sys',
  sysPassword: process.env.ORACLE_SYS_PASSWORD ?? '',
  /** Tablespace the benchmark user gets quota on. DATA on Autonomous, USERS on the container. */
  tablespace: process.env.ORACLE_TABLESPACE ?? '',
  /** Filesystem path the ONNX directory object points at, inside the database host. */
  onnxPath: process.env.ORACLE_ONNX_PATH ?? '/opt/oracle/onnx',
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
  /**
   * The expression the control statement evaluates in place of the cross-encoder. It has to
   * read the same text the model would read, so that the paired difference is inference and
   * not CLOB access, and it has to reference :qtext so the statement binds identically.
   */
  indbControlExpr: process.env.ORACLE_INDB_CONTROL_EXPR ?? '',
  schemaPrefix: process.env.ORACLE_SCHEMA_PREFIX ?? 'BENCH',
  /** 'local' loads ONNX files from a directory object; 'adb' loads them from Object Storage. */
  target: (process.env.ORACLE_TARGET ?? 'local') as 'local' | 'adb',
  onnxDirectory: process.env.ORACLE_ONNX_DIRECTORY ?? 'ONNX_DIR',
  /** Pre-authenticated request base URL for the models bucket (infra/oci output). Must end with '/'. */
  modelsParUrl: process.env.ORACLE_MODELS_PAR_URL ?? '',
  embedFile: process.env.ORACLE_EMBED_FILE ?? 'all_MiniLM_L12_v2.onnx',
  /**
   * Instruction prefixes some embedding models require. E5 wants 'query: ' and 'passage: ';
   * BGE wants an instruction on the query only; MiniLM and GTE want neither. Getting this
   * wrong does not error - it quietly costs recall - so it is configuration rather than a
   * hardcoded assumption. Applied to the embedding model only; cross-encoders never use them.
   */
  embedQueryPrefix: process.env.ORACLE_EMBED_QUERY_PREFIX ?? '',
  embedDocPrefix: process.env.ORACLE_EMBED_DOC_PREFIX ?? '',
  rerankFile: process.env.ORACLE_RERANK_FILE ?? 'bge_reranker_base.onnx',
  /**
   * How LOAD_ONNX_MODEL maps the cross-encoder's graph inputs to SQL arguments.
   *
   * The default is the single-input form that scripts/augment_reranker_onnx.py produces,
   * where the query and passage are packed into one string in SQL. A model built with
   * Oracle's own two-input converter wants
   *   { "input": ["FIRST_INPUT", "SECOND_INPUT"] }
   * paired with the matching ORACLE_INDB_SCORE_EXPR.
   */
  rerankInputSpec: process.env.ORACLE_RERANK_INPUT_SPEC ?? '{ "input": ["DATA"] }',
} as const;

export const app = {
  /**
   * Local directory holding the exported cross-encoder (config.json, tokenizer.json,
   * onnx/model.onnx). Produced by `npm run export:app-model` so that the app side runs the
   * *same weights* the database does.
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
  /** Where the database-side ONNX files live. docker-compose mounts this into the container. */
  oracleModels: resolve(process.env.ORACLE_MODELS_DIR ?? './models/oracle'),
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
    repeats: num(process.env.BENCH_REPEATS, 1),
    quiesceMs: num(process.env.BENCH_QUIESCE_MS, 5000),
    resetCommand: process.env.BENCH_RESET_CMD ?? '',
  };
  return { ...base, ...overrides };
}

/**
 * Expand a run config into the concrete stages to measure.
 *
 * Per retrieval strategy: one baseline at top-K depth (what you would ship with no reranker),
 * and per candidate depth N, a treatment and a control for every reranker. The control is the
 * treatment with scoring removed, at the same depth, with the same projection, so that the
 * paired difference is the cost of scoring and nothing else.
 *
 * Stages are assigned to isolation batches by arm: everything in-database, then everything
 * application-side, then a transfer batch holding only the two controls. Pairs are interleaved
 * within a batch; batches are separated by a reset. The transfer controls are duplicates of
 * the arm controls on purpose - neither runs inference, so interleaving them carries nothing
 * over, and it keeps the app-minus-in-DB transfer number paired.
 */
export function buildStages(cfg: RunConfig): Stage[] {
  const stages: Stage[] = [];
  const rerankers = cfg.rerankers.filter((r) => r !== 'none');
  const depths = cfg.candidateCounts.filter((n) => n >= cfg.topK);

  for (const retrieval of cfg.retrievals) {
    if (cfg.rerankers.includes('none')) {
      stages.push({
        id: `${retrieval}`,
        label: `${labelFor(retrieval)} (no rerank)`,
        retrieval,
        reranker: 'none',
        candidateCount: cfg.topK,
        topK: cfg.topK,
        role: 'baseline',
        batch: 'baseline',
      });
    }
  }

  for (const reranker of rerankers) {
    for (const retrieval of cfg.retrievals) {
      for (const n of depths) {
        const group = `${reranker}:${retrieval}@${n}`;
        stages.push({
          id: `${retrieval}+rerank-${reranker}@${n}`,
          label: `${labelFor(retrieval)} + ${labelFor(reranker)} rerank (N=${n})`,
          retrieval,
          reranker,
          candidateCount: n,
          topK: cfg.topK,
          role: 'treatment',
          group,
          batch: reranker,
        });
        stages.push({
          id: `${retrieval}+control-${reranker}@${n}`,
          label: `${labelFor(retrieval)} + ${labelFor(reranker)} control, no scoring (N=${n})`,
          retrieval,
          reranker,
          candidateCount: n,
          topK: cfg.topK,
          role: 'control',
          group,
          batch: reranker,
        });
      }
    }
  }

  if (rerankers.includes('in-db') && rerankers.includes('app')) {
    for (const retrieval of cfg.retrievals) {
      for (const n of depths) {
        const group = `transfer:${retrieval}@${n}`;
        for (const reranker of ['in-db', 'app'] as const) {
          stages.push({
            id: `${retrieval}+transfer-${reranker}@${n}`,
            label: `${labelFor(retrieval)} + ${labelFor(reranker)} control for transfer (N=${n})`,
            retrieval,
            reranker,
            candidateCount: n,
            topK: cfg.topK,
            role: 'control',
            group,
            batch: 'transfer',
          });
        }
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
