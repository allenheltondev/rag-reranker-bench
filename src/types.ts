/**
 * Shared vocabulary for the benchmark.
 *
 * The whole point of this repo is a controlled comparison, so the types are
 * deliberately built around one rule: candidate generation happens once per
 * (query, candidateCount) and the *identical* candidate list is handed to every
 * reranker under test. If a ranking changes, it changed because of the reranker.
 */

/** A chunk of text that can be retrieved. Metadata mirrors what an agent-memory store would carry. */
export interface Chunk {
  id: string;
  docId: string;
  title: string;
  content: string;
  /** Tenant the row belongs to. Retrieval filters on this the way row-level security would. */
  tenant: string;
  /** Owning user/agent, or null for tenant-wide knowledge. */
  owner: string | null;
  /** ISO date. */
  createdAt: string;
  /** ISO date, or null when the fact does not expire. Expired rows must never reach the model. */
  expiresAt: string | null;
  /** Free-form labels used by the corpus generator and for error analysis. */
  tags: string[];
}

/** Graded relevance judgment: 0 irrelevant, 1 marginal, 2 relevant, 3 the answer. */
export type Grade = 0 | 1 | 2 | 3;

export interface Query {
  id: string;
  text: string;
  /** Tenant/user issuing the query. Candidate generation filters to what this principal may see. */
  tenant: string;
  owner: string | null;
  /** Which retrieval failure mode this query is designed to stress. */
  kind: 'conceptual' | 'exact-identifier' | 'scoped' | 'mixed';
  /** chunkId -> grade. Chunks not listed are graded 0. */
  judgments: Record<string, Grade>;
  /** Human note explaining what makes the query hard. Surfaced in the report. */
  note?: string;
}

/** One candidate produced by retrieval, before any reranking. */
export interface Candidate {
  chunkId: string;
  title: string;
  content: string;
  /** Rank in the fused candidate list, 1-based. */
  rank: number;
  /** Fusion score (RRF) or raw retrieval score, depending on strategy. */
  score: number;
  /** Component ranks, when the candidate came from hybrid retrieval. */
  vectorRank?: number;
  lexicalRank?: number;
}

/** A reranked result: the same chunk, with a cross-encoder score and a new position. */
export interface RankedResult {
  chunkId: string;
  rank: number;
  score: number;
}

/** Where a reranker ran. */
export type RerankerId = 'none' | 'in-db' | 'app' | 'fixture';

/** Timing breakdown for a single measured iteration, in milliseconds. */
export interface Timings {
  /** Candidate generation: the hybrid/vector/lexical SQL, including fetch. */
  candidates: number;
  /**
   * Bytes-on-the-wire phase. Zero for in-database reranking by construction:
   * the candidate text never leaves the database.
   */
  transfer: number;
  /** Tokenization, app-side only. In-database this is inside `infer`. */
  tokenize: number;
  /** Cross-encoder scoring. */
  infer: number;
  /** Sorting and truncation to top-K. */
  sort: number;
  /** End to end, measured around the whole stage. Not necessarily the sum of the parts. */
  total: number;
}

export interface IterationResult {
  queryId: string;
  iteration: number;
  /** Which full repetition of the protocol this observation belongs to. */
  repeat: number;
  /**
   * Whether the timing breakdown is real or whether only the total is trustworthy.
   * In-database reranking is one statement, so its stages cannot be observed from outside;
   * reporting zeros for its sub-phases would invent precision that does not exist.
   */
  attribution: 'split' | 'total-only';
  timings: Timings;
  /** Final ordering handed to the model, truncated to topK. */
  results: RankedResult[];
  /** Bytes of candidate text that crossed the database boundary during this iteration. */
  bytesFromDb: number;
  /** Number of candidates actually scored. */
  candidatesScored: number;
  /** Absent on legacy files where the in-database count was only the requested limit. */
  candidateCountSource?: 'measured';
}

/**
 * A named pipeline configuration under test.
 *
 * Stages come in three roles. A baseline is plain retrieval at the depth you would deploy
 * without a reranker. A treatment is a reranked pipeline. A control is the treatment's exact
 * pipeline with the scoring step removed and nothing else changed, so that
 * treatment - control isolates the cost of scoring. Treatments and controls that share a
 * group are measured interleaved, and their differences are computed pairwise.
 */
export interface Stage {
  /** Stable id used in reports and JSON, e.g. "hybrid-rrf+rerank-in-db@40". */
  id: string;
  label: string;
  retrieval: 'vector' | 'lexical' | 'hybrid-rrf';
  reranker: RerankerId;
  /** Candidates generated and fed to the reranker. For baselines this equals topK. */
  candidateCount: number;
  /** Results returned to the model. */
  topK: number;
  role: 'baseline' | 'treatment' | 'control';
  /** Stages sharing a group are run interleaved so their timings are paired. */
  group?: string;
  /**
   * Isolation batch. Batches run one after another with a full reset between them (pool
   * closed, model disposed, quiesce, optional external command), so that nothing one arm
   * does can carry over into the other's measurements. Groups never span batches.
   */
  batch: string;
}

export interface StageRun {
  stage: Stage;
  iterations: IterationResult[];
}

export interface BenchRun {
  startedAt: string;
  finishedAt: string;
  config: RunConfig;
  environment: EnvironmentInfo;
  stages: StageRun[];
}

export interface RunConfig {
  backend: 'oracle' | 'fixture';
  queries: number;
  iterations: number;
  warmup: number;
  topK: number;
  candidateCounts: number[];
  rerankers: RerankerId[];
  retrievals: Array<Stage['retrieval']>;
  rrfK: number;
  corpusSize: number;
  seed: number;
  /** Full repetitions of the whole protocol, each with resets. Between-repeat spread is the repeatability figure. */
  repeats: number;
  /** Idle time after a reset before measuring resumes. */
  quiesceMs: number;
  /** Optional shell command run at each reset, e.g. a container restart. */
  resetCommand: string;
}

export interface EnvironmentInfo {
  node: string;
  platform: string;
  arch: string;
  cpus: number;
  cpuModel: string;
  totalMemMb: number;
  /** Populated when the Oracle backend is in use. */
  oracle?: {
    version: string;
    banner: string;
    clientMode: string;
    embedModel: string;
    queryEmbedding?: string;
    /** CPUs the database has, which on a container is its share and not the host's. */
    cpuCount: number | null;
    pgaTargetMb: number | null;
    pgaLimitMb: number | null;
    /** Present only when in-database reranking actually ran. */
    rerankModel?: string;
    indbRerankApi?: string;
  };
  /** Populated when the app reranker is in use. */
  app?: {
    modelPath: string;
    executionProviders: string[];
    intraOpThreads: number | 'default';
    dtype: string;
    batchSize?: number;
    maxLength?: number;
  };
}
