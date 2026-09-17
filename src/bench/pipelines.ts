import type { IterationResult, Query, Stage, Timings } from '../types.js';
import type { CandidateSource } from '../retrieval/candidates.js';
import type { AppReranker } from '../rerank/app.js';
import type { FixtureReranker } from '../rerank/fixture.js';
import type { InDbReranker } from '../rerank/indb.js';

const zero = (): Timings => ({ candidates: 0, transfer: 0, tokenize: 0, infer: 0, sort: 0, total: 0 });

export interface Pipeline {
  readonly stage: Stage;
  run(query: Query): Promise<IterationResult>;
  /** The SQL this pipeline executes, for `--dump-sql`. Undefined for the fixture backend. */
  sql?(): string | undefined;
}

/** Retrieval with no reranking: the baseline every reranked stage has to beat to be worth it. */
export class RetrievalOnlyPipeline implements Pipeline {
  constructor(readonly stage: Stage, private readonly source: CandidateSource) {}

  sql(): string | undefined {
    return 'sqlFor' in this.source
      ? (this.source as { sqlFor(r: Stage['retrieval']): string }).sqlFor(this.stage.retrieval)
      : undefined;
  }

  async run(query: Query): Promise<IterationResult> {
    const started = performance.now();
    const batch = await this.source.generate(query, this.stage.retrieval, this.stage.candidateCount);
    const timings = zero();
    timings.candidates = batch.ms;
    timings.transfer = 0;
    timings.total = performance.now() - started;
    return {
      queryId: query.id,
      iteration: 0,
      attribution: 'split',
      timings,
      results: batch.candidates.slice(0, this.stage.topK).map((c, i) => ({
        chunkId: c.chunkId,
        rank: i + 1,
        score: c.score,
      })),
      bytesFromDb: batch.bytes,
      candidatesScored: 0,
    };
  }
}

/**
 * Retrieve in the database, rerank in the application.
 *
 * The candidate text is fetched across the connection before it can be scored, so this is the
 * pipeline where `bytesFromDb` is a real architectural cost rather than a rounding error.
 */
export class AppRerankPipeline implements Pipeline {
  constructor(
    readonly stage: Stage,
    private readonly source: CandidateSource,
    private readonly reranker: AppReranker | FixtureReranker,
  ) {}

  sql(): string | undefined {
    return 'sqlFor' in this.source
      ? (this.source as { sqlFor(r: Stage['retrieval']): string }).sqlFor(this.stage.retrieval)
      : undefined;
  }

  async run(query: Query): Promise<IterationResult> {
    const started = performance.now();
    const batch = await this.source.generate(query, this.stage.retrieval, this.stage.candidateCount);
    const outcome = await this.reranker.rerank(query.text, batch.candidates, this.stage.topK);
    const total = performance.now() - started;

    const timings: Timings = {
      candidates: batch.ms,
      // Serialisation and network time is inside the candidate query's wall time; naming it
      // separately would double-count. What is attributable is the volume, reported as bytes.
      transfer: 0,
      tokenize: outcome.timings.tokenize,
      infer: outcome.timings.infer,
      sort: outcome.timings.sort,
      total,
    };

    return {
      queryId: query.id,
      iteration: 0,
      attribution: 'split',
      timings,
      results: outcome.results,
      bytesFromDb: batch.bytes,
      candidatesScored: batch.candidates.length,
    };
  }
}

/**
 * Retrieve and rerank in the database: one statement, identifiers and scores come back.
 * With `control` set, the same statement runs without the cross-encoder.
 */
export class InDbRerankPipeline implements Pipeline {
  constructor(
    readonly stage: Stage,
    private readonly reranker: InDbReranker,
    private readonly control = false,
  ) {}

  sql(): string {
    return this.reranker.sqlFor(this.stage.retrieval, this.control);
  }

  async run(query: Query): Promise<IterationResult> {
    const started = performance.now();
    const outcome = await this.reranker.rerank(
      query,
      this.stage.retrieval,
      this.stage.candidateCount,
      this.stage.topK,
      this.control,
    );
    const timings = zero();
    timings.total = performance.now() - started;
    return {
      queryId: query.id,
      iteration: 0,
      attribution: 'total-only',
      timings,
      results: outcome.results,
      bytesFromDb: outcome.bytes,
      candidatesScored: this.control ? 0 : this.stage.candidateCount,
    };
  }
}
