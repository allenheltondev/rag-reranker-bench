import oracledb from 'oracledb';
import { oracle } from '../config.js';
import { assertIdentifier, withConnection } from '../db/oracle.js';
import { loadSql } from '../db/sql.js';
import { arms, bindsFor } from '../retrieval/candidates.js';
import type { Query, RankedResult, Stage } from '../types.js';

/**
 * The default scoring expression, matching Oracle's own hybrid-retrieval example: the query is
 * the first input, the candidate text is the second, and the model is a database object.
 */
export function defaultScoreExpr(model: string): string {
  return `PREDICTION(${model} USING :qtext AS FIRST_INPUT, TITLE || '. ' || CONTENT AS SECOND_INPUT)`;
}

export function scoreExpr(): string {
  const model = assertIdentifier(oracle.rerankModel, 'ORACLE_RERANK_MODEL');
  return oracle.indbScoreExpr || defaultScoreExpr(model);
}

export interface InDbOutcome {
  results: RankedResult[];
  /** Wall time for the single statement: filter, retrieve, fuse, rerank, return. */
  ms: number;
  /** Bytes returned to the application: identifiers and scores only. */
  bytes: number;
}

/**
 * In-database reranking.
 *
 * One statement does the whole pipeline. There is no candidate list in application memory at
 * any point, which is the property being measured - and also the reason this class cannot
 * report a tokenize/infer split the way the application path can. The harness derives the
 * cost of the reranking stage by differencing against the unreranked stage at the same
 * candidate depth, which is an estimate and is labelled as one in the report.
 */
export class InDbReranker {
  constructor(private readonly rrfK: number) {}

  sqlFor(retrieval: Stage['retrieval']): string {
    return loadSql('rerank_indb_prediction.sql', {
      ...arms(retrieval, oracle.schemaPrefix.toUpperCase()),
      SCORE_EXPR: scoreExpr(),
    });
  }

  async rerank(query: Query, retrieval: Stage['retrieval'], n: number, topK: number): Promise<InDbOutcome> {
    const sql = this.sqlFor(retrieval);
    const binds = bindsFor(query, n, this.rrfK, topK);
    return withConnection(async (conn) => {
      const started = performance.now();
      const res = await conn.execute<{ ID: string; SCORE: number }>(sql, binds, {
        outFormat: oracledb.OUT_FORMAT_OBJECT,
      });
      const ms = performance.now() - started;
      const rows = res.rows ?? [];
      const results: RankedResult[] = rows.map((r, i) => ({
        chunkId: r.ID,
        rank: i + 1,
        score: r.SCORE,
      }));
      // 8 bytes for a double, plus the identifier. The candidate text is not in this number
      // because it never left the database.
      const bytes = rows.reduce((acc, r) => acc + Buffer.byteLength(r.ID, 'utf8') + 8, 0);
      return { results, ms, bytes };
    });
  }
}
