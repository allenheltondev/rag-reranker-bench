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

/**
 * The control's stand-in for the cross-encoder.
 *
 * It must (a) read the same text the model reads, so CLOB access is on both sides of the
 * subtraction, (b) reference :qtext, so the statement binds identically, and (c) be evaluated
 * for every candidate, which putting it in the ORDER BY guarantees. What it must not do is any
 * inference. Everything else about the statement is byte-identical to the treatment.
 */
export function defaultControlExpr(): string {
  return `LENGTH(:qtext || TITLE || '. ' || CONTENT)`;
}

export function controlExpr(): string {
  return oracle.indbControlExpr || defaultControlExpr();
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
 * report a tokenize/infer split the way the application path can. The cost of scoring is
 * instead obtained by subtraction: the same statement is run with `control = true`, which
 * swaps the cross-encoder for a cheap expression over the same text, and the harness pairs
 * the two timings per query and iteration. See README, "How the reranking cost is calculated".
 */
export class InDbReranker {
  constructor(private readonly rrfK: number) {}

  sqlFor(retrieval: Stage['retrieval'], control = false): string {
    return loadSql('rerank_indb_prediction.sql', {
      ...arms(retrieval, oracle.schemaPrefix.toUpperCase()),
      SCORE_EXPR: control ? controlExpr() : scoreExpr(),
    });
  }

  async rerank(
    query: Query,
    retrieval: Stage['retrieval'],
    n: number,
    topK: number,
    control = false,
  ): Promise<InDbOutcome> {
    const sql = this.sqlFor(retrieval, control);
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
