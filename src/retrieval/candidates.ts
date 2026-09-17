import oracledb from 'oracledb';
import { oracle } from '../config.js';
import { toContainsExpression, withConnection } from '../db/oracle.js';
import { loadSql } from '../db/sql.js';
import type { Candidate, Query, Stage } from '../types.js';

export interface CandidateBatch {
  candidates: Candidate[];
  /** UTF-8 bytes of candidate text that crossed the database boundary. */
  bytes: number;
  /** Wall time for the candidate query, including fetch. */
  ms: number;
}

export interface CandidateSource {
  readonly kind: 'oracle' | 'fixture';
  generate(query: Query, retrieval: Stage['retrieval'], n: number): Promise<CandidateBatch>;
  close(): Promise<void>;
}

const SCOPE = `
    WHERE TENANT = :tenant
      AND (OWNER_ID IS NULL OR OWNER_ID = :owner)
      AND (EXPIRES_AT IS NULL OR EXPIRES_AT > SYSDATE)`;

/**
 * The retrieval arms, kept side by side so the differences between strategies are one glance
 * rather than three files. A disabled arm is a subquery that returns no rows and touches no
 * index, because leaving an unused CTE in place would bill its scan to the wrong measurement.
 */
export function arms(retrieval: Stage['retrieval'], prefix: string): Record<string, string> {
  const NO_ROWS = `    SELECT ID FROM ${prefix}_CHUNKS WHERE 1 = 0`;

  const vecSource = `    SELECT ID
    FROM ${prefix}_CHUNKS${SCOPE}
    ORDER BY VECTOR_DISTANCE(EMBEDDING, (SELECT V FROM qv), COSINE), ID
    FETCH FIRST :pool ROWS ONLY`;

  const lexSource = `    SELECT ID
    FROM ${prefix}_CHUNKS${SCOPE}
      AND CONTAINS(CONTENT, :contains, 1) > 0
    ORDER BY SCORE(1) DESC, ID
    FETCH FIRST :pool ROWS ONLY`;

  const fusedHybrid = `  SELECT
    NVL(v.ID, l.ID) AS ID,
    NVL(1 / (:rrfk + v.RNK), 0) + NVL(1 / (:rrfk + l.RNK), 0) AS SCORE,
    v.RNK AS VRANK,
    l.RNK AS LRANK
  FROM vec v FULL OUTER JOIN lex l ON v.ID = l.ID`;

  // Single-retriever stages still express their score as 1/(k+rank). The value is arbitrary
  // for a single list, but keeping the shape identical means the ordering logic below and the
  // downstream code do not branch on strategy.
  const fusedVector = `  SELECT v.ID AS ID, 1 / (:rrfk + v.RNK) AS SCORE, v.RNK AS VRANK, TO_NUMBER(NULL) AS LRANK
  FROM vec v`;
  const fusedLexical = `  SELECT l.ID AS ID, 1 / (:rrfk + l.RNK) AS SCORE, TO_NUMBER(NULL) AS VRANK, l.RNK AS LRANK
  FROM lex l`;

  switch (retrieval) {
    case 'vector':
      return { VEC_SOURCE: vecSource, LEX_SOURCE: NO_ROWS, FUSED_BODY: fusedVector };
    case 'lexical':
      return { VEC_SOURCE: NO_ROWS, LEX_SOURCE: lexSource, FUSED_BODY: fusedLexical };
    case 'hybrid-rrf':
      return { VEC_SOURCE: vecSource, LEX_SOURCE: lexSource, FUSED_BODY: fusedHybrid };
  }
}

interface CandidateRow {
  ID: string;
  TITLE: string;
  CONTENT: string;
  SCORE: number;
  VRANK: number | null;
  LRANK: number | null;
}

/**
 * How deep each retriever goes before fusion.
 *
 * Fusion needs more input than output or it cannot reorder anything, so each arm retrieves
 * the requested candidate count. Both arms use the same depth; weighting them differently
 * would be a second variable.
 */
export const poolFor = (n: number): number => n;

export function bindsFor(query: Query, n: number, rrfK: number, topK?: number): Record<string, string | number> {
  const binds: Record<string, string | number> = {
    qtext: query.text,
    contains: toContainsExpression(query.text),
    tenant: query.tenant,
    // A null owner would make `OWNER_ID = :owner` unknown for every row, so tenant-wide
    // queries bind a sentinel that matches nothing and rely on the IS NULL arm.
    owner: query.owner ?? '__none__',
    pool: poolFor(n),
    n,
    rrfk: rrfK,
  };
  if (topK !== undefined) binds['topk'] = topK;
  return binds;
}

export class OracleCandidateSource implements CandidateSource {
  readonly kind = 'oracle' as const;
  constructor(private readonly rrfK: number) {}

  sqlFor(retrieval: Stage['retrieval']): string {
    return loadSql('query_candidates.sql', arms(retrieval, oracle.schemaPrefix.toUpperCase()));
  }

  async generate(query: Query, retrieval: Stage['retrieval'], n: number): Promise<CandidateBatch> {
    const sql = this.sqlFor(retrieval);
    const binds = bindsFor(query, n, this.rrfK);
    return withConnection(async (conn) => {
      const started = performance.now();
      const res = await conn.execute<CandidateRow>(sql, binds, {
        outFormat: oracledb.OUT_FORMAT_OBJECT,
        fetchInfo: { CONTENT: { type: oracledb.STRING } },
      });
      const ms = performance.now() - started;
      const rows = res.rows ?? [];
      let bytes = 0;
      const candidates: Candidate[] = rows.map((r, i) => {
        bytes += Buffer.byteLength(r.TITLE, 'utf8') + Buffer.byteLength(r.CONTENT, 'utf8')
          + Buffer.byteLength(r.ID, 'utf8');
        const c: Candidate = {
          chunkId: r.ID,
          title: r.TITLE,
          content: r.CONTENT,
          rank: i + 1,
          score: r.SCORE,
        };
        if (r.VRANK !== null) c.vectorRank = r.VRANK;
        if (r.LRANK !== null) c.lexicalRank = r.LRANK;
        return c;
      });
      return { candidates, bytes, ms };
    });
  }

  async close(): Promise<void> {
    // The pool is owned by db/oracle.ts and closed once at the end of the run.
  }
}
