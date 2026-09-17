import oracledb from 'oracledb';
import { oracle } from '../config.js';
import type { Chunk } from '../types.js';
import { withConnection } from './oracle.js';
import { loadSql, splitStatements } from './sql.js';

/** Run a .sql script, statement by statement, reporting which statement failed. */
export async function runScript(name: string, extra: Record<string, string> = {}): Promise<void> {
  const statements = splitStatements(loadSql(name, extra));
  await withConnection(async (conn) => {
    for (const [i, stmt] of statements.entries()) {
      try {
        await conn.execute(stmt);
      } catch (err) {
        const head = stmt.split('\n').filter((l) => !l.trim().startsWith('--')).slice(0, 3).join('\n');
        throw new Error(`${name}: statement ${i + 1} failed.\n${head}\n\n${(err as Error).message}`);
      }
    }
    await conn.commit();
  });
}

/**
 * Insert chunks and embed them in the database.
 *
 * Embedding happens in SQL via VECTOR_EMBEDDING rather than in the application, for the same
 * reason the benchmark exists: it is the version of the architecture where text does not leave
 * the database. It also guarantees the stored vectors match what query-time embedding produces.
 */
export async function loadChunks(chunks: Chunk[], batchSize = 50): Promise<number> {
  const sql = `
    INSERT INTO ${oracle.schemaPrefix.toUpperCase()}_CHUNKS
      (ID, DOC_ID, TITLE, CONTENT, TENANT, OWNER_ID, CREATED_AT, EXPIRES_AT, TAGS, EMBEDDING)
    VALUES
      (:id, :docId, :title, :content, :tenant, :owner,
       TO_DATE(:createdAt, 'YYYY-MM-DD'),
       CASE WHEN :expiresAt IS NULL THEN NULL ELSE TO_DATE(:expiresAt, 'YYYY-MM-DD') END,
       :tags,
       VECTOR_EMBEDDING(${oracle.embedModel.toUpperCase()} USING :embedText AS DATA))`;

  let inserted = 0;
  await withConnection(async (conn) => {
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      const binds = batch.map((c) => ({
        id: c.id,
        docId: c.docId,
        title: c.title,
        content: c.content,
        tenant: c.tenant,
        owner: c.owner,
        createdAt: c.createdAt,
        expiresAt: c.expiresAt,
        tags: c.tags.join(','),
        // Title and body are embedded together, matching what the reranker is shown at
        // query time. Embedding a different string than you rerank is a classic quiet bug.
        embedText: `${c.title}. ${c.content}`,
      }));
      const res = await conn.executeMany(sql, binds, {
        autoCommit: false,
        bindDefs: {
          id: { type: oracledb.STRING, maxSize: 64 },
          docId: { type: oracledb.STRING, maxSize: 64 },
          title: { type: oracledb.STRING, maxSize: 400 },
          content: { type: oracledb.STRING, maxSize: 32000 },
          tenant: { type: oracledb.STRING, maxSize: 64 },
          owner: { type: oracledb.STRING, maxSize: 64 },
          createdAt: { type: oracledb.STRING, maxSize: 10 },
          expiresAt: { type: oracledb.STRING, maxSize: 10 },
          tags: { type: oracledb.STRING, maxSize: 400 },
          embedText: { type: oracledb.STRING, maxSize: 32000 },
        },
      });
      inserted += res.rowsAffected ?? 0;
    }
    await conn.commit();
  });
  return inserted;
}

export async function countChunks(): Promise<number> {
  return withConnection(async (conn) => {
    const r = await conn.execute<[number]>(
      `SELECT COUNT(*) FROM ${oracle.schemaPrefix.toUpperCase()}_CHUNKS`,
    );
    return r.rows?.[0]?.[0] ?? 0;
  });
}
