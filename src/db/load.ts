import oracledb from 'oracledb';
import { oracle } from '../config.js';
import type { Chunk } from '../types.js';
import { withConnection, withElevatedConnection } from './oracle.js';
import { loadSql, splitStatements } from './sql.js';

/**
 * Run a .sql script, statement by statement, reporting which statement failed.
 *
 * `as` selects the credentials: the benchmark user for everything except creating that user.
 */
export async function runScript(
  name: string,
  extra: Record<string, string> = {},
  as: 'bench' | 'elevated' = 'bench',
  only?: readonly number[],
): Promise<void> {
  const all = splitStatements(loadSql(name, extra));
  const statements = only ? only.map((i) => {
    const stmt = all[i];
    if (stmt === undefined) throw new Error(`${name} has no statement ${i} (found ${all.length}).`);
    return stmt;
  }) : all;
  const runner = as === 'elevated' ? withElevatedConnection : withConnection;
  await runner(async (conn) => {
    for (const [i, stmt] of statements.entries()) {
      try {
        await conn.execute(stmt);
      } catch (err) {
        const message = (err as Error).message;
        const head = stmt.split('\n').filter((l) => !l.trim().startsWith('--')).slice(0, 3).join('\n');
        // ORA-22288 on a model load means the database cannot see the file. It is always a
        // path problem on the database host, never a problem with the ONNX file itself, and
        // the raw error does not say which path it looked in.
        // ORA-54466 names the MGA, not shared memory, so the actual remedy is not obvious
        // from the error or from anything it links to.
        const mga = message.includes('ORA-54466') || message.includes('sskgm_mga_cr')
          ? `\n\nThe database could not allocate memory to hold the model. Loading an ONNX model
places it in the MGA, which is carved out of the container's shared memory, and Docker's
default /dev/shm is 64 MB - far less than a cross-encoder needs.

docker-compose.yml now sets shm_size: 4gb. Apply it with:
  docker compose up -d --force-recreate oracle
The data volume survives, so the schema and corpus are still there afterwards.

If it still fails, the model itself is too large for the memory available. Rebuild it smaller:
  npm run augment:rerank-model -- --quantize
and set APP_RERANK_DTYPE=q8 in .env so BOTH arms run the same weights - otherwise the two
sides are no longer comparable, which is the one thing this benchmark cannot tolerate.`
          : '';
        const vectorMemory = message.includes('ORA-51962')
          ? `\n\nThe database has no vector memory configured, so it cannot build an approximate
index. This is OPTIONAL: the benchmark uses exact search by default precisely so that ANN
tuning is not a second variable, and your data is loaded and usable right now. To enable it
anyway, on the container:
  docker compose exec oracle sqlplus -s "sys/<pw>@localhost:1521/FREE as sysdba"
  ALTER SYSTEM SET vector_memory_size = 512M SCOPE=SPFILE;
  SHUTDOWN IMMEDIATE; STARTUP;
then re-run with --vector-index.`
          : '';
        const hint = mga || vectorMemory || (message.includes('ORA-22288')
          ? `\n\nThe database could not open that file. It looks inside the directory object on the
DATABASE host, not on your machine. Check what it can actually see:
  docker compose exec oracle ls -l /opt/oracle/onnx
Files go in ./models/oracle on the host, which docker-compose mounts there. If that listing is
empty, the file is not where you think it is; if the name differs, set ORACLE_EMBED_FILE or
ORACLE_RERANK_FILE in .env to match.`
          : '');
        throw new Error(`${name}: statement ${i + 1} failed.\n${head}\n\n${message}${hint}`);
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
        // The document prefix is whatever the chosen embedding model asks for, or nothing.
        embedText: `${oracle.embedDocPrefix}${c.title}. ${c.content}`,
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
