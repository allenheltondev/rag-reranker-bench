import oracledb from 'oracledb';
import { oracle } from '../config.js';
import { closeEmbeddingPool } from './query-vector.js';

let pool: oracledb.Pool | null = null;

/**
 * Turn an Oracle error into something actionable.
 *
 * These are the failures this benchmark actually provokes, and each names a resource without
 * saying which knob controls it: the model is held per session in the PGA, loaded through the
 * container's shared memory, and read from a directory on the database host.
 */
export function hintFor(message: string): string {
  if (message.includes('ORA-04036')) {
    return `\n\nThe database ran out of PGA. Every session that runs PREDICTION holds the model in
its own PGA, so the aggregate target has to exceed the model's size with room to work.

IMPORTANT: the container image applies INIT_PGA_SIZE only when it CREATES the database. If
your volume already had one, recreating the container did NOT change it. Check what is really
in force - \`npm run doctor\` prints it - and if it is still small, set it on the live database:

  docker compose exec oracle sqlplus -s "sys/<pw>@localhost:1521/FREE as sysdba"
    ALTER SYSTEM SET pga_aggregate_target = 3G SCOPE=SPFILE;
    ALTER SYSTEM SET pga_aggregate_limit  = 6G SCOPE=SPFILE;
    SHUTDOWN IMMEDIATE;
    STARTUP;

The Free edition caps total memory, so those values may be refused or silently clamped. When
they are, the model is simply too large to score inside this edition, and the fix is to make
it smaller:

  npm run augment:rerank-model -- --quantize

then re-upload it (npm run models:rerank) and set APP_RERANK_DTYPE=q8 so BOTH arms run the
same weights. Quantizing one side only would end the comparison.`;
  }
  if (message.includes('ORA-54466') || message.includes('sskgm_mga_cr')) {
    return `\n\nThe database could not allocate memory to hold the model. Loading an ONNX model places
it in the MGA, which is carved out of the container's shared memory, and Docker's default
/dev/shm is 64 MB - far less than a cross-encoder needs.

docker-compose.yml now sets shm_size: 4gb. Apply it with:
  docker compose up -d --force-recreate oracle
The data volume survives, so the schema and corpus are still there afterwards.

If it still fails, rebuild the model smaller:
  npm run augment:rerank-model -- --quantize
and set APP_RERANK_DTYPE=q8 in .env so BOTH arms run the same weights - otherwise the two
sides are no longer comparable, which is the one thing this benchmark cannot tolerate.`;
  }
  if (message.includes('ORA-51962')) {
    return `\n\nThe database has no vector memory configured, so it cannot build an approximate index.
This is OPTIONAL: the benchmark uses exact search by default precisely so that ANN tuning is
not a second variable, and your data is loaded and usable right now. To enable it anyway:
  ALTER SYSTEM SET vector_memory_size = 512M SCOPE=SPFILE;
  SHUTDOWN IMMEDIATE; STARTUP;
then re-run with --vector-index.`;
  }
  if (message.includes('ORA-22288')) {
    return `\n\nThe database could not open that file. It looks inside the directory object on the
DATABASE host, not on your machine. Check what it can actually see:
  docker compose exec oracle ls -l /opt/oracle/onnx
Files go in ./models/oracle on the host, which docker-compose mounts there.`;
  }
  return '';
}

/** Oracle identifiers are interpolated into SQL (they cannot be bound), so validate them hard. */
export function assertIdentifier(name: string, what: string): string {
  if (!/^[A-Za-z][A-Za-z0-9_$#]{0,127}$/.test(name)) {
    throw new Error(`Unsafe ${what}: ${JSON.stringify(name)}. Expected a plain Oracle identifier.`);
  }
  return name.toUpperCase();
}

export async function getPool(): Promise<oracledb.Pool> {
  if (pool) return pool;
  if (!oracle.user || !oracle.password) {
    throw new Error('ORACLE_USER and ORACLE_PASSWORD must be set. Copy .env.example to .env first.');
  }
  // Thin mode is the default and needs no Instant Client. Thick mode is opt-in because some
  // deployments require it for older auth schemes.
  if (oracle.clientLibDir) {
    oracledb.initOracleClient({ libDir: oracle.clientLibDir });
  }
  oracledb.fetchAsString = [oracledb.CLOB];
  pool = await oracledb.createPool({
    user: oracle.user,
    password: oracle.password,
    connectString: oracle.connectString,
    poolMin: oracle.poolMin,
    poolMax: oracle.poolMax,
    poolIncrement: 1,
  });
  return pool;
}

export async function withConnection<T>(fn: (conn: oracledb.Connection) => Promise<T>): Promise<T> {
  const p = await getPool();
  const conn = await p.getConnection();
  try {
    return await fn(conn);
  } finally {
    await conn.close();
  }
}

/**
 * A standalone connection using elevated credentials, for the one operation the benchmark user
 * cannot perform: creating itself. Not pooled, and never reused by the harness.
 */
export async function withElevatedConnection<T>(
  fn: (conn: oracledb.Connection) => Promise<T>,
): Promise<T> {
  if (!oracle.sysPassword) {
    throw new Error(
      'ORACLE_SYS_PASSWORD must be set to create the benchmark user. For the container it is the '
      + 'password you gave docker compose; for Autonomous Database it is the ADMIN password.',
    );
  }
  if (oracle.clientLibDir) oracledb.initOracleClient({ libDir: oracle.clientLibDir });
  const attrs: oracledb.ConnectionAttributes = {
    user: oracle.sysUser,
    password: oracle.sysPassword,
    connectString: oracle.connectString,
  };
  // Autonomous Database has no SYSDBA for customers; ADMIN is an ordinary privileged user.
  if (oracle.target !== 'adb') attrs.privilege = oracledb.SYSDBA;

  let conn: oracledb.Connection;
  try {
    conn = await oracledb.getConnection(attrs);
  } catch (err) {
    const message = (err as Error).message;
    if (message.includes('ORA-01017')) {
      throw new Error(`${message}\n\nCheck ORACLE_SYS_USER / ORACLE_SYS_PASSWORD in .env.`);
    }
    throw err;
  }
  try {
    return await fn(conn);
  } finally {
    await conn.close();
  }
}

export async function closePool(): Promise<void> {
  await closeEmbeddingPool();
  if (pool) {
    await pool.close(5);
    pool = null;
  }
}

export interface OracleInfo {
  version: string;
  banner: string;
  clientMode: string;
  models: string[];
  /** CPUs the database believes it has. Not necessarily the host's, and that matters. */
  cpuCount: number | null;
  /** PGA target and hard limit, in MB. The model is held per session in the PGA. */
  pgaTargetMb: number | null;
  pgaLimitMb: number | null;
}

export async function describeOracle(): Promise<OracleInfo> {
  return withConnection(async (conn) => {
    const banner = await conn.execute<{ BANNER: string }>(
      `SELECT BANNER FROM V$VERSION WHERE ROWNUM = 1`,
      {},
      { outFormat: oracledb.OUT_FORMAT_OBJECT },
    );
    let models: string[] = [];
    try {
      const m = await conn.execute<{ MODEL_NAME: string }>(
        `SELECT MODEL_NAME FROM USER_MINING_MODELS ORDER BY MODEL_NAME`,
        {},
        { outFormat: oracledb.OUT_FORMAT_OBJECT },
      );
      models = (m.rows ?? []).map((r) => r.MODEL_NAME);
    } catch {
      // USER_MINING_MODELS is unavailable on some editions; the doctor command reports this.
      models = [];
    }
    // The database's own CPU count, which on a container is its share rather than the host's.
    // An application reranker using every core while the database has a fraction of them is
    // not a comparison of where inference runs; it is a comparison of how much CPU each got.
    //
    // V$PARAMETER returns values as strings, which matters: pga_aggregate_target is a byte
    // count that overflows the BINARY_INTEGER the DBMS_UTILITY route hands back. That route
    // stays as a fallback for a user without the catalog grant.
    const param = async (name: string): Promise<number | null> => {
      try {
        const r = await conn.execute<{ VALUE: string }>(
          `SELECT VALUE FROM V$PARAMETER WHERE NAME = :n`,
          { n: name },
          { outFormat: oracledb.OUT_FORMAT_OBJECT },
        );
        const raw = r.rows?.[0]?.VALUE;
        if (raw !== undefined && raw !== null && raw !== '') {
          const v = Number(raw);
          if (!Number.isNaN(v)) return v;
        }
      } catch {
        // No catalog grant; try the package instead.
      }
      try {
        const r = await conn.execute<{ out: number }>(
          `DECLARE n BINARY_INTEGER; s VARCHAR2(4000); t BINARY_INTEGER;
           BEGIN t := DBMS_UTILITY.GET_PARAMETER_VALUE(:p, n, s); :out := n; END;`,
          { p: name, out: { dir: oracledb.BIND_OUT, type: oracledb.NUMBER } },
        );
        const v = (r.outBinds as { out: number } | undefined)?.out;
        const n = v === undefined || v === null ? NaN : Number(v);
        return Number.isNaN(n) || n === 0 ? null : n;
      } catch {
        return null;
      }
    };

    const asMb = (bytes: number | null): number | null =>
      bytes === null ? null : Math.round(bytes / 1024 / 1024);

    return {
      version: conn.oracleServerVersionString,
      banner: banner.rows?.[0]?.BANNER ?? 'unknown',
      clientMode: oracledb.thin ? 'thin' : 'thick',
      models,
      cpuCount: await param('cpu_count'),
      // The container image applies INIT_PGA_SIZE only when it CREATES the database, so a
      // recreate against an existing volume silently keeps the old value. Report what is in
      // force rather than what the compose file asked for.
      pgaTargetMb: asMb(await param('pga_aggregate_target')),
      pgaLimitMb: asMb(await param('pga_aggregate_limit')),
    };
  });
}

/**
 * Build a safe Oracle Text CONTAINS expression from free-form query text.
 *
 * Terms are stripped to word characters and wrapped in braces so that reserved words
 * (AND, NEAR, ABOUT, hyphens inside identifiers like INC-4821) are treated as literals.
 * Joined with OR so the expression behaves like a bag-of-words retriever rather than
 * requiring every term.
 */
export function toContainsExpression(text: string): string {
  const terms = text
    .toLowerCase()
    .split(/[^a-z0-9_-]+/)
    .map((t) => t.replace(/[{}]/g, ''))
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
  if (terms.length === 0) return '{search}';
  return [...new Set(terms)].map((t) => `{${t}}`).join(' OR ');
}

const STOPWORDS = new Set([
  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'our', 'was', 'what', 'when', 'how', 'why',
  'does', 'did', 'this', 'that', 'with', 'from', 'into', 'about', 'should', 'would', 'could',
  'have', 'has', 'had', 'its', 'it', 'is', 'do', 'we', 'i', 'a', 'an', 'of', 'on', 'in', 'to',
  'keep', 'still', 'some', 'them', 'they', 'their', 'there', 'then', 'than', 'at', 'be', 'by',
]);
