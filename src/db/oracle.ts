import oracledb from 'oracledb';
import { oracle } from '../config.js';

let pool: oracledb.Pool | null = null;

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

export async function closePool(): Promise<void> {
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
    return {
      version: conn.oracleServerVersionString,
      banner: banner.rows?.[0]?.BANNER ?? 'unknown',
      clientMode: oracledb.thin ? 'thin' : 'thick',
      models,
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
