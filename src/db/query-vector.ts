import oracledb from 'oracledb';
import { oracle } from '../config.js';
import { assertIdentifier } from './oracle.js';
import { queryEmbedInput } from './sql.js';

let embeddingPool: oracledb.Pool | undefined;

export function queryVectorExpression(mode = oracle.queryEmbedding): string {
  if (mode === 'separate-session') return ':qvec';
  if (mode !== 'inline') throw new Error('ORACLE_QUERY_EMBEDDING must be inline or separate-session');
  return `VECTOR_EMBEDDING(${assertIdentifier(oracle.embedModel, 'embedding model')} USING ${queryEmbedInput()} AS DATA)`;
}

/** No cache: each request includes real embedding work and its network round trip. */
export async function queryVectorBinds(retrieval: string, text: string, mode = oracle.queryEmbedding): Promise<Record<string, oracledb.BindParameter>> {
  queryVectorExpression(mode); // Validate even for lexical retrieval.
  if (mode === 'inline' || retrieval === 'lexical') return {};
  embeddingPool ??= await oracledb.createPool({
    user: oracle.user, password: oracle.password, connectString: oracle.connectString,
    poolMin: 1, poolMax: 1, poolIncrement: 1,
  });
  const conn = await embeddingPool.getConnection();
  try {
    const result = await conn.execute<{ V: Float32Array }>(
      `SELECT ${queryVectorExpression('inline')} AS V FROM DUAL`, { qtext: text },
      { outFormat: oracledb.OUT_FORMAT_OBJECT },
    );
    const qvec = result.rows?.[0]?.V;
    if (!(qvec instanceof Float32Array) || qvec.length !== oracle.embedDims) {
      throw new Error(`Expected a ${oracle.embedDims}-dimension FLOAT32 query embedding`);
    }
    return { qvec: { type: oracledb.DB_TYPE_VECTOR, val: qvec } };
  } finally { await conn.close(); }
}

export async function closeEmbeddingPool(): Promise<void> {
  if (embeddingPool) { await embeddingPool.close(5); embeddingPool = undefined; }
}
