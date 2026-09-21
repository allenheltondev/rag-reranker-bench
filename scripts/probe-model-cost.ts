import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import oracledb from 'oracledb';
import { oracle, paths } from '../src/config.js';
import { assertIdentifier, closePool, withConnection } from '../src/db/oracle.js';
import { loadSql, usedBinds } from '../src/db/sql.js';
import { queryVectorExpression } from '../src/db/query-vector.js';
import { arms, bindsFor, OracleCandidateSource } from '../src/retrieval/candidates.js';
import { InDbReranker } from '../src/rerank/indb.js';
import type { Query } from '../src/types.js';

const queries = JSON.parse(readFileSync(paths.queries, 'utf8')) as Query[];
const query = queries[0]!;
const embed = queryVectorExpression('inline');
const output: unknown[] = [];
const startedAt = new Date().toISOString();
const sqls: Record<string, string> = {};
const expected = new Map<string, { chunkId: string; score: number }[]>();
try {
  await withConnection(async conn => {
    conn.callTimeout = 120000;
    const vec = await conn.execute<{ V: Float32Array }>(`SELECT ${embed} V FROM DUAL`, { qtext: query.text }, { outFormat: oracledb.OUT_FORMAT_OBJECT });
    const qvec = vec.rows![0]!.V;
    for (const retrieval of ['lexical', 'vector', 'hybrid-rrf'] as const) {
      for (const mode of ['inline', 'bound', 'separate-each-time', 'same-model', 'control'] as const) {
        if (retrieval === 'lexical' && mode === 'same-model') continue;
        let sql = new InDbReranker(60, 'inline').sqlFor(retrieval, mode === 'control');
        if (mode === 'bound' || mode === 'separate-each-time') sql = sql.replace(embed, ':qvec');
        if (mode === 'same-model') sql = loadSql('rerank_indb_prediction.sql', {
          ...arms(retrieval, assertIdentifier(oracle.schemaPrefix, 'schema prefix')),
          QUERY_VECTOR: embed,
          SCORE_EXPR: `VECTOR_DISTANCE(VECTOR_EMBEDDING(${assertIdentifier(oracle.embedModel, 'embedding model')} USING TITLE || '. ' || CONTENT AS DATA), (SELECT V FROM qv), COSINE)`,
        });
        sqls[`${retrieval}/${mode}`] = sql;
        const runs = [];
        for (let i = 0; i < 4; i++) {
          const start = performance.now();
          if (mode === 'separate-each-time') await conn.execute(`SELECT ${embed} V FROM DUAL`, { qtext: query.text });
          const res = await conn.execute<{ ID: string; SCORE: number }>(sql, usedBinds(sql, { ...bindsFor(query, 1, 60, 1), qvec: { type: oracledb.DB_TYPE_VECTOR, val: qvec } }), { outFormat: oracledb.OUT_FORMAT_OBJECT });
          runs.push({ ms: performance.now() - start, rows: res.rows });
          const scores = (res.rows ?? []).map(r => ({ chunkId: r.ID, score: r.SCORE }));
          if (mode === 'inline' && i === 0) expected.set(retrieval, scores);
          if (mode !== 'same-model' && mode !== 'control') assert.deepEqual(scores, expected.get(retrieval), `${retrieval}/${mode}: score parity`);
        }
        const result = { retrieval, mode, runs };
        output.push(result);
        console.log(JSON.stringify(result));
      }
    }
  });
  await closePool();
  for (const retrieval of ['lexical', 'vector', 'hybrid-rrf'] as const) {
    sqls[`${retrieval}/dedicated-session`] = new InDbReranker(60, 'separate-session').sqlFor(retrieval);
    const runs = [];
    for (let i = 0; i < 4; i++) {
      const res = await new InDbReranker(60, 'separate-session').rerank(query, retrieval, 1, 1);
      runs.push({ ms: res.ms, rows: res.results });
      assert.deepEqual(res.results.map(r => ({ chunkId: r.chunkId, score: r.score })), expected.get(retrieval), `${retrieval}/dedicated-session: score parity`);
    }
    const result = { retrieval, mode: 'dedicated-session', runs };
    output.push(result);
    console.log(JSON.stringify(result));
  }
  await closePool();
  const inlineSource = new OracleCandidateSource(60, 'inline');
  const splitSource = new OracleCandidateSource(60, 'separate-session');
  let checked = 0;
  for (const retrieval of ['vector', 'lexical', 'hybrid-rrf'] as const) {
    for (const n of [10, 20, 40, 80]) {
      for (const q of queries) {
        const inline = await inlineSource.generate(q, retrieval, n);
        const split = await splitSource.generate(q, retrieval, n);
        assert.deepEqual(split.candidates, inline.candidates, `${q.id}/${retrieval}/${n}: embedding-mode parity`);
        checked++;
      }
    }
  }
  output.push({ embeddingModeParity: { checked, mismatches: 0 } });
  console.log(`Embedding-mode parity: ${checked} candidate lists identical, including order, scores and text.`);
} finally {
  mkdirSync('results/model-cost-probe', { recursive: true });
  writeFileSync('results/model-cost-probe/raw.json', JSON.stringify(output, null, 2));
  writeFileSync('results/model-cost-probe/sql.json', JSON.stringify(sqls, null, 2));
  writeFileSync('results/model-cost-probe/metadata.json', JSON.stringify({ startedAt, finishedAt: new Date().toISOString(), queryId: query.id, embeddingModel: oracle.embedModel, rerankerModel: oracle.rerankModel, dimensions: oracle.embedDims }, null, 2));
  await closePool();
}
