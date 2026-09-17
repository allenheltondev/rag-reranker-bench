import test from 'node:test';
import assert from 'node:assert/strict';
import { buildStages, runConfigFromEnv } from '../src/config.js';
import { arms } from '../src/retrieval/candidates.js';
import { loadSql, render, splitStatements } from '../src/db/sql.js';
import { assertIdentifier } from '../src/db/oracle.js';
import { defaultScoreExpr, scoreExpr } from '../src/rerank/indb.js';
import type { Stage } from '../src/types.js';

const RETRIEVALS: Array<Stage['retrieval']> = ['vector', 'lexical', 'hybrid-rrf'];

/** Comment lines keep their placeholders on purpose, so assertions look at the SQL only. */
const executable = (sql: string): string =>
  sql.split('\n').filter((line) => !line.trim().startsWith('--')).join('\n');

test('render substitutes tokens and refuses to leave any behind', () => {
  assert.equal(render('a ${X} b', { X: '1' }), 'a 1 b');
  assert.throws(() => render('a ${X} b', {}), /Unsubstituted SQL tokens: \$\{X\}/);
});

test('render leaves comment lines alone, including their placeholders', () => {
  assert.equal(render('-- uses ${X}\nSELECT ${X}', { X: '1' }), '-- uses ${X}\nSELECT 1');
  // A placeholder that only ever appears in a comment must not fail the render.
  assert.equal(render('-- ${UNKNOWN}\nSELECT 1', {}), '-- ${UNKNOWN}\nSELECT 1');
});

test('every retrieval template renders completely', () => {
  for (const retrieval of RETRIEVALS) {
    const sql = executable(loadSql('query_candidates.sql', arms(retrieval, 'BENCH')));
    assert.ok(!/\$\{/.test(sql), `${retrieval}: unsubstituted token`);
    assert.ok(sql.includes('BENCH_CHUNKS'), `${retrieval}: table name missing`);
  }
});

test('every in-database rerank template renders completely', () => {
  for (const retrieval of RETRIEVALS) {
    const sql = executable(loadSql('rerank_indb_prediction.sql', {
      ...arms(retrieval, 'BENCH'),
      SCORE_EXPR: scoreExpr(),
    }));
    assert.ok(!/\$\{/.test(sql), `${retrieval}: unsubstituted token`);
    assert.ok(sql.includes('PREDICTION('), `${retrieval}: scoring expression missing`);
  }
});

test('schema and teardown scripts render and split into statements', () => {
  const schema = splitStatements(loadSql('01_schema.sql'));
  assert.ok(schema.length >= 3, `expected several statements, got ${schema.length}`);
  assert.ok(schema.some((s) => s.includes('CREATE TABLE BENCH_CHUNKS')));
  assert.ok(schema.some((s) => s.includes('CTXSYS.CONTEXT')), 'no text index');
  assert.ok(schema.some((s) => s.includes('VECTOR(')), 'no vector column');
  assert.equal(splitStatements(loadSql('99_teardown.sql')).length, 1);
});

test('splitStatements drops comment-only fragments', () => {
  assert.deepEqual(splitStatements('-- just a comment\n/\nSELECT 1 FROM DUAL\n/'), ['SELECT 1 FROM DUAL']);
});

test('rendered SQL has balanced parentheses', () => {
  for (const retrieval of RETRIEVALS) {
    for (const name of ['query_candidates.sql', 'rerank_indb_prediction.sql']) {
      const body = executable(loadSql(name, { ...arms(retrieval, 'BENCH'), SCORE_EXPR: scoreExpr() }));
      let depth = 0;
      for (const ch of body) {
        if (ch === '(') depth++;
        else if (ch === ')') depth--;
        assert.ok(depth >= 0, `${name}/${retrieval}: unbalanced parentheses`);
      }
      assert.equal(depth, 0, `${name}/${retrieval}: unbalanced parentheses`);
    }
  }
});

test('the vector-only arm does not scan the text index, and vice versa', () => {
  const vectorOnly = executable(loadSql('query_candidates.sql', arms('vector', 'BENCH')));
  assert.ok(!vectorOnly.includes('CONTAINS(CONTENT'), 'vector-only still runs a lexical scan');
  const lexicalOnly = executable(loadSql('query_candidates.sql', arms('lexical', 'BENCH')));
  assert.ok(!lexicalOnly.includes('VECTOR_DISTANCE'), 'lexical-only still runs a vector scan');
  const hybrid = executable(loadSql('query_candidates.sql', arms('hybrid-rrf', 'BENCH')));
  assert.ok(hybrid.includes('CONTAINS(CONTENT') && hybrid.includes('VECTOR_DISTANCE'));
});

test('model names are validated before reaching SQL', () => {
  // The model name cannot be bound, so it is interpolated. Anything that is not a plain
  // Oracle identifier has to be refused rather than concatenated into a statement.
  assert.equal(assertIdentifier('bge_reranker', 'model'), 'BGE_RERANKER');
  for (const bad of ['bge reranker', 'bge;DROP', "bge'", '1bge', '', 'a'.repeat(129)]) {
    assert.throws(() => assertIdentifier(bad, 'model'), /Unsafe model/, `accepted ${JSON.stringify(bad)}`);
  }
});

test('the default scoring expression passes the query and the candidate text as the two inputs', () => {
  const expr = defaultScoreExpr('BGE_RERANKER');
  assert.match(expr, /PREDICTION\(BGE_RERANKER USING :qtext AS FIRST_INPUT/);
  assert.match(expr, /AS SECOND_INPUT\)$/);
});

test('stage expansion covers the sweep and skips depths below top-K', () => {
  const cfg = runConfigFromEnv({
    retrievals: ['hybrid-rrf'],
    rerankers: ['none', 'in-db', 'app'],
    candidateCounts: [5, 10, 40],
    topK: 10,
  });
  const stages = buildStages(cfg);
  const ids = stages.map((s) => s.id);
  assert.ok(ids.includes('hybrid-rrf'));
  assert.ok(ids.includes('hybrid-rrf+rerank-in-db@40'));
  assert.ok(ids.includes('hybrid-rrf+rerank-app@40'));
  assert.ok(!ids.some((id) => id.endsWith('@5')), 'a candidate depth below top-K was kept');
  // The in-database and application stages must exist in matching pairs, or the report has
  // nothing to compare.
  const inDb = ids.filter((id) => id.includes('rerank-in-db'));
  const appSide = ids.filter((id) => id.includes('rerank-app'));
  assert.equal(inDb.length, appSide.length);
});
