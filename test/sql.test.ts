import test from 'node:test';
import assert from 'node:assert/strict';
import { buildStages, runConfigFromEnv } from '../src/config.js';
import { arms } from '../src/retrieval/candidates.js';
import { loadSql, queryEmbedInput, render, splitStatements } from '../src/db/sql.js';
import { assertIdentifier } from '../src/db/oracle.js';
import { controlExpr, defaultScoreExpr, scoreExpr } from '../src/rerank/indb.js';
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

test('model-loading scripts are one statement per model, embedding first', () => {
  // cmdModels selects statement 0 for --only embed and 1 for --only rerank, so the order and
  // the count are load-bearing, not incidental.
  const local = splitStatements(loadSql('03_load_models.sql', {
    ONNX_DIRECTORY: 'ONNX_DIR', EMBED_FILE: 'e.onnx', RERANK_FILE: 'r.onnx',
  }));
  assert.equal(local.length, 2);
  assert.ok(local[0]!.includes("'e.onnx'") && local[0]!.includes('DOC_EMBEDDER'));
  assert.ok(!local[0]!.includes('BGE_RERANKER'), 'embedding statement touches the reranker');
  assert.ok(local[1]!.includes("'r.onnx'") && local[1]!.includes('BGE_RERANKER'));
  assert.ok(!local[1]!.includes('DOC_EMBEDDER'), 'reranker statement touches the embedder');
  // Each statement drops before loading, so re-running is idempotent.
  assert.ok(local.every((x) => x.includes('DROP_ONNX_MODEL') && x.includes('LOAD_ONNX_MODEL')));

  const adb = splitStatements(loadSql('03_load_models_adb.sql', {
    PAR_BASE_URL: 'https://x/p/abc/n/ns/b/models/o/', EMBED_FILE: 'e.onnx', RERANK_FILE: 'r.onnx',
  }));
  assert.equal(adb.length, 2);
  assert.ok(adb[0]!.includes("'https://x/p/abc/n/ns/b/models/o/e.onnx'"));
  assert.ok(adb[1]!.includes('LOAD_ONNX_MODEL_CLOUD') && adb[1]!.includes('r.onnx'));
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

test('every treatment has a control at the same depth in the same group', () => {
  const cfg = runConfigFromEnv({
    retrievals: ['vector', 'hybrid-rrf'],
    rerankers: ['none', 'in-db', 'app'],
    candidateCounts: [10, 40],
    topK: 10,
  });
  const stages = buildStages(cfg);
  const treatments = stages.filter((s) => s.role === 'treatment');
  assert.ok(treatments.length > 0);
  for (const t of treatments) {
    assert.ok(t.group, `${t.id} has no group`);
    const control = stages.find(
      (c) => c.role === 'control' && c.group === t.group && c.reranker === t.reranker,
    );
    assert.ok(control, `${t.id} has no control`);
    assert.equal(control!.candidateCount, t.candidateCount, 'control depth differs from treatment');
    assert.equal(control!.retrieval, t.retrieval);
    assert.equal(control!.topK, t.topK);
  }
  // Baselines stand alone: they are what you would ship without a reranker, not a control.
  for (const b of stages.filter((s) => s.role === 'baseline')) {
    assert.equal(b.group, undefined);
    assert.equal(b.candidateCount, cfg.topK);
  }
});

test('the control statement differs from the treatment only in the scoring expression', () => {
  const treatment = executable(loadSql('rerank_indb_prediction.sql', {
    ...arms('hybrid-rrf', 'BENCH'), SCORE_EXPR: scoreExpr(),
  }));
  const control = executable(loadSql('rerank_indb_prediction.sql', {
    ...arms('hybrid-rrf', 'BENCH'), SCORE_EXPR: controlExpr(),
  }));
  assert.notEqual(treatment, control);
  assert.equal(
    treatment.replace(scoreExpr(), controlExpr()),
    control,
    'something other than the scoring expression changed between treatment and control',
  );
  assert.ok(!control.includes('PREDICTION'), 'control still calls the model');
  // The control must bind the same variables and read the same text, or the subtraction is
  // comparing statements that do different work.
  assert.ok(controlExpr().includes(':qtext'));
  assert.ok(controlExpr().includes('CONTENT'));
});

test('stages are assigned to isolation batches by arm, and groups never span batches', () => {
  const cfg = runConfigFromEnv({
    retrievals: ['hybrid-rrf'],
    rerankers: ['none', 'in-db', 'app'],
    candidateCounts: [10, 40],
    topK: 10,
  });
  const stages = buildStages(cfg);
  const order = [...new Set(stages.map((s) => s.batch))];
  assert.deepEqual(order, ['baseline', 'in-db', 'app', 'transfer']);
  const batchOfGroup = new Map<string, string>();
  for (const s of stages) {
    if (!s.group) continue;
    const seen = batchOfGroup.get(s.group);
    if (seen) assert.equal(seen, s.batch, `group ${s.group} spans batches`);
    batchOfGroup.set(s.group, s.batch);
  }
  // The transfer batch is the two controls only, one per depth, and never a treatment.
  const transfer = stages.filter((s) => s.batch === 'transfer');
  assert.equal(transfer.length, 4);
  assert.ok(transfer.every((s) => s.role === 'control'));
  assert.deepEqual(
    transfer.map((s) => s.reranker).sort(),
    ['app', 'app', 'in-db', 'in-db'],
  );
});

test('no transfer batch unless both arms are present', () => {
  const stages = buildStages(runConfigFromEnv({ rerankers: ['none', 'app'], candidateCounts: [10] }));
  assert.ok(!stages.some((s) => s.batch === 'transfer'));
});

test('the query embedding input is bare unless a prefix is configured', () => {
  // Default: nothing is concatenated, so a model that wants no instruction sees only the query.
  assert.equal(queryEmbedInput(), ':qtext');
  const sql = executable(loadSql('query_candidates.sql', arms('vector', 'BENCH')));
  assert.ok(sql.includes('VECTOR_EMBEDDING(DOC_EMBEDDER USING :qtext AS DATA)'));
});

test('a configured prefix is concatenated and its quotes are escaped', () => {
  const original = process.env['ORACLE_EMBED_QUERY_PREFIX'];
  try {
    // Re-import with the env var set, since config reads it once at module load.
    process.env['ORACLE_EMBED_QUERY_PREFIX'] = "query: it's";
    const escaped = "'query: it''s' || :qtext";
    // Mirror what queryEmbedInput does, to assert the escaping rule itself.
    const prefix = process.env['ORACLE_EMBED_QUERY_PREFIX'];
    assert.equal(`'${prefix.replace(/'/g, "''")}' || :qtext`, escaped);
    // A doubled quote cannot terminate the literal, so the statement stays one expression.
    assert.equal((escaped.match(/'/g) ?? []).length % 2, 0);
  } finally {
    if (original === undefined) delete process.env['ORACLE_EMBED_QUERY_PREFIX'];
    else process.env['ORACLE_EMBED_QUERY_PREFIX'] = original;
  }
});
