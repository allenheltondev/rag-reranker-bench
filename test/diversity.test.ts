import test from 'node:test';
import assert from 'node:assert/strict';
import {
  compareDiversity, cosine, intraListSimilarity, jaccard, measureDiversity, renderDiversity, tokenSet,
} from '../src/bench/diversity.js';
import type { BenchRun, Chunk, IterationResult, Stage } from '../src/types.js';

test('cosine is 1 for parallel vectors and 0 for orthogonal ones', () => {
  assert.equal(cosine([1, 2, 3], [2, 4, 6]).toFixed(6), '1.000000');
  assert.equal(cosine([1, 0], [0, 1]), 0);
  assert.ok(Number.isNaN(cosine([0, 0], [1, 1])));
});

test('word sets keep identifiers whole and drop filler', () => {
  const t = tokenSet('What was the root cause of INC-4821? The cause was a bad deploy.');
  assert.ok(t.has('inc-4821'));
  assert.ok(t.has('cause'));
  assert.ok(!t.has('the'));
  assert.ok(!t.has('was'));
  assert.ok(!t.has('of'));
});

test('jaccard is overlap over union', () => {
  assert.equal(jaccard(new Set(['a1', 'b1']), new Set(['b1', 'c1'])), 1 / 3);
  assert.equal(jaccard(new Set(['a1']), new Set(['a1'])), 1);
});

test('intra-list similarity looks only at the top k and needs two items', () => {
  const sim = (a: number, b: number) => (a === b ? 1 : 0);
  assert.equal(intraListSimilarity([7, 7, 9], 2, sim), 1);
  assert.equal(intraListSimilarity([7, 7, 9], 3, sim), 1 / 3);
  assert.ok(Number.isNaN(intraListSimilarity([7], 5, sim)));
});

// A tiny world: vector retrieval returns five paraphrases of one memory, hybrid returns five
// different memories. That is the flat-drafts problem in miniature.
const QUERIES = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'];
const dupIds = (q: string) => [0, 1, 2, 3, 4].map((i) => `${q}-dup${i}`);
const varIds = (q: string) => [0, 1, 2, 3, 4].map((i) => `${q}-var${i}`);

function chunk(id: string, content: string): Chunk {
  return {
    id, docId: 'd', title: '', content, tenant: 't', owner: null,
    createdAt: '2025-01-01', expiresAt: null, tags: [],
  };
}

const TOPICS = ['serverless cold starts latency', 'dynamodb single table design', 'eventbridge schema registry',
  'step functions retries backoff', 'lambda powertools tracing'];
const chunks: Chunk[] = QUERIES.flatMap((q) => [
  ...dupIds(q).map((id, i) => chunk(id, `serverless cold starts latency matter ${i === 0 ? '' : 'really'}`)),
  ...varIds(q).map((id, i) => chunk(id, TOPICS[i]!)),
]);

const embeddings = new Map<string, number[]>();
for (const q of QUERIES) {
  dupIds(q).forEach((id, i) => embeddings.set(id, [1, 0.01 * i, 0, 0, 0]));
  varIds(q).forEach((id, i) => embeddings.set(id, [0, 0, 0, 0, 0].map((_, j) => (j === i ? 1 : 0))));
}

function stage(over: Partial<Stage>): Stage {
  return { id: 'x', label: 'x', retrieval: 'vector', reranker: 'none', candidateCount: 10, topK: 10, role: 'baseline', ...over } as Stage;
}
function it(queryId: string, ids: string[]): IterationResult {
  return {
    queryId, iteration: 0, repeat: 0, attribution: 'split',
    timings: { candidates: 0, transfer: 0, tokenize: 0, infer: 0, sort: 0, total: 1 },
    results: ids.map((chunkId, i) => ({ chunkId, rank: i + 1, score: 1 - i / 10 })),
    bytesFromDb: 0, candidatesScored: 0,
  };
}

const run = {
  stages: [
    { stage: stage({ id: 'vector', label: 'Vector (no rerank)', retrieval: 'vector' }), iterations: QUERIES.map((q) => it(q, dupIds(q))) },
    { stage: stage({ id: 'hybrid-rrf', label: 'Hybrid RRF (no rerank)', retrieval: 'hybrid-rrf' }), iterations: QUERIES.map((q) => it(q, varIds(q))) },
    // A reranker that keeps vector's order exactly: no change in variety.
    { stage: stage({ id: 'vector+rerank-app@10', label: 'Vector + app rerank (N=10)', retrieval: 'vector', reranker: 'app', role: 'treatment' }), iterations: QUERIES.map((q) => it(q, dupIds(q))) },
    // A control sorts by a placeholder and must never be treated as a ranking.
    { stage: stage({ id: 'vector+control-app@10', label: 'control', retrieval: 'vector', reranker: 'app', role: 'control' }), iterations: QUERIES.map((q) => it(q, varIds(q))) },
  ],
} as unknown as BenchRun;

test('controls are left out; they order by a placeholder, not relevance', () => {
  const stages = measureDiversity(run, chunks, embeddings);
  assert.ok(!stages.some((s) => s.stage.role === 'control'));
  assert.equal(stages.length, 3);
});

test('near-duplicate results measure as far more alike than varied ones, on both measures', () => {
  const [vector, hybrid] = measureDiversity(run, chunks, embeddings);
  const avg = (m: Map<string, number> | undefined) => [...(m?.values() ?? [])].reduce((a, b) => a + b, 0) / (m?.size ?? 1);
  assert.ok(avg(vector!.embedding!.get(5)) > 0.99);
  assert.ok(avg(hybrid!.embedding!.get(5)) < 0.01);
  assert.ok(avg(vector!.words.get(5)) > avg(hybrid!.words.get(5)));
});

test('fusion that spreads results out is detected as more varied, with both measures agreeing', () => {
  const comps = compareDiversity(measureDiversity(run, chunks, embeddings));
  const c = comps.find((x) => x.family === 'fusion' && x.label.includes('vector') && x.k === 5)!;
  assert.equal(c.embedding!.verdict, 'more varied');
  assert.equal(c.words.verdict, 'more varied');
  assert.ok(c.embedding!.ci.upper < 0);
  assert.equal(c.words.n, QUERIES.length);
});

test('a reranker that keeps the same order reads as no detectable change', () => {
  const comps = compareDiversity(measureDiversity(run, chunks, embeddings));
  const c = comps.find((x) => x.family === 'rerank')!;
  assert.equal(c.embedding!.verdict, "can't tell");
  assert.equal(c.words.verdict, "can't tell");
  assert.equal(c.embedding!.mean, 0);
});

test('without embeddings it reports word overlap alone and says so', () => {
  const md = renderDiversity(measureDiversity(run, chunks, null));
  assert.match(md, /Embeddings were not loaded/);
  assert.match(md, /words only/);
  assert.doesNotMatch(md, /Embedding @3/);
});

test('when the two measures disagree the report says so instead of picking one', () => {
  // Same words, different vectors: embedding says varied, word overlap says identical.
  const same = QUERIES.flatMap((q) => varIds(q).map((id) => chunk(id, 'identical words every time')))
    .concat(QUERIES.flatMap((q) => dupIds(q).map((id) => chunk(id, 'identical words every time'))));
  const comps = compareDiversity(measureDiversity(run, same, embeddings));
  const c = comps.find((x) => x.family === 'fusion' && x.label.includes('vector') && x.k === 5)!;
  assert.equal(c.embedding!.verdict, 'more varied');
  assert.equal(c.words.verdict, "can't tell");
  const md = renderDiversity(measureDiversity(run, same, embeddings));
  assert.match(md, /measures disagree/);
});

test('the rendered report leads with the two questions the article needs answered', () => {
  const md = renderDiversity(measureDiversity(run, chunks, embeddings));
  assert.match(md, /Did fusion make the top 5 more varied/);
  assert.match(md, /Did reranking change how varied the top 5 is/);
  assert.match(md, /Both measures agree: \*\*more varied\*\*/);
  assert.match(md, /No detectable change on either measure/);
});

test('a words-only summary never claims two measures agreed', () => {
  const md = renderDiversity(measureDiversity(run, chunks, null));
  assert.doesNotMatch(md, /either measure|Both measures agree/);
  assert.match(md, /one measure, not two/);
});
