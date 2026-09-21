import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectStage, renderInspection } from '../src/bench/inspect.js';
import type { BenchRun, IterationResult, Stage, StageRun } from '../src/types.js';

const stage = (over: Partial<Stage> = {}): Stage => ({
  id: 'vector+rerank-app@40',
  label: 'Vector + application rerank (N=40)',
  retrieval: 'vector',
  reranker: 'app',
  candidateCount: 40,
  topK: 10,
  role: 'treatment',
  ...over,
} as Stage);

const it = (
  queryId: string,
  iteration: number,
  total: number,
  scored: number,
  order: string[],
): IterationResult => ({
  queryId,
  iteration,
  repeat: 0,
  attribution: 'split',
  timings: { candidates: 0, transfer: 0, tokenize: 0, infer: total, sort: 0, total },
  results: order.map((chunkId, i) => ({ chunkId, rank: i + 1, score: 1 - i / 100 })),
  bytesFromDb: 0,
  candidatesScored: scored,
});

const run = (iterations: IterationResult[], over: Partial<Stage> = {}): StageRun =>
  ({ stage: stage(over), iterations }) as StageRun;

test('per-candidate cost divides by candidates scored, not candidates requested', () => {
  // Retrieval saturated at 20 even though the stage asked for 40. Dividing by 40
  // would report half the true cost.
  const r = inspectStage(run([
    it('q1', 0, 1000, 20, ['a']),
    it('q1', 1, 1000, 20, ['a']),
  ]));
  assert.equal(r.requestedCandidates, 40);
  assert.equal(r.scoredMedian, 20);
  assert.equal(r.msPerCandidate, 50);
});

test('a stage scoring different depths on different iterations is flagged', () => {
  const r = inspectStage(run([
    it('q1', 0, 1000, 40, ['a']),
    it('q1', 1, 500, 20, ['a']),
  ]));
  assert.equal(r.scoredMin, 20);
  assert.equal(r.scoredMax, 40);
});

test('an ordering that changes between iterations is caught', () => {
  const r = inspectStage(run([
    it('q1', 0, 100, 40, ['a', 'b']),
    it('q1', 1, 100, 40, ['b', 'a']),
    it('q2', 0, 100, 40, ['c', 'd']),
    it('q2', 1, 100, 40, ['c', 'd']),
  ]));
  assert.deepEqual(r.nondeterministic, ['q1']);
});

test('a single pathological iteration is reported with its absolute times', () => {
  // Nineteen iterations at 2,600 ms and one at 60,000 ms. Across many queries this
  // sits above p99 and never reaches the summary report.
  const iterations = Array.from({ length: 19 }, (_, i) => it('q12', i, 2600, 40, ['a']));
  iterations.push(it('q12', 19, 60000, 40, ['a']));
  const r = inspectStage(run(iterations));
  assert.equal(r.outliers.length, 1);
  assert.equal(r.outliers[0]!.queryId, 'q12');
  assert.equal(r.outliers[0]!.maxMs, 60000);
  assert.equal(r.outliers[0]!.minMs, 2600);
  assert.ok(r.outliers[0]!.spread > 20);
});

test('a quiet stage produces no outliers and no warnings', () => {
  const r = inspectStage(run([
    it('q1', 0, 100, 40, ['a']),
    it('q1', 1, 104, 40, ['a']),
  ]));
  assert.deepEqual(r.outliers, []);
  assert.deepEqual(r.nondeterministic, []);
});

test('the rendered report states plainly when nothing is wrong', () => {
  const benchRun = {
    stages: [run([it('q1', 0, 100, 40, ['a']), it('q1', 1, 101, 40, ['a'])])],
  } as unknown as BenchRun;
  const md = renderInspection(benchRun);
  assert.match(md, /Every stage scored the same number of candidates/);
  assert.match(md, /not first-iteration artifacts/);
  assert.match(md, /None beyond 5x/);
});
