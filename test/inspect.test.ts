import test from 'node:test';
import assert from 'node:assert/strict';
import { costModels, inspectStage, renderInspection } from '../src/bench/inspect.js';
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
  order: string[] = ['a'],
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

const bench = (stages: StageRun[]): BenchRun => ({ stages }) as unknown as BenchRun;

const noMeasured = new Map<string, Map<string, number>>();

test('new in-database observations use their measured counts without an application arm', () => {
  const observation = { ...it('q1', 0, 1900, 19), candidateCountSource: 'measured' as const };
  const r = inspectStage(run([observation], { reranker: 'in-db' }), noMeasured);
  assert.equal(r.countsFrom, 'measured');
  assert.equal(r.scoredMedian, 19);
});

test('per-candidate cost divides by candidates scored, not candidates requested', () => {
  const r = inspectStage(run([it('q1', 0, 1000, 20), it('q1', 1, 1000, 20)]), noMeasured);
  assert.equal(r.requestedCandidates, 40);
  assert.equal(r.scoredMedian, 20);
  assert.equal(r.msPerCandidate, 50);
  assert.equal(r.countsFrom, 'measured');
});

test('different queries scoring different depths is not flagged as instability', () => {
  // Lexical retrieval hands every query a different number of matches. That is retrieval
  // working, not a stage mixing populations.
  const r = inspectStage(run([
    it('q1', 0, 200, 2), it('q1', 1, 200, 2),
    it('q2', 0, 4000, 40), it('q2', 1, 4000, 40),
  ]), noMeasured);
  assert.deepEqual(r.unstableCounts, []);
  assert.equal(r.scoredMin, 2);
  assert.equal(r.scoredMax, 40);
});

test('one query handed different depths across its own iterations is flagged', () => {
  const r = inspectStage(run([it('q1', 0, 4000, 40), it('q1', 1, 2000, 20)]), noMeasured);
  assert.deepEqual(r.unstableCounts, ['q1']);
});

test('the in-database arm borrows measured counts from the paired application stage', () => {
  const appStage = run([it('q1', 0, 1300, 19), it('q1', 1, 1300, 19)], {
    id: 'lexical+rerank-app@40', retrieval: 'lexical', reranker: 'app',
  });
  // The in-database arm recorded the requested depth, 40, because it cannot see its own rows.
  const dbStage = run([it('q1', 0, 1460, 40), it('q1', 1, 1460, 40)], {
    id: 'lexical+rerank-in-db@40', retrieval: 'lexical', reranker: 'in-db',
  });
  const md = renderInspection(bench([appStage, dbStage]));
  assert.match(md, /lexical\+rerank-in-db@40.*borrowed/);
  // 1460 / 19 = 76.8, not 1460 / 40 = 36.5.
  assert.match(md, /lexical\+rerank-in-db@40.*76\.84/);
});

test('a fixed cost per statement is recovered from a run that has one', () => {
  // 2,400 ms before scoring anything, then 75 ms per candidate.
  const stages = [10, 20, 40, 80].map((n) => run(
    [it('q1', 0, 2400 + 75 * n, n), it('q1', 1, 2400 + 75 * n, n)],
    { id: `vector+rerank-in-db@${n}`, reranker: 'in-db', candidateCount: n },
  ));
  const model = costModels(bench(stages)).find((m) => m.key.includes('in database'));
  assert.ok(model);
  assert.ok(Math.abs(model.interceptMs - 2400) < 1, `intercept ${model.interceptMs}`);
  assert.ok(Math.abs(model.slopeMsPerCandidate - 75) < 0.1, `slope ${model.slopeMsPerCandidate}`);
  assert.ok(model.r2 > 0.999);
});

test('a path that pays only for work it does fits an intercept near zero', () => {
  const stages = [10, 20, 40, 80].map((n) => run(
    [it('q1', 0, 63 * n, n), it('q1', 1, 63 * n, n)],
    { id: `vector+rerank-app@${n}`, candidateCount: n },
  ));
  const model = costModels(bench(stages)).find((m) => m.key.includes('application'));
  assert.ok(model);
  assert.ok(Math.abs(model.interceptMs) < 1, `intercept ${model.interceptMs}`);
  assert.ok(Math.abs(model.slopeMsPerCandidate - 63) < 0.1);
});

test('a fit from a single candidate depth reports no slope rather than a fabricated one', () => {
  const model = costModels(bench([run([it('q1', 0, 1000, 40), it('q2', 0, 1200, 40)])]))[0]!;
  assert.equal(model.distinctCandidateCounts, 1);
  assert.ok(Number.isNaN(model.slopeMsPerCandidate));
  assert.ok(Number.isNaN(model.interceptMs));
});

test('a stalled iteration is reported with its absolute times and does not move the median', () => {
  const iterations = Array.from({ length: 19 }, (_, i) => it('q12', i, 2600, 40));
  iterations.push(it('q12', 19, 3_637_452, 40));
  const r = inspectStage(run(iterations), noMeasured);
  assert.equal(r.outliers.length, 1);
  assert.equal(r.outliers[0]!.maxMs, 3_637_452);
  assert.equal(r.outliers[0]!.medianMs, 2600);
  assert.equal(r.p50Ms, 2600);
});

test('an ordering that changes between iterations is caught', () => {
  const r = inspectStage(run([
    it('q1', 0, 100, 40, ['a', 'b']),
    it('q1', 1, 100, 40, ['b', 'a']),
    it('q2', 0, 100, 40, ['c', 'd']),
    it('q2', 1, 100, 40, ['c', 'd']),
  ]), noMeasured);
  assert.deepEqual(r.nondeterministic, ['q1']);
});

test('baselines and controls are left out of the scoring table', () => {
  const control = run([it('q1', 0, 20, 0)], { id: 'vector+control-app@40', role: 'control' });
  const treatment = run([it('q1', 0, 2520, 40)]);
  const md = renderInspection(bench([control, treatment]));
  assert.ok(!md.includes('vector+control-app@40 |'), 'control should not appear as a scoring row');
  assert.match(md, /vector\+rerank-app@40/);
});

test('the rendered report states plainly when nothing is wrong', () => {
  const md = renderInspection(bench([run([it('q1', 0, 100, 40), it('q1', 1, 101, 40)])]));
  assert.match(md, /Every query was handed the same number of candidates/);
  assert.match(md, /not first-iteration artifacts/);
  assert.match(md, /None beyond 5x/);
});

test('with no application arm the fit falls back to requested depth and says so', () => {
  // The probe runs `--rerankers in-db` alone. Vector retrieval always fills the pool, so
  // requested equals actual there and the fit is sound - but the source has to be visible.
  const stages = [10, 20, 40].map((n) => run(
    [it('q1', 0, 2400 + 75 * n, n)],
    { id: `vector+rerank-in-db@${n}`, reranker: 'in-db', candidateCount: n },
  ));
  const model = costModels(bench(stages)).find((m) => m.key.includes('in database'));
  assert.ok(model);
  assert.ok(Math.abs(model.interceptMs - 2400) < 1, `intercept ${model.interceptMs}`);
  assert.match(renderInspection(bench(stages)), /requested/);
});
