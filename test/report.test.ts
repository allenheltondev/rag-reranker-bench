import test from 'node:test';
import assert from 'node:assert/strict';
import { scoringCosts, transferCosts } from '../src/bench/report.js';
import { runConfigFromEnv } from '../src/config.js';
import type { BenchRun, IterationResult, Stage, StageRun } from '../src/types.js';

const stage = (over: Partial<Stage>): Stage => ({
  id: 'x', label: 'x', retrieval: 'hybrid-rrf', reranker: 'in-db', candidateCount: 40, topK: 10,
  role: 'treatment', batch: 'in-db', ...over,
});

const obs = (queryId: string, repeat: number, iteration: number, total: number): IterationResult => ({
  queryId, repeat, iteration, attribution: 'total-only',
  timings: { candidates: 0, transfer: 0, tokenize: 0, infer: 0, sort: 0, total },
  results: [], bytesFromDb: 0, candidatesScored: 0,
});

function run(stages: StageRun[], repeats = 1): BenchRun {
  return {
    startedAt: '', finishedAt: '',
    config: runConfigFromEnv({ repeats, seed: 1 }),
    environment: { node: '', platform: '', arch: '', cpus: 1, cpuModel: '', totalMemMb: 0 },
    stages,
  };
}

test('scoring cost is the median of per-observation differences, matched on query, repeat and iteration', () => {
  const treatment = stage({ id: 't', group: 'in-db:hybrid-rrf@40' });
  const control = stage({ id: 'c', role: 'control', group: 'in-db:hybrid-rrf@40' });
  const r = run([
    // Drift climbs across the run; per-observation added cost is 100 in repeat 0 and 120 in repeat 1.
    { stage: treatment, iterations: [obs('q', 0, 0, 1100), obs('q', 0, 1, 1600), obs('q', 1, 0, 2120), obs('q', 1, 1, 2620)] },
    { stage: control, iterations: [obs('q', 0, 0, 1000), obs('q', 0, 1, 1500), obs('q', 1, 0, 2000), obs('q', 1, 1, 2500)] },
  ], 2);
  const [all] = scoringCosts(r);
  assert.equal(all!.pairs, 4);
  assert.equal(all!.medianDelta, 110);
  assert.equal(all!.meanDelta, 110);
  assert.ok(all!.ci.lower <= 110 && 110 <= all!.ci.upper);
  assert.equal(scoringCosts(r, 0)[0]!.medianDelta, 100);
  assert.equal(scoringCosts(r, 1)[0]!.medianDelta, 120);
});

test('an observation without a partner is dropped rather than paired with a neighbour', () => {
  const treatment = stage({ id: 't', group: 'g' });
  const control = stage({ id: 'c', role: 'control', group: 'g' });
  const r = run([
    { stage: treatment, iterations: [obs('q', 0, 0, 150), obs('q', 0, 1, 999)] },
    { stage: control, iterations: [obs('q', 0, 0, 100)] },
  ]);
  assert.equal(scoringCosts(r)[0]!.pairs, 1);
  assert.equal(scoringCosts(r)[0]!.medianDelta, 50);
});

test('transfer cost comes from the transfer batch only, never from the arm controls', () => {
  const armApp = stage({ id: 'ca', role: 'control', reranker: 'app', batch: 'app', group: 'app:hybrid-rrf@40' });
  const armDb = stage({ id: 'cd', role: 'control', reranker: 'in-db', batch: 'in-db', group: 'in-db:hybrid-rrf@40' });
  const xferApp = stage({ id: 'xa', role: 'control', reranker: 'app', batch: 'transfer', group: 'transfer:hybrid-rrf@40' });
  const xferDb = stage({ id: 'xd', role: 'control', reranker: 'in-db', batch: 'transfer', group: 'transfer:hybrid-rrf@40' });
  const r = run([
    { stage: armApp, iterations: [obs('q', 0, 0, 900)] },
    { stage: armDb, iterations: [obs('q', 0, 0, 100)] },
    { stage: xferApp, iterations: [obs('q', 0, 0, 130)] },
    { stage: xferDb, iterations: [obs('q', 0, 0, 100)] },
  ]);
  const rows = transferCosts(r);
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.medianDelta, 30);
});
