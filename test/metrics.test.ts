import test from 'node:test';
import assert from 'node:assert/strict';
import {
  dcg, jaccardAt, kendallTau, mean, mrrAt, ndcgAt, percentile, rankOfBest, recallAt, stddev,
} from '../src/bench/metrics.js';
import type { Grade, RankedResult } from '../src/types.js';

const ranked = (...ids: string[]): RankedResult[] =>
  ids.map((chunkId, i) => ({ chunkId, rank: i + 1, score: 1 - i / 100 }));

test('percentile interpolates and handles the edges', () => {
  const xs = [1, 2, 3, 4];
  assert.equal(percentile(xs, 0), 1);
  assert.equal(percentile(xs, 100), 4);
  assert.equal(percentile(xs, 50), 2.5);
  assert.equal(percentile([5], 50), 5);
  assert.ok(Number.isNaN(percentile([], 50)));
});

test('percentile is order independent', () => {
  assert.equal(percentile([9, 1, 5, 3], 50), percentile([1, 3, 5, 9], 50));
});

test('mean and stddev use the sample standard deviation', () => {
  assert.equal(mean([2, 4, 6]), 4);
  assert.equal(stddev([2, 4, 6]), 2); // sample sd of {2,4,6} is 2
  assert.equal(stddev([7]), 0);
});

test('dcg uses exponential gain with log2 discount', () => {
  // grade 3 at rank 1 -> (2^3-1)/log2(2) = 7 ; grade 1 at rank 2 -> (2^1-1)/log2(3)
  const expected = 7 + 1 / Math.log2(3);
  assert.ok(Math.abs(dcg([3, 1], 2) - expected) < 1e-12);
});

test('ndcg is 1 for the ideal ordering and lower when the best result is buried', () => {
  const judgments: Record<string, Grade> = { a: 3, b: 2, c: 1 };
  assert.ok(Math.abs(ndcgAt(ranked('a', 'b', 'c'), judgments, 3) - 1) < 1e-12);
  const buried = ndcgAt(ranked('c', 'b', 'a'), judgments, 3);
  assert.ok(buried < 1 && buried > 0, `expected 0 < ${buried} < 1`);
});

test('ndcg respects the cutoff', () => {
  const judgments: Record<string, Grade> = { a: 3 };
  assert.equal(ndcgAt(ranked('x', 'a'), judgments, 1), 0);
  assert.ok(ndcgAt(ranked('x', 'a'), judgments, 2) > 0);
});

test('ndcg returns NaN when a query has no positive judgments', () => {
  assert.ok(Number.isNaN(ndcgAt(ranked('a'), {}, 10)));
});

test('recall counts grade 2 and above only', () => {
  const judgments: Record<string, Grade> = { a: 3, b: 2, c: 1 };
  assert.equal(recallAt(ranked('a', 'c'), judgments, 10), 0.5); // found a, missed b; c does not count
  assert.equal(recallAt(ranked('a', 'b'), judgments, 10), 1);
  assert.equal(recallAt(ranked('c'), judgments, 10), 0);
});

test('mrr is the reciprocal rank of the first relevant hit', () => {
  const judgments: Record<string, Grade> = { a: 3, b: 2, c: 1 };
  assert.equal(mrrAt(ranked('a'), judgments, 10), 1);
  assert.equal(mrrAt(ranked('c', 'b'), judgments, 10), 0.5);
  assert.equal(mrrAt(ranked('c', 'x'), judgments, 10), 0);
});

test('rankOfBest locates the highest graded chunk', () => {
  const judgments: Record<string, Grade> = { a: 3, b: 2 };
  assert.equal(rankOfBest(ranked('b', 'a'), judgments), 2);
  assert.equal(rankOfBest(ranked('b'), judgments), null);
});

test('jaccard compares top-k membership', () => {
  assert.equal(jaccardAt(ranked('a', 'b'), ranked('b', 'a'), 2), 1);
  assert.equal(jaccardAt(ranked('a', 'b'), ranked('a', 'c'), 2), 1 / 3);
  assert.equal(jaccardAt(ranked('a'), ranked('z'), 1), 0);
});

test('kendall tau is 1 for identical orderings and -1 for reversed', () => {
  assert.equal(kendallTau(ranked('a', 'b', 'c'), ranked('a', 'b', 'c')), 1);
  assert.equal(kendallTau(ranked('a', 'b', 'c'), ranked('c', 'b', 'a')), -1);
  assert.ok(Number.isNaN(kendallTau(ranked('a'), ranked('a'))));
});

test('kendall tau ignores items missing from one list', () => {
  // common items are a,b in the same relative order
  assert.equal(kendallTau(ranked('a', 'z', 'b'), ranked('a', 'b'), ), 1);
});

import { bootstrapCI, cv, median, pairedDifferences } from '../src/bench/metrics.js';

test('median and cv', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([1, 2, 3, 4]), 2.5);
  assert.equal(cv([2, 4, 6]), 2 / 4);
  assert.ok(Number.isNaN(cv([])));
});

test('bootstrap CI brackets the statistic, is seeded, and narrows with more data', () => {
  const xs = Array.from({ length: 40 }, (_, i) => 100 + (i % 7));
  const ci = bootstrapCI(xs, median, 500, 7);
  assert.ok(ci.lower <= median(xs) && median(xs) <= ci.upper);
  assert.deepEqual(bootstrapCI(xs, median, 500, 7), ci, 'same seed must give the same interval');
  const wide = bootstrapCI([100, 130, 90, 140, 105], median, 500, 7);
  assert.ok(wide.upper - wide.lower > ci.upper - ci.lower);
  assert.ok(Number.isNaN(bootstrapCI([], median).lower));
});

test('paired differences match observations by key and skip unmatched ones', () => {
  const a = new Map([['q1:0', 10], ['q2:0', 20], ['q3:0', 30]]);
  const b = new Map([['q1:0', 4], ['q2:0', 5]]);
  assert.deepEqual(pairedDifferences(a, b), [6, 15]);
});

test('median of paired differences is not the difference of medians', () => {
  const control = new Map([['a', 100], ['b', 200], ['c', 300]]);
  const treatment = new Map([['a', 180], ['b', 210], ['c', 340]]);
  // Per-observation added cost is [80, 10, 40]; its median is 40.
  assert.equal(median(pairedDifferences(treatment, control)), 40);
  // The marginal medians are 210 and 200, so the naive subtraction says 10.
  assert.equal(median([...treatment.values()]) - median([...control.values()]), 10);
});
