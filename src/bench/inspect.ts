/**
 * Per-iteration forensics on a finished run.
 *
 * The summary report aggregates. Aggregates hide two failure modes that would
 * invalidate a comparison without ever looking wrong:
 *
 *  - A stage that scores a different number of candidates than its name claims.
 *    `candidateCount` is what was *requested*; retrieval may return fewer, and a
 *    per-candidate cost computed from the requested depth would then be wrong.
 *  - A stage whose ordering is not deterministic across iterations. Quality in
 *    the summary is read from the first iteration only, so a pipeline that
 *    silently degrades on later iterations still reports the first one's nDCG.
 *
 * Both are checked here against the raw per-iteration record, not re-derived.
 */
import { median } from './metrics.js';
import type { BenchRun, IterationResult, StageRun } from '../types.js';

export interface QueryOutlier {
  queryId: string;
  n: number;
  minMs: number;
  medianMs: number;
  maxMs: number;
  /** max/min. A deterministic pipeline on a quiet box sits near 1. */
  spread: number;
}

export interface StageInspection {
  stageId: string;
  requestedCandidates: number;
  /** What retrieval actually handed the scorer, across every iteration. */
  scoredMin: number;
  scoredMedian: number;
  scoredMax: number;
  /** Median end-to-end, and that divided by candidates actually scored. */
  p50Ms: number;
  msPerCandidate: number;
  /** Queries whose orderings are not identical across every iteration. */
  nondeterministic: string[];
  /** Queries whose slowest iteration is more than `spreadThreshold` times their fastest. */
  outliers: QueryOutlier[];
}

const byQuery = (run: StageRun): Map<string, IterationResult[]> => {
  const map = new Map<string, IterationResult[]>();
  for (const it of run.iterations) {
    const arr = map.get(it.queryId) ?? [];
    arr.push(it);
    map.set(it.queryId, arr);
  }
  return map;
};

const ordering = (it: IterationResult): string => it.results.map((r) => r.chunkId).join(',');

export function inspectStage(run: StageRun, spreadThreshold = 5): StageInspection {
  const scored = run.iterations.map((i) => i.candidatesScored);
  const totals = run.iterations.map((i) => i.timings.total);
  const nondeterministic: string[] = [];
  const outliers: QueryOutlier[] = [];

  for (const [queryId, its] of byQuery(run)) {
    if (new Set(its.map(ordering)).size > 1) nondeterministic.push(queryId);
    const ts = its.map((i) => i.timings.total);
    const minMs = Math.min(...ts);
    const maxMs = Math.max(...ts);
    const spread = minMs > 0 ? maxMs / minMs : Infinity;
    if (spread > spreadThreshold) {
      outliers.push({ queryId, n: ts.length, minMs, medianMs: median(ts), maxMs, spread });
    }
  }
  outliers.sort((a, b) => b.spread - a.spread);

  const scoredMedian = median(scored);
  const p50Ms = median(totals);
  return {
    stageId: run.stage.id,
    requestedCandidates: run.stage.candidateCount,
    scoredMin: Math.min(...scored),
    scoredMedian,
    scoredMax: Math.max(...scored),
    p50Ms,
    msPerCandidate: scoredMedian > 0 ? p50Ms / scoredMedian : NaN,
    nondeterministic,
    outliers,
  };
}

const fmt = (n: number, d = 1): string =>
  Number.isFinite(n) ? n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—';

export function renderInspection(run: BenchRun): string {
  const stages = run.stages.filter((s) => s.iterations.length > 0);
  const rows = stages.map((s) => inspectStage(s));
  const out: string[] = [];

  out.push('# Per-iteration inspection');
  out.push('');
  out.push('## Candidates actually scored, and cost per candidate');
  out.push('');
  out.push('`Requested` is the depth the stage asked for. `Scored` is what retrieval returned.');
  out.push('Lexical retrieval runs out of matches before deep N, so a stage can saturate: two');
  out.push('rows with the same `Scored` are measuring the same work whatever their names say.');
  out.push('The last column is the only per-candidate number in this repo derived from candidates');
  out.push('that were really scored rather than from the requested depth.');
  out.push('');
  out.push('| Stage | Requested | Scored min/med/max | p50 (ms) | ms per candidate |');
  out.push('|---|---:|---:|---:|---:|');
  for (const r of rows) {
    const scoredCol = r.scoredMin === r.scoredMax
      ? `${r.scoredMin}`
      : `${r.scoredMin} / ${r.scoredMedian} / ${r.scoredMax}`;
    out.push(`| ${r.stageId} | ${r.requestedCandidates} | ${scoredCol} | ${fmt(r.p50Ms)} | ${fmt(r.msPerCandidate, 2)} |`);
  }
  out.push('');

  const varying = rows.filter((r) => r.scoredMin !== r.scoredMax);
  out.push('## Stages whose scored-candidate count varied between iterations');
  out.push('');
  if (varying.length === 0) {
    out.push('None. Every stage scored the same number of candidates on every iteration, so the');
    out.push('per-candidate costs above are computed against a constant.');
  } else {
    out.push('A stage that scores a different number of candidates on different iterations is not');
    out.push('one measurement. Its p50 mixes two populations and its paired differences mix them too.');
    out.push('');
    out.push('| Stage | min | median | max |');
    out.push('|---|---:|---:|---:|');
    for (const r of varying) out.push(`| ${r.stageId} | ${r.scoredMin} | ${r.scoredMedian} | ${r.scoredMax} |`);
  }
  out.push('');

  const nd = rows.filter((r) => r.nondeterministic.length > 0);
  out.push('## Stages that did not return the same ordering every iteration');
  out.push('');
  out.push('The summary report reads quality from the first iteration of each query, on the stated');
  out.push('assumption that these pipelines are deterministic. Where that assumption holds, the');
  out.push('reported nDCG describes every iteration. Where it does not, it describes one.');
  out.push('');
  if (nd.length === 0) {
    out.push('None. Every stage returned an identical ordering on every iteration of every query,');
    out.push('so the quality figures in the summary report are not first-iteration artifacts.');
  } else {
    out.push('| Stage | Queries that varied |');
    out.push('|---|---|');
    for (const r of nd) out.push(`| ${r.stageId} | ${r.nondeterministic.join(', ')} |`);
  }
  out.push('');

  const withOutliers = rows.filter((r) => r.outliers.length > 0);
  out.push('## Queries whose slowest iteration far exceeded their fastest');
  out.push('');
  out.push('p95 and p99 are taken across every query and iteration together, so with hundreds of');
  out.push('observations a single pathological iteration sits below p99 and never appears in the');
  out.push('summary. The within-query coefficient of variation in the stability table is the only');
  out.push('place it shows, and there it shows as a percentage with no absolute times beside it.');
  out.push('This table gives the times.');
  out.push('');
  if (withOutliers.length === 0) {
    out.push('None beyond 5x. No iteration was pathological enough to distort a within-query spread.');
  } else {
    out.push('| Stage | Query | n | min (ms) | median (ms) | max (ms) | max/min |');
    out.push('|---|---|---:|---:|---:|---:|---:|');
    for (const r of withOutliers) {
      for (const o of r.outliers) {
        out.push(`| ${r.stageId} | ${o.queryId} | ${o.n} | ${fmt(o.minMs)} | ${fmt(o.medianMs)} | ${fmt(o.maxMs)} | ${fmt(o.spread, 1)}x |`);
      }
    }
  }
  out.push('');
  return out.join('\n');
}
