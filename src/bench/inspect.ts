/**
 * Per-iteration forensics on a finished run.
 *
 * The summary report aggregates, and three things are invisible in an aggregate.
 *
 * How many candidates were really scored. `Stage.candidateCount` is what a stage
 * *asked* for; lexical retrieval routinely returns fewer, and it returns a different
 * number for every query. A per-candidate cost divided by the requested depth is
 * therefore wrong, and wrong by a factor that grows with depth.
 *
 * Whether a stage was deterministic. Quality in the summary is read from the first
 * iteration of each query on the stated assumption that it was. Nothing checked.
 *
 * What the slowest iteration actually cost. p95 and p99 span every query and
 * iteration together, so with hundreds of observations one pathological iteration
 * sits below p99 and never surfaces.
 *
 * The in-database path cannot count its own scored rows: it is one statement and only
 * the top-K comes back. It records the requested depth instead. Where a run also
 * measured the application arm at the same retrieval and depth, and candidate parity
 * confirmed the two arms saw identical candidates, the measured count is borrowed
 * from there. That is stated in the output wherever it happens, never assumed.
 */
import { median } from './metrics.js';
import type { BenchRun, IterationResult, Stage, StageRun } from '../types.js';

/**
 * Where a stage's candidate counts came from.
 *
 * `measured` - the application arm counted the candidates it was handed.
 * `borrowed` - taken from the application arm at the same retrieval and depth, which
 *   candidate parity confirmed saw an identical candidate set.
 * `requested` - the depth the stage asked for, because no application arm ran. Correct
 *   where retrieval always fills the pool, an overstatement where it runs out of matches.
 */
export type CountSource = 'measured' | 'borrowed' | 'requested';

/** ms = intercept + slope * candidates, fitted by least squares. */
export interface CostModel {
  key: string;
  /** Fixed cost per statement, independent of how many candidates are scored. */
  interceptMs: number;
  /** Marginal cost of one more candidate. */
  slopeMsPerCandidate: number;
  r2: number;
  points: number;
  /** Distinct candidate counts the fit is built from. One value cannot separate a+bx. */
  distinctCandidateCounts: number;
}

export interface QueryOutlier {
  stageId: string;
  queryId: string;
  n: number;
  minMs: number;
  medianMs: number;
  maxMs: number;
  spread: number;
}

export interface StageInspection {
  stageId: string;
  stage: Stage;
  requestedCandidates: number;
  /** Measured, or borrowed from the paired application stage. Null when neither exists. */
  scoredMedian: number | null;
  scoredMin: number | null;
  scoredMax: number | null;
  countsFrom: CountSource;
  /** Queries whose scored-candidate count changed between iterations of that same query. */
  unstableCounts: string[];
  nondeterministic: string[];
  outliers: QueryOutlier[];
  p50Ms: number;
  msPerCandidate: number | null;
}

const groupByQuery = (run: StageRun): Map<string, IterationResult[]> => {
  const map = new Map<string, IterationResult[]>();
  for (const it of run.iterations) {
    const arr = map.get(it.queryId) ?? [];
    arr.push(it);
    map.set(it.queryId, arr);
  }
  return map;
};

const ordering = (it: IterationResult): string => it.results.map((r) => r.chunkId).join(',');

const pairKey = (s: Stage): string => `${s.retrieval}|${s.candidateCount}|${s.topK}`;

/** Stages that really score candidates. Baselines and controls do not, by construction. */
const scores = (s: Stage): boolean => s.role === 'treatment' && s.reranker !== 'none';

/**
 * Per-query measured candidate counts, keyed by retrieval and depth, taken from the
 * application arm - the only arm that can see how many candidates it was handed.
 */
function measuredByPair(run: BenchRun): Map<string, Map<string, number>> {
  const out = new Map<string, Map<string, number>>();
  for (const sr of run.stages) {
    if (!scores(sr.stage) || sr.stage.reranker !== 'app') continue;
    const perQuery = new Map<string, number>();
    for (const [queryId, its] of groupByQuery(sr)) {
      perQuery.set(queryId, median(its.map((i) => i.candidatesScored)));
    }
    out.set(pairKey(sr.stage), perQuery);
  }
  return out;
}

function fit(points: readonly { x: number; y: number }[]): Omit<CostModel, 'key'> {
  const n = points.length;
  const xs = points.map((p) => p.x);
  const distinct = new Set(xs).size;
  const base = { points: n, distinctCandidateCounts: distinct };
  if (n < 2 || distinct < 2) {
    return { interceptMs: NaN, slopeMsPerCandidate: NaN, r2: NaN, ...base };
  }
  const xbar = xs.reduce((a, b) => a + b, 0) / n;
  const ybar = points.reduce((a, p) => a + p.y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (const p of points) {
    sxy += (p.x - xbar) * (p.y - ybar);
    sxx += (p.x - xbar) ** 2;
  }
  const slope = sxy / sxx;
  const intercept = ybar - slope * xbar;
  let ssRes = 0;
  let ssTot = 0;
  for (const p of points) {
    ssRes += (p.y - (intercept + slope * p.x)) ** 2;
    ssTot += (p.y - ybar) ** 2;
  }
  return {
    interceptMs: intercept,
    slopeMsPerCandidate: slope,
    r2: ssTot === 0 ? NaN : 1 - ssRes / ssTot,
    ...base,
  };
}

function resolveCounts(
  run: StageRun,
  measured: Map<string, Map<string, number>>,
): { perQuery: Map<string, number>; source: CountSource } {
  const byQuery = groupByQuery(run);
  if (run.stage.reranker === 'app' && scores(run.stage)) {
    const perQuery = new Map<string, number>();
    for (const [q, its] of byQuery) perQuery.set(q, median(its.map((i) => i.candidatesScored)));
    return { perQuery, source: 'measured' };
  }
  const borrowed = measured.get(pairKey(run.stage));
  if (borrowed && borrowed.size > 0) return { perQuery: borrowed, source: 'borrowed' };
  const perQuery = new Map<string, number>();
  for (const [q, its] of byQuery) perQuery.set(q, median(its.map((i) => i.candidatesScored)));
  return { perQuery, source: 'requested' };
}

export function inspectStage(
  run: StageRun,
  measured: Map<string, Map<string, number>>,
  spreadThreshold = 5,
): StageInspection {
  const byQuery = groupByQuery(run);
  const totals = run.iterations.map((i) => i.timings.total);
  const { perQuery, source } = resolveCounts(run, measured);
  const ownCounts = source === 'measured';

  const nondeterministic: string[] = [];
  const unstableCounts: string[] = [];
  const outliers: QueryOutlier[] = [];
  const counts: number[] = [];

  for (const [queryId, its] of byQuery) {
    if (new Set(its.map(ordering)).size > 1) nondeterministic.push(queryId);
    // Only variation *within* one query is a defect. Different queries matching
    // different numbers of documents is what lexical retrieval does.
    if (ownCounts && new Set(its.map((i) => i.candidatesScored)).size > 1) {
      unstableCounts.push(queryId);
    }
    const ts = its.map((i) => i.timings.total);
    const minMs = Math.min(...ts);
    const maxMs = Math.max(...ts);
    const spread = minMs > 0 ? maxMs / minMs : Infinity;
    if (spread > spreadThreshold) {
      outliers.push({ stageId: run.stage.id, queryId, n: ts.length, minMs, medianMs: median(ts), maxMs, spread });
    }
    const c = perQuery.get(queryId);
    if (c !== undefined && c > 0) counts.push(c);
  }
  outliers.sort((a, b) => b.spread - a.spread);

  const scoredMedian = counts.length > 0 ? median(counts) : null;
  const p50Ms = median(totals);
  return {
    stageId: run.stage.id,
    stage: run.stage,
    requestedCandidates: run.stage.candidateCount,
    scoredMedian,
    scoredMin: counts.length > 0 ? Math.min(...counts) : null,
    scoredMax: counts.length > 0 ? Math.max(...counts) : null,
    countsFrom: source,
    unstableCounts,
    nondeterministic,
    outliers,
    p50Ms,
    msPerCandidate: scoredMedian && scoredMedian > 0 ? p50Ms / scoredMedian : null,
  };
}

/**
 * Fit cost per candidate and cost per statement, separately for each retrieval and
 * execution location.
 *
 * Every (query, depth) pair contributes a point: how many candidates that query really
 * had, against that query's median time. Lexical retrieval returns a different number
 * of candidates for every query, so its points span a range of candidate counts inside
 * a single stage - which means the fit has something to separate a fixed cost from a
 * per-candidate one even before comparing across depths.
 */
export function costModels(run: BenchRun): CostModel[] {
  const measured = measuredByPair(run);
  const buckets = new Map<string, { x: number; y: number }[]>();

  for (const sr of run.stages) {
    if (!scores(sr.stage)) continue;
    const { perQuery } = resolveCounts(sr, measured);
    const key = `${sr.stage.retrieval} · ${sr.stage.reranker === 'app' ? 'application' : 'in database'}`;
    const pts = buckets.get(key) ?? [];
    for (const [queryId, its] of groupByQuery(sr)) {
      const x = perQuery.get(queryId);
      if (x === undefined || x <= 0) continue;
      // The median over a query's iterations, so one stalled iteration cannot move a point.
      pts.push({ x, y: median(its.map((i) => i.timings.total)) });
    }
    buckets.set(key, pts);
  }

  return [...buckets.entries()]
    .map(([key, pts]) => ({ key, ...fit(pts) }))
    .sort((a, b) => a.key.localeCompare(b.key));
}

const fmt = (n: number | null, d = 1): string =>
  n !== null && Number.isFinite(n)
    ? n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })
    : '—';

export function renderInspection(run: BenchRun): string {
  const measured = measuredByPair(run);
  const all = run.stages.filter((s) => s.iterations.length > 0).map((s) => inspectStage(s, measured));
  const scoring = all.filter((r) => scores(r.stage));
  const out: string[] = [];

  out.push('# Per-iteration inspection');
  out.push('');

  out.push('## Cost model: fixed cost per statement against cost per candidate');
  out.push('');
  out.push('Least squares over every (query, depth) pair: the candidates that query really had,');
  out.push('against that query\'s median time. A path that only pays for the work it does fits an');
  out.push('intercept near zero. A large intercept is a cost the statement pays before it scores');
  out.push('anything, and no amount of batching or bigger hardware divides it away.');
  out.push('');
  out.push('| Retrieval and location | Fixed (ms) | Per candidate (ms) | R² | Points | Distinct depths |');
  out.push('|---|---:|---:|---:|---:|---:|');
  for (const m of costModels(run)) {
    out.push(`| ${m.key} | **${fmt(m.interceptMs)}** | **${fmt(m.slopeMsPerCandidate, 2)}** | ${fmt(m.r2, 3)} | ${m.points} | ${m.distinctCandidateCounts} |`);
  }
  out.push('');

  out.push('## Candidates requested against candidates scored');
  out.push('');
  out.push('Only stages that score are listed; baselines and controls do not run the model.');
  out.push('`Scored` is measured for the application arm. The in-database arm is one statement');
  out.push('that returns only its top-K, so it cannot count its own scored rows; where the run');
  out.push('measured the application arm at the same retrieval and depth, and candidate parity');
  out.push('confirmed both arms saw identical candidates, that measured count is shown and');
  out.push('marked borrowed. Where no application arm ran, the requested depth is shown and');
  out.push('marked as such: correct where retrieval always fills the pool, an overstatement');
  out.push('where it runs out of matches.');
  out.push('');
  out.push('| Stage | Requested | Scored min/med/max | Source | p50 (ms) | ms per candidate |');
  out.push('|---|---:|---:|---|---:|---:|');
  for (const r of scoring) {
    const span = r.scoredMedian === null
      ? '—'
      : r.scoredMin === r.scoredMax
        ? `${fmt(r.scoredMedian, 0)}`
        : `${fmt(r.scoredMin, 0)} / ${fmt(r.scoredMedian, 0)} / ${fmt(r.scoredMax, 0)}`;
    const source = r.scoredMedian === null ? '—' : r.countsFrom;
    const short = r.requestedCandidates !== r.scoredMedian && r.scoredMedian !== null ? ' ⚠' : '';
    out.push(`| ${r.stageId} | ${r.requestedCandidates} | ${span}${short} | ${source} | ${fmt(r.p50Ms)} | ${fmt(r.msPerCandidate, 2)} |`);
  }
  out.push('');
  out.push('⚠ marks a stage that scored fewer candidates than its name claims, because retrieval');
  out.push('ran out of matches. Two such stages at different depths measure the same work.');
  out.push('');

  const unstable = scoring.filter((r) => r.unstableCounts.length > 0);
  out.push('## Queries whose candidate count changed between iterations');
  out.push('');
  out.push('Different queries matching different numbers of documents is retrieval working');
  out.push('correctly. The same query being handed a different number of candidates on a later');
  out.push('iteration is not, and would mean a stage mixes two populations in one median.');
  out.push('');
  if (unstable.length === 0) {
    out.push('None. Every query was handed the same number of candidates on every iteration.');
  } else {
    out.push('| Stage | Queries |');
    out.push('|---|---|');
    for (const r of unstable) out.push(`| ${r.stageId} | ${r.unstableCounts.join(', ')} |`);
  }
  out.push('');

  const nd = all.filter((r) => r.nondeterministic.length > 0);
  out.push('## Stages that did not return the same ordering every iteration');
  out.push('');
  out.push('The summary reads quality from the first iteration of each query, on the assumption');
  out.push('that these pipelines are deterministic. Where that holds, the reported nDCG describes');
  out.push('every iteration. Where it does not, it describes one.');
  out.push('');
  if (nd.length === 0) {
    out.push('None. Every stage returned an identical ordering on every iteration of every query,');
    out.push('so the quality figures in the summary are not first-iteration artifacts.');
  } else {
    out.push('| Stage | Queries that varied |');
    out.push('|---|---|');
    for (const r of nd) out.push(`| ${r.stageId} | ${r.nondeterministic.join(', ')} |`);
  }
  out.push('');

  const outliers = all.flatMap((r) => r.outliers).sort((a, b) => b.spread - a.spread);
  out.push('## Queries whose slowest iteration far exceeded their fastest');
  out.push('');
  out.push('p95 and p99 span every query and iteration together, so with hundreds of observations');
  out.push('a single stalled iteration sits below p99 and never reaches the summary. Every figure');
  out.push('this benchmark reports is a median or a median of paired differences, so one stalled');
  out.push('iteration does not move them - but a stall of minutes is a fact about the machine');
  out.push('during the run, and it belongs on the record rather than inside a percentage.');
  out.push('');
  if (outliers.length === 0) {
    out.push('None beyond 5x.');
  } else {
    out.push('| Stage | Query | n | min (ms) | median (ms) | max (ms) | max/min |');
    out.push('|---|---|---:|---:|---:|---:|---:|');
    for (const o of outliers) {
      out.push(`| ${o.stageId} | ${o.queryId} | ${o.n} | ${fmt(o.minMs)} | ${fmt(o.medianMs)} | ${fmt(o.maxMs)} | ${fmt(o.spread)}x |`);
    }
  }
  out.push('');
  return out.join('\n');
}
