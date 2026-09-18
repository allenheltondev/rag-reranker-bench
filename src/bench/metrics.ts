import type { Grade, Query, RankedResult } from '../types.js';

/**
 * A chunk counts as relevant for the binary metrics (recall, MRR) at grade 2 or above.
 * Grade 1 chunks are "related but not an answer" and only contribute partial gain to nDCG.
 */
export const RELEVANT_AT: Grade = 2;

/** Linear-interpolated percentile over an unsorted sample. Returns NaN for an empty sample. */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return NaN;
  if (p <= 0) return Math.min(...values);
  if (p >= 100) return Math.max(...values);
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (idx - lo);
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return NaN;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function stddev(values: readonly number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((acc, v) => acc + (v - m) ** 2, 0) / (values.length - 1));
}

const gain = (g: number): number => 2 ** g - 1;
const discount = (rank0: number): number => Math.log2(rank0 + 2);

/** Discounted cumulative gain over the first k results, using exponential gain. */
export function dcg(grades: readonly number[], k: number): number {
  let sum = 0;
  for (let i = 0; i < Math.min(k, grades.length); i++) sum += gain(grades[i]!) / discount(i);
  return sum;
}

/**
 * nDCG@k against a judgment map. Queries with no positive judgments return NaN rather than
 * 0 or 1, so they can be excluded from averages instead of silently distorting them.
 */
export function ndcgAt(results: readonly RankedResult[], judgments: Record<string, Grade>, k: number): number {
  const ideal = Object.values(judgments).sort((a, b) => b - a);
  if (ideal.length === 0) return NaN;
  const idealDcg = dcg(ideal, k);
  if (idealDcg === 0) return NaN;
  const actual = results.slice(0, k).map((r) => judgments[r.chunkId] ?? 0);
  return dcg(actual, k) / idealDcg;
}

/** Fraction of relevant chunks (grade >= RELEVANT_AT) that appear in the top k. */
export function recallAt(results: readonly RankedResult[], judgments: Record<string, Grade>, k: number): number {
  const relevant = Object.entries(judgments).filter(([, g]) => g >= RELEVANT_AT).map(([id]) => id);
  if (relevant.length === 0) return NaN;
  const top = new Set(results.slice(0, k).map((r) => r.chunkId));
  return relevant.filter((id) => top.has(id)).length / relevant.length;
}

/** Reciprocal rank of the first relevant chunk within the top k; 0 when none is present. */
export function mrrAt(results: readonly RankedResult[], judgments: Record<string, Grade>, k: number): number {
  const hasRelevant = Object.values(judgments).some((g) => g >= RELEVANT_AT);
  if (!hasRelevant) return NaN;
  for (let i = 0; i < Math.min(k, results.length); i++) {
    if ((judgments[results[i]!.chunkId] ?? 0) >= RELEVANT_AT) return 1 / (i + 1);
  }
  return 0;
}

/** Rank of the best-graded chunk in the result list, 1-based; null if absent. */
export function rankOfBest(results: readonly RankedResult[], judgments: Record<string, Grade>): number | null {
  let bestGrade = 0;
  let bestId: string | null = null;
  for (const [id, g] of Object.entries(judgments)) {
    if (g > bestGrade) { bestGrade = g; bestId = id; }
  }
  if (bestId === null) return null;
  const idx = results.findIndex((r) => r.chunkId === bestId);
  return idx === -1 ? null : idx + 1;
}

/** Set overlap of two top-k lists, |A ∩ B| / |A ∪ B|. */
export function jaccardAt(a: readonly RankedResult[], b: readonly RankedResult[], k: number): number {
  const setA = new Set(a.slice(0, k).map((r) => r.chunkId));
  const setB = new Set(b.slice(0, k).map((r) => r.chunkId));
  if (setA.size === 0 && setB.size === 0) return 1;
  let inter = 0;
  for (const id of setA) if (setB.has(id)) inter++;
  return inter / (setA.size + setB.size - inter);
}

/**
 * Kendall tau-a over the items both lists contain, comparing their orderings.
 * Used as an equivalence check: two runs of the same model over the same candidates
 * should agree at tau ~1.0, and anything lower points at preprocessing differences.
 */
export function kendallTau(a: readonly RankedResult[], b: readonly RankedResult[]): number {
  const posA = new Map(a.map((r, i) => [r.chunkId, i]));
  const posB = new Map(b.map((r, i) => [r.chunkId, i]));
  const common = [...posA.keys()].filter((id) => posB.has(id));
  const n = common.length;
  if (n < 2) return NaN;
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const x = posA.get(common[i]!)! - posA.get(common[j]!)!;
      const y = posB.get(common[i]!)! - posB.get(common[j]!)!;
      const s = Math.sign(x) * Math.sign(y);
      if (s > 0) concordant++;
      else if (s < 0) discordant++;
    }
  }
  return (concordant - discordant) / ((n * (n - 1)) / 2);
}

export interface QualitySummary {
  ndcg: number;
  recall: number;
  mrr: number;
  /**
   * Confidence interval on the mean nDCG, from resampling the QUERY SET.
   *
   * A deterministic pipeline returns the same ordering every iteration, so repeating the run
   * does not make a quality number more certain. The uncertainty is which queries you happened
   * to choose, and on a set this size that uncertainty is large. Reporting nDCG without it
   * invites reading a difference that a different sixteen queries would not reproduce.
   */
  ndcgCI: Interval;
  /** nDCG at several context budgets, keyed by cutoff. */
  ndcgByK: Record<number, number>;
  /** Per-query nDCG at the primary cutoff, for per-query breakdowns and paired comparisons. */
  perQuery: Record<string, number>;
}

/**
 * Average quality across queries. Deterministic pipelines produce one ordering per query,
 * so this takes the first iteration's results for each query rather than averaging
 * identical numbers.
 */
export function summariseQuality(
  resultsByQuery: Map<string, RankedResult[]>,
  queries: readonly Query[],
  k: number,
  cutoffs: readonly number[] = [3, 5, 10],
  seed = 1,
): QualitySummary {
  const ndcgs: number[] = [];
  const recalls: number[] = [];
  const mrrs: number[] = [];
  const perQuery: Record<string, number> = {};
  const byK = new Map<number, number[]>(cutoffs.map((c) => [c, []]));

  for (const q of queries) {
    const results = resultsByQuery.get(q.id);
    if (!results) continue;
    const n = ndcgAt(results, q.judgments, k);
    const r = recallAt(results, q.judgments, k);
    const m = mrrAt(results, q.judgments, k);
    if (!Number.isNaN(n)) { ndcgs.push(n); perQuery[q.id] = n; }
    if (!Number.isNaN(r)) recalls.push(r);
    if (!Number.isNaN(m)) mrrs.push(m);
    for (const cutoff of cutoffs) {
      const v = ndcgAt(results, q.judgments, cutoff);
      if (!Number.isNaN(v)) byK.get(cutoff)!.push(v);
    }
  }

  const ndcgByK: Record<number, number> = {};
  for (const [cutoff, values] of byK) ndcgByK[cutoff] = mean(values);

  return {
    ndcg: mean(ndcgs),
    recall: mean(recalls),
    mrr: mean(mrrs),
    // Resampling queries, not iterations: see the note on ndcgCI.
    ndcgCI: bootstrapCI(ndcgs, mean, 2000, seed),
    ndcgByK,
    perQuery,
  };
}

/**
 * Rows that should never have been retrievable at all.
 *
 * This is not a quality metric. A chunk belonging to another tenant, another user, or an
 * expired fact reaching the context window is a correctness failure, and it stays a failure
 * however well the reranker ordered it. Counting it separately keeps it from being averaged
 * into an nDCG that looks fine.
 */
export function scopeViolations(
  results: readonly RankedResult[],
  query: Query,
  chunkById: Map<string, { tenant: string; owner: string | null; expiresAt: string | null }>,
  now: Date = new Date(),
): { tenant: number; owner: number; expired: number } {
  const counts = { tenant: 0, owner: 0, expired: 0 };
  for (const r of results) {
    const chunk = chunkById.get(r.chunkId);
    if (!chunk) continue;
    if (chunk.tenant !== query.tenant) counts.tenant++;
    if (chunk.owner !== null && chunk.owner !== query.owner) counts.owner++;
    if (chunk.expiresAt !== null && new Date(chunk.expiresAt) <= now) counts.expired++;
  }
  return counts;
}

export function median(values: readonly number[]): number {
  return percentile(values, 50);
}

/** Coefficient of variation, stddev / mean. A run-to-run stability figure for one stage. */
export function cv(values: readonly number[]): number {
  const m = mean(values);
  return m === 0 || Number.isNaN(m) ? NaN : stddev(values) / m;
}

/** mulberry32, so a confidence interval is reproducible for a given seed. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Interval {
  lower: number;
  upper: number;
}

/**
 * Percentile bootstrap confidence interval for a statistic of a sample.
 *
 * Latency differences are not normally distributed and the sample is small, so a t-interval
 * would be the wrong tool. Resampling with replacement, recomputing the statistic each time,
 * and reading off the 2.5th and 97.5th percentiles makes no distributional assumption. The
 * resampling is seeded so that re-rendering a report from the same raw data gives the same
 * interval.
 */
export function bootstrapCI(
  values: readonly number[],
  stat: (xs: readonly number[]) => number = median,
  resamples = 2000,
  seed = 1,
  level = 0.95,
): Interval {
  if (values.length === 0) return { lower: NaN, upper: NaN };
  const rand = seeded(seed);
  const n = values.length;
  const stats: number[] = new Array(resamples);
  const sample: number[] = new Array(n);
  for (let r = 0; r < resamples; r++) {
    for (let i = 0; i < n; i++) sample[i] = values[Math.floor(rand() * n)]!;
    stats[r] = stat(sample);
  }
  const alpha = (1 - level) / 2;
  return { lower: percentile(stats, alpha * 100), upper: percentile(stats, (1 - alpha) * 100) };
}

/**
 * Element-wise a[i] - b[i] over observations matched by key.
 *
 * This is the whole reason grouped stages run interleaved: a treatment timing minus the control
 * timing taken moments earlier under the same conditions is a single observation of the cost of
 * what the treatment adds. The median of those observations is the estimate; median(a) -
 * median(b) is not the same quantity and is not what gets reported.
 */
export function pairedDifferences<K>(
  a: ReadonlyMap<K, number>,
  b: ReadonlyMap<K, number>,
): number[] {
  const out: number[] = [];
  for (const [key, va] of a) {
    const vb = b.get(key);
    if (vb !== undefined) out.push(va - vb);
  }
  return out;
}
