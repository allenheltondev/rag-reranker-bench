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
  /** Per-query nDCG, keyed by query id, for the per-query breakdown in the report. */
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
): QualitySummary {
  const ndcgs: number[] = [];
  const recalls: number[] = [];
  const mrrs: number[] = [];
  const perQuery: Record<string, number> = {};

  for (const q of queries) {
    const results = resultsByQuery.get(q.id);
    if (!results) continue;
    const n = ndcgAt(results, q.judgments, k);
    const r = recallAt(results, q.judgments, k);
    const m = mrrAt(results, q.judgments, k);
    if (!Number.isNaN(n)) { ndcgs.push(n); perQuery[q.id] = n; }
    if (!Number.isNaN(r)) recalls.push(r);
    if (!Number.isNaN(m)) mrrs.push(m);
  }

  return { ndcg: mean(ndcgs), recall: mean(recalls), mrr: mean(mrrs), perQuery };
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
