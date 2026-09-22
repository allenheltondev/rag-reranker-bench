/**
 * How varied are the results each stage hands the model?
 *
 * Every other number in this repo asks whether the right memories reach the top. This asks
 * whether the memories that reach the top are different from each other. The two come apart
 * exactly when a store is full of near-duplicates: five paraphrases of one good answer can
 * score perfectly on relevance and still give a few-shot prompt nothing to learn range from.
 *
 * The measure is intra-list similarity, the mean pairwise similarity across a stage's top k.
 * Lower is more varied. It is computed two independent ways, because either alone can be
 * argued with:
 *
 *  - Embedding similarity uses the same vectors vector retrieval ranks by. That makes it the
 *    most faithful view of "these are the same idea", and also biased against vector
 *    retrieval, whose results are close to the query in that space and so tend to be close
 *    to each other. On its own it would partly be measuring how retrieval was done.
 *  - Word overlap (Jaccard over content words) knows nothing about any embedding model.
 *
 * A difference both measures agree on is a difference in the results. One they disagree on
 * is a statement about the instrument.
 *
 * Nothing here re-runs anything. Rankings in a run are deterministic (the inspector checks),
 * so the first iteration's top k for each query is the ranking, and the embeddings are
 * already stored beside the text.
 */
import { bootstrapCI, mean, pairedDifferences, type Interval } from './metrics.js';
import type { BenchRun, Chunk, Stage, StageRun } from '../types.js';

export type Vec = Float32Array | readonly number[];

export const K_VALUES = [3, 5, 10] as const;
/** The agent that motivated this pulls five few-shot examples, so this is the headline k. */
export const HEADLINE_K = 5;

export function cosine(a: Vec, b: Vec): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na === 0 || nb === 0 ? NaN : dot / Math.sqrt(na * nb);
}

// Words that appear in almost every memory and would make any two look alike.
const STOP = new Set([
  'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were', 'but', 'not', 'you',
  'your', 'have', 'has', 'had', 'from', 'they', 'their', 'there', 'then', 'than', 'when',
  'what', 'which', 'who', 'will', 'would', 'can', 'could', 'should', 'into', 'about', 'its',
  'our', 'all', 'any', 'one', 'also', 'been', 'being', 'each', 'more', 'most', 'some', 'such',
  'only', 'over', 'very', 'just', 'use', 'used', 'using',
]);

/** Content words, lowercased. Hyphenated identifiers such as INC-4821 stay whole. */
export function tokenSet(text: string): Set<string> {
  const out = new Set<string>();
  for (const t of text.toLowerCase().match(/[a-z0-9]+(?:-[a-z0-9]+)*/g) ?? []) {
    if (t.length >= 3 && !STOP.has(t)) out.add(t);
  }
  return out;
}

export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 && b.size === 0) return NaN;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  return inter / (a.size + b.size - inter);
}

/** Mean pairwise similarity over the first k items. NaN when fewer than two are available. */
export function intraListSimilarity<T>(
  items: readonly T[],
  k: number,
  sim: (a: T, b: T) => number,
): number {
  const top = items.slice(0, k);
  if (top.length < 2) return NaN;
  const pairs: number[] = [];
  for (let i = 0; i < top.length; i++) {
    for (let j = i + 1; j < top.length; j++) {
      const s = sim(top[i]!, top[j]!);
      if (!Number.isNaN(s)) pairs.push(s);
    }
  }
  return pairs.length === 0 ? NaN : mean(pairs);
}

type PerQuery = Map<string, number>;
export interface StageDiversity {
  stage: Stage;
  /** k -> query -> intra-list similarity. Absent where a query returned fewer than 2 items. */
  embedding: Map<number, PerQuery> | null;
  words: Map<number, PerQuery>;
}

/** Stages whose ordering means something. Controls sort by a placeholder, not relevance. */
const ranks = (s: Stage): boolean => s.role !== 'control';

function firstRankings(run: StageRun): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const it of run.iterations) {
    if (!out.has(it.queryId)) out.set(it.queryId, it.results.map((r) => r.chunkId));
  }
  return out;
}

export function measureDiversity(
  run: BenchRun,
  chunks: readonly Chunk[],
  embeddings: ReadonlyMap<string, Vec> | null,
): StageDiversity[] {
  const words = new Map(chunks.map((c) => [c.id, tokenSet(`${c.title} ${c.content}`)]));
  const out: StageDiversity[] = [];
  for (const sr of run.stages) {
    if (!ranks(sr.stage) || sr.iterations.length === 0) continue;
    const rankings = firstRankings(sr);
    const byKWords = new Map<number, PerQuery>();
    const byKEmb = embeddings ? new Map<number, PerQuery>() : null;
    for (const k of K_VALUES) {
      const w: PerQuery = new Map();
      const e: PerQuery = new Map();
      for (const [q, ids] of rankings) {
        const ws = ids.map((id) => words.get(id)).filter((x): x is Set<string> => x !== undefined);
        const wv = intraListSimilarity(ws, k, jaccard);
        if (!Number.isNaN(wv)) w.set(q, wv);
        if (embeddings) {
          const vs = ids.map((id) => embeddings.get(id)).filter((x): x is Vec => x !== undefined);
          const ev = intraListSimilarity(vs, k, cosine);
          if (!Number.isNaN(ev)) e.set(q, ev);
        }
      }
      byKWords.set(k, w);
      byKEmb?.set(k, e);
    }
    out.push({ stage: sr.stage, embedding: byKEmb, words: byKWords });
  }
  return out;
}

export type Verdict = 'more varied' | 'less varied' | "can't tell";

export interface DiversityDelta {
  /** Mean of (subject - reference) over queries both answered. Negative is more varied. */
  mean: number;
  ci: Interval;
  n: number;
  verdict: Verdict;
}

export interface Comparison {
  label: string;
  family: 'fusion' | 'rerank';
  k: number;
  embedding: DiversityDelta | null;
  words: DiversityDelta;
}

function delta(subject: PerQuery | undefined, reference: PerQuery | undefined): DiversityDelta {
  const diffs = subject && reference ? pairedDifferences(subject, reference) : [];
  const ci = bootstrapCI(diffs, mean, 2000, 1);
  const verdict: Verdict = ci.upper < 0 ? 'more varied' : ci.lower > 0 ? 'less varied' : "can't tell";
  return { mean: mean(diffs), ci, n: diffs.length, verdict };
}

export function compareDiversity(stages: readonly StageDiversity[]): Comparison[] {
  const baseline = (r: Stage['retrieval']) =>
    stages.find((s) => s.stage.role === 'baseline' && s.stage.retrieval === r);
  const out: Comparison[] = [];

  // Fusion: does adding a second signal change how alike the results are?
  const hybrid = baseline('hybrid-rrf');
  for (const ref of ['vector', 'lexical'] as const) {
    const b = baseline(ref);
    if (!hybrid || !b) continue;
    for (const k of K_VALUES) {
      out.push({
        label: `Hybrid RRF vs ${ref} alone`,
        family: 'fusion',
        k,
        embedding: hybrid.embedding && b.embedding ? delta(hybrid.embedding.get(k), b.embedding.get(k)) : null,
        words: delta(hybrid.words.get(k), b.words.get(k)),
      });
    }
  }

  // Reranking: does reordering the pool with a cross-encoder change it?
  for (const s of stages) {
    if (s.stage.role !== 'treatment') continue;
    const b = baseline(s.stage.retrieval);
    if (!b) continue;
    const k = HEADLINE_K;
    out.push({
      label: `${s.stage.label} vs no rerank`,
      family: 'rerank',
      k,
      embedding: s.embedding && b.embedding ? delta(s.embedding.get(k), b.embedding.get(k)) : null,
      words: delta(s.words.get(k), b.words.get(k)),
    });
  }
  return out;
}

const f3 = (n: number): string => (Number.isFinite(n) ? n.toFixed(3) : '—');
const signed = (n: number): string => (Number.isFinite(n) ? `${n >= 0 ? '+' : ''}${n.toFixed(3)}` : '—');
const cell = (d: DiversityDelta | null): string =>
  d === null || d.n === 0 ? '—' : `${signed(d.mean)} [${signed(d.ci.lower)}, ${signed(d.ci.upper)}]`;
const verdictCell = (c: Comparison): string => {
  const w = c.words.verdict;
  if (!c.embedding) return `${w} (words only)`;
  const e = c.embedding.verdict;
  return e === w ? `**${e}**` : `${e} / ${w} (measures disagree)`;
};

function answer(comps: readonly Comparison[], family: Comparison['family'], pick: (c: Comparison) => boolean): string {
  const rows = comps.filter((c) => c.family === family && pick(c));
  if (rows.length === 0) return 'Not measured in this run.';
  const agreed = rows.filter((c) => !c.embedding || c.embedding.verdict === c.words.verdict);
  const verdicts = new Set(agreed.map((c) => c.words.verdict));
  const both = rows.every((c) => c.embedding !== null);
  if (agreed.length === rows.length && verdicts.size === 1) {
    const v = [...verdicts][0]!;
    if (!both) {
      return v === "can't tell"
        ? 'No detectable change in word overlap. Embeddings were not loaded, so this is one measure, not two.'
        : `Word overlap says **${v}** in every comparison. Embeddings were not loaded, so this is one measure, not two.`;
    }
    return v === "can't tell"
      ? 'No detectable change on either measure.'
      : `Both measures agree: **${v}** in every comparison.`;
  }
  return 'Mixed: the comparisons or the two measures do not agree. Read the rows below before drawing a conclusion.';
}

export function renderDiversity(stages: readonly StageDiversity[]): string {
  const comps = compareDiversity(stages);
  const hasEmb = stages.some((s) => s.embedding !== null);
  const out: string[] = [];
  const K = HEADLINE_K;

  out.push('# Result diversity');
  out.push('');
  out.push('Intra-list similarity: the mean pairwise similarity across each stage\'s top k. **Lower is');
  out.push('more varied.** Measured by embedding cosine and, independently, by word overlap. The');
  out.push('embedding measure uses the same space vector retrieval ranks in, so it is biased against');
  out.push('vector retrieval; word overlap is not. Trust a difference when both agree.');
  if (!hasEmb) {
    out.push('');
    out.push('> Embeddings were not loaded, so only word overlap is reported. Run without `--no-db`');
    out.push('> against the database that produced this run to get both measures.');
  }
  out.push('');

  out.push('## The two questions');
  out.push('');
  out.push(`**Did fusion make the top ${K} more varied than a single signal?** `
    + answer(comps, 'fusion', (c) => c.k === K));
  out.push('');
  out.push(`**Did reranking change how varied the top ${K} is?** `
    + answer(comps, 'rerank', () => true));
  out.push('');

  out.push('## Similarity by stage');
  out.push('');
  const head = hasEmb
    ? '| Stage | Embedding @3 | @5 | @10 | Words @3 | @5 | @10 | Queries @5 |'
    : '| Stage | Words @3 | @5 | @10 | Queries @5 |';
  out.push(head);
  out.push(hasEmb ? '|---|---:|---:|---:|---:|---:|---:|---:|' : '|---|---:|---:|---:|---:|');
  for (const s of stages) {
    const m = (pk: Map<number, PerQuery> | null, k: number) => f3(mean([...(pk?.get(k)?.values() ?? [])]));
    const n = s.words.get(K)?.size ?? 0;
    out.push(hasEmb
      ? `| ${s.stage.label} | ${m(s.embedding, 3)} | ${m(s.embedding, 5)} | ${m(s.embedding, 10)} | ${m(s.words, 3)} | ${m(s.words, 5)} | ${m(s.words, 10)} | ${n} |`
      : `| ${s.stage.label} | ${m(s.words, 3)} | ${m(s.words, 5)} | ${m(s.words, 10)} | ${n} |`);
  }
  out.push('');
  out.push('A query counts at a given k only if the stage returned at least two results. Lexical');
  out.push('retrieval sometimes returns fewer than k, so its query counts can be lower.');
  out.push('');

  const table = (family: Comparison['family'], title: string, blurb: string[]) => {
    out.push(`## ${title}`);
    out.push('');
    for (const l of blurb) out.push(l);
    out.push('');
    out.push('| Comparison | k | Δ embedding similarity [95% CI] | Δ word similarity [95% CI] | Queries | Verdict |');
    out.push('|---|---:|---|---|---:|---|');
    for (const c of comps.filter((x) => x.family === family)) {
      out.push(`| ${c.label} | ${c.k} | ${cell(c.embedding)} | ${cell(c.words)} | ${c.words.n} | ${verdictCell(c)} |`);
    }
    out.push('');
  };
  table('fusion', 'Fusion against a single signal', [
    'Paired per query: hybrid\'s similarity minus the single signal\'s, averaged, with a bootstrap',
    'interval over queries. Negative means hybrid\'s results are less alike.',
  ]);
  table('rerank', `Reranking against no reranking, top ${K}`, [
    'Each reranked stage against the same retrieval without a reranker. A cross-encoder scores',
    'each candidate on its own, so there is no mechanism by which it should spread results out;',
    'near-duplicates that are all relevant should score alike and stay together. This table',
    'checks that expectation rather than assuming it.',
  ]);

  out.push('## Reading this honestly');
  out.push('');
  out.push('Every interval here is over the same judged query set as the relevance numbers, so it');
  out.push('has the same limit: few queries means wide intervals, and "can\'t tell" is a statement');
  out.push('about sample size, not proof of no effect. Diversity is also not the goal on its own. Five');
  out.push('unrelated memories are maximally varied and useless. Read these next to the relevance');
  out.push('numbers: the result that matters is a stage that is more varied without being less relevant.');
  out.push('');
  return out.join('\n');
}
