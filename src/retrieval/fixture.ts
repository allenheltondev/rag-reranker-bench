import type { Candidate, Chunk, Query, Stage } from '../types.js';
import type { CandidateBatch, CandidateSource } from './candidates.js';

/**
 * A local, dependency-free stand-in for the Oracle backend.
 *
 * It exists so the harness, the metrics and the report can be developed and tested on a laptop
 * with no database and no model weights. It is a SELF-TEST, NOT A SIMULATION: the "vector" arm
 * is hashed TF-IDF, not embeddings, so it does not reproduce the paraphrase behaviour of a real
 * embedding model. Never quote a number produced by this backend as a result.
 */

const STOPWORDS = new Set([
  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'our', 'was', 'what', 'when', 'how', 'why',
  'does', 'did', 'this', 'that', 'with', 'from', 'into', 'about', 'should', 'would', 'could',
  'have', 'has', 'had', 'its', 'it', 'is', 'do', 'we', 'i', 'a', 'an', 'of', 'on', 'in', 'to',
  'keep', 'still', 'some', 'them', 'they', 'their', 'there', 'then', 'than', 'at', 'be', 'by',
]);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_-]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

const DIMS = 384;

function hashVector(tokens: string[], idf: Map<string, number>): Float64Array {
  const v = new Float64Array(DIMS);
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  for (const [term, count] of tf) {
    let h = 2166136261;
    for (let i = 0; i < term.length; i++) {
      h ^= term.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const dim = Math.abs(h) % DIMS;
    const sign = (h & 1) === 0 ? 1 : -1;
    v[dim] = v[dim]! + sign * (1 + Math.log(count)) * (idf.get(term) ?? 1);
  }
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < DIMS; i++) v[i]! /= norm;
  return v;
}

const dot = (a: Float64Array, b: Float64Array): number => {
  let s = 0;
  for (let i = 0; i < DIMS; i++) s += a[i]! * b[i]!;
  return s;
};

interface Indexed {
  chunk: Chunk;
  tokens: string[];
  vector: Float64Array;
  length: number;
}

export class FixtureCandidateSource implements CandidateSource {
  readonly kind = 'fixture' as const;
  private readonly docs: Indexed[] = [];
  private readonly idf = new Map<string, number>();
  private avgLen = 0;
  private readonly today = new Date();

  constructor(chunks: readonly Chunk[], private readonly rrfK: number) {
    const df = new Map<string, number>();
    const tokenized = chunks.map((c) => ({ chunk: c, tokens: tokenize(`${c.title}. ${c.content}`) }));
    for (const { tokens } of tokenized) {
      for (const t of new Set(tokens)) df.set(t, (df.get(t) ?? 0) + 1);
    }
    const n = tokenized.length;
    for (const [term, count] of df) this.idf.set(term, Math.log(1 + (n - count + 0.5) / (count + 0.5)));
    this.avgLen = tokenized.reduce((a, d) => a + d.tokens.length, 0) / Math.max(1, n);
    for (const { chunk, tokens } of tokenized) {
      this.docs.push({ chunk, tokens, vector: hashVector(tokens, this.idf), length: tokens.length });
    }
  }

  /** Same scope predicate as the SQL: tenant, owner visibility, and lifecycle. */
  private visible(d: Indexed, query: Query): boolean {
    if (d.chunk.tenant !== query.tenant) return false;
    if (d.chunk.owner !== null && d.chunk.owner !== query.owner) return false;
    if (d.chunk.expiresAt !== null && new Date(d.chunk.expiresAt) <= this.today) return false;
    return true;
  }

  private bm25(d: Indexed, queryTokens: string[]): number {
    const k1 = 1.2, b = 0.75;
    const tf = new Map<string, number>();
    for (const t of d.tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const q of new Set(queryTokens)) {
      const f = tf.get(q);
      if (!f) continue;
      const idf = this.idf.get(q) ?? 0;
      score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * d.length) / this.avgLen)));
    }
    return score;
  }

  async generate(query: Query, retrieval: Stage['retrieval'], n: number): Promise<CandidateBatch> {
    const started = performance.now();
    const qTokens = tokenize(query.text);
    const qVector = hashVector(qTokens, this.idf);
    const pool = this.docs.filter((d) => this.visible(d, query));

    const rank = (scored: Array<{ id: string; s: number }>): Map<string, number> => {
      const ordered = scored
        .filter((x) => x.s > 0)
        .sort((a, b) => (b.s - a.s) || a.id.localeCompare(b.id))
        .slice(0, n);
      return new Map(ordered.map((x, i) => [x.id, i + 1]));
    };

    const vecRanks = retrieval === 'lexical'
      ? new Map<string, number>()
      : rank(pool.map((d) => ({ id: d.chunk.id, s: dot(qVector, d.vector) })));
    const lexRanks = retrieval === 'vector'
      ? new Map<string, number>()
      : rank(pool.map((d) => ({ id: d.chunk.id, s: this.bm25(d, qTokens) })));

    const ids = new Set([...vecRanks.keys(), ...lexRanks.keys()]);
    const byId = new Map(pool.map((d) => [d.chunk.id, d.chunk]));

    const fused = [...ids].map((id) => {
      const vr = vecRanks.get(id);
      const lr = lexRanks.get(id);
      const score = (vr ? 1 / (this.rrfK + vr) : 0) + (lr ? 1 / (this.rrfK + lr) : 0);
      return { id, score, vr, lr };
    })
      .sort((a, b) => (b.score - a.score) || a.id.localeCompare(b.id))
      .slice(0, n);

    let bytes = 0;
    const candidates: Candidate[] = fused.map((f, i) => {
      const chunk = byId.get(f.id)!;
      bytes += Buffer.byteLength(chunk.title, 'utf8') + Buffer.byteLength(chunk.content, 'utf8')
        + Buffer.byteLength(chunk.id, 'utf8');
      const c: Candidate = {
        chunkId: f.id,
        title: chunk.title,
        content: chunk.content,
        rank: i + 1,
        score: f.score,
      };
      if (f.vr !== undefined) c.vectorRank = f.vr;
      if (f.lr !== undefined) c.lexicalRank = f.lr;
      return c;
    });

    return { candidates, bytes, ms: performance.now() - started };
  }

  async close(): Promise<void> {}
}
