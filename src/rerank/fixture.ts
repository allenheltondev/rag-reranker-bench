import type { Candidate, RankedResult } from '../types.js';
import { tokenize } from '../retrieval/fixture.js';
import type { AppRerankOutcome } from './app.js';

/**
 * A stand-in cross-encoder for harness development.
 *
 * It scores each pair independently, like a real cross-encoder, and it deliberately disagrees
 * with the retrieval ranking so that reranking visibly changes the order. It is NOT a model and
 * its quality numbers mean nothing; its job is to prove the harness, timing and reporting work
 * before a real model is attached.
 *
 * FIXTURE_RERANK_US_PER_PAIR simulates per-pair inference cost, which makes the candidate-count
 * sweep behave like the real thing while the plumbing is being tested.
 */
export class FixtureReranker {
  private readonly usPerPair: number;

  constructor(usPerPair = Number(process.env.FIXTURE_RERANK_US_PER_PAIR ?? 250)) {
    this.usPerPair = usPerPair;
  }

  private busyWait(us: number): void {
    if (us <= 0) return;
    const end = performance.now() + us / 1000;
    while (performance.now() < end) { /* deliberate spin: simulates CPU-bound inference */ }
  }

  async rerank(queryText: string, candidates: readonly Candidate[], topK: number): Promise<AppRerankOutcome> {
    const qTokens = new Set(tokenize(queryText));

    const t0 = performance.now();
    const docTokens = candidates.map((c) => tokenize(`${c.title}. ${c.content}`));
    const tokenize_ = performance.now() - t0;

    const t1 = performance.now();
    const scores = candidates.map((c, i) => {
      this.busyWait(this.usPerPair);
      const tokens = docTokens[i]!;
      const titleTokens = new Set(tokenize(c.title));
      let overlap = 0;
      for (const t of new Set(tokens)) if (qTokens.has(t)) overlap++;
      let titleHits = 0;
      for (const t of qTokens) if (titleTokens.has(t)) titleHits++;
      // An exact identifier in the body is weighted heavily, which is the behaviour a real
      // cross-encoder has and a bi-encoder does not.
      let exact = 0;
      for (const t of qTokens) if (/\d/.test(t) && tokens.includes(t)) exact += 2;
      return (overlap / Math.max(1, qTokens.size)) + 0.5 * titleHits + exact;
    });
    const infer = performance.now() - t1;

    const t2 = performance.now();
    const results = candidates
      .map((c, i) => ({ chunkId: c.chunkId, score: scores[i]! }))
      .sort((a, b) => (b.score - a.score) || a.chunkId.localeCompare(b.chunkId))
      .slice(0, topK)
      .map((r, i): RankedResult => ({ chunkId: r.chunkId, rank: i + 1, score: r.score }));
    const sort = performance.now() - t2;

    return { results, timings: { tokenize: tokenize_, infer, sort } };
  }

  async close(): Promise<void> {}
}
