import test from 'node:test';
import assert from 'node:assert/strict';
import { generateCorpus } from '../src/corpus/generate.js';
import { FixtureCandidateSource } from '../src/retrieval/fixture.js';

const { chunks, queries } = generateCorpus(480, 20250917);
const byId = new Map(chunks.map((c) => [c.id, c]));

test('generation is deterministic for a seed', () => {
  const again = generateCorpus(480, 20250917);
  assert.deepEqual(again.chunks, chunks);
  assert.deepEqual(again.queries, queries);
});

test('chunk ids are unique', () => {
  assert.equal(new Set(chunks.map((c) => c.id)).size, chunks.length);
});

test('every judgment points at a chunk that exists', () => {
  for (const q of queries) {
    for (const id of Object.keys(q.judgments)) {
      assert.ok(byId.has(id), `${q.id} judges missing chunk ${id}`);
    }
  }
});

test('every query has at least one chunk graded relevant', () => {
  for (const q of queries) {
    const relevant = Object.values(q.judgments).filter((g) => g >= 2);
    assert.ok(relevant.length > 0, `${q.id} has no chunk graded 2 or higher`);
  }
});

/**
 * The bug this catches: a gold chunk scoped to a user the query is not issued by is filtered
 * out before retrieval ever runs, so the query is unanswerable and every strategy scores zero
 * on it for a reason that has nothing to do with retrieval.
 */
test('every relevant chunk is visible to the query that needs it', () => {
  const now = new Date();
  for (const q of queries) {
    for (const [id, grade] of Object.entries(q.judgments)) {
      if (grade < 2) continue;
      const c = byId.get(id)!;
      assert.equal(c.tenant, q.tenant, `${q.id}: relevant chunk ${id} belongs to tenant ${c.tenant}`);
      assert.ok(
        c.owner === null || c.owner === q.owner,
        `${q.id}: relevant chunk ${id} is owned by ${c.owner} but the query is issued by ${q.owner}`,
      );
      assert.ok(
        c.expiresAt === null || new Date(c.expiresAt) > now,
        `${q.id}: relevant chunk ${id} expired on ${c.expiresAt}`,
      );
    }
  }
});

test('the corpus contains the hazards the benchmark claims to test', () => {
  assert.ok(chunks.some((c) => c.tenant !== 'acme'), 'no other-tenant chunks');
  assert.ok(chunks.some((c) => c.expiresAt !== null), 'no expiring chunks');
  assert.ok(chunks.some((c) => c.owner !== null), 'no owner-scoped chunks');
  const kinds = new Set(queries.map((q) => q.kind));
  for (const kind of ['exact-identifier', 'conceptual', 'scoped', 'mixed']) {
    assert.ok(kinds.has(kind as never), `no ${kind} queries`);
  }
});

test('retrieval never returns a chunk the query may not see', async () => {
  const source = new FixtureCandidateSource(chunks, 60);
  const now = new Date();
  for (const q of queries) {
    const batch = await source.generate(q, 'hybrid-rrf', 40);
    for (const c of batch.candidates) {
      const chunk = byId.get(c.chunkId)!;
      assert.equal(chunk.tenant, q.tenant, `${q.id}: leaked tenant ${chunk.tenant}`);
      assert.ok(chunk.owner === null || chunk.owner === q.owner, `${q.id}: leaked owner ${chunk.owner}`);
      assert.ok(chunk.expiresAt === null || new Date(chunk.expiresAt) > now, `${q.id}: leaked expired chunk`);
    }
  }
});

test('candidate generation respects the requested depth and is ordered', async () => {
  const source = new FixtureCandidateSource(chunks, 60);
  const batch = await source.generate(queries[0]!, 'hybrid-rrf', 20);
  assert.ok(batch.candidates.length <= 20);
  assert.ok(batch.candidates.length > 0);
  batch.candidates.forEach((c, i) => assert.equal(c.rank, i + 1));
  for (let i = 1; i < batch.candidates.length; i++) {
    assert.ok(batch.candidates[i - 1]!.score >= batch.candidates[i]!.score, 'candidates are not sorted');
  }
});
