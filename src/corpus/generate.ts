import type { Chunk, Grade, Query } from '../types.js';
import { scenarios, type AuthoredChunk } from './scenarios.js';

/** Small deterministic PRNG (mulberry32) so a seed reproduces a corpus exactly. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = <T,>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)]!;

const DEFAULT_TENANT = 'acme';

/**
 * Filler exists to give the retriever a realistic haystack. It is deliberately drawn from
 * the same operational vocabulary as the scenarios (services, regions, incidents, policies)
 * so that it competes for similarity, but no filler chunk answers any benchmark query.
 */
const SERVICES = ['payments', 'checkout', 'ledger', 'notifications', 'search', 'identity', 'billing', 'ingest', 'reporting', 'gateway'] as const;
const REGIONS = ['us-east-2', 'us-west-2', 'eu-west-1', 'ap-southeast-1'] as const;
const TEAMS = ['platform', 'payments', 'growth', 'data platform', 'security', 'developer experience'] as const;
const OWNERS = ['u-allen', 'u-priya', 'u-dana', 'u-marco', null] as const;

const FILLER_TEMPLATES: Array<(r: () => number, i: number) => { title: string; content: string; tags: string[] }> = [
  (r, i) => {
    const svc = pick(r, SERVICES), region = pick(r, REGIONS), team = pick(r, TEAMS);
    return {
      title: `Service record: ${svc} (${region})`,
      content: `The ${svc} service is owned by the ${team} team and runs in ${region} on ${3 + Math.floor(r() * 12)} instances. It exposes ${2 + Math.floor(r() * 6)} public endpoints and depends on the identity service for authentication. Scaling is horizontal with a target CPU utilisation of ${50 + Math.floor(r() * 25)} percent. Service record ${1000 + i} is reviewed each quarter by the owning team.`,
      tags: ['service-record'],
    };
  },
  (r, i) => {
    const svc = pick(r, SERVICES);
    return {
      title: `SLO definition: ${svc} availability`,
      content: `The ${svc} service targets ${99 + r()} percent availability measured over a rolling 30 day window. Error budget burn above 2x triggers a freeze on feature deployments until the burn rate returns to normal. SLO record ${2000 + i} is owned by the service team and reviewed monthly. Availability is measured at the load balancer, not at the client.`,
      tags: ['slo'],
    };
  },
  (r, i) => {
    const svc = pick(r, SERVICES), region = pick(r, REGIONS);
    return {
      title: `INC-${4600 + (i % 180)} postmortem: ${svc} degradation`,
      content: `INC-${4600 + (i % 180)} affected the ${svc} service in ${region} for ${8 + Math.floor(r() * 90)} minutes. Root cause was ${pick(r, ['a failed dependency health check', 'an expired certificate', 'a misconfigured autoscaling policy', 'a noisy neighbour on shared infrastructure', 'a memory leak in a background worker'])}. Customer impact was limited to elevated latency. Remediation items were filed and completed within the following sprint.`,
      tags: ['incident', 'postmortem'],
    };
  },
  (r, i) => {
    const team = pick(r, TEAMS);
    return {
      title: `Weekly notes: ${team} team`,
      content: `The ${team} team reviewed ${2 + Math.floor(r() * 5)} open items this week. Progress on the migration is on track, dependency upgrades are queued behind the release, and one item was deferred to next quarter for capacity reasons. Notes record ${3000 + i}. Action items are tracked in the team board rather than in these notes.`,
      tags: ['notes'],
    };
  },
  (r, i) => {
    const svc = pick(r, SERVICES);
    return {
      title: `Runbook: restarting ${svc} workers`,
      content: `To restart ${svc} workers, drain the instance from the load balancer, wait for in-flight requests to complete, then issue the restart. Never restart more than a third of the pool at once. If the worker does not come back within ${60 + Math.floor(r() * 120)} seconds, capture a thread dump before retrying. Runbook ${4000 + i} is validated during each game day.`,
      tags: ['runbook'],
    };
  },
  (r, i) => ({
    title: `Cost report: ${pick(r, ['compute', 'storage', 'network egress', 'managed database'])} spend`,
    content: `Spend in this category changed by ${(r() * 30 - 10).toFixed(1)} percent month over month. The largest contributor was the ${pick(r, SERVICES)} service. Finance requires an explanation for any category moving more than 15 percent. Report ${5000 + i} covers the current billing period only and excludes committed use discounts.`,
    tags: ['cost'],
  }),
  (r, i) => ({
    title: `Onboarding: ${pick(r, ['local environment setup', 'first deployment', 'access requests', 'on-call shadowing'])}`,
    content: `New engineers complete this step in their first two weeks. It requires an approved access request, a working local environment, and a buddy from the owning team. Onboarding step ${6000 + i} is tracked in the checklist and signed off by the buddy rather than the manager.`,
    tags: ['onboarding'],
  }),
  (r, i) => ({
    title: `Dependency upgrade: ${pick(r, ['driver', 'runtime', 'framework', 'client library'])} ${1 + Math.floor(r() * 9)}.${Math.floor(r() * 20)}`,
    content: `This upgrade lands behind a feature flag and is rolled out region by region. It changes default timeouts, so services relying on the previous defaults must set them explicitly. Upgrade record ${7000 + i}. No API changes are expected, and the rollback path is a redeploy of the previous image.`,
    tags: ['upgrade'],
  }),
];

export interface GeneratedCorpus {
  chunks: Chunk[];
  queries: Query[];
}

export function generateCorpus(size: number, seed: number): GeneratedCorpus {
  const r = rng(seed);
  const chunks: Chunk[] = [];
  const queries: Query[] = [];

  for (const s of scenarios) {
    const tenant = s.tenant ?? DEFAULT_TENANT;
    const owner = s.owner === undefined ? null : s.owner;
    const judgments: Record<string, Grade> = {};

    s.chunks.forEach((c: AuthoredChunk, idx) => {
      const id = `${s.id}-c${String(idx + 1).padStart(2, '0')}`;
      chunks.push({
        id,
        docId: `${s.id}-doc`,
        title: c.title,
        content: c.content,
        tenant: c.tenant ?? tenant,
        owner: c.owner === undefined ? null : c.owner,
        createdAt: c.createdAt ?? '2025-05-01',
        expiresAt: c.expiresAt === undefined ? null : c.expiresAt,
        tags: c.tags ?? [],
      });
      // Grade 0 chunks are recorded implicitly; only positives go in the judgment map.
      if (c.grade > 0) judgments[id] = c.grade;
    });

    queries.push({
      id: s.id,
      text: s.text,
      tenant,
      owner,
      kind: s.kind,
      judgments,
      note: s.note,
    });
  }

  const fillerCount = Math.max(0, size - chunks.length);
  for (let i = 0; i < fillerCount; i++) {
    const tpl = FILLER_TEMPLATES[i % FILLER_TEMPLATES.length]!;
    const { title, content, tags } = tpl(r, i);
    const year = 2024 + Math.floor(r() * 2);
    const month = 1 + Math.floor(r() * 12);
    const day = 1 + Math.floor(r() * 28);
    chunks.push({
      id: `flr-${String(i + 1).padStart(4, '0')}`,
      docId: `flr-doc-${Math.floor(i / 4)}`,
      title,
      content,
      // A slice of filler belongs to the other tenant so that tenant filtering has real work to do.
      tenant: r() < 0.15 ? 'globex' : DEFAULT_TENANT,
      owner: pick(r, OWNERS),
      createdAt: `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`,
      // A slice of filler is expired, so the lifecycle filter is exercised on every query.
      expiresAt: r() < 0.08 ? `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}` : null,
      tags,
    });
  }

  return { chunks, queries };
}
