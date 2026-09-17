import type { Grade, Query } from '../types.js';

/**
 * Hand-authored retrieval scenarios.
 *
 * Every scenario is a query plus the chunks that exist because of it. Judgments are
 * derived by construction: a chunk written to answer the query is graded 2-3, a chunk
 * written to be *almost* right is graded 0-1. That is honest only if the distractors are
 * genuinely hard, so each one is a real topical neighbour rather than filler.
 *
 * Four hazard classes are represented, because they fail differently:
 *   exact-identifier - vector search blurs IDs; lexical nails them
 *   conceptual       - the answer shares no vocabulary with the question
 *   scoped           - the right text exists for the wrong tenant, or has expired
 *   mixed            - topically dense neighbourhood where fusion ranks badly and only a
 *                      cross-encoder recovers the answer. These are the reranker's case.
 */
export interface AuthoredChunk {
  title: string;
  content: string;
  grade: Grade;
  tags?: string[];
  tenant?: string;
  owner?: string | null;
  createdAt?: string;
  expiresAt?: string | null;
}

export interface Scenario {
  id: string;
  kind: Query['kind'];
  text: string;
  tenant?: string;
  owner?: string | null;
  note: string;
  chunks: AuthoredChunk[];
}

const t = (tenant: string) => tenant;

export const scenarios: Scenario[] = [
  {
    id: 'q01',
    kind: 'exact-identifier',
    text: 'What was the root cause of INC-4821?',
    note: 'Nine incident reports share structure and vocabulary. Only the ID separates them, so vector similarity is nearly flat across all of them.',
    chunks: [
      {
        title: 'INC-4821 postmortem: checkout latency spike',
        content:
          'INC-4821 began at 03:14 UTC when p99 checkout latency rose from 240ms to 9.1s in us-east-2. Root cause was a connection pool exhaustion in the payments service: a deploy the previous evening lowered the pool ceiling from 200 to 20 while leaving the acquire timeout at 30 seconds, so every request queued behind a saturated pool instead of failing fast. Traffic was healthy throughout; the incident was entirely self-inflicted by the configuration change. Mitigation was a rollback of the pool setting at 03:52 UTC.',
        grade: 3,
        tags: ['incident', 'postmortem'],
      },
      {
        title: 'INC-4822 postmortem: checkout latency spike (follow-on)',
        content:
          'INC-4822 opened at 06:40 UTC the same morning as a follow-on to the earlier checkout event. p99 latency in us-east-2 briefly returned to 3.4s while the payments service redeployed. Root cause was a stale canary instance still running the previous configuration. No customer-visible failures were recorded.',
        grade: 0,
        tags: ['incident', 'postmortem'],
      },
      {
        title: 'INC-4819 postmortem: checkout errors in eu-west-1',
        content:
          'INC-4819 began at 22:05 UTC when checkout error rate in eu-west-1 reached 4.2%. Root cause was connection pool exhaustion in the payments service caused by a slow downstream dependency holding connections open. Latency recovered after the dependency timeout was reduced.',
        grade: 0,
        tags: ['incident', 'postmortem'],
      },
      {
        title: 'INC-4831 postmortem: payments pool exhaustion',
        content:
          'INC-4831 was declared at 11:20 UTC after the payments service again exhausted its connection pool, this time under genuine peak traffic. Root cause was capacity, not configuration: the pool ceiling was correct but undersized for a 3x promotional spike. Remediation was a permanent increase to 320 connections.',
        grade: 0,
        tags: ['incident', 'postmortem'],
      },
      {
        title: 'Incident index: Q3 checkout incidents',
        content:
          'Checkout-related incidents this quarter: INC-4819 (eu-west-1 error rate), INC-4821 (us-east-2 latency), INC-4822 (follow-on), INC-4831 (pool capacity). Each has a linked postmortem document with root cause and remediation. Use this index to find the right postmortem before paging the on-call team.',
        grade: 1,
        tags: ['index'],
      },
    ],
  },
  {
    id: 'q02',
    kind: 'exact-identifier',
    text: 'Which region did we move the ledger replicas to, us-east-2 or us-west-2?',
    note: 'Both region strings appear across the corpus with near-identical surrounding language. An embedding treats them as the same token neighbourhood.',
    chunks: [
      {
        title: 'Ledger replica migration completed',
        content:
          'The ledger read replicas were moved from us-east-2 to us-west-2 in August to sit closer to the reconciliation workers. Cross-region replication lag dropped from 900ms to 40ms. The primary remains in us-east-2; only the read replicas moved. Any runbook that still points reconciliation at us-east-2 replicas is out of date.',
        grade: 3,
        tags: ['architecture', 'migration'],
      },
      {
        title: 'Reconciliation workers deployment',
        content:
          'Reconciliation workers run in us-west-2 on a six-node pool. They read from the nearest ledger replica and write settlement records back to the primary. The workers are latency sensitive and will shed load if a read exceeds 250ms.',
        grade: 1,
        tags: ['architecture'],
      },
      {
        title: 'Ledger primary topology',
        content:
          'The ledger primary runs in us-east-2 with synchronous standby in the same region. Failover is automatic within the region and manual across regions. Do not point write traffic at any other region.',
        grade: 1,
        tags: ['architecture'],
      },
      {
        title: 'Search index replica placement',
        content:
          'The search index replicas were moved from us-west-2 to us-east-2 last quarter to reduce query latency for the largest tenants. This migration is unrelated to the ledger and is frequently confused with it in planning docs.',
        grade: 0,
        tags: ['architecture', 'migration'],
      },
    ],
  },
  {
    id: 'q03',
    kind: 'conceptual',
    text: 'How should I write code samples for the developer blog?',
    // The answering chunk is a personal memory, so the query has to be issued by its owner.
    // Without this the gold chunk is correctly filtered out and the query is unanswerable.
    owner: 'u-allen',
    note: 'The answering chunk never uses the words "code sample" or "blog". Pure vocabulary mismatch; lexical retrieval cannot find it.',
    chunks: [
      {
        title: 'Allen: writing preferences',
        content:
          'Allen prefers examples that get to the point. Show the smallest thing that actually runs, skip the ceremony, and never pad an article with configuration the reader can infer. He would rather see one honest snippet with its output than three abstract ones. He dislikes screenshots of terminals and prefers pasted text. Opinions in the intro, evidence in the middle, a measurement at the end.',
        grade: 3,
        owner: 'u-allen',
        tags: ['preference', 'memory'],
      },
      {
        title: 'Editorial standards for published tutorials',
        content:
          'Every published tutorial must include a runnable repository link, pinned dependency versions, and a statement of what was measured rather than assumed. Tutorials that only describe an approach without demonstrating it are returned for revision.',
        grade: 2,
        tags: ['process'],
      },
      {
        title: 'Blog post submission checklist',
        content:
          'Before submitting a post: confirm the title is under 60 characters, add alt text for every image, verify outbound links resolve, and tag the post with at most three categories. Submissions missing alt text are rejected automatically by the CMS.',
        grade: 0,
        tags: ['process'],
      },
      {
        title: 'Code formatting configuration',
        content:
          'The repository formats code samples with Prettier using two-space indentation, single quotes, and an 100 character print width. Run the formatter before committing; CI rejects unformatted samples.',
        grade: 0,
        tags: ['process'],
      },
    ],
  },
  {
    id: 'q04',
    kind: 'conceptual',
    text: 'Why does the agent keep forgetting things between sessions?',
    note: 'The answer is a design note about summarisation dropping durable facts. Shares almost no terms with the question.',
    chunks: [
      {
        title: 'Design note: durable facts are lost during summarisation',
        content:
          'When a conversation exceeds the context budget we summarise older turns. The summariser optimises for narrative continuity, not for durable facts, so specifics like account identifiers, stated preferences, and decisions with no follow-up discussion are the first things dropped. The result looks like amnesia to the user even though nothing was deleted from storage. The fix is to extract durable facts into the memory store before summarisation runs, not after.',
        grade: 3,
        tags: ['design', 'memory'],
      },
      {
        title: 'Context budget allocation',
        content:
          'Each turn allocates 8k tokens to retrieved memories, 4k to the running summary, and the remainder to the live conversation. When retrieval returns more than the memory allocation, the lowest ranked items are truncated without warning.',
        grade: 2,
        tags: ['design'],
      },
      {
        title: 'Session storage retention policy',
        content:
          'Raw session transcripts are retained for 90 days and then deleted. Extracted memories are retained indefinitely unless an expiry is set. Deletion of a transcript does not delete memories derived from it.',
        grade: 1,
        tags: ['policy'],
      },
      {
        title: 'Session handoff between devices',
        content:
          'A session started on one device can be resumed on another. The session token carries the conversation identifier; if the token is not refreshed within 30 days the session is archived and a new one begins.',
        grade: 0,
        tags: ['product'],
      },
    ],
  },
  {
    id: 'q05',
    kind: 'scoped',
    text: 'What is our current deployment approval policy?',
    note: 'Three versions of this policy exist: the current one, an expired one, and another tenant\'s. Only metadata separates them; the text is nearly identical.',
    chunks: [
      {
        title: 'Deployment approval policy (current)',
        content:
          'Production deployments require one approval from a service owner. Deployments during the change freeze window require two approvals, one of which must come from the on-call engineer. Approvals are recorded in the deployment record and cannot be granted by the author of the change.',
        grade: 3,
        tags: ['policy'],
        createdAt: '2025-07-01',
      },
      {
        title: 'Deployment approval policy (superseded)',
        content:
          'Production deployments require two approvals from service owners. Deployments during the change freeze window are prohibited entirely. Approvals are recorded in the deployment record and may be granted by any engineer on the owning team, including the author.',
        grade: 0,
        tags: ['policy', 'expired'],
        createdAt: '2024-02-01',
        expiresAt: '2025-06-30',
      },
      {
        title: 'Deployment approval policy',
        content:
          'Production deployments require approval from the platform review board. The board meets twice weekly and deployments outside those windows require an emergency exception signed by a director.',
        grade: 0,
        tenant: t('globex'),
        tags: ['policy'],
      },
      {
        title: 'Change freeze calendar',
        content:
          'Change freeze windows run from the 20th of December through the 2nd of January, and for 72 hours around each major product launch. The calendar is published in the engineering handbook and mirrored to the deployment tool.',
        grade: 1,
        tags: ['policy'],
      },
    ],
  },
  {
    id: 'q06',
    kind: 'scoped',
    text: 'What database does the billing service use?',
    note: 'The stale answer is more strongly worded than the current one, so a reranker that ignores metadata will confidently promote the wrong chunk.',
    chunks: [
      {
        title: 'Billing service data store',
        content:
          'Billing moved to Oracle AI Database 26ai in June. Invoice documents, the customer metadata they are scoped by, and their embeddings all live in the same database, which is why invoice search no longer needs a separate vector store. The migration retired the previous Postgres cluster.',
        grade: 3,
        tags: ['architecture'],
        createdAt: '2025-06-15',
      },
      {
        title: 'Billing service data store (archived)',
        content:
          'Billing runs on a dedicated Postgres 14 cluster with a read replica per region. This is the authoritative record for invoices and must not be bypassed. All invoice reads go through the billing API rather than querying the cluster directly.',
        grade: 0,
        tags: ['architecture', 'expired'],
        createdAt: '2023-09-01',
        expiresAt: '2025-06-14',
      },
      {
        title: 'Invoice search implementation',
        content:
          'Invoice search filters by customer and date range before ranking, so a tenant never sees another tenant\'s invoices even if the text matches. Ranking combines a text index with vector similarity over the invoice body.',
        grade: 2,
        tags: ['architecture'],
      },
      {
        title: 'Reporting warehouse',
        content:
          'Nightly exports land in the reporting warehouse for finance dashboards. The warehouse is not authoritative and lags production by up to 24 hours. Do not use it to answer customer questions about current balances.',
        grade: 0,
        tags: ['architecture'],
      },
    ],
  },
  {
    id: 'q07',
    kind: 'mixed',
    text: 'How do I roll back a canary that is failing health checks in us-east-2?',
    note: 'THE RERANKER CASE. Eight chunks discuss canaries in us-east-2 and score high on both vector and lexical retrieval, but only one describes the rollback procedure. Fusion ranks it mid-pack.',
    chunks: [
      {
        title: 'Rolling back a failing canary',
        content:
          'To roll back a canary that is failing health checks, run `deployctl canary abort --service <name> --region <region>`. This shifts all traffic back to the stable revision within 30 seconds and leaves the canary instances running for inspection. Do not delete the canary instances until you have captured their logs, because the abort command does not preserve them. If the abort command fails, scale the canary deployment to zero and let the load balancer drain it, which takes up to five minutes.',
        grade: 3,
        tags: ['runbook'],
      },
      {
        title: 'Canary health check configuration',
        content:
          'Canary deployments in us-east-2 are evaluated against three health checks: HTTP 200 rate above 99.5%, p99 latency under 800ms, and error budget burn below 2x. A canary failing any check for two consecutive minutes is marked unhealthy and paged to the deploying engineer.',
        grade: 1,
        tags: ['runbook'],
      },
      {
        title: 'Canary traffic ramp schedule',
        content:
          'Canaries in us-east-2 receive 1% of traffic for ten minutes, then 5%, 25%, and 50% at ten minute intervals before full promotion. The ramp pauses automatically if health checks degrade, but a paused canary still serves its current traffic share.',
        grade: 1,
        tags: ['runbook'],
      },
      {
        title: 'Why our canaries fail health checks',
        content:
          'The most common cause of canary health check failures in us-east-2 is cold cache: a fresh instance serves the first few hundred requests without a warm local cache and misses the p99 latency threshold. This is expected and usually clears within four minutes.',
        grade: 1,
        tags: ['runbook'],
      },
      {
        title: 'Canary promotion checklist',
        content:
          'Before promoting a canary in us-east-2, confirm all three health checks have been green for fifteen minutes, confirm no open incidents touch the service, and record the promotion in the deployment log. Promotion is irreversible without a new deployment.',
        grade: 0,
        tags: ['runbook'],
      },
      {
        title: 'Canary deployment concepts',
        content:
          'A canary deployment routes a small share of production traffic to a new revision so that failures are contained. Canaries in us-east-2 use the same load balancer as the stable revision, distinguished by a weighted target group.',
        grade: 0,
        tags: ['reference'],
      },
    ],
  },
  {
    id: 'q08',
    kind: 'mixed',
    text: 'What should I do when the vector index build is stuck at 90 percent?',
    note: 'Dense neighbourhood of index-related chunks. The procedural answer uses different phrasing than the question; the topically closest chunks are explanations rather than procedures.',
    chunks: [
      {
        title: 'Unsticking a stalled index build',
        content:
          'An index build that stops advancing near completion is almost always waiting on a lock held by a long-running query. Identify the blocking session and either wait for it or terminate it, then the build resumes on its own. Do not drop and recreate the index: the rebuild restarts from zero and the same lock will block it again. If no blocking session exists, the build is genuinely progressing and the progress figure is an estimate that flattens near the end.',
        grade: 3,
        tags: ['runbook'],
      },
      {
        title: 'Vector index build progress reporting',
        content:
          'Index build progress is reported as an estimate based on rows processed against the table row count. Because the final phase merges partitions rather than processing rows, progress appears to stall near 90 percent even on a healthy build.',
        grade: 2,
        tags: ['reference'],
      },
      {
        title: 'Vector index types and tradeoffs',
        content:
          'HNSW indexes build quickly and query fast but are memory resident. IVF indexes tolerate larger datasets on disk with a small recall cost. Choose based on whether the working set fits in memory, not on build time alone.',
        grade: 0,
        tags: ['reference'],
      },
      {
        title: 'Index build resource limits',
        content:
          'Index builds are limited to four parallel workers by default. Raising the limit speeds up large builds but competes with query traffic for CPU, which is why the default is conservative on shared instances.',
        grade: 1,
        tags: ['reference'],
      },
      {
        title: 'Rebuilding an index after bulk load',
        content:
          'After a bulk load, rebuild the vector index rather than relying on incremental maintenance. Incremental maintenance during a bulk load produces a fragmented structure with degraded recall.',
        grade: 0,
        tags: ['runbook'],
      },
    ],
  },
  {
    id: 'q09',
    kind: 'mixed',
    text: 'Our retrieval returns the right documents but the model still answers wrong. What do we check?',
    note: 'The best answer is about ordering and context budget. Several chunks about retrieval quality score higher on similarity but answer a different question.',
    chunks: [
      {
        title: 'When retrieval is right but the answer is wrong',
        content:
          'If the correct passage is in the candidate set but the model ignores it, the problem is ordering, not recall. Models weight early context more heavily, and a correct passage sitting at position nine behind eight plausible neighbours is effectively invisible. Check the rank of the correct passage before you touch the retriever. If recall is fine and precision at the top is not, add a reranking stage rather than retrieving more.',
        grade: 3,
        tags: ['design', 'retrieval'],
      },
      {
        title: 'Context budget truncation',
        content:
          'Retrieved passages are truncated to fit the context allocation. Truncation happens from the end of the ranked list, so a correct passage ranked last may never reach the model at all. Log what was dropped, not just what was retrieved.',
        grade: 2,
        tags: ['design', 'retrieval'],
      },
      {
        title: 'Improving embedding recall',
        content:
          'Recall problems are addressed by improving the embedding model, chunking strategy, or candidate depth. Measure recall at the candidate depth you actually use before changing models, because a larger candidate pool often recovers more than a better embedding does.',
        grade: 1,
        tags: ['design', 'retrieval'],
      },
      {
        title: 'Chunking strategy',
        content:
          'Chunks of 400 to 600 tokens with 15 percent overlap work well for documentation. Smaller chunks raise precision but fragment procedures across boundaries, which hurts answers that require several consecutive steps.',
        grade: 0,
        tags: ['design', 'retrieval'],
      },
      {
        title: 'Evaluating retrieval quality',
        content:
          'Retrieval quality is measured with recall at the candidate depth and nDCG at the context depth. Reporting only one of them hides the failure mode where recall is high and ordering is poor.',
        grade: 2,
        tags: ['design', 'retrieval'],
      },
    ],
  },
  {
    id: 'q10',
    kind: 'exact-identifier',
    text: 'What does error code ORA-00060 mean in our ingest pipeline?',
    note: 'Error codes are the canonical lexical win: ORA-00060 and ORA-00054 are one character apart to an embedding.',
    chunks: [
      {
        title: 'ORA-00060 in the ingest pipeline',
        content:
          'ORA-00060 is a deadlock detected while waiting for a resource. In the ingest pipeline it appears when two workers update the same document row in different orders during a re-ingest. The pipeline retries the transaction once and then routes the batch to the dead letter queue. Fix the ordering in the worker rather than raising the retry count, because a retry storm makes the deadlock more likely.',
        grade: 3,
        tags: ['runbook', 'errors'],
      },
      {
        title: 'ORA-00054 in the ingest pipeline',
        content:
          'ORA-00054 means a resource is busy and acquisition was requested with NOWAIT. In the ingest pipeline it appears when a schema change runs against a table with active writers. Pause ingest before applying migrations.',
        grade: 0,
        tags: ['runbook', 'errors'],
      },
      {
        title: 'Ingest pipeline dead letter queue',
        content:
          'Batches that fail twice are written to the dead letter queue with the originating error code and the document identifiers. The queue is drained manually; there is no automatic reprocessing.',
        grade: 1,
        tags: ['runbook'],
      },
      {
        title: 'Common database error codes',
        content:
          'The errors seen most often across services are ORA-00060 (deadlock), ORA-00054 (resource busy), ORA-01555 (snapshot too old), and ORA-12170 (connect timeout). Each has a linked runbook section.',
        grade: 1,
        tags: ['reference', 'errors'],
      },
    ],
  },
  {
    id: 'q11',
    kind: 'scoped',
    text: 'What is Priya working on this quarter?',
    note: 'Owner-scoped memory. An identical-shaped note exists for another user; without an owner filter both are equally retrievable.',
    tenant: 'acme',
    owner: 'u-priya',
    chunks: [
      {
        title: 'Priya: current focus',
        content:
          'Priya is leading the retrieval quality workstream this quarter. Her goal is to get nDCG at 10 above 0.80 on the internal evaluation set and to publish the evaluation harness so other teams can run it. She is explicitly not taking on ingestion work this quarter.',
        grade: 3,
        owner: 'u-priya',
        tags: ['memory', 'people'],
      },
      {
        title: 'Allen: current focus',
        content:
          'Allen is writing the developer education series this quarter, with a benchmark-backed article on retrieval architecture as the centrepiece. He is not taking on platform on-call rotations during this period.',
        grade: 0,
        owner: 'u-allen',
        tags: ['memory', 'people'],
      },
      {
        title: 'Retrieval quality workstream charter',
        content:
          'The retrieval quality workstream owns the evaluation harness, the judgment set, and the weekly quality report. It does not own the ingestion pipeline or the embedding model selection, which sit with the platform team.',
        grade: 2,
        tags: ['process'],
      },
      {
        title: 'Quarterly planning process',
        content:
          'Each engineer publishes a focus statement at the start of the quarter and reviews it at the midpoint. Focus statements are visible to the whole tenant and are expected to change at most once per quarter.',
        grade: 0,
        tags: ['process'],
      },
    ],
  },
  {
    id: 'q12',
    kind: 'conceptual',
    text: 'Is it worth adding another stage to our search pipeline?',
    note: 'Abstract question, concrete answer. The answering chunk is about measurement discipline and never says "stage" or "pipeline".',
    chunks: [
      {
        title: 'Every retrieval component must earn its place',
        content:
          'Adding a component to a search system is only justified if it improves the queries you actually serve. Measure the current ordering, add the component, and measure again on the same queries. If the improvement is inside the noise, you have bought latency and operational surface for nothing. This applies as much to fusion as to neural scoring: hybrid retrieval that widens the candidate pool without improving the order is a cost with no benefit.',
        grade: 3,
        tags: ['design', 'retrieval'],
      },
      {
        title: 'Latency budget for interactive search',
        content:
          'Interactive search has a 400ms p95 budget end to end. Retrieval is allocated 120ms of that. Anything that pushes retrieval past its allocation must be justified by a measured quality gain, and the measurement must use production query distribution rather than a curated set.',
        grade: 2,
        tags: ['design'],
      },
      {
        title: 'Search pipeline architecture overview',
        content:
          'The search pipeline has four stages: filtering, candidate generation, fusion, and final ranking. Each stage narrows the working set, and each stage is independently instrumented.',
        grade: 1,
        tags: ['architecture'],
      },
      {
        title: 'Adding a new microservice',
        content:
          'New services require a service record, an on-call rotation, an SLO, and a decommission plan before they can receive production traffic. Services without an owning team are decommissioned automatically after 90 days.',
        grade: 0,
        tags: ['process'],
      },
    ],
  },
  {
    id: 'q13',
    kind: 'mixed',
    text: 'Why did the nightly reconciliation job start taking four hours?',
    note: 'The answer requires connecting a job slowdown to a replica migration described elsewhere in different words.',
    chunks: [
      {
        title: 'Reconciliation runtime regression',
        content:
          'The nightly reconciliation job grew from 40 minutes to just over four hours after the read replicas it queries were relocated. The job issues roughly two million small reads, so an increase in per-read latency multiplies directly into runtime. Batching the reads into ranges brought the job back to 55 minutes without moving anything.',
        grade: 3,
        tags: ['incident', 'performance'],
      },
      {
        title: 'Reconciliation job schedule',
        content:
          'Reconciliation runs nightly at 02:00 UTC and must complete before the 07:00 UTC finance export. A run that exceeds four hours triggers a page to the data platform on-call.',
        grade: 1,
        tags: ['runbook'],
      },
      {
        title: 'Read amplification in batch jobs',
        content:
          'Batch jobs that issue many small reads are dominated by per-read latency rather than throughput. A job making a million reads at 2ms takes 33 minutes; the same job at 8ms takes over two hours. Always check read count before blaming the database.',
        grade: 2,
        tags: ['reference', 'performance'],
      },
      {
        title: 'Nightly export job runtime',
        content:
          'The finance export job runs for 20 to 25 minutes and has not regressed. It reads from the warehouse rather than production replicas, which is why replica changes do not affect it.',
        grade: 0,
        tags: ['runbook'],
      },
    ],
  },
  {
    id: 'q14',
    kind: 'scoped',
    text: 'What are the rate limits on the public API?',
    note: 'The other tenant\'s limits are stated more specifically, which is exactly the kind of confident wrong answer a metadata filter has to prevent.',
    chunks: [
      {
        title: 'Public API rate limits',
        content:
          'The public API allows 600 requests per minute per API key with a burst of 100. Exceeding the limit returns HTTP 429 with a Retry-After header. Limits are applied per key rather than per account, so an account with several keys has a correspondingly higher ceiling.',
        grade: 3,
        tags: ['product', 'api'],
      },
      {
        title: 'Public API rate limits',
        content:
          'The public API allows 120 requests per minute per account with no burst allowance. Exceeding the limit returns HTTP 429 and repeated violations suspend the key for one hour. Enterprise plans may negotiate a higher ceiling in their contract.',
        grade: 0,
        tenant: t('globex'),
        tags: ['product', 'api'],
      },
      {
        title: 'API key management',
        content:
          'API keys are created per environment and can be rotated without downtime by creating a second key, migrating traffic, and revoking the first. Keys inherit the rate limit of the account that owns them.',
        grade: 1,
        tags: ['product', 'api'],
      },
      {
        title: 'Webhook delivery limits',
        content:
          'Webhook deliveries are retried with exponential backoff for up to 24 hours. Endpoints that fail for a full day are disabled and the account owner is notified by email.',
        grade: 0,
        tags: ['product', 'api'],
      },
    ],
  },
  {
    id: 'q15',
    kind: 'conceptual',
    text: 'What did we decide about storing conversation history for enterprise customers?',
    note: 'A decision record phrased as an outcome, surrounded by policy chunks that describe mechanisms rather than the decision.',
    chunks: [
      {
        title: 'Decision: conversation history stays in the customer region',
        content:
          'We decided that enterprise conversation history will be stored in the customer\'s own region and never replicated across regions, even for disaster recovery. The tradeoff accepted was a slower recovery objective in exchange for a residency guarantee we can state in the contract without qualification. This decision supersedes the earlier plan to replicate to a single central region.',
        grade: 3,
        tags: ['decision'],
      },
      {
        title: 'Data residency mechanism',
        content:
          'Region pinning is enforced at the storage layer using a residency tag on every record. Writes carrying a tag that does not match the local region are rejected rather than forwarded, which makes accidental cross-region writes impossible.',
        grade: 2,
        tags: ['architecture'],
      },
      {
        title: 'Disaster recovery objectives',
        content:
          'The recovery point objective is 15 minutes and the recovery time objective is four hours for standard accounts. Enterprise accounts with region pinning have a longer recovery time objective because no cross-region standby exists.',
        grade: 2,
        tags: ['policy'],
      },
      {
        title: 'Conversation history retention',
        content:
          'Conversation history is retained for the term of the contract plus 30 days, after which it is deleted along with derived indexes. Customers may request earlier deletion through support.',
        grade: 1,
        tags: ['policy'],
      },
    ],
  },
  {
    id: 'q16',
    kind: 'mixed',
    text: 'The reranker made our results worse on some queries. Is that expected?',
    note: 'Self-referential and deliberately included: the honest answer is nuanced, and the neighbouring chunks are enthusiastic about reranking.',
    chunks: [
      {
        title: 'When reranking hurts',
        content:
          'A cross-encoder can reorder a candidate list worse than it found it. It happens when the candidate pool is already precise, when the query is a keyword lookup whose answer has low semantic overlap with the query text, and when the model was trained on a domain unlike yours. Reranking is a bet that ordering is your problem; if recall is your problem, it cannot help, and it will happily promote a fluent passage over the one containing the identifier you asked for.',
        grade: 3,
        tags: ['design', 'retrieval'],
      },
      {
        title: 'Cross-encoder scoring explained',
        content:
          'A cross-encoder scores the query and a candidate together in a single forward pass, so it can model term interactions that a bi-encoder cannot. The cost is that every candidate requires its own pass, which is why cross-encoders sit after retrieval rather than replacing it.',
        grade: 2,
        tags: ['reference', 'retrieval'],
      },
      {
        title: 'Reciprocal rank fusion',
        content:
          'Reciprocal rank fusion combines ranked lists by summing 1/(k + rank) across lists, with k commonly set to 60. It requires no score calibration between retrievers, which is why it survives mixing lexical and vector results.',
        grade: 1,
        tags: ['reference', 'retrieval'],
      },
      {
        title: 'Choosing a reranking model',
        content:
          'Model choice is dominated by sequence length and domain fit rather than parameter count. A base model with a 512 token window usually beats a larger model that truncates your passages in half.',
        grade: 1,
        tags: ['reference', 'retrieval'],
      },
    ],
  },
];
