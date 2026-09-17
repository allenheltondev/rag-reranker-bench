import type { BenchRun, Chunk, Query, RankedResult, Stage, StageRun } from '../types.js';
import {
  jaccardAt, kendallTau, mean, percentile, rankOfBest, scopeViolations, summariseQuality,
  type QualitySummary,
} from './metrics.js';
import type { ParityReport } from './harness.js';

export interface StageSummary {
  stage: Stage;
  p50: number;
  p95: number;
  p99: number;
  meanMs: number;
  /** Mean bytes of candidate text crossing the database boundary per query. */
  bytesPerQuery: number;
  quality: QualitySummary;
  attribution: 'split' | 'total-only';
  /** Mean sub-phase timings, only meaningful when attribution is 'split'. */
  phases?: { candidates: number; tokenize: number; infer: number; sort: number };
  /** Out-of-scope rows that reached the context window. Any non-zero value is a bug. */
  violations: { tenant: number; owner: number; expired: number };
}

const fmt = (n: number, digits = 1): string =>
  Number.isNaN(n) ? 'n/a' : n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });

const fmtBytes = (n: number): string => {
  if (Number.isNaN(n)) return 'n/a';
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
};

/** Deterministic pipelines return one ordering per query; take the first iteration's. */
function resultsByQuery(run: StageRun): Map<string, RankedResult[]> {
  const map = new Map<string, RankedResult[]>();
  for (const it of run.iterations) {
    if (!map.has(it.queryId)) map.set(it.queryId, it.results);
  }
  return map;
}

export function summariseStage(
  run: StageRun,
  queries: readonly Query[],
  topK: number,
  chunkById?: Map<string, Chunk>,
): StageSummary {
  const totals = run.iterations.map((i) => i.timings.total);
  const perQueryResults = resultsByQuery(run);
  const violations = { tenant: 0, owner: 0, expired: 0 };
  if (chunkById) {
    for (const q of queries) {
      const results = perQueryResults.get(q.id);
      if (!results) continue;
      const v = scopeViolations(results.slice(0, topK), q, chunkById);
      violations.tenant += v.tenant;
      violations.owner += v.owner;
      violations.expired += v.expired;
    }
  }
  const summary: StageSummary = {
    stage: run.stage,
    p50: percentile(totals, 50),
    p95: percentile(totals, 95),
    p99: percentile(totals, 99),
    meanMs: mean(totals),
    bytesPerQuery: mean(run.iterations.map((i) => i.bytesFromDb)),
    quality: summariseQuality(perQueryResults, queries, topK),
    attribution: run.iterations[0]?.attribution ?? 'split',
    violations,
  };
  if (summary.attribution === 'split') {
    summary.phases = {
      candidates: mean(run.iterations.map((i) => i.timings.candidates)),
      tokenize: mean(run.iterations.map((i) => i.timings.tokenize)),
      infer: mean(run.iterations.map((i) => i.timings.infer)),
      sort: mean(run.iterations.map((i) => i.timings.sort)),
    };
  }
  return summary;
}

export interface AgreementRow {
  retrieval: Stage['retrieval'];
  candidateCount: number;
  /** Top-K set overlap between the in-database and application orderings. */
  jaccard: number;
  /** Rank correlation over the chunks both top-K lists contain. */
  tau: number;
  /** Queries where the two paths disagreed about the top result. */
  topOneDisagreements: string[];
}

/**
 * Compare the two reranking paths.
 *
 * They run the same model over the same candidates, so they should agree almost perfectly.
 * They are not required to be bit-identical - tokenizer truncation and float handling can
 * differ - so this is reported as a number rather than asserted. A low value is a finding
 * about the implementations, not evidence that one of them ranks better.
 */
export function compareRerankers(run: BenchRun, topK: number): AgreementRow[] {
  const rows: AgreementRow[] = [];
  const byId = new Map(run.stages.map((s) => [s.stage.id, s]));

  for (const stageRun of run.stages) {
    if (stageRun.stage.reranker !== 'in-db') continue;
    const appId = stageRun.stage.id.replace('rerank-in-db', 'rerank-app');
    const appRun = byId.get(appId);
    if (!appRun) continue;

    const dbResults = resultsByQuery(stageRun);
    const appResults = resultsByQuery(appRun);
    const jaccards: number[] = [];
    const taus: number[] = [];
    const disagreements: string[] = [];

    for (const [queryId, dbList] of dbResults) {
      const appList = appResults.get(queryId);
      if (!appList) continue;
      jaccards.push(jaccardAt(dbList, appList, topK));
      const tau = kendallTau(dbList.slice(0, topK), appList.slice(0, topK));
      if (!Number.isNaN(tau)) taus.push(tau);
      if (dbList[0]?.chunkId !== appList[0]?.chunkId) disagreements.push(queryId);
    }

    rows.push({
      retrieval: stageRun.stage.retrieval,
      candidateCount: stageRun.stage.candidateCount,
      jaccard: mean(jaccards),
      tau: mean(taus),
      topOneDisagreements: disagreements,
    });
  }
  return rows;
}

export function renderMarkdown(
  run: BenchRun,
  queries: readonly Query[],
  parity?: ParityReport,
  chunks?: readonly Chunk[],
): string {
  const k = run.config.topK;
  const chunkById = chunks ? new Map(chunks.map((c) => [c.id, c])) : undefined;
  const summaries = run.stages.map((s) => summariseStage(s, queries, k, chunkById));
  const out: string[] = [];

  out.push('# Reranker placement benchmark');
  out.push('');
  out.push(`Run started ${run.startedAt}, finished ${run.finishedAt}.`);
  out.push('');

  if (run.config.backend === 'fixture') {
    out.push('> **This run used the fixture backend.** Retrieval is hashed TF-IDF and BM25 over a');
    out.push('> local corpus, and the "reranker" is a scoring function, not a model. These numbers');
    out.push('> validate the harness. They are not results and must not be quoted as any.');
    out.push('');
  }

  out.push('## Environment');
  out.push('');
  out.push('| | |');
  out.push('|---|---|');
  out.push(`| Node | ${run.environment.node} |`);
  out.push(`| Platform | ${run.environment.platform}/${run.environment.arch} |`);
  out.push(`| CPU | ${run.environment.cpuModel} (${run.environment.cpus} vCPU) |`);
  out.push(`| Memory | ${run.environment.totalMemMb} MB |`);
  if (run.environment.oracle) {
    out.push(`| Oracle | ${run.environment.oracle.version} (${run.environment.oracle.clientMode} client) |`);
    out.push(`| In-DB rerank model | ${run.environment.oracle.rerankModel} via ${run.environment.oracle.indbRerankApi} |`);
    out.push(`| Embedding model | ${run.environment.oracle.embedModel} |`);
  }
  if (run.environment.app) {
    out.push(`| App rerank model | ${run.environment.app.modelPath} |`);
    out.push(`| App execution | ${run.environment.app.executionProviders.join(', ')}, dtype ${run.environment.app.dtype}, intra-op threads ${run.environment.app.intraOpThreads} |`);
  }
  out.push('');
  out.push(`Queries: ${queries.length} · iterations: ${run.config.iterations} · warmup: ${run.config.warmup} · top-K: ${k} · RRF k: ${run.config.rrfK}`);
  out.push('');

  if (parity) {
    out.push('## Candidate parity');
    out.push('');
    if (parity.checked === 0) {
      out.push('Not checked (requires the Oracle backend with in-database reranking enabled).');
    } else if (parity.mismatches.length === 0) {
      out.push(`All ${parity.checked} (query, candidate depth) combinations produced identical candidate sets on both paths. Any ranking difference below is the reranker, not retrieval.`);
    } else {
      out.push(`**${parity.mismatches.length} of ${parity.checked} combinations disagreed.** The comparison below is not apples-to-apples until this is resolved:`);
      out.push('');
      for (const m of parity.mismatches.slice(0, 10)) {
        out.push(`- \`${m.queryId}\` @ N=${m.candidateCount}: ${m.onlyInDb.length} only in DB, ${m.onlyInApp.length} only in app`);
      }
    }
    out.push('');
  }

  out.push('## Latency and quality by stage');
  out.push('');
  out.push(`| Stage | p50 (ms) | p95 (ms) | p99 (ms) | nDCG@${k} | Recall@${k} | MRR@${k} | Bytes from DB |`);
  out.push('|---|---:|---:|---:|---:|---:|---:|---:|');
  for (const s of summaries) {
    out.push(`| ${s.stage.label} | ${fmt(s.p50)} | ${fmt(s.p95)} | ${fmt(s.p99)} | ${fmt(s.quality.ndcg, 3)} | ${fmt(s.quality.recall, 3)} | ${fmt(s.quality.mrr, 3)} | ${fmtBytes(s.bytesPerQuery)} |`);
  }
  out.push('');

  if (chunkById) {
    const leaks = summaries.filter((s) => s.violations.tenant + s.violations.owner + s.violations.expired > 0);
    out.push('## Scope violations');
    out.push('');
    if (leaks.length === 0) {
      out.push(`No stage returned a row belonging to another tenant, another user, or an expired fact. ${queries.length} queries x ${summaries.length} stages checked.`);
    } else {
      out.push('**Out-of-scope rows reached the context window.** This is a correctness failure, not a');
      out.push('ranking one: no amount of reranking makes another tenant\'s row acceptable.');
      out.push('');
      out.push('| Stage | Other tenant | Other user | Expired |');
      out.push('|---|---:|---:|---:|');
      for (const s of leaks) {
        out.push(`| ${s.stage.label} | ${s.violations.tenant} | ${s.violations.owner} | ${s.violations.expired} |`);
      }
    }
    out.push('');
  }

  // The candidate sweep is the table the article is actually about.
  const sweep = summaries.filter((s) => s.stage.reranker === 'in-db' || s.stage.reranker === 'app');
  if (sweep.length > 0) {
    out.push('## Cost of reranking as the candidate pool grows');
    out.push('');
    out.push('Added latency is measured against the same retrieval strategy with no reranking, so it');
    out.push('isolates the scoring stage from candidate generation.');
    out.push('');
    out.push(`| Retrieval | N | Where | p50 (ms) | p95 (ms) | Added p50 vs no rerank | nDCG@${k} | Δ nDCG |`);
    out.push('|---|---:|---|---:|---:|---:|---:|---:|');
    for (const s of sweep) {
      const baseline = summaries.find(
        (b) => b.stage.reranker === 'none' && b.stage.retrieval === s.stage.retrieval,
      );
      const addedP50 = baseline ? s.p50 - baseline.p50 : NaN;
      const deltaNdcg = baseline ? s.quality.ndcg - baseline.quality.ndcg : NaN;
      const where = s.stage.reranker === 'in-db' ? 'in database' : 'application';
      out.push(`| ${s.stage.retrieval} | ${s.stage.candidateCount} | ${where} | ${fmt(s.p50)} | ${fmt(s.p95)} | ${fmt(addedP50)} | ${fmt(s.quality.ndcg, 3)} | ${deltaNdcg >= 0 ? '+' : ''}${fmt(deltaNdcg, 3)} |`);
    }
    out.push('');
  }

  const splits = summaries.filter((s) => s.phases && (s.stage.reranker === 'app'));
  if (splits.length > 0) {
    out.push('## Where the application path spends its time');
    out.push('');
    out.push('The in-database path is a single statement and cannot be broken down from outside the');
    out.push('database, so it is absent from this table by construction rather than by omission.');
    out.push('');
    out.push('| Stage | Candidates (ms) | Tokenize (ms) | Infer (ms) | Sort (ms) | Total p50 (ms) |');
    out.push('|---|---:|---:|---:|---:|---:|');
    for (const s of splits) {
      const p = s.phases!;
      out.push(`| ${s.stage.label} | ${fmt(p.candidates)} | ${fmt(p.tokenize)} | ${fmt(p.infer)} | ${fmt(p.sort, 2)} | ${fmt(s.p50)} |`);
    }
    out.push('');
  }

  const agreement = compareRerankers(run, k);
  if (agreement.length > 0) {
    out.push('## Do the two paths rank the same way?');
    out.push('');
    out.push('Same model, same candidates, different execution location. These should agree.');
    out.push('');
    out.push(`| Retrieval | N | Top-${k} Jaccard | Kendall tau | Queries with a different top result |`);
    out.push('|---|---:|---:|---:|---|');
    for (const a of agreement) {
      const d = a.topOneDisagreements.length === 0 ? 'none' : a.topOneDisagreements.join(', ');
      out.push(`| ${a.retrieval} | ${a.candidateCount} | ${fmt(a.jaccard, 3)} | ${fmt(a.tau, 3)} | ${d} |`);
    }
    out.push('');
  }

  out.push('## Per-query detail');
  out.push('');
  out.push('Where the best-graded chunk ended up, per stage. This is where a reranker earning or');
  out.push('not earning its place becomes visible on individual queries rather than in an average.');
  out.push('');
  const header = ['Query', 'Kind', ...summaries.map((s) => s.stage.id)];
  out.push(`| ${header.join(' | ')} |`);
  out.push(`|${header.map(() => '---').join('|')}|`);
  for (const q of queries) {
    const cells = summaries.map((s) => {
      const stageRun = run.stages.find((r) => r.stage.id === s.stage.id)!;
      const results = resultsByQuery(stageRun).get(q.id) ?? [];
      const rank = rankOfBest(results, q.judgments);
      return rank === null ? '—' : String(rank);
    });
    out.push(`| ${q.id} | ${q.kind} | ${cells.join(' | ')} |`);
  }
  out.push('');
  out.push('`—` means the best-graded chunk for that query never reached the context window.');
  out.push('');

  return out.join('\n');
}

export function renderCsv(run: BenchRun, queries: readonly Query[], chunks?: readonly Chunk[]): string {
  const k = run.config.topK;
  const chunkById = chunks ? new Map(chunks.map((c) => [c.id, c])) : undefined;
  const rows = [
    'stage_id,retrieval,reranker,candidate_count,top_k,p50_ms,p95_ms,p99_ms,mean_ms,bytes_per_query,ndcg,recall,mrr,scope_violations',
  ];
  for (const stageRun of run.stages) {
    const s = summariseStage(stageRun, queries, k, chunkById);
    rows.push([
      s.stage.id, s.stage.retrieval, s.stage.reranker, s.stage.candidateCount, s.stage.topK,
      s.p50.toFixed(3), s.p95.toFixed(3), s.p99.toFixed(3), s.meanMs.toFixed(3),
      s.bytesPerQuery.toFixed(0), s.quality.ndcg.toFixed(4), s.quality.recall.toFixed(4),
      s.quality.mrr.toFixed(4),
      String(s.violations.tenant + s.violations.owner + s.violations.expired),
    ].join(','));
  }
  return rows.join('\n');
}
