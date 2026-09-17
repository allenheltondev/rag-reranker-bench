import type { BenchRun, Chunk, Query, RankedResult, Stage, StageRun } from '../types.js';
import {
  bootstrapCI, cv, jaccardAt, kendallTau, mean, median, pairedDifferences, percentile,
  rankOfBest, scopeViolations, summariseQuality, type Interval, type QualitySummary,
} from './metrics.js';
import type { ParityReport } from './harness.js';

export interface StageSummary {
  stage: Stage;
  p50: number;
  p95: number;
  p99: number;
  meanMs: number;
  /**
   * Run-to-run noise: for each query, the coefficient of variation of its total across
   * iterations; then the median of those across queries. Computed within a query on purpose,
   * so that legitimately different costs between queries do not read as instability.
   */
  cv: number;
  /** The worst single query's within-query CV, and which query it was. */
  worstQueryCv: { queryId: string; cv: number };
  /** Mean bytes of candidate text crossing the database boundary per query. */
  bytesPerQuery: number;
  quality: QualitySummary;
  attribution: 'split' | 'total-only';
  /** Median sub-phase timings, only meaningful when attribution is 'split'. */
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

  const totalsByQuery = new Map<string, number[]>();
  for (const i of run.iterations) {
    const arr = totalsByQuery.get(i.queryId) ?? [];
    arr.push(i.timings.total);
    totalsByQuery.set(i.queryId, arr);
  }
  const perQueryCv = [...totalsByQuery.entries()]
    .map(([queryId, xs]) => ({ queryId, cv: cv(xs) }))
    .filter((x) => !Number.isNaN(x.cv));
  const worstQueryCv = perQueryCv.reduce(
    (a, b) => (b.cv > a.cv ? b : a),
    perQueryCv[0] ?? { queryId: 'n/a', cv: NaN },
  );
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
    cv: median(perQueryCv.map((x) => x.cv)),
    worstQueryCv,
    bytesPerQuery: mean(run.iterations.map((i) => i.bytesFromDb)),
    quality: summariseQuality(perQueryResults, queries, topK),
    attribution: run.iterations[0]?.attribution ?? 'split',
    violations,
  };
  if (summary.attribution === 'split') {
    summary.phases = {
      candidates: median(run.iterations.map((i) => i.timings.candidates)),
      tokenize: median(run.iterations.map((i) => i.timings.tokenize)),
      infer: median(run.iterations.map((i) => i.timings.infer)),
      sort: median(run.iterations.map((i) => i.timings.sort)),
    };
  }
  return summary;
}

export interface ScoringCost {
  retrieval: Stage['retrieval'];
  candidateCount: number;
  reranker: Stage['reranker'];
  treatmentId: string;
  controlId: string;
  /** Number of paired observations (queries x iterations). */
  pairs: number;
  controlP50: number;
  treatmentP50: number;
  /** Median of per-observation (treatment - control). This is the reported cost of scoring. */
  medianDelta: number;
  meanDelta: number;
  ci: Interval;
  /**
   * Application path only: the same quantity measured directly with clocks around tokenize,
   * infer and sort. If subtraction is a sound way to measure scoring cost, this agrees with
   * medianDelta; the report prints both so the reader can see whether it does.
   */
  directMedian?: number;
}

const obsKey = (i: { queryId: string; repeat: number; iteration: number }): string =>
  `${i.queryId}:${i.repeat}:${i.iteration}`;

function totalsByObservation(run: StageRun, repeat?: number): Map<string, number> {
  const its = repeat === undefined ? run.iterations : run.iterations.filter((i) => i.repeat === repeat);
  return new Map(its.map((i) => [obsKey(i), i.timings.total]));
}

/**
 * The cost of scoring, per treatment, by subtraction against its control.
 *
 * Control and treatment ran interleaved, so every observation of the treatment has a partner
 * observation of the control taken under the same conditions. The difference of each pair is
 * one measurement of "what scoring added". The median of those is the estimate and a seeded
 * bootstrap gives its interval.
 */
export function scoringCosts(run: BenchRun, repeat?: number): ScoringCost[] {
  const rows: ScoringCost[] = [];
  for (const treatment of run.stages) {
    if (treatment.stage.role !== 'treatment' || !treatment.stage.group) continue;
    const control = run.stages.find(
      (c) => c.stage.role === 'control' && c.stage.group === treatment.stage.group
        && c.stage.reranker === treatment.stage.reranker,
    );
    if (!control) continue;

    const deltas = pairedDifferences(
      totalsByObservation(treatment, repeat),
      totalsByObservation(control, repeat),
    );
    if (deltas.length === 0) continue;

    const row: ScoringCost = {
      retrieval: treatment.stage.retrieval,
      candidateCount: treatment.stage.candidateCount,
      reranker: treatment.stage.reranker,
      treatmentId: treatment.stage.id,
      controlId: control.stage.id,
      pairs: deltas.length,
      controlP50: percentile(control.iterations.map((i) => i.timings.total), 50),
      treatmentP50: percentile(treatment.iterations.map((i) => i.timings.total), 50),
      medianDelta: median(deltas),
      meanDelta: mean(deltas),
      ci: bootstrapCI(deltas, median, 2000, run.config.seed),
    };
    if (treatment.iterations[0]?.attribution === 'split' && treatment.stage.reranker !== 'none') {
      row.directMedian = median(
        treatment.iterations
          .filter((i) => repeat === undefined || i.repeat === repeat)
          .map((i) => i.timings.tokenize + i.timings.infer + i.timings.sort),
      );
    }
    rows.push(row);
  }
  return rows;
}

export interface TransferCost {
  retrieval: Stage['retrieval'];
  candidateCount: number;
  pairs: number;
  /** Paired median of (app control - in-DB control): the cost of moving candidate text out. */
  medianDelta: number;
  ci: Interval;
  appBytes: number;
  indbBytes: number;
}

/**
 * What it costs to bring the candidate text to the application.
 *
 * Measured from the transfer batch, which holds only the two controls, run interleaved: they
 * do identical work up to the projection - one returns identifiers and a number, the other
 * identifiers and the full text - and neither runs inference, so there is nothing to carry
 * over between them. Their paired difference is the serialisation and transfer of that text.
 */
export function transferCosts(run: BenchRun): TransferCost[] {
  const rows: TransferCost[] = [];
  const groups = new Set(
    run.stages.map((s) => s.stage.group).filter((g): g is string => !!g && g.startsWith('transfer:')),
  );
  for (const group of groups) {
    const app = run.stages.find((s) => s.stage.group === group && s.stage.role === 'control' && s.stage.reranker === 'app');
    const indb = run.stages.find((s) => s.stage.group === group && s.stage.role === 'control' && s.stage.reranker === 'in-db');
    if (!app || !indb) continue;
    const deltas = pairedDifferences(totalsByObservation(app), totalsByObservation(indb));
    if (deltas.length === 0) continue;
    rows.push({
      retrieval: app.stage.retrieval,
      candidateCount: app.stage.candidateCount,
      pairs: deltas.length,
      medianDelta: median(deltas),
      ci: bootstrapCI(deltas, median, 2000, run.config.seed),
      appBytes: mean(app.iterations.map((i) => i.bytesFromDb)),
      indbBytes: mean(indb.iterations.map((i) => i.bytesFromDb)),
    });
  }
  return rows;
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
  out.push(`Queries: ${queries.length} · iterations: ${run.config.iterations} · warmup: ${run.config.warmup} · repeats: ${run.config.repeats} · top-K: ${k} · RRF k: ${run.config.rrfK}`);
  const batches = [...new Set(run.stages.map((s) => s.stage.batch))];
  out.push('');
  out.push(`Isolation batches, in order, with a reset between each: ${batches.map((b) => `\`${b}\``).join(' → ')}. Quiesce ${run.config.quiesceMs} ms${run.config.resetCommand ? `, reset command \`${run.config.resetCommand}\`` : ''}.`);
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
  const costs = scoringCosts(run);
  if (costs.length > 0) {
    out.push('## Cost of scoring as the candidate pool grows');
    out.push('');
    out.push('Each row is a treatment minus its control. The control is the same pipeline at the same');
    out.push('depth with the scoring step removed and nothing else changed, run interleaved with the');
    out.push('treatment. "Scoring" is the median of the per-observation differences, with a 95%');
    out.push('bootstrap interval. Quality deltas are against the no-rerank baseline at top-K.');
    out.push('');
    out.push(`| Retrieval | N | Where | Control p50 (ms) | Treatment p50 (ms) | Scoring, median Δ (ms) | 95% CI | Pairs | nDCG@${k} | Δ nDCG |`);
    out.push('|---|---:|---|---:|---:|---:|---|---:|---:|---:|');
    for (const c of costs) {
      const treat = summaries.find((x) => x.stage.id === c.treatmentId)!;
      const baseline = summaries.find((b) => b.stage.role === 'baseline' && b.stage.retrieval === c.retrieval);
      const deltaNdcg = baseline ? treat.quality.ndcg - baseline.quality.ndcg : NaN;
      const where = c.reranker === 'in-db' ? 'in database' : 'application';
      out.push(`| ${c.retrieval} | ${c.candidateCount} | ${where} | ${fmt(c.controlP50)} | ${fmt(c.treatmentP50)} | **${fmt(c.medianDelta)}** | [${fmt(c.ci.lower)}, ${fmt(c.ci.upper)}] | ${c.pairs} | ${fmt(treat.quality.ndcg, 3)} | ${deltaNdcg >= 0 ? '+' : ''}${fmt(deltaNdcg, 3)} |`);
    }
    out.push('');

    const checks = costs.filter((c) => c.directMedian !== undefined);
    if (checks.length > 0) {
      out.push('### Is subtraction a valid way to measure this?');
      out.push('');
      out.push('The application path can be timed both ways: by subtraction, exactly as the in-database');
      out.push('path has to be, and directly with clocks around tokenize, infer and sort. If the two');
      out.push('agree, the subtraction method is sound and the in-database figures above can be trusted');
      out.push('to the same degree. If they do not, the gap is measurement error and it applies to every');
      out.push('subtracted number in this report.');
      out.push('');
      out.push('| Retrieval | N | By subtraction (ms) | Measured directly (ms) | Gap (ms) | Gap as % of direct |');
      out.push('|---|---:|---:|---:|---:|---:|');
      for (const c of checks) {
        const gap = c.medianDelta - c.directMedian!;
        const pct = c.directMedian! === 0 ? NaN : (gap / c.directMedian!) * 100;
        out.push(`| ${c.retrieval} | ${c.candidateCount} | ${fmt(c.medianDelta, 2)} | ${fmt(c.directMedian!, 2)} | ${gap >= 0 ? '+' : ''}${fmt(gap, 2)} | ${pct >= 0 ? '+' : ''}${fmt(pct)}% |`);
      }
      out.push('');
    }
  }

  const transfers = transferCosts(run);
  if (transfers.length > 0) {
    out.push('## Cost of moving the candidate text to the application');
    out.push('');
    out.push('From the transfer batch: the two controls alone, run interleaved. They do identical work');
    out.push('up to the projection - one returns identifiers and a number, the other identifiers and');
    out.push('every candidate\'s full text - and neither runs inference. Their paired difference is');
    out.push('the cost of that text leaving the database.');
    out.push('');
    out.push('| Retrieval | N | Text returned (app) | Returned (in-DB) | Transfer, median Δ (ms) | 95% CI | Pairs |');
    out.push('|---|---:|---:|---:|---:|---|---:|');
    for (const t of transfers) {
      out.push(`| ${t.retrieval} | ${t.candidateCount} | ${fmtBytes(t.appBytes)} | ${fmtBytes(t.indbBytes)} | ${fmt(t.medianDelta, 2)} | [${fmt(t.ci.lower, 2)}, ${fmt(t.ci.upper, 2)}] | ${t.pairs} |`);
    }
    out.push('');
  }

  const splits = summaries.filter((s) => s.phases && s.stage.role === 'treatment' && s.stage.reranker === 'app');
  if (splits.length > 0) {
    out.push('## Where the application path spends its time');
    out.push('');
    out.push('The in-database path is a single statement and cannot be broken down from outside the');
    out.push('database, so it is absent from this table by construction rather than by omission.');
    out.push('');
    out.push('| Stage | Candidates p50 (ms) | Tokenize p50 (ms) | Infer p50 (ms) | Sort p50 (ms) | Total p50 (ms) |');
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
  const shown = summaries.filter((s) => s.stage.role !== 'control');
  const header = ['Query', 'Kind', ...shown.map((s) => s.stage.id)];
  out.push(`| ${header.join(' | ')} |`);
  out.push(`|${header.map(() => '---').join('|')}|`);
  for (const q of queries) {
    const cells = shown.map((s) => {
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

  if (run.config.repeats > 1) {
    out.push('## Repeatability across repeats');
    out.push('');
    out.push('The scoring cost recomputed from each full repetition of the protocol on its own. The');
    out.push('spread between repeats is what the number would do if you ran the benchmark again.');
    out.push('');
    const perRepeat = Array.from({ length: run.config.repeats }, (_, r) => scoringCosts(run, r));
    out.push(`| Retrieval | N | Where | ${perRepeat.map((_, r) => `Repeat ${r + 1} (ms)`).join(' | ')} | Spread (ms) | Spread % |`);
    out.push(`|---|---:|---|${perRepeat.map(() => '---:').join('|')}|---:|---:|`);
    for (const c of scoringCosts(run)) {
      const vals = perRepeat.map((rows) => rows.find((x) => x.treatmentId === c.treatmentId)?.medianDelta ?? NaN);
      const spread = Math.max(...vals) - Math.min(...vals);
      const pct = c.medianDelta === 0 ? NaN : (spread / c.medianDelta) * 100;
      const where = c.reranker === 'in-db' ? 'in database' : 'application';
      out.push(`| ${c.retrieval} | ${c.candidateCount} | ${where} | ${vals.map((v) => fmt(v, 2)).join(' | ')} | ${fmt(spread, 2)} | ${fmt(pct)}% |`);
    }
    out.push('');
  }

  out.push('## Stability');
  out.push('');
  out.push('Run-to-run noise, measured within each query: the coefficient of variation of a query\'s');
  out.push('end-to-end time across iterations, summarised across queries. It is computed this way so');
  out.push('that queries which legitimately cost different amounts do not read as instability. Stages');
  out.push('under a millisecond will show large percentages from timer resolution alone; read those in');
  out.push('absolute terms.');
  out.push('');
  out.push('| Stage | p50 (ms) | Median within-query CV | Worst query | Its CV |');
  out.push('|---|---:|---:|---|---:|');
  for (const s of summaries) {
    const flag = s.cv > 0.25 && s.p50 >= 1 ? ' ⚠' : '';
    out.push(`| ${s.stage.id} | ${fmt(s.p50)} | ${fmt(s.cv * 100)}%${flag} | ${s.worstQueryCv.queryId} | ${fmt(s.worstQueryCv.cv * 100)}% |`);
  }
  out.push('');
  const unstable = summaries.filter((s) => s.cv > 0.25 && s.p50 >= 1);
  out.push(unstable.length === 0
    ? 'No stage over a millisecond exceeds 25% within-query variation; the intervals above are as tight as the iteration count allows.'
    : `⚠ ${unstable.length} stage(s) over a millisecond exceed 25% within-query variation. More iterations, more repeats, or a quieter machine will tighten their intervals.`);
  out.push('');

  return out.join('\n');
}

export function renderCsv(run: BenchRun, queries: readonly Query[], chunks?: readonly Chunk[]): string {
  const k = run.config.topK;
  const chunkById = chunks ? new Map(chunks.map((c) => [c.id, c])) : undefined;
  const rows = [
    'stage_id,role,retrieval,reranker,candidate_count,top_k,p50_ms,p95_ms,p99_ms,mean_ms,cv,bytes_per_query,ndcg,recall,mrr,scope_violations',
  ];
  for (const stageRun of run.stages) {
    const s = summariseStage(stageRun, queries, k, chunkById);
    rows.push([
      s.stage.id, s.stage.role, s.stage.retrieval, s.stage.reranker, s.stage.candidateCount, s.stage.topK,
      s.p50.toFixed(3), s.p95.toFixed(3), s.p99.toFixed(3), s.meanMs.toFixed(3), s.cv.toFixed(4),
      s.bytesPerQuery.toFixed(0), s.quality.ndcg.toFixed(4), s.quality.recall.toFixed(4),
      s.quality.mrr.toFixed(4),
      String(s.violations.tenant + s.violations.owner + s.violations.expired),
    ].join(','));
  }
  return rows.join('\n');
}

export function renderCostsCsv(run: BenchRun): string {
  const rows = [
    'retrieval,candidate_count,where,pairs,control_p50_ms,treatment_p50_ms,scoring_median_delta_ms,scoring_mean_delta_ms,ci_lower_ms,ci_upper_ms,direct_median_ms',
  ];
  for (const c of scoringCosts(run)) {
    rows.push([
      c.retrieval, c.candidateCount, c.reranker, c.pairs,
      c.controlP50.toFixed(3), c.treatmentP50.toFixed(3),
      c.medianDelta.toFixed(3), c.meanDelta.toFixed(3), c.ci.lower.toFixed(3), c.ci.upper.toFixed(3),
      c.directMedian === undefined ? '' : c.directMedian.toFixed(3),
    ].join(','));
  }
  for (const t of transferCosts(run)) {
    rows.push([
      t.retrieval, t.candidateCount, 'transfer', t.pairs, '', '',
      t.medianDelta.toFixed(3), '', t.ci.lower.toFixed(3), t.ci.upper.toFixed(3), '',
    ].join(','));
  }
  return rows.join('\n');
}
