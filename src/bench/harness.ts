import { execSync } from 'node:child_process';
import { cpus, totalmem } from 'node:os';
import { setTimeout as sleep } from 'node:timers/promises';
import { app, oracle } from '../config.js';
import { closePool, describeOracle } from '../db/oracle.js';
import { AppReranker } from '../rerank/app.js';
import { FixtureReranker } from '../rerank/fixture.js';
import { InDbReranker } from '../rerank/indb.js';
import { OracleCandidateSource, type CandidateSource } from '../retrieval/candidates.js';
import { FixtureCandidateSource } from '../retrieval/fixture.js';
import type {
  BenchRun, Chunk, EnvironmentInfo, IterationResult, Query, RunConfig, Stage,
} from '../types.js';
import { AppRerankPipeline, InDbRerankPipeline, RetrievalOnlyPipeline, type Pipeline } from './pipelines.js';

export interface HarnessDeps {
  source: CandidateSource;
  appReranker?: AppReranker | FixtureReranker;
  indbReranker?: InDbReranker;
}

export function buildDeps(cfg: RunConfig, chunks: readonly Chunk[]): HarnessDeps {
  if (cfg.backend === 'fixture') {
    return {
      source: new FixtureCandidateSource(chunks, cfg.rrfK),
      appReranker: new FixtureReranker(),
    };
  }
  const deps: HarnessDeps = { source: new OracleCandidateSource(cfg.rrfK) };
  if (cfg.rerankers.includes('app')) deps.appReranker = new AppReranker();
  if (cfg.rerankers.includes('in-db')) deps.indbReranker = new InDbReranker(cfg.rrfK);
  return deps;
}

export function pipelineFor(stage: Stage, deps: HarnessDeps): Pipeline {
  if (stage.role === 'control') {
    // A control is the treatment without scoring. In the database that is the same statement
    // with the model call swapped out; in the application it is the candidate fetch alone,
    // because the app-side treatment is exactly that fetch followed by scoring.
    if (stage.reranker === 'in-db') {
      if (!deps.indbReranker) throw new Error('In-database reranking requires the oracle backend.');
      return new InDbRerankPipeline(stage, deps.indbReranker, true);
    }
    return new RetrievalOnlyPipeline(stage, deps.source);
  }
  switch (stage.reranker) {
    case 'none':
      return new RetrievalOnlyPipeline(stage, deps.source);
    case 'in-db': {
      if (!deps.indbReranker) throw new Error('In-database reranking requires the oracle backend.');
      return new InDbRerankPipeline(stage, deps.indbReranker);
    }
    case 'app':
    case 'fixture': {
      // The fixture backend substitutes its own reranker behind the same stage id, so reports
      // from both backends have the same shape.
      if (!deps.appReranker) throw new Error(`No ${stage.reranker} reranker available.`);
      return new AppRerankPipeline(stage, deps.source, deps.appReranker);
    }
  }
}

export async function captureEnvironment(cfg: RunConfig): Promise<EnvironmentInfo> {
  const cpuList = cpus();
  const env: EnvironmentInfo = {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: cpuList.length,
    cpuModel: cpuList[0]?.model ?? 'unknown',
    totalMemMb: Math.round(totalmem() / 1024 / 1024),
  };
  if (cfg.backend === 'oracle') {
    const info = await describeOracle();
    env.oracle = {
      version: info.version,
      banner: info.banner,
      clientMode: info.clientMode,
      rerankModel: oracle.rerankModel,
      embedModel: oracle.embedModel,
      indbRerankApi: oracle.indbRerankApi,
    };
  }
  if (cfg.rerankers.includes('app')) {
    env.app = {
      modelPath: cfg.backend === 'fixture' ? '(fixture reranker, no model)' : app.modelPath,
      executionProviders: cfg.backend === 'fixture' ? ['fixture'] : [app.device],
      intraOpThreads: app.intraOpThreads ?? 'default',
      dtype: cfg.backend === 'fixture' ? 'n/a' : app.dtype,
    };
  }
  return env;
}

export interface ParityReport {
  checked: number;
  mismatches: Array<{ queryId: string; candidateCount: number; onlyInDb: string[]; onlyInApp: string[] }>;
}

/**
 * Prove the two reranking paths are scoring the same candidates.
 *
 * Without this, a difference in the final ordering could be retrieval drift rather than the
 * reranker, and the whole comparison would be worthless. The in-database statement is asked to
 * return its full candidate depth instead of the top K, and the resulting identifier set is
 * compared to what the application path fetched.
 */
export async function verifyCandidateParity(
  queries: readonly Query[],
  cfg: RunConfig,
  deps: HarnessDeps,
): Promise<ParityReport> {
  const report: ParityReport = { checked: 0, mismatches: [] };
  if (!deps.indbReranker || cfg.backend !== 'oracle') return report;

  for (const retrieval of cfg.retrievals) {
    for (const n of cfg.candidateCounts) {
      if (n < cfg.topK) continue;
      for (const query of queries) {
        const appBatch = await deps.source.generate(query, retrieval, n);
        // topK = n makes the reranked statement return its entire candidate set.
        const indb = await deps.indbReranker.rerank(query, retrieval, n, n);
        const appIds = new Set(appBatch.candidates.map((c) => c.chunkId));
        const dbIds = new Set(indb.results.map((r) => r.chunkId));
        const onlyInDb = [...dbIds].filter((id) => !appIds.has(id));
        const onlyInApp = [...appIds].filter((id) => !dbIds.has(id));
        report.checked++;
        if (onlyInDb.length || onlyInApp.length) {
          report.mismatches.push({ queryId: query.id, candidateCount: n, onlyInDb, onlyInApp });
        }
      }
    }
  }
  return report;
}

/** Compact duration for progress lines: seconds under a minute, else m:ss. */
function secs(ms: number): string {
  const total = Math.round(ms / 1000);
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m${String(total % 60).padStart(2, '0')}s`;
}

export interface RunOptions {
  onProgress?: (msg: string) => void;
}

/**
 * Put the process and its connections back to a cold-ish state between isolation batches.
 *
 * The connection pool is closed (the next statement opens a fresh one), the application model
 * is disposed (the next scoring call reloads it), the heap is collected if the runtime allows,
 * an external command runs if one is configured - a container restart is the obvious use -
 * and then the harness idles so that spinning ONNX Runtime threads, CPU boost state and the
 * like have settled. Every unit warms up again after this, so none of it lands in a measurement.
 */
export async function resetBetweenBatches(
  cfg: RunConfig,
  deps: HarnessDeps,
  log: (msg: string) => void,
): Promise<void> {
  log(`  reset: closing connections and disposing the model`);
  if (cfg.backend === 'oracle') await closePool();
  if (deps.appReranker) await deps.appReranker.close();
  const gc = (globalThis as { gc?: () => void }).gc;
  if (gc) gc();
  if (cfg.resetCommand) {
    log(`  reset: running ${JSON.stringify(cfg.resetCommand)}`);
    execSync(cfg.resetCommand, { stdio: 'inherit' });
  }
  if (cfg.quiesceMs > 0) {
    log(`  reset: quiescing for ${cfg.quiesceMs} ms`);
    await sleep(cfg.quiesceMs);
  }
}

/** Grouped stages become one unit; ungrouped stages are a unit of one. Declaration order is kept. */
function unitsOf(stages: readonly Stage[]): Stage[][] {
  const units: Stage[][] = [];
  const groupIndex = new Map<string, number>();
  for (const stage of stages) {
    if (!stage.group) { units.push([stage]); continue; }
    const at = groupIndex.get(stage.group);
    if (at === undefined) { groupIndex.set(stage.group, units.length); units.push([stage]); }
    else units[at]!.push(stage);
  }
  return units;
}

/**
 * Run every stage and collect timings.
 *
 * The protocol, per repeat: batches run in order with a reset between them. Within a batch,
 * a group (a treatment and its control) is run interleaved: within each iteration every
 * query goes through every pipeline in the group back to back, order rotating per iteration.
 * That is what makes the differences between them paired - each treatment timing has a
 * control timing taken moments apart under the same load, cache state and clock speed, so
 * drift over the course of a run cancels instead of becoming a systematic error.
 *
 * Batches are what keep the arms apart. Nothing is subtracted across a batch boundary, so
 * whatever one arm leaves behind - spinning inference threads, a hot buffer cache - is reset
 * before the other is measured.
 */
export async function runBenchmark(
  stages: readonly Stage[],
  queries: readonly Query[],
  cfg: RunConfig,
  deps: HarnessDeps,
  options: RunOptions = {},
): Promise<BenchRun> {
  const log = options.onProgress ?? (() => {});
  const startedAt = new Date().toISOString();
  const environment = await captureEnvironment(cfg);
  const iterationsByStage = new Map<string, IterationResult[]>(stages.map((s) => [s.id, []]));

  const batches: string[] = [];
  for (const s of stages) if (!batches.includes(s.batch)) batches.push(s.batch);

  let first = true;
  for (let repeat = 0; repeat < cfg.repeats; repeat++) {
    if (cfg.repeats > 1) log(`--- repeat ${repeat + 1}/${cfg.repeats} ---`);
    for (const batch of batches) {
      if (!first) await resetBetweenBatches(cfg, deps, log);
      first = false;
      const batchStages = stages.filter((s) => s.batch === batch);
      log(`batch "${batch}": ${batchStages.length} stages`);

      for (const unit of unitsOf(batchStages)) {
        const pipelines = unit.map((stage) => ({ stage, pipeline: pipelineFor(stage, deps) }));
        log(`  ${unit.map((s) => s.label).join('  |  ')}`);

        // Warmup is per unit, after any reset: the first execution of each statement pays for
        // a hard parse, and the first inference pays for model load and allocator warmup.
        const warmupStarted = performance.now();
        for (let w = 0; w < cfg.warmup; w++) {
          for (const query of queries) {
            for (const { pipeline } of pipelines) await pipeline.run(query);
          }
        }
        if (cfg.warmup > 0) {
          log(`    warmup: ${cfg.warmup} pass(es) in ${secs(performance.now() - warmupStarted)}`);
        }

        const measureStarted = performance.now();
        for (let i = 0; i < cfg.iterations; i++) {
          const offset = i % pipelines.length;
          const order = [...pipelines.slice(offset), ...pipelines.slice(0, offset)];
          for (const query of queries) {
            for (const { stage, pipeline } of order) {
              const result = await pipeline.run(query);
              result.iteration = i;
              result.repeat = repeat;
              iterationsByStage.get(stage.id)!.push(result);
            }
          }
          // A cross-encoder over a deep candidate pool takes minutes per unit. Without a line
          // per iteration there is no way to tell a slow run from a hung one.
          const elapsed = performance.now() - measureStarted;
          const remaining = (elapsed / (i + 1)) * (cfg.iterations - i - 1);
          log(
            `    iteration ${i + 1}/${cfg.iterations} · ${secs(elapsed)} elapsed`
            + (i + 1 < cfg.iterations ? ` · ~${secs(remaining)} left` : ''),
          );
        }
      }
    }
  }

  return {
    startedAt,
    finishedAt: new Date().toISOString(),
    config: cfg,
    environment,
    stages: stages.map((stage) => ({ stage, iterations: iterationsByStage.get(stage.id)! })),
  };
}
