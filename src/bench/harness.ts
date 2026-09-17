import { cpus, totalmem } from 'node:os';
import { app, oracle } from '../config.js';
import { describeOracle } from '../db/oracle.js';
import { AppReranker } from '../rerank/app.js';
import { FixtureReranker } from '../rerank/fixture.js';
import { InDbReranker } from '../rerank/indb.js';
import { OracleCandidateSource, type CandidateSource } from '../retrieval/candidates.js';
import { FixtureCandidateSource } from '../retrieval/fixture.js';
import type {
  BenchRun, Chunk, EnvironmentInfo, Query, RunConfig, Stage, StageRun,
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

export interface RunOptions {
  onProgress?: (msg: string) => void;
}

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
  const stageRuns: StageRun[] = [];

  for (const [index, stage] of stages.entries()) {
    const pipeline = pipelineFor(stage, deps);
    log(`[${index + 1}/${stages.length}] ${stage.label}`);

    // Warmup is per stage, not per run: the first execution of each distinct statement pays
    // for a hard parse, and the first inference pays for model load and allocator warmup.
    for (let w = 0; w < cfg.warmup; w++) {
      for (const query of queries) await pipeline.run(query);
    }

    const iterations = [];
    for (let i = 0; i < cfg.iterations; i++) {
      for (const query of queries) {
        const result = await pipeline.run(query);
        result.iteration = i;
        iterations.push(result);
      }
    }
    stageRuns.push({ stage, iterations });
  }

  return {
    startedAt,
    finishedAt: new Date().toISOString(),
    config: cfg,
    environment,
    stages: stageRuns,
  };
}
