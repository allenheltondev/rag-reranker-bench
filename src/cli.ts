import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { app, buildStages, oracle, paths, runConfigFromEnv } from './config.js';
import { generateCorpus } from './corpus/generate.js';
import { closePool, describeOracle, withConnection } from './db/oracle.js';
import { countChunks, loadChunks, runScript } from './db/load.js';
import { AppReranker } from './rerank/app.js';
import { scoreExpr } from './rerank/indb.js';
import { buildDeps, pipelineFor, runBenchmark, verifyCandidateParity } from './bench/harness.js';
import { renderCostsCsv, renderCsv, renderMarkdown } from './bench/report.js';
import type { BenchRun, Chunk, Query, RunConfig } from './types.js';

const args = process.argv.slice(2);
const command = args[0] ?? 'help';
const flag = (name: string): boolean => args.includes(`--${name}`);
const opt = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};

const log = (msg = ''): void => { process.stdout.write(`${msg}\n`); };

function overridesFromFlags(): Partial<RunConfig> {
  const o: Partial<RunConfig> = {};
  const backend = opt('backend');
  if (backend === 'oracle' || backend === 'fixture') o.backend = backend;
  const iterations = opt('iterations');
  if (iterations) o.iterations = Number(iterations);
  const warmup = opt('warmup');
  if (warmup) o.warmup = Number(warmup);
  const topK = opt('top-k');
  if (topK) o.topK = Number(topK);
  const candidates = opt('candidates');
  if (candidates) o.candidateCounts = candidates.split(',').map(Number);
  const rerankers = opt('rerankers');
  if (rerankers) o.rerankers = rerankers.split(',') as RunConfig['rerankers'];
  const retrievals = opt('retrievals');
  if (retrievals) o.retrievals = retrievals.split(',') as RunConfig['retrievals'];
  const queries = opt('queries');
  if (queries) o.queries = Number(queries);
  return o;
}

function readCorpus(): { chunks: Chunk[]; queries: Query[] } {
  try {
    return {
      chunks: JSON.parse(readFileSync(paths.corpus, 'utf8')) as Chunk[],
      queries: JSON.parse(readFileSync(paths.queries, 'utf8')) as Query[],
    };
  } catch {
    throw new Error(`Corpus not found at ${paths.corpus}. Run \`npm run corpus\` first.`);
  }
}

async function cmdCorpus(): Promise<void> {
  const cfg = runConfigFromEnv(overridesFromFlags());
  const { chunks, queries } = generateCorpus(cfg.corpusSize, cfg.seed);
  mkdirSync(resolve(paths.corpus, '..'), { recursive: true });
  writeFileSync(paths.corpus, `${JSON.stringify(chunks, null, 2)}\n`);
  writeFileSync(paths.queries, `${JSON.stringify(queries, null, 2)}\n`);

  const graded = queries.reduce((a, q) => a + Object.keys(q.judgments).length, 0);
  const expired = chunks.filter((c) => c.expiresAt !== null).length;
  const otherTenant = chunks.filter((c) => c.tenant !== 'acme').length;
  log(`Wrote ${chunks.length} chunks to ${paths.corpus}`);
  log(`Wrote ${queries.length} queries to ${paths.queries}`);
  log(`  graded chunks: ${graded}`);
  log(`  expired chunks (must be filtered out): ${expired}`);
  log(`  other-tenant chunks (must never be returned): ${otherTenant}`);
  const byKind = queries.reduce<Record<string, number>>((acc, q) => {
    acc[q.kind] = (acc[q.kind] ?? 0) + 1;
    return acc;
  }, {});
  log(`  queries by hazard: ${Object.entries(byKind).map(([k, v]) => `${k}=${v}`).join(' ')}`);
}

async function cmdLoad(): Promise<void> {
  const { chunks } = readCorpus();
  log('Creating schema...');
  await runScript('01_schema.sql');
  log(`Loading ${chunks.length} chunks and embedding them with ${oracle.embedModel}...`);
  const inserted = await loadChunks(chunks);
  log(`Inserted ${inserted} rows; table now holds ${await countChunks()}.`);
  if (flag('vector-index')) {
    log('Creating the approximate vector index...');
    await runScript('02_vector_index.sql');
  } else {
    log('Skipped the approximate vector index (exact search keeps recall out of the comparison).');
    log('Pass --vector-index to create it anyway.');
  }
}

/** Check every moving part before a run, and say precisely which one is not ready. */
async function cmdDoctor(): Promise<void> {
  let failures = 0;
  const check = async (name: string, fn: () => Promise<string>): Promise<void> => {
    try {
      log(`  ok   ${name}: ${await fn()}`);
    } catch (err) {
      failures++;
      log(`  FAIL ${name}: ${(err as Error).message.split('\n')[0]}`);
    }
  };

  log('Corpus');
  await check('data files', async () => {
    const { chunks, queries } = readCorpus();
    return `${chunks.length} chunks, ${queries.length} queries`;
  });

  if (!flag('skip-oracle')) {
    log('Oracle');
    await check('connection', async () => {
      const info = await describeOracle();
      return `${info.version} (${info.clientMode} client)`;
    });
    await check('chunk table', async () => `${await countChunks()} rows`);
    await check('models loaded', async () => {
      const info = await describeOracle();
      const want = [oracle.embedModel.toUpperCase(), oracle.rerankModel.toUpperCase()];
      const missing = want.filter((m) => !info.models.includes(m));
      if (missing.length) throw new Error(`missing ${missing.join(', ')} — run sql/03_load_models.sql`);
      return info.models.join(', ');
    });
    await check('embedding works', async () => withConnection(async (conn) => {
      const r = await conn.execute<[unknown]>(
        `SELECT VECTOR_EMBEDDING(${oracle.embedModel.toUpperCase()} USING 'hello' AS DATA) FROM DUAL`,
      );
      return r.rows?.[0] ? `returned a vector (expecting ${oracle.embedDims} dims)` : 'no row';
    }));
    // The most likely thing to be wrong, and the most expensive to discover mid-run.
    await check(`in-DB scoring via ${oracle.indbRerankApi}`, async () => withConnection(async (conn) => {
      const r = await conn.execute<[number]>(
        `SELECT ${scoreExpr().replace(/TITLE \|\| '\. ' \|\| CONTENT/, `'the sky is blue'`)} FROM DUAL`,
        { qtext: 'what colour is the sky' },
      );
      const score = r.rows?.[0]?.[0];
      if (typeof score !== 'number') throw new Error(`expected a number, got ${typeof score}`);
      return `scored ${score.toFixed(4)}`;
    }));
  }

  if (!flag('skip-app')) {
    log('Application reranker');
    await check('model loads', async () => {
      const reranker = new AppReranker();
      await reranker.load();
      const out = await reranker.rerank('what colour is the sky', [
        { chunkId: 'a', title: 'Sky', content: 'The sky is blue.', rank: 1, score: 1 },
        { chunkId: 'b', title: 'Grass', content: 'Grass is green.', rank: 2, score: 0.5 },
      ], 2);
      await reranker.close();
      const top = out.results[0]?.chunkId;
      if (top !== 'a') throw new Error(`ranked '${top}' above the matching passage — check the score expression`);
      return `${app.modelPath} scored 2 pairs correctly`;
    });
  }

  log('');
  log(failures === 0 ? 'All checks passed.' : `${failures} check(s) failed.`);
  if (failures > 0) process.exitCode = 1;
}

async function cmdBench(): Promise<void> {
  const cfg = runConfigFromEnv(overridesFromFlags());
  // There is no database in fixture mode, so there is nothing for an in-database stage to
  // mean. Drop it loudly rather than failing several stages into a run.
  if (cfg.backend === 'fixture' && cfg.rerankers.includes('in-db')) {
    cfg.rerankers = cfg.rerankers.filter((r) => r !== 'in-db');
    log('Fixture backend: skipping in-database stages (they need Oracle).');
  }
  const { chunks, queries: allQueries } = readCorpus();
  const queries = cfg.queries > 0 ? allQueries.slice(0, cfg.queries) : allQueries;
  const stages = buildStages(cfg);
  const deps = buildDeps(cfg, chunks);

  if (flag('dump-sql')) {
    for (const stage of stages) {
      const sql = pipelineFor(stage, deps).sql?.();
      log(`\n${'='.repeat(78)}\n-- ${stage.id}\n${'='.repeat(78)}`);
      log(sql ?? '(fixture backend: no SQL)');
    }
    await closePool().catch(() => {});
    return;
  }

  log(`Backend: ${cfg.backend} · ${queries.length} queries · ${stages.length} stages · ${cfg.iterations} iterations (+${cfg.warmup} warmup)`);
  log('');

  let parity;
  if (cfg.backend === 'oracle' && cfg.rerankers.includes('in-db') && cfg.rerankers.includes('app')) {
    log('Verifying both reranking paths see identical candidates...');
    parity = await verifyCandidateParity(queries, cfg, deps);
    log(parity.mismatches.length === 0
      ? `  ${parity.checked} combinations checked, all identical.`
      : `  WARNING: ${parity.mismatches.length}/${parity.checked} combinations differ. The comparison is not controlled.`);
    log('');
  }

  const run = await runBenchmark(stages, queries, cfg, deps, { onProgress: log });

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const dir = resolve(paths.results, stamp);
  mkdirSync(dir, { recursive: true });
  const payload: BenchRun & { parity?: typeof parity } = { ...run };
  if (parity) payload.parity = parity;
  writeFileSync(resolve(dir, 'raw.json'), `${JSON.stringify(payload, null, 2)}\n`);
  writeFileSync(resolve(dir, 'summary.md'), renderMarkdown(run, queries, parity, chunks));
  writeFileSync(resolve(dir, 'summary.csv'), `${renderCsv(run, queries, chunks)}\n`);
  writeFileSync(resolve(dir, 'scoring-costs.csv'), `${renderCostsCsv(run)}\n`);

  log('');
  log(`Results written to ${dir}`);
  log('');
  log(renderMarkdown(run, queries, parity, chunks));
  await closePool().catch(() => {});
}

async function cmdReport(): Promise<void> {
  const file = opt('run') ?? args[1];
  if (!file) throw new Error('Usage: npm run report -- --run results/<stamp>/raw.json');
  const run = JSON.parse(readFileSync(resolve(file), 'utf8')) as BenchRun & { parity?: never };
  const { chunks, queries } = readCorpus();
  log(renderMarkdown(run, queries, run.parity, chunks));
}

function cmdHelp(): void {
  log(`rag-reranker-bench

  npm run corpus                      Generate the corpus and query set into data/
  npm run load                        Create the schema and load the corpus into Oracle
  npm run doctor                      Check Oracle, the models, and the app reranker
  npm run bench                       Run the benchmark
  npm run report -- --run <raw.json>  Re-render a report from a previous run

Flags for bench:
  --backend oracle|fixture   Where retrieval and reranking run (default: oracle)
  --retrievals a,b,c         vector, lexical, hybrid-rrf
  --rerankers a,b,c          none, in-db, app
  --candidates 10,20,40,80   Candidate pool sizes to sweep
  --top-k 10                 Results returned to the model
  --iterations 20            Measured iterations per query per stage
  --warmup 5                 Unmeasured iterations per query per stage
  --queries N                Use only the first N queries
  --dump-sql                 Print the rendered SQL for every stage and exit

Flags for doctor:
  --skip-oracle              Skip the database checks
  --skip-app                 Skip the application reranker checks
`);
}

const commands: Record<string, () => Promise<void> | void> = {
  corpus: cmdCorpus,
  load: cmdLoad,
  doctor: cmdDoctor,
  bench: cmdBench,
  report: cmdReport,
  help: cmdHelp,
};

const handler = commands[command];
if (!handler) {
  log(`Unknown command: ${command}`);
  cmdHelp();
  process.exit(1);
}

try {
  await handler();
} catch (err) {
  log('');
  log(`Error: ${(err as Error).message}`);
  await closePool().catch(() => {});
  process.exit(1);
}
